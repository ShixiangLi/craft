"""使用真实 Crafter 和模拟 Ollama 响应验证实验边界；不消耗模型资源。"""
from contextlib import redirect_stdout
from io import BytesIO, StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError

from agents.naive import NaiveAgent
from engine.environment import create_environment
from engine.evaluator import task_succeeded
from engine.experiment import run_experiment
from modules.common.llm import LLMClient
from modules.common.observation import describe_observation
from utils.config import PROJECT_ROOT, load_config


def ollama_response(action="noop"):
    return BytesIO(json.dumps({"message": {"content": json.dumps({"action": action})},
                              "done": True, "model": "test", "prompt_eval_count": 10,
                              "eval_count": 3}).encode())


class NaiveFlowTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = load_config(PROJECT_ROOT / "configs/naive.yaml")
        self.config["output_dir"] = self.temp.name
        self.config["experiment"].update(max_steps=3, max_model_calls=3)
        self.config["task"]["success_condition"] = None

    def run_quietly(self):
        with redirect_stdout(StringIO()):
            return run_experiment(self.config)

    def test_real_environment_flow_and_step_limit(self):
        self.config["experiment"]["seeds"] = [0, 1]
        with patch("modules.common.llm.opener.open", side_effect=lambda *a, **k: ollama_response()) as send:
            summary = self.run_quietly()
        self.assertEqual(send.call_count, 6)
        self.assertEqual(summary["episodes"], 2)
        self.assertIsNone(summary["success_rate"])
        self.assertEqual(summary["total_input_tokens"], 60)
        output = Path(summary["output_dir"])
        self.assertTrue((output / "config.yaml").is_file())
        self.assertTrue((output / "metadata.json").is_file())
        self.assertTrue((output / "summary.json").is_file())
        for result in summary["results"]:
            self.assertEqual(result["stop_reason"], "max_steps")
            records = [json.loads(line) for line in (output / result["episode_id"] / "trajectory.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 4)
            self.assertEqual(records[0]["type"], "reset")
            self.assertTrue(records[-1]["truncated"])
            self.assertFalse(records[-1]["terminated"])
            self.assertEqual(records[-1]["observation"]["step"], 3)
        request = send.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(request.full_url, "http://localhost:11434/api/chat")
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["options"]["seed"], 1)
        self.assertIn("noop", payload["format"]["properties"]["action"]["enum"])

    def test_call_limit_stops_before_another_request(self):
        self.config["experiment"]["max_model_calls"] = 1
        with patch("modules.common.llm.opener.open", side_effect=lambda *a, **k: ollama_response()) as send:
            summary = self.run_quietly()
        self.assertEqual(send.call_count, 1)
        self.assertEqual(summary["results"][0]["stop_reason"], "max_model_calls")
        self.assertEqual(summary["results"][0]["steps"], 1)

    def test_invalid_action_keeps_partial_trajectory_and_error(self):
        with patch("modules.common.llm.opener.open", side_effect=[ollama_response(), ollama_response("teleport")]):
            with self.assertRaisesRegex(ValueError, "非法动作"):
                self.run_quietly()
        run_dir = next(Path(self.temp.name).iterdir())
        self.assertTrue((run_dir / "error.json").is_file())
        self.assertFalse((run_dir / "summary.json").exists())
        records = [json.loads(line) for line in next(run_dir.glob("*/trajectory.jsonl")).read_text().splitlines()]
        self.assertEqual([r["type"] for r in records], ["reset", "step", "error"])
        self.assertIn("teleport", records[-1]["decision"]["response"])

    def test_connection_failure_is_not_reported_as_success(self):
        with patch("modules.common.llm.opener.open", side_effect=URLError("offline")):
            with self.assertRaisesRegex(RuntimeError, "无法调用 Ollama"):
                self.run_quietly()
        run_dir = next(Path(self.temp.name).iterdir())
        self.assertTrue((run_dir / "error.json").exists())
        self.assertFalse((run_dir / "summary.json").exists())

    def test_same_seed_resets_world_and_local_observation(self):
        env = create_environment(self.config["environment"])
        self.addCleanup(env.close)
        first = env.reset(seed=7)
        env.step(1)
        second = env.reset(seed=7)
        self.assertEqual(first, second)
        self.assertEqual(len(first["local_map"]), 7)
        self.assertEqual(len(first["local_map"][0]), 9)
        self.assertEqual(first["local_map"][3][4], "player")
        self.assertNotIn("player_pos", first)
        self.assertNotIn("semantic", first)

    def test_success_and_death_stop_without_extra_actions(self):
        # 用终止反馈夹具覆盖真实环境的一步，验证 runner 对停止条件的处理。
        original_step = __import__("engine.environment", fromlist=["CrafterEnvironment"]).CrafterEnvironment.step
        for terminal in (False, True):
            with self.subTest(terminated=terminal):
                self.config["task"]["success_condition"] = {"achievement": "collect_wood", "count": 1}

                def step(env, action):
                    transition = original_step(env, action)
                    if terminal:
                        transition["terminated"] = True
                        transition["observation"]["inventory"]["health"] = 0
                    else:
                        transition["observation"]["achievements"]["collect_wood"] = 1
                    return transition

                with patch("engine.environment.CrafterEnvironment.step", step), patch("modules.common.llm.opener.open", side_effect=lambda *a, **k: ollama_response()) as send:
                    summary = self.run_quietly()
                result = summary["results"][0]
                self.assertEqual(send.call_count, 1)
                self.assertEqual(result["success"], not terminal)
                self.assertEqual(result["stop_reason"], "terminated" if terminal else "success")

    def test_agent_reset_clears_previous_feedback(self):
        agent = NaiveAgent(self.config["agent"], LLMClient(self.config["model"]))
        agent.reset(self.config["task"], seed=0)
        agent.observe({"action": "noop", "reward": 0, "new_achievements": []})
        agent.llm.calls = 7
        agent.reset(self.config["task"], seed=1)
        self.assertIsNone(agent.previous)
        self.assertEqual(agent.llm.calls, 0)

    def test_text_observation_preserves_orientation_and_state(self):
        env = create_environment(self.config["environment"])
        self.addCleanup(env.close)
        obs = env.reset(seed=0)
        # 不对称地图夹具可发现 x/y 颠倒或上下左右互换。
        obs["local_map"] = [["grass", "tree", "grass"],
                            ["water", "player", "stone"],
                            ["grass", "cow", "grass"]]
        for facing, target in [([0, -1], "tree"), ([0, 1], "cow"),
                               ([-1, 0], "water"), ([1, 0], "stone")]:
            obs["facing"] = facing
            text = describe_observation(obs)
            self.assertIn(f"adjacent target: {target}", text)
            self.assertIn("dy=-1: grass | tree | grass", text)
            self.assertIn("dy=+1: grass | cow | grass", text)
            self.assertIn("health=9/9", text)
        self.assertEqual(obs["inventory"]["wood"], 0)
        self.assertEqual(obs["achievements"]["collect_wood"], 0)

    def test_prompt_contains_configured_goal_and_success_condition(self):
        env = create_environment(self.config["environment"])
        self.addCleanup(env.close)
        observation = env.reset(seed=0)
        task = {"description": "Collect two wood", "success_condition": {
            "achievement": "collect_wood", "count": 2}}
        agent = NaiveAgent(self.config["agent"], LLMClient(self.config["model"]))
        agent.reset(task, seed=0)
        with patch("modules.common.llm.opener.open", side_effect=lambda *a, **k: ollama_response()) as send:
            agent.act(observation)
        content = json.loads(send.call_args.args[0].data)["messages"][-1]["content"]
        self.assertIn(task["description"], content)
        self.assertIn('"count": 2', content)
        self.assertIn("adjacent target:", content)
        self.assertNotIn("$observation", content)
        self.assertNotIn("$success_condition", content)

    def test_success_uses_environment_counts(self):
        task = {"success_condition": {"achievement": "collect_wood", "count": 2}}
        self.assertFalse(task_succeeded({"achievements": {"collect_wood": 1}}, task))
        self.assertTrue(task_succeeded({"achievements": {"collect_wood": 2}}, task))


if __name__ == "__main__":
    unittest.main()
