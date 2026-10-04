"""ADaPT 执行优先、递归短路、真实反馈与统一实验预算的行为验证。"""
from contextlib import redirect_stdout
from copy import deepcopy
from io import BytesIO, StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agents.adapt import AdaptAgent
from agents.base import AgentFinished
from engine.environment import create_environment
from engine.experiment import run_experiment
from modules.common.llm import ModelCallBudgetExceeded
from utils.config import PROJECT_ROOT, load_config


GOAL = "Collect at least one unit of diamond while staying alive."


def decision(action, thought=""):
    return {"thought": thought, "action": action}


def plan(subtasks, logic):
    return {"thought": "Decompose after the executor could not finish.",
            "subtasks": subtasks, "logic": logic}


class FakeLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []
        self.max_calls = None
        self.reset_stats()

    def reset_stats(self):
        self.calls = 0

    def generate(self, messages, **kwargs):
        if self.max_calls is not None and self.calls >= self.max_calls:
            raise ModelCallBudgetExceeded("test call budget")
        self.calls += 1
        self.requests.append((deepcopy(messages), kwargs))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response if isinstance(response, str) else json.dumps(response)


def make_agent(responses, **params):
    config = load_config(PROJECT_ROOT / "configs/adapt.yaml")["agent"]
    config["params"].update(max_depth=3, max_executor_calls=5, max_subtasks=5,
                            max_history_steps=None)
    config["params"].update(params)
    agent = AdaptAgent(config, FakeLLM(responses))
    agent.reset({"description": GOAL, "success_condition": {
        "achievement": "collect_diamond", "count": 1}}, seed=7)
    return agent


def prompt_text(request):
    return "\n".join(message["content"] for message in request[0])


class AdaptAgentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env = create_environment({"name": "crafter", "params": {}})
        try:
            cls.initial_observation = env.reset(seed=0)
        finally:
            env.close()

    def setUp(self):
        self.observation = deepcopy(self.initial_observation)

    def feedback(self, agent, observation, action_id, **inventory):
        observation = deepcopy(observation)
        observation["step"] += 1
        observation["inventory"].update(inventory)
        agent.observe({"observation": observation,
                       "action": observation["actions"][action_id],
                       "reward": 0.25, "new_achievements": [],
                       "terminated": False, "truncated": False})
        return observation

    def assert_finished(self, agent, observation, completed):
        with self.assertRaises(AgentFinished) as stopped:
            agent.act(observation)
        self.assertIs(stopped.exception.completed, completed)

    def test_executor_runs_before_planning_and_completion_is_internal(self):
        agent = make_agent([decision("task_completed", "The observed task is complete.")])
        self.assert_finished(agent, self.observation, True)
        self.assertEqual(agent.llm.calls, 1)
        self.assertEqual(agent.llm.requests[0][1]["label"], "adapt_execute")
        schema = agent.llm.requests[0][1]["response_schema"]
        choices = schema["properties"]["action"]["enum"]
        self.assertEqual(set(choices), set(self.observation["actions"]) |
                         {"task_completed", "task_failed"})
        self.assertEqual(agent.history, [])
        self.assertIsNone(agent.pending)

    def test_mixed_logic_short_circuits_without_spending_recursion_depth(self):
        names = ["Locate a safe tree.", "Use route alpha.",
                 "Use route beta.", "Collect a wood unit."]
        agent = make_agent([
            decision("task_failed"), plan(names, "1 AND (2 OR 3) AND 4"),
            decision("task_completed"), decision("task_failed"),
            decision("task_completed"), decision("noop"),
            decision("task_completed"),
        ], max_depth=2)
        action = agent.act(self.observation)
        self.assertEqual(self.observation["actions"][action], "noop")
        self.assertEqual(agent.task_path, [GOAL, names[3]])
        observation = self.feedback(agent, self.observation, action)
        self.assert_finished(agent, observation, True)
        labels = [kwargs["label"] for _, kwargs in agent.llm.requests]
        self.assertEqual(labels, ["adapt_execute", "adapt_plan"] + ["adapt_execute"] * 5)
        # A single successful decomposition returns to the caller directly;
        # the parent executor is not replayed and no extra plan is generated.
        self.assertEqual(agent.llm.calls, 7)

    def test_failed_and_branch_skips_remaining_child_and_tries_or_alternative(self):
        names = ["Attempt route alpha.", "This dependent route must be skipped.",
                 "Try an independent route beta."]
        agent = make_agent([
            decision("task_failed"), plan(names, "(1 AND 2) OR 3"),
            decision("task_failed"), decision("task_completed"),
        ], max_depth=2)
        self.assert_finished(agent, self.observation, True)
        self.assertEqual(agent.llm.calls, 4)
        last = prompt_text(agent.llm.requests[-1])
        self.assertIn(names[2], last)
        self.assertEqual(sum(kw["label"] == "adapt_plan"
                             for _, kw in agent.llm.requests), 1)

    def test_successful_or_branch_skips_alternative(self):
        agent = make_agent([
            decision("task_failed"), plan(["Try first route.", "Try second route."], "1 OR 2"),
            decision("task_completed"),
        ], max_depth=2)
        self.assert_finished(agent, self.observation, True)
        self.assertEqual(agent.llm.calls, 3)

    def test_depth_limit_counts_tasks_with_root_at_one(self):
        agent = make_agent([decision("task_failed")], max_depth=1)
        self.assert_finished(agent, self.observation, False)
        self.assertEqual(agent.llm.calls, 1)
        agent = make_agent([
            decision("task_failed"), plan(["Get a tool."], "1"),
            decision("task_failed"), plan(["Collect wood."], "1"),
            decision("task_failed"),
        ], max_depth=3)
        self.assert_finished(agent, self.observation, False)
        self.assertEqual(agent.llm.calls, 5)
        last = prompt_text(agent.llm.requests[-1])
        for task in (GOAL, "Get a tool.", "Collect wood."):
            self.assertIn(task, last)

    def test_executor_budget_triggers_one_plan_and_resets_child_history(self):
        root_thought = "Unique root-only trajectory thought."
        agent = make_agent([
            decision("noop", root_thought), plan(["Gather some wood."], "1"),
            decision("noop", "Work on the child task."),
        ], max_executor_calls=1, max_depth=2)
        action = agent.act(self.observation)
        observation = self.feedback(agent, self.observation, action, wood=2)
        action = agent.act(observation)
        self.assertEqual(agent.task_path, [GOAL, "Gather some wood."])
        self.assertEqual(agent.history, [])
        planner = prompt_text(agent.llm.requests[1])
        child = prompt_text(agent.llm.requests[2])
        self.assertIn(root_thought, planner)
        self.assertIn('"reward": 0.25', planner)
        self.assertIn("Step: 1", planner)
        self.assertIn("wood=2", planner)
        self.assertNotIn(root_thought, child)
        self.assertIn("wood=2", child)
        self.assertIn(GOAL, child)
        observation = self.feedback(agent, observation, action)
        self.assert_finished(agent, observation, False)
        self.assertEqual(agent.llm.calls, 3)

    def test_or_fallback_observes_environment_changes_without_rollback(self):
        agent = make_agent([
            decision("task_failed"), plan(["Try route alpha.", "Try route beta."], "1 OR 2"),
            decision("noop", "Attempt alpha."), decision("task_failed"),
            decision("noop", "Attempt beta."),
        ], max_depth=2)
        action = agent.act(self.observation)
        observation = self.feedback(agent, self.observation, action, wood=3, health=5)
        agent.act(observation)
        self.assertEqual(agent.task_path, [GOAL, "Try route beta."])
        self.assertEqual(agent.history, [])
        fallback = prompt_text(agent.llm.requests[-1])
        self.assertIn("Step: 1", fallback)
        self.assertIn("wood=3", fallback)
        self.assertIn("health=5/9", fallback)
        self.assertNotIn("Thought: Attempt alpha.", fallback)

    def test_real_feedback_pending_guard_and_history_window(self):
        agent = make_agent([decision("noop", f"Unique observation thought {i}.")
                            for i in range(3)], max_history_steps=1)
        with self.assertRaisesRegex(RuntimeError, "act"):
            agent.observe({})
        action = agent.act(self.observation)
        self.assertEqual(agent.history, [])
        with self.assertRaisesRegex(RuntimeError, "observe"):
            agent.act(self.observation)
        self.assertEqual(agent.llm.calls, 1)
        observation = self.feedback(agent, self.observation, action)
        self.assertEqual(agent.history[0]["feedback"]["reward"], 0.25)
        action = agent.act(observation)
        next_prompt = prompt_text(agent.llm.requests[-1])
        self.assertIn("Unique observation thought 0.", next_prompt)
        self.assertIn('"reward": 0.25', next_prompt)
        observation = self.feedback(agent, observation, action)
        agent.act(observation)
        next_prompt = prompt_text(agent.llm.requests[-1])
        self.assertNotIn("Unique observation thought 0.", next_prompt)
        self.assertIn("Unique observation thought 1.", next_prompt)

    def test_reset_discards_suspended_plan_and_unexecuted_action(self):
        agent = make_agent([
            decision("task_failed"), plan(["Old child task."], "1"),
            decision("task_failed"), plan(["Old nested task."], "1"),
            decision("noop"), decision("task_completed"),
        ])
        agent.act(self.observation)
        self.assertIsNotNone(agent.pending)
        self.assertEqual(len(agent.task_path), 3)
        agent.reset({"description": "A completely new goal."}, seed=12)
        self.assertEqual(agent.history, [])
        self.assertIsNone(agent.pending)
        self.assertIsNone(agent.previous)
        self.assertEqual(agent.task_path, [])
        self.assertEqual(agent.llm.calls, 0)
        self.assert_finished(agent, self.observation, True)
        last = prompt_text(agent.llm.requests[-1])
        self.assertIn("A completely new goal.", last)
        self.assertNotIn("Old child task.", last)
        self.assertNotIn("Old nested task.", last)
        self.assertEqual(agent.llm.calls, 1)

    def test_global_budget_includes_executor_and_planner_calls(self):
        for budget in (1, 2):
            with self.subTest(budget=budget):
                agent = make_agent([
                    decision("task_failed"), plan(["Gather wood."], "1"),
                    decision("noop"),
                ])
                agent.llm.max_calls = budget
                with self.assertRaises(ModelCallBudgetExceeded):
                    agent.act(self.observation)
                self.assertEqual(agent.llm.calls, budget)
                self.assertIsNone(agent.pending)
                self.assertEqual(agent.history, [])

    def test_invalid_responses_and_api_errors_propagate_without_fallback(self):
        cases = [
            ([decision("teleport")], ValueError),
            ([{"thought": "", "action": "noop", "observation": "fake reward"}], ValueError),
            ([decision("task_failed"), plan(["Gather wood."], "1 OR 2")], ValueError),
            ([decision("task_failed"), plan(["Gather wood."], "1; import os")], ValueError),
            ([RuntimeError("HTTP unavailable")], RuntimeError),
        ]
        for responses, error in cases:
            with self.subTest(responses=responses):
                agent = make_agent(responses)
                with self.assertRaises(error):
                    agent.act(self.observation)
                self.assertIsNone(agent.pending)
                self.assertEqual(agent.llm.calls, len(responses))

    def test_invalid_limits_are_rejected(self):
        for field in ("max_depth", "max_executor_calls", "max_subtasks"):
            for value in (0, -1, True, 1.5, "2"):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        make_agent([], **{field: value})


class AdaptFlowTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def run_responses(self, responses, *, max_depth=2, max_steps=5, max_model_calls=20):
        config = load_config(PROJECT_ROOT / "configs/adapt.yaml")
        config["output_dir"] = self.temp.name
        config["model"].update(provider="ollama", base_url="http://localhost:11434",
                               api_key="", api_key_env=None)
        config["agent"]["params"].update(max_depth=max_depth, max_executor_calls=5)
        config["experiment"].update(seeds=[0], episodes_per_seed=1,
                                     max_steps=max_steps, max_model_calls=max_model_calls)
        replies = iter(responses)

        def reply(request, **kwargs):
            payload = json.loads(request.data)
            self.assertIn("format", payload)
            return BytesIO(json.dumps({
                "model": "test", "message": {"content": json.dumps(next(replies))},
                "done": True, "done_reason": "stop",
                "prompt_eval_count": 10, "eval_count": 5,
            }).encode())

        # Existing shared-flow tests cover the plotting implementation; this
        # suite exercises real Crafter, recording, and controller termination.
        with redirect_stdout(StringIO()), \
                patch("modules.common.llm.opener.open", side_effect=reply) as http, \
                patch("utils.visualization.visualize_run", return_value=[]):
            result = run_experiment(config)
        episode = result["results"][0]
        directory = Path(result["output_dir"])
        path = directory / episode["episode_id"]
        calls = [json.loads(line) for line in (path / "model_calls.jsonl").read_text().splitlines()]
        trajectory = [json.loads(line) for line in (path / "trajectory.jsonl").read_text().splitlines()]
        self.assertEqual(len(calls), http.call_count)
        self.assertEqual(episode["model_calls"], http.call_count)
        self.assertEqual(episode["input_tokens"], 10 * http.call_count)
        self.assertTrue(all(call["status"] == "ok" for call in calls))
        self.assertTrue((directory / "summary.json").is_file())
        self.assertFalse((directory / "error.json").exists())
        self.assertEqual([call["call"] for call in calls], list(range(1, len(calls) + 1)))
        return episode, calls, trajectory

    def test_model_claim_cannot_make_unfulfilled_environment_goal_successful(self):
        episode, calls, trajectory = self.run_responses([decision("task_completed")])
        self.assertEqual(episode["steps"], 0)
        self.assertEqual(episode["stop_reason"], "agent_completed")
        self.assertFalse(episode["success"])
        self.assertEqual(episode["achievements"]["collect_diamond"], 0)
        self.assertEqual(trajectory[-1]["type"], "agent_stopped")
        self.assertEqual(calls[0]["label"], "adapt_execute")

    def test_executor_failure_is_normal_termination_at_depth_limit(self):
        episode, calls, trajectory = self.run_responses([decision("task_failed")], max_depth=1)
        self.assertEqual(episode["stop_reason"], "agent_failed")
        self.assertFalse(episode["success"])
        self.assertEqual(episode["steps"], 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(trajectory[-1]["type"], "agent_stopped")

    def test_planning_and_control_markers_consume_calls_but_only_actions_consume_steps(self):
        episode, calls, trajectory = self.run_responses([
            decision("task_failed"), plan(["Explore the surroundings."], "1"),
            decision("noop"), decision("task_completed"),
        ])
        self.assertEqual(episode["steps"], 1)
        self.assertEqual(episode["model_calls"], 4)
        self.assertEqual(episode["stop_reason"], "agent_completed")
        self.assertFalse(episode["success"])
        self.assertEqual([row["type"] for row in trajectory], ["reset", "step", "agent_stopped"])
        self.assertEqual(trajectory[1]["action"], "noop")
        self.assertEqual([call["label"] for call in calls],
                         ["adapt_execute", "adapt_plan", "adapt_execute", "adapt_execute"])

    def test_global_budget_stops_mid_decomposition_without_executing_a_step(self):
        for budget in (1, 2):
            with self.subTest(budget=budget):
                episode, calls, trajectory = self.run_responses([
                    decision("task_failed"), plan(["Explore the surroundings."], "1"),
                ], max_model_calls=budget)
                self.assertEqual(episode["stop_reason"], "max_model_calls")
                self.assertEqual(episode["steps"], 0)
                self.assertEqual(len(calls), budget)
                self.assertEqual(trajectory[-1]["type"], "budget_exhausted")

    def test_step_limit_stops_before_unnecessary_completion_or_planning_call(self):
        episode, calls, trajectory = self.run_responses([decision("noop")], max_steps=1)
        self.assertEqual(episode["stop_reason"], "max_steps")
        self.assertEqual(episode["steps"], 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(trajectory[-1]["type"], "step")
        self.assertTrue(trajectory[-1]["truncated"])

    def test_deepseek_json_mode_supports_recursive_planning_and_real_action(self):
        config = load_config(PROJECT_ROOT / "configs/adapt.yaml")
        config["output_dir"] = self.temp.name
        config["model"].update(provider="auto", base_url="https://api.deepseek.com",
                               name="test-model", api_key="fake-adapt-api-key", api_key_env=None)
        config["experiment"].update(seeds=[0], episodes_per_seed=1, max_steps=1,
                                     max_model_calls=10)
        replies = iter([decision("task_failed"), plan(["Explore the surroundings."], "1"),
                        decision("noop")])
        schemas = []

        def reply(request, **kwargs):
            self.assertEqual(request.full_url, "https://api.deepseek.com/chat/completions")
            self.assertEqual(request.get_header("Authorization"), "Bearer fake-adapt-api-key")
            payload = json.loads(request.data)
            self.assertEqual(payload["response_format"], {"type": "json_object"})
            self.assertNotIn("format", payload)
            schema_text = payload["messages"][0]["content"].split(
                "Return only a JSON object matching this JSON schema:\n", 1)[1]
            schemas.append(json.loads(schema_text))
            return BytesIO(json.dumps({
                "choices": [{"message": {"content": json.dumps(next(replies))},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }).encode())

        with redirect_stdout(StringIO()), \
                patch("modules.common.llm.opener.open", side_effect=reply) as http, \
                patch("utils.visualization.visualize_run", return_value=[]):
            result = run_experiment(config)
        episode = result["results"][0]
        self.assertEqual(http.call_count, 3)
        self.assertEqual(episode["model_calls"], 3)
        self.assertEqual(episode["steps"], 1)
        self.assertEqual(episode["stop_reason"], "max_steps")
        self.assertEqual(episode["input_tokens"], 30)
        self.assertEqual(episode["output_tokens"], 15)
        self.assertEqual(schemas[0], schemas[2])
        self.assertEqual(set(schemas[0]["required"]), {"thought", "action"})
        self.assertIn("task_failed", schemas[0]["properties"]["action"]["enum"])
        self.assertEqual(set(schemas[1]["required"]), {"thought", "subtasks", "logic"})
        directory = Path(result["output_dir"])
        calls_text = (directory / episode["episode_id"] / "model_calls.jsonl").read_text()
        calls = [json.loads(line) for line in calls_text.splitlines()]
        self.assertEqual([call["label"] for call in calls],
                         ["adapt_execute", "adapt_plan", "adapt_execute"])
        self.assertTrue(all(call["provider"] == "deepseek" and call["status"] == "ok"
                            for call in calls))
        self.assertNotIn("fake-adapt-api-key", calls_text)
        self.assertNotIn("fake-adapt-api-key", (directory / "config.yaml").read_text())


if __name__ == "__main__":
    unittest.main()
