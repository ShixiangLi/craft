"""SPRING 协议测试：真实 Crafter 观测和 HTTP 响应夹具，不消耗模型。"""
from io import BytesIO
import json
import unittest
from unittest.mock import patch

from agents.spring import SpringAgent
from engine.environment import create_environment
from modules.common.llm import LLMClient, ModelCallBudgetExceeded
from utils.config import PROJECT_ROOT, load_config


class SpringTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config(PROJECT_ROOT / "configs/spring.yaml")
        self.client = LLMClient(self.config["model"])
        self.agent = SpringAgent(self.config["agent"], self.client)
        self.agent.reset(self.config["task"], seed=0)
        self.env = create_environment(self.config["environment"])
        self.addCleanup(self.env.close)
        self.obs = self.env.reset(seed=0)
        self.requests = []

    def respond(self, request, **kwargs):
        payload = json.loads(request.data)
        self.requests.append(payload)
        index = len(self.requests)
        content = json.dumps({"action": "noop"}) if index % 9 == 0 else f"answer-{index}"
        return BytesIO(json.dumps({"message": {"content": content}, "done": True,
                                   "done_reason": "stop", "prompt_eval_count": 10,
                                   "eval_count": 3}).encode())

    def test_nine_calls_and_only_direct_parent_answers(self):
        # Explicit expected edges are independently transcribed from the official
        # notebook; do not import the implementation graph to validate itself.
        expected_parents = [[], [], [1], [2], [1, 3], [5], [6], [7], [2, 4, 7, 8]]
        with patch("modules.common.llm.opener.open", side_effect=self.respond):
            action = self.agent.act(self.obs)
        self.assertEqual(self.obs["actions"][action], "noop")
        self.assertEqual(self.client.calls, 9)
        self.assertEqual(self.client.input_tokens, 90)
        self.assertEqual(self.client.output_tokens, 27)
        for i, (request, parents) in enumerate(zip(self.requests, expected_parents)):
            messages = request["messages"]
            self.assertEqual([m["role"] for m in messages[:3]], ["system"] * 3)
            self.assertEqual(messages[1]["content"], self.agent.prompts["knowledge"])
            self.assertIn(self.config["task"]["description"], messages[2]["content"])
            self.assertEqual([m["content"] for m in messages if m["role"] == "assistant"],
                             [f"answer-{parent}" for parent in parents])
            self.assertEqual(len(messages), 4 + 2 * len(parents))
            if i < 8:
                self.assertNotIn("format", request)
            else:
                self.assertEqual(request["format"]["properties"]["action"]["enum"],
                                 self.obs["actions"])
        self.assertEqual([node["id"] for node in self.agent.last_decision["nodes"]],
                         ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "qa"])

    def test_two_frames_and_reset_do_not_retain_old_nodes(self):
        with patch("modules.common.llm.opener.open", side_effect=self.respond):
            for step in range(3):
                obs = {**self.obs, "step": step}
                self.agent.act(obs)
                self.agent.observe({"action": "noop", "reward": 123,
                                    "new_achievements": ["synthetic_achievement"]})
        context = self.requests[-1]["messages"][2]["content"]
        self.assertNotIn("Player Observation Step 0:", context)
        self.assertLess(context.index("Player Observation Step 1:"),
                        context.index("Player Observation Step 2:"))
        self.assertIn("Last player action: noop", context)
        self.assertNotIn("synthetic_achievement", context)
        self.assertNotIn("reward", context)
        # Last decision's qa has only its current parents, never previous QA.
        self.assertEqual([m["content"] for m in self.requests[-1]["messages"]
                          if m["role"] == "assistant"],
                         ["answer-20", "answer-22", "answer-25", "answer-26"])
        self.agent.reset(self.config["task"], seed=7)
        self.assertEqual(len(self.agent.observations), 0)
        self.assertEqual(self.client.calls, 0)
        with patch("modules.common.llm.opener.open", side_effect=self.respond):
            self.agent.act(self.obs)
        context = self.requests[-1]["messages"][2]["content"]
        self.assertIn("Player Observation Step 0:", context)
        self.assertNotIn("Player Observation Step 1:", context)
        self.assertIn("Last player action: None (episode start)", context)
        self.assertEqual(self.requests[-1]["options"]["seed"], 7)

    def test_insufficient_budget_never_starts_partial_graph(self):
        self.client.max_calls = 8
        with patch("modules.common.llm.opener.open", side_effect=self.respond) as send:
            with self.assertRaises(ModelCallBudgetExceeded):
                self.agent.act(self.obs)
            send.assert_not_called()
        self.assertEqual(len(self.agent.observations), 0)
        self.client.max_calls = 17
        with patch("modules.common.llm.opener.open", side_effect=self.respond) as send:
            self.agent.act(self.obs)
            with self.assertRaises(ModelCallBudgetExceeded):
                self.agent.act({**self.obs, "step": 1})
            self.assertEqual(send.call_count, 9)
        self.assertEqual(self.agent.last_decision["calls"], [])
        self.assertEqual(len(self.agent.observations), 1)

    def test_failure_keeps_completed_nodes_and_never_falls_back_to_do(self):
        def malformed_final(request, **kwargs):
            payload = json.loads(request.data)
            if "format" in payload:
                return BytesIO(json.dumps({"message": {"content": '{"action":"teleport"}'},
                                           "done": True}).encode())
            return self.respond(request, **kwargs)

        records = []
        self.client.on_call = records.append
        with patch("modules.common.llm.opener.open", side_effect=malformed_final):
            with self.assertRaisesRegex(ValueError, "非法动作"):
                self.agent.act(self.obs)
        self.assertEqual(len(records), 9)
        self.assertEqual(len(self.agent.last_decision["nodes"]), 9)
        self.assertIn("teleport", self.agent.last_decision["response"])

    def test_node_failure_preserves_prior_calls(self):
        def empty_third_node(request, **kwargs):
            if len(self.requests) == 2:
                return BytesIO(json.dumps({"message": {"content": ""}, "done": True}).encode())
            return self.respond(request, **kwargs)

        records = []
        self.client.on_call = records.append
        with patch("modules.common.llm.opener.open", side_effect=empty_third_node):
            with self.assertRaisesRegex(ValueError, "message.content"):
                self.agent.act(self.obs)
        self.assertEqual(self.client.calls, 3)
        self.assertEqual([r["label"] for r in records], ["q1", "q2", "q3"])
        self.assertEqual(records[-1]["status"], "error")
        self.assertEqual(len(self.agent.last_decision["nodes"]), 2)


if __name__ == "__main__":
    unittest.main()
