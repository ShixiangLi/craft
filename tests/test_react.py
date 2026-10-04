"""验证 ReAct 的真实反馈历史、稀疏思考、回合隔离和 Crafter 示例。"""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from agents.react import ReActAgent
from engine.environment import create_environment
from modules.react.components import parse_react, render_history


ROOT = Path(__file__).resolve().parents[1]


class FakeLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []
        self.reset_stats()

    def reset_stats(self):
        self.calls = 0

    def generate(self, messages, **kwargs):
        self.calls += 1
        self.requests.append((deepcopy(messages), kwargs))
        return json.dumps(next(self.responses))


def make_agent(responses, **params):
    prompts = {name: str(ROOT / "prompts/react" / f"{name}.txt")
               for name in ("system", "step", "examples")}
    prompts["rules"] = str(ROOT / "prompts/common/crafter_rules.txt")
    agent = ReActAgent({"prompts": prompts, "params": params}, FakeLLM(responses))
    agent.reset({"description": "Collect wood", "success_condition": {
        "achievement": "collect_wood", "count": 1}}, seed=17)
    return agent


class ReActTest(unittest.TestCase):
    def setUp(self):
        self.env = create_environment({"name": "crafter", "params": {}})
        self.addCleanup(self.env.close)
        self.observation = self.env.reset(seed=0)

    def execute(self, agent, observation):
        action = agent.act(observation)
        transition = self.env.step(action)
        transition.update(action=observation["actions"][action], new_achievements=[])
        agent.observe(transition)
        return transition["observation"]

    def test_thought_action_and_real_feedback_reach_next_decision(self):
        agent = make_agent([{"thought": "Look for a reachable tree.", "action": "noop"},
                            {"thought": "", "action": "move_left"}])
        next_observation = self.execute(agent, self.observation)
        self.assertEqual(len(agent.history), 1)
        agent.act(next_observation)
        content = agent.llm.requests[1][0][-1]["content"]
        self.assertIn("Thought: Look for a reachable tree.", content)
        self.assertIn("Action: noop", content)
        self.assertIn("Environment feedback:", content)
        self.assertIn("Observation 0:", content)
        self.assertIn("Step: 1", content)
        self.assertIn("Collect wood", content)
        self.assertEqual(agent.llm.calls, 2)
        self.assertEqual(agent.last_decision["thought"], "")
        schema = agent.llm.requests[0][1]["response_schema"]
        self.assertEqual(schema["required"], ["thought", "action"])
        self.assertEqual(schema["properties"]["action"]["enum"], self.observation["actions"])

    def test_predictions_are_not_environment_observations_and_reset_isolates_episodes(self):
        agent = make_agent([{"thought": "Try moving.", "action": "noop"}])
        agent.act(self.observation)
        self.assertEqual(agent.history, [])
        with self.assertRaisesRegex(RuntimeError, "observe"):
            agent.act(self.observation)
        agent.reset({"description": "A different goal"}, seed=1)
        self.assertEqual(agent.history, [])
        self.assertIsNone(agent.pending)
        self.assertIsNone(agent.previous)
        self.assertEqual(agent.llm.calls, 0)

    def test_full_history_default_and_explicit_complete_interaction_window(self):
        responses = [{"thought": f"Unique thought {i}.", "action": "noop"} for i in range(4)]
        for window in (None, 1):
            with self.subTest(window=window):
                agent = make_agent(responses, max_history_steps=window)
                observation = self.env.reset(seed=0)
                for _ in range(3):
                    observation = self.execute(agent, observation)
                agent.act(observation)
                content = agent.llm.requests[-1][0][-1]["content"]
                self.assertIn("Unique thought 2.", content)
                if window is None:
                    self.assertIn("Unique thought 0.", content)
                    self.assertNotIn("interactions omitted", content)
                else:
                    self.assertNotIn("Unique thought 0.", content)
                    self.assertIn("Earlier 2 interactions omitted", content)
                    self.assertIn("Observation 2:", content)
                    self.assertNotIn("Observation 1:", content)

    def test_invalid_outputs_do_not_become_actions_or_fabricated_feedback(self):
        invalid = [
            {"action": "noop"}, {"thought": [], "action": "noop"},
            {"thought": "", "action": "teleport"},
            {"thought": "", "action": "noop", "observation": "I got diamond"},
        ]
        for response in invalid:
            with self.subTest(response=response):
                agent = make_agent([response])
                with self.assertRaises(ValueError):
                    agent.act(self.observation)
                self.assertEqual(agent.history, [])
                self.assertIsNone(agent.pending)
        self.assertEqual(parse_react('{"thought":"", "action":"noop"}', ["noop"]), ("", 0))

    def test_history_uses_feedback_even_when_reward_is_zero(self):
        history = [{"step": 4, "observation": "wood=1; table north", "thought": "",
                    "action": "make_wood_pickaxe", "feedback": {
                        "action": "make_wood_pickaxe", "reward": 0,
                        "new_achievements": []}}]
        text = render_history(history)
        self.assertIn('"reward": 0', text)
        self.assertNotIn("failed", text)
        self.assertNotIn("Thought:", text)

    def test_history_window_rejects_invalid_values(self):
        for value in (0, -1, True, 1.5, "3"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "max_history_steps"):
                    make_agent([], max_history_steps=value)

    def controlled_tree_scene(self, wood=0, stone=0):
        """控制 few-shot 起始条件；不向 agent 公开环境私有状态。"""
        self.env.reset(seed=0)
        player = self.env.env._player
        world = player.world
        for obj in tuple(world.objects):
            if obj is not player:
                world.remove(obj)
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                world[player.pos + (dx, dy)] = "grass"
        world[player.pos + (0, -1)] = "tree"
        player.inventory.update(wood=wood, stone=stone)
        return player

    def test_both_handwritten_demonstrations_are_valid_in_real_crafter(self):
        player = self.controlled_tree_scene()
        start = player.pos.copy()
        player.facing = (0, 1)
        actions = self.observation["actions"]
        obs = self.env.step(actions.index("move_up"))["observation"]
        self.assertEqual(list(player.pos), list(start))
        self.assertEqual(obs["facing"], [0, -1])
        self.assertEqual(obs["inventory"]["wood"], 0)
        obs = self.env.step(actions.index("do"))["observation"]
        self.assertEqual(obs["inventory"]["wood"], 1)
        self.assertEqual(obs["achievements"]["collect_wood"], 1)

        player = self.controlled_tree_scene(wood=2, stone=1)
        player.facing = (0, -1)
        for action, expected_wood in (("do", 3), ("place_table", 1), ("make_stone_pickaxe", 0)):
            obs = self.env.step(actions.index(action))["observation"]
            self.assertEqual(obs["inventory"]["wood"], expected_wood)
        self.assertEqual(obs["inventory"]["stone"], 0)
        self.assertEqual(obs["inventory"]["stone_pickaxe"], 1)
        self.assertEqual(obs["achievements"]["make_stone_pickaxe"], 1)


if __name__ == "__main__":
    unittest.main()
