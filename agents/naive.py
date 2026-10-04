"""当前观测 + 最终目标 + 上一步反馈 -> 一个动作。"""
from string import Template
from typing import Any, Mapping

from agents.llm_agent import LLMBaseAgent
from modules.common.actions import parse_action


class NaiveAgent(LLMBaseAgent):
    def act(self, observation: Mapping[str, Any]) -> int:
        self.begin_decision()
        messages = [
            {"role": "system", "content": self.prompts["system"]},
            {"role": "user", "content": Template(self.prompts["step"]).substitute(self.context(observation))},
        ]
        raw = self.query(messages, actions=observation["actions"])
        return parse_action(raw, observation["actions"])
