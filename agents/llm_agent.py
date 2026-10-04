"""LLM 智能体的共享生命周期、文本观测、模型调用和诊断记录。"""
import json
from typing import Any, Mapping

from agents.base import BaseAgent
from modules.common.llm import LLMClient
from modules.common.observation import describe_observation
from utils.prompt import load_prompt


class LLMBaseAgent(BaseAgent):
    def __init__(self, config: Mapping[str, Any], llm: LLMClient):
        self.config = config
        self.llm = llm
        self.prompts = {key: load_prompt(path) for key, path in config["prompts"].items()}
        self.last_decision = {}

    def reset(self, task: Mapping[str, Any], *, seed: int) -> None:
        self.task = dict(task)
        self.seed = seed
        self.previous = None
        self.last_decision = {}
        self.llm.reset_stats()

    def context(self, observation: Mapping[str, Any]) -> dict[str, str]:
        condition = self.task.get("success_condition")
        return {
            "goal": self.task["description"],
            "success_condition": json.dumps(condition, ensure_ascii=False) if condition is not None else "Not configured; pursue the goal until the episode ends.",
            "observation": describe_observation(dict(observation)),
            "previous": json.dumps(self.previous, ensure_ascii=False) if self.previous is not None else "No previous action (episode start).",
        }

    def begin_decision(self) -> None:
        self.last_decision = {"calls": []}

    def query(self, messages: list[dict[str, str]], *, actions: list[str] | None = None,
              response_schema: dict | None = None, label: str = "decision") -> str:
        self.last_decision["messages"] = messages
        self.last_decision.pop("response", None)
        raw = self.llm.generate(messages, seed=self.seed, actions=actions,
                                response_schema=response_schema, label=label)
        self.last_decision["response"] = raw
        self.last_decision.setdefault("calls", []).append({
            "call": self.llm.calls, "label": label, "response": raw})
        return raw

    def observe(self, transition: Mapping[str, Any]) -> None:
        self.previous = {
            "action": transition["action"], "reward": transition["reward"],
            "new_achievements": transition["new_achievements"],
        }
