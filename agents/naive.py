"""最小文本智能体：当前观测 + 最终目标 + 上一步反馈 -> 一个动作。"""
import json
from typing import Any, Mapping

from agents.base import BaseAgent
from modules.common.llm import LLMClient
from modules.common.observation import describe_observation
from utils.prompt import load_prompt


class NaiveAgent(BaseAgent):
    def __init__(self, config: Mapping[str, Any], llm: LLMClient):
        self.llm = llm
        self.system_prompt = load_prompt(config["prompts"]["system"])
        self.step_prompt_path = config["prompts"]["step"]
        self.last_decision = {}

    def reset(self, task: Mapping[str, Any], *, seed: int) -> None:
        self.task = dict(task)
        self.seed = seed
        self.previous = None
        self.last_decision = {}
        self.llm.reset_stats()

    def act(self, observation: Mapping[str, Any]) -> int:
        actions = observation["actions"]
        content = load_prompt(self.step_prompt_path, {
            "goal": self.task["description"],
            "success_condition": json.dumps(self.task.get("success_condition"), ensure_ascii=False) if self.task.get("success_condition") is not None else "Not configured; pursue the goal until the episode ends.",
            "observation": describe_observation(dict(observation)),
            "previous": json.dumps(self.previous, ensure_ascii=False) if self.previous is not None else "No previous action (episode start).",
        })
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": content}]
        self.last_decision = {"messages": messages}
        raw = self.llm.generate(messages, seed=self.seed, actions=actions)
        self.last_decision["response"] = raw
        try:
            parsed = json.loads(raw)
            action = parsed.get("action") if isinstance(parsed, dict) else None
        except json.JSONDecodeError as exc:
            raise ValueError(f"模型输出不是 JSON: {raw[:500]}") from exc
        if action not in actions:
            raise ValueError(f"模型返回非法动作: {action!r}")
        return actions.index(action)

    def observe(self, transition: Mapping[str, Any]) -> None:
        self.previous = {
            "action": transition["action"], "reward": transition["reward"],
            "new_achievements": transition["new_achievements"],
        }
