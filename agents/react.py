"""ReAct 的 Crafter 移植：稀疏显式思考、单步动作、真实反馈与回合轨迹。"""
from string import Template
from typing import Any, Mapping

from agents.llm_agent import LLMBaseAgent
from modules.common.actions import action_schema
from modules.react.components import parse_react, render_history


class ReActAgent(LLMBaseAgent):
    def __init__(self, config: Mapping[str, Any], llm):
        super().__init__(config, llm)
        self.max_history_steps = config.get("params", {}).get("max_history_steps")
        if self.max_history_steps is not None and (
            type(self.max_history_steps) is not int or self.max_history_steps < 1
        ):
            raise ValueError("react.max_history_steps 必须是正整数或 null")
        self.system_prompt = Template(self.prompts["system"]).substitute(
            rules=self.prompts["rules"], examples=self.prompts["examples"]
        )

    def reset(self, task: Mapping[str, Any], *, seed: int) -> None:
        super().reset(task, seed=seed)
        self.history = []
        self.pending = None

    def act(self, observation: Mapping[str, Any]) -> int:
        if self.pending is not None:
            raise RuntimeError("ReAct 必须先 observe 实际环境反馈，再进行下一次 act")
        self.begin_decision()
        action = self._react_decision(observation)
        return observation["actions"].index(action)

    def _react_decision(self, observation: Mapping[str, Any], *,
                        extra_context: dict | None = None, extra_actions=(),
                        label: str = "react") -> str:
        """共享执行器；额外控制动作只返回给调用者，不提交环境历史。"""
        context = self.context(observation)
        context["history"] = render_history(self.history, self.max_history_steps)
        context.update(extra_context or {})
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": Template(self.prompts["step"]).substitute(context)},
        ]
        actions = list(observation["actions"]) + list(extra_actions)
        schema = action_schema(actions, extra_properties={"thought": {"type": "string"}})
        raw = self.query(messages, response_schema=schema, label=label)
        thought, action_id = parse_react(raw, actions)
        action = actions[action_id]
        self.last_decision.update(thought=thought, action=action,
                                  history_steps=len(self.history) if self.max_history_steps is None
                                  else min(len(self.history), self.max_history_steps))
        # 仅在 observe 收到实际环境结果后提交，模型不能伪造 Observation。
        if action in observation["actions"]:
            self.pending = {"step": observation["step"], "observation": context["observation"],
                            "thought": thought, "action": action}
        return action

    def observe(self, transition: Mapping[str, Any]) -> None:
        if self.pending is None:
            raise RuntimeError("ReAct observe 之前必须有待执行的 act")
        super().observe(transition)
        self.history.append({**self.pending, "feedback": dict(self.previous)})
        self.pending = None
