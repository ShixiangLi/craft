"""ReActAgent 算法占位；具体流程尚未实现。"""
from typing import Any, Mapping
from agents.base import BaseAgent


class ReActAgent(BaseAgent):
    def __init__(self, config: Mapping[str, Any]):
        raise NotImplementedError("ReActAgent 尚未实现")

    def reset(self, task: Mapping[str, Any], *, seed: int) -> None:
        raise NotImplementedError("回合初始化尚未实现")

    def act(self, observation: Mapping[str, Any]) -> int:
        raise NotImplementedError("动作决策尚未实现")

    def observe(self, transition: Mapping[str, Any]) -> None:
        raise NotImplementedError("反馈处理尚未实现")
