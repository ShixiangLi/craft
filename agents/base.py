"""所有智能体共用的外部接口；不限制内部算法结构。"""
from abc import ABC, abstractmethod
from typing import Any, Mapping


class AgentFinished(Exception):
    """智能体主动结束规划；completed 是自报结果，不代替环境评测。"""
    def __init__(self, completed: bool):
        self.completed = completed
        super().__init__("agent_completed" if completed else "agent_failed")


class BaseAgent(ABC):
    @abstractmethod
    def reset(self, task: Mapping[str, Any], *, seed: int) -> None:
        """开始新回合，接收最终任务并清空回合状态。"""
        raise NotImplementedError

    @abstractmethod
    def act(self, observation: Mapping[str, Any]) -> int:
        """返回环境动作编号；主动结束可抛出 AgentFinished。"""
        raise NotImplementedError

    @abstractmethod
    def observe(self, transition: Mapping[str, Any]) -> None:
        """接收动作后的反馈，包括终止状态；具体字段待环境适配时确定。"""
        raise NotImplementedError
