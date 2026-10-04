"""所有智能体共用的外部接口；不限制内部算法结构。"""
from abc import ABC, abstractmethod
from typing import Any, Mapping


class BaseAgent(ABC):
    @abstractmethod
    def reset(self, task: Mapping[str, Any], *, seed: int) -> None:
        """开始新回合，接收最终任务并清空回合状态。"""
        raise NotImplementedError

    @abstractmethod
    def act(self, observation: Mapping[str, Any]) -> int:
        """根据观测返回一个环境动作编号。"""
        raise NotImplementedError

    @abstractmethod
    def observe(self, transition: Mapping[str, Any]) -> None:
        """接收动作后的反馈，包括终止状态；具体字段待环境适配时确定。"""
        raise NotImplementedError
