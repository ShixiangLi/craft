"""动态任务图中的可复用行为节点；环境条件统一由边携带。"""
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum

from .edges import TaskEdge, _nonempty_string


class IncomingLogic(str, Enum):
    """AND 要求所有入边满足；OR 要求至少一条完整入边满足。"""

    AND = "and"
    OR = "or"


class NodeStatus(str, Enum):
    """由可信执行代码维护，模型建图工具不能写入。"""

    PENDING = "pending"
    DOING = "doing"
    FINISH = "finish"
    SUSPENDED = "suspended"
    FAILED = "failed"


@dataclass
class TaskNode:
    """描述一个具体行为；finish 后仍可因新的边需求重新激活。

    incoming_logic 组合普通入边；中断边另作强制门槛。前后序由边推导，
    节点不保存库存目标、完成谓词或重复的连接关系。
    recipe_action 可绑定原生制作/放置动作，供 Planner 校验输入与产物归属。
    """

    id: str
    task: str
    incoming_logic: IncomingLogic = IncomingLogic.AND
    status: NodeStatus = NodeStatus.PENDING
    recipe_action: str | None = None

    def __post_init__(self) -> None:
        _nonempty_string(self.id, "node.id")
        _nonempty_string(self.task, "node.task")
        if self.recipe_action is not None:
            _nonempty_string(self.recipe_action, "node.recipe_action")
        try:
            self.incoming_logic = IncomingLogic(self.incoming_logic)
        except (ValueError, TypeError) as exc:
            raise ValueError("node.incoming_logic must be and or or") from exc
        try:
            self.status = NodeStatus(self.status)
        except (ValueError, TypeError) as exc:
            raise ValueError("node.status must be pending, doing, finish, suspended or failed") from exc

    def predecessor_ids(self, edges: Iterable[TaskEdge]) -> list[str]:
        """按边顺序返回前序节点 ID；多条边连接同一前序时只列出一次。"""
        return list(dict.fromkeys(edge.source for edge in edges if edge.target == self.id))

    def successor_ids(self, edges: Iterable[TaskEdge]) -> list[str]:
        """按边顺序返回后继节点 ID；无后继时返回空列表。"""
        return list(dict.fromkeys(edge.target for edge in edges if edge.source == self.id))

    def to_dict(self) -> dict:
        result = {"id": self.id, "task": self.task,
                  "incoming_logic": self.incoming_logic.value, "status": self.status.value}
        if self.recipe_action is not None:
            result["recipe_action"] = self.recipe_action
        return result

    @classmethod
    def from_dict(cls, data: Mapping) -> "TaskNode":
        required = {"id", "task"}
        allowed = required | {"incoming_logic", "status", "recipe_action"}
        if not isinstance(data, Mapping) or not required <= set(data) <= allowed:
            raise ValueError("A node requires id and task; incoming_logic, status and recipe_action are optional; no other fields are accepted")
        return cls(**data)
