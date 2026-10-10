"""任务之间的有向边；条件是边属性，不是独立图节点。

一条边可转移，只要求 conditions 中全部条件经当前观测验证为 True。
conditions 不能为空；源节点的历史状态不参与条件求值。
目标节点用 incoming_logic 组合多条完整入边。
本模块只定义条件格式，不读取环境、不缓存条件真假，也不执行转移。
"""
from collections.abc import Mapping
from dataclasses import dataclass
import math


def _nonempty_string(value: object, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")


@dataclass(frozen=True)
class EdgeCondition:
    """结构化比较谓词，例如 inventory.wood >= 2。

    source/key 对应的数据源及字段是否受支持，由后续 evaluator 决定。
    此处不把 LLM 提供的条件定义当作已经成立的环境事实。
    """

    source: str
    key: str
    op: str
    value: int | float | str | bool | None

    def __post_init__(self) -> None:
        _nonempty_string(self.source, "condition.source")
        _nonempty_string(self.key, "condition.key")
        if self.op not in ("==", "!=", ">", ">=", "<", "<="):
            raise ValueError("condition.op must be ==, !=, >, >=, < or <=")
        if self.value is not None and type(self.value) not in (int, float, str, bool):
            raise ValueError("condition.value must be a JSON scalar")
        if isinstance(self.value, float) and not math.isfinite(self.value):
            raise ValueError("condition.value cannot be NaN or infinity")
        if self.op in (">", ">=", "<", "<=") and type(self.value) not in (int, float):
            raise ValueError("Numeric comparisons require a numeric condition.value, not a boolean")

    def to_dict(self) -> dict:
        return {"source": self.source, "key": self.key, "op": self.op, "value": self.value}

    @classmethod
    def from_dict(cls, data: Mapping) -> "EdgeCondition":
        if not isinstance(data, Mapping) or set(data) != {"source", "key", "op", "value"}:
            raise ValueError("An edge condition must contain exactly source, key, op and value")
        return cls(**data)


@dataclass(frozen=True)
class TaskEdge:
    """source 与 target 是任务节点 ID，连接关系由边统一保存。

    不在这里决定节点是否存在、边 ID 是否重复或图是否有环；这些需要完整图，
    应由后续图容器校验。条件的实际真假也不属于边的静态定义。
    """

    id: str
    source: str
    target: str
    conditions: tuple[EdgeCondition, ...]
    kind: str = "dependency"

    def __post_init__(self) -> None:
        _nonempty_string(self.id, "edge.id")
        _nonempty_string(self.source, "edge.source")
        _nonempty_string(self.target, "edge.target")
        if self.kind not in ("dependency", "interrupt"):
            raise ValueError("edge.kind must be dependency or interrupt")
        if not isinstance(self.conditions, (list, tuple)) or not self.conditions or any(
                not isinstance(condition, EdgeCondition) for condition in self.conditions):
            raise ValueError("edge.conditions must be a nonempty list or tuple of EdgeCondition objects")
        object.__setattr__(self, "conditions", tuple(self.conditions))

    def to_dict(self) -> dict:
        result = {"id": self.id, "source": self.source, "target": self.target,
                  "conditions": [condition.to_dict() for condition in self.conditions]}
        if self.kind != "dependency":
            result["kind"] = self.kind
        return result

    @classmethod
    def from_dict(cls, data: Mapping) -> "TaskEdge":
        required = {"id", "source", "target", "conditions"}
        if not isinstance(data, Mapping) or not required <= set(data) <= required | {"kind"}:
            raise ValueError("An edge requires id, source, target and conditions; kind is optional; no other fields are accepted")
        conditions = data["conditions"]
        if not isinstance(conditions, list):
            raise ValueError("edge.conditions must be a JSON array")
        return cls(id=data["id"], source=data["source"], target=data["target"],
                   conditions=tuple(EdgeCondition.from_dict(condition) for condition in conditions),
                   kind=data.get("kind", "dependency"))
