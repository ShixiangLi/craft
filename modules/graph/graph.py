"""任务图容器：校验后提交单次增删改，返回副本，防止绕过接口修改。"""
from collections.abc import Mapping
from copy import deepcopy

from .edges import TaskEdge, _nonempty_string
from .nodes import TaskNode


class GraphError(ValueError):
    """可供工具层返回给智能体的稳定错误码。"""

    def __init__(self, code: str, message: str, *, details=None):
        self.code = code
        self.details = details
        super().__init__(message)


class TaskGraph:
    def __init__(self):
        self._nodes: dict[str, TaskNode] = {}
        self._edges: dict[str, TaskEdge] = {}
        self._version = 0

    @property
    def version(self) -> int:
        return self._version

    def get_node(self, node_id: str) -> TaskNode:
        _nonempty_string(node_id, "node_id")
        if node_id not in self._nodes:
            raise GraphError("node_not_found", f"Node not found: {node_id}")
        return deepcopy(self._nodes[node_id])

    def get_edge(self, edge_id: str) -> TaskEdge:
        _nonempty_string(edge_id, "edge_id")
        if edge_id not in self._edges:
            raise GraphError("edge_not_found", f"Edge not found: {edge_id}")
        return deepcopy(self._edges[edge_id])

    def add_node(self, node: TaskNode) -> TaskNode:
        if not isinstance(node, TaskNode):
            raise ValueError("node must be a TaskNode")
        candidate = TaskNode.from_dict(node.to_dict())
        if candidate.id in self._nodes:
            raise GraphError("duplicate_node", f"Node ID already exists: {candidate.id}")
        self._nodes[candidate.id] = candidate
        self._version += 1
        return deepcopy(candidate)

    def update_node(self, node_id: str, changes: Mapping) -> TaskNode:
        """部分更新；ID 不可改。status 仅允许可信代码使用，不对模型开放。"""
        current = self.get_node(node_id)
        self._validate_changes(changes, {"task", "incoming_logic", "status", "recipe_action"})
        candidate = TaskNode.from_dict({**current.to_dict(), **changes})
        if candidate != current:
            self._nodes[node_id] = candidate
            self._version += 1
        return deepcopy(candidate)

    def delete_node(self, node_id: str, *, cascade: bool = False) -> dict:
        """默认拒绝产生悬空边；显式 cascade 时一并删除所有入边和出边。"""
        current = self.get_node(node_id)
        if type(cascade) is not bool:
            raise ValueError("cascade must be a boolean")
        related = [edge for edge in self._edges.values()
                   if edge.source == node_id or edge.target == node_id]
        if related and not cascade:
            raise GraphError("node_has_edges", f"Node {node_id} has incident edges; delete them first or set cascade=true")
        result = {"node": current.to_dict(), "deleted_edges": [edge.to_dict() for edge in related]}
        for edge in related:
            del self._edges[edge.id]
        del self._nodes[node_id]
        self._version += 1
        return result

    def _validate_endpoints(self, edge: TaskEdge) -> None:
        for node_id in (edge.source, edge.target):
            if node_id not in self._nodes:
                raise GraphError("node_not_found", f"Edge references a missing node: {node_id}")
        if edge.source == edge.target:
            raise GraphError("self_dependency", "A task cannot depend on itself")

    def add_edge(self, edge: TaskEdge) -> TaskEdge:
        if not isinstance(edge, TaskEdge):
            raise ValueError("edge must be a TaskEdge")
        candidate = TaskEdge.from_dict(edge.to_dict())
        if candidate.id in self._edges:
            raise GraphError("duplicate_edge", f"Edge ID already exists: {candidate.id}")
        self._validate_endpoints(candidate)
        self._edges[candidate.id] = candidate
        self._version += 1
        return deepcopy(candidate)

    def update_edge(self, edge_id: str, changes: Mapping) -> TaskEdge:
        """部分更新端点或完整替换条件列表；候选边全部验证通过后才替换。"""
        current = self.get_edge(edge_id)
        self._validate_changes(changes, {"source", "target", "conditions", "kind"})
        data = {**current.to_dict(), **changes}
        candidate = TaskEdge.from_dict(data)
        self._validate_endpoints(candidate)
        if candidate != current:
            self._edges[edge_id] = candidate
            self._version += 1
        return deepcopy(candidate)

    def delete_edge(self, edge_id: str) -> TaskEdge:
        current = self.get_edge(edge_id)
        del self._edges[edge_id]
        self._version += 1
        return current

    @staticmethod
    def _validate_changes(changes: Mapping, allowed: set[str]) -> None:
        if not isinstance(changes, Mapping) or not changes or not set(changes) <= allowed:
            raise ValueError(f"changes must be a nonempty object containing only: {', '.join(sorted(allowed))}")

    def predecessor_ids(self, node_id: str) -> list[str]:
        return self.get_node(node_id).predecessor_ids(self._edges.values())

    def successor_ids(self, node_id: str) -> list[str]:
        return self.get_node(node_id).successor_ids(self._edges.values())

    def get_node_context(self, node_id: str) -> dict:
        """一次返回节点及直接邻接任务和边，不递归、不判断可执行性。"""
        node = self.get_node(node_id)
        incoming = [edge for edge in self._edges.values() if edge.target == node_id]
        outgoing = [edge for edge in self._edges.values() if edge.source == node_id]
        return {
            "node": node.to_dict(),
            "predecessors": [self._nodes[identifier].to_dict()
                             for identifier in dict.fromkeys(edge.source for edge in incoming)],
            "successors": [self._nodes[identifier].to_dict()
                           for identifier in dict.fromkeys(edge.target for edge in outgoing)],
            "incoming_edges": [edge.to_dict() for edge in incoming],
            "outgoing_edges": [edge.to_dict() for edge in outgoing],
        }

    def snapshot(self) -> dict:
        return {"version": self.version,
                "nodes": {key: node.to_dict() for key, node in self._nodes.items()},
                "edges": {key: edge.to_dict() for key, edge in self._edges.items()}}

    def commit(self, candidate: "TaskGraph", *, expected_version: int) -> None:
        """可信模块提交暂存修改；不作为模型工具开放。"""
        if self.version != expected_version:
            raise GraphError("version_conflict", "The graph changed during planning; replan against the current version")
        if not isinstance(candidate, TaskGraph) or candidate.version < self.version:
            raise ValueError("candidate must be a TaskGraph copied and edited from the current graph")
        self._nodes = deepcopy(candidate._nodes)
        self._edges = deepcopy(candidate._edges)
        self._version = candidate.version
