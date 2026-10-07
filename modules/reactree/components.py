"""独立实现 ReAcTree 的有界决策协议、节点状态和上下文呈现。"""
from dataclasses import dataclass, field
import json

CONTROL_FLOWS = ("sequence", "fallback", "parallel")
DECISION_PROTOCOL = "type_first_v1"
DEFAULTS = {
    "max_depth": 6, "max_subgoals": 5, "max_nodes": 128,
    "max_history_steps": None, "parallel_policy": "all",
    "planning_error_policy": "node_failure",
    "working_memory": True, "working_memory_mode": "spatial", "max_recall_locations": 8,
    "episodic_memory": True, "episodic_memory_path": None, "episodic_memory_sources": [],
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "embedding_device": "cpu", "max_examples": 3, "max_example_chars": 20000,
}
MAX_THOUGHT_CHARS = 1024
MAX_GOAL_CHARS = 500


def validate_params(params=None):
    if params is not None and not isinstance(params, dict):
        raise ValueError("reactree.params 必须是映射")
    params = {**DEFAULTS, **(params or {})}
    if set(params) != set(DEFAULTS):
        raise ValueError(f"未知 reactree 参数: {sorted(set(params) - set(DEFAULTS))}")
    for name in ("max_depth", "max_subgoals", "max_nodes", "max_history_steps",
                 "max_examples", "max_example_chars", "max_recall_locations"):
        value = params[name]
        if name == "max_history_steps" and value is None:
            continue
        if type(value) is not int or value < 1:
            raise ValueError(f"reactree.{name} 必须是正整数" +
                             ("或 null" if name == "max_history_steps" else ""))
    for name in ("working_memory", "episodic_memory"):
        if type(params[name]) is not bool:
            raise ValueError(f"reactree.{name} 必须是布尔值")
    if params["parallel_policy"] not in ("all", "majority"):
        raise ValueError("reactree.parallel_policy 必须是 all 或 majority")
    if params["planning_error_policy"] not in ("node_failure", "abort"):
        raise ValueError("reactree.planning_error_policy 必须是 node_failure 或 abort")
    if params["working_memory_mode"] not in ("spatial", "last_seen"):
        raise ValueError("reactree.working_memory_mode 必须是 spatial 或 last_seen")
    for name in ("embedding_model", "embedding_device"):
        if not isinstance(params[name], str) or not params[name].strip():
            raise ValueError(f"reactree.{name} 必须是非空字符串")
    path = params["episodic_memory_path"]
    if path is not None and (not isinstance(path, str) or not path.strip()):
        raise ValueError("reactree.episodic_memory_path 必须是路径字符串或 null")
    sources = params["episodic_memory_sources"]
    if not isinstance(sources, list) or any(not isinstance(p, str) or not p.strip() for p in sources):
        raise ValueError("reactree.episodic_memory_sources 必须是训练运行/回合目录的字符串列表")
    if path is not None and sources:
        raise ValueError("episodic_memory_path 与 episodic_memory_sources 只能配置一项")
    if sources and not params["episodic_memory"]:
        raise ValueError("配置 episodic_memory_sources 时必须启用 episodic_memory")
    return params


@dataclass
class Node:
    id: str
    kind: str
    content: str
    depth: int
    parent: str | None = None
    children: list[str] = field(default_factory=list)
    status: str = "pending"
    termination: str | None = None
    decisions: int = 0
    history: list[dict] = field(default_factory=list)
    examples: list[dict] = field(default_factory=list)
    failure_reason: str | None = None

    def snapshot(self):
        return {"id": self.id, "kind": self.kind, "content": self.content,
                "depth": self.depth, "parent": self.parent,
                "children": list(self.children), "status": self.status,
                "termination": self.termination, "decisions": self.decisions,
                "history_entries": len(self.history), "examples": self.examples,
                "failure_reason": self.failure_reason}


def decision_schema(actions, *, can_expand, max_subgoals, recall_targets=()):
    def branch(kind, **arguments):
        properties = {
            "type": {"type": "string", "enum": [kind]},
            **arguments,
            "thought": {"type": "string", "maxLength": MAX_THOUGHT_CHARS},
        }
        return {"type": "object", "properties": properties,
                "required": list(properties), "additionalProperties": False}

    # 生成的首字段只选择决策类别；具体游戏动作仅在 Act 类别内选择。
    # 一个 JSON 响应完成类别与内容生成，不增加一次独立的分类模型调用。
    branches = [branch("Think"),
                branch("Act", action={"type": "string", "enum": list(actions) + ["done", "failure"]})]
    if can_expand:
        branches.append(branch("Expand",
            control_flow={"type": "string", "enum": list(CONTROL_FLOWS)},
            subgoals={"type": "array", "minItems": 1, "maxItems": max_subgoals,
                      "items": {"type": "string", "minLength": 1,
                                "maxLength": MAX_GOAL_CHARS}}))
    if recall_targets:
        branches.append(branch("Act", action={"type": "string", "enum": ["recall_observation"]},
                              target={"type": "string", "enum": list(recall_targets)}))
    return {"oneOf": branches}


def parse_decision(raw, actions, *, can_expand, max_subgoals, recall_targets=()):
    data = json.loads(raw)
    if not isinstance(data, dict) or data.get("type") not in ("Think", "Act", "Expand"):
        raise ValueError("ReAcTree 输出必须包含 type: Think、Act 或 Expand；不接受旧 action-first 协议")
    if next(iter(data)) != "type":
        raise ValueError("ReAcTree 必须先生成 type，再生成对应内容")
    kind = data["type"]
    keys = {"type", "thought"}
    if kind == "Expand":
        if not can_expand:
            raise ValueError("当前深度或节点预算不允许 expand")
        keys |= {"control_flow", "subgoals"}
        if data.get("control_flow") not in CONTROL_FLOWS:
            raise ValueError("非法 ReAcTree control_flow")
        goals = data.get("subgoals")
        if not isinstance(goals, list) or not 1 <= len(goals) <= max_subgoals or any(
            not isinstance(g, str) or not g.strip() or len(g) > MAX_GOAL_CHARS
            for g in goals
        ):
            raise ValueError(f"subgoals 必须有 1–{max_subgoals} 个非空子目标，每个至多 {MAX_GOAL_CHARS} 字符")
        data["subgoals"] = [g.strip() for g in goals]
    elif kind == "Act":
        keys.add("action")
        action = data.get("action")
        if not isinstance(action, str):
            raise ValueError("Act 必须包含字符串 action")
        if action == "recall_observation":
            keys.add("target")
            if data.get("target") not in recall_targets:
                raise ValueError("recall_observation 只能查询已见过的目标标签")
        elif action not in list(actions) + ["done", "failure"]:
            raise ValueError(f"非法 ReAcTree Act 动作: {action[:100]!r}")
    if set(data) != keys or not isinstance(data.get("thought"), str):
        raise ValueError(f"{kind} 必须且仅包含字段 {sorted(keys)}，thought 必须是字符串")
    if len(data["thought"]) > MAX_THOUGHT_CHARS:
        raise ValueError(f"thought 至多 {MAX_THOUGHT_CHARS} 字符")
    return data


def render_game_examples(text):
    """只转换共享原生动作示例的响应格式，不新增事实、动作或分解演示。"""
    lines = []
    for line in text.splitlines():
        if line.startswith("Response: "):
            response = json.loads(line.removeprefix("Response: "))
            if (not isinstance(response, dict) or set(response) != {"action", "thought"}
                    or not all(isinstance(response[key], str) for key in ("action", "thought"))):
                raise ValueError("共享游戏示例 Response 必须包含 action/thought 字符串")
            line = "Response: " + json.dumps({"type": "Act", "action": response["action"],
                                            "thought": response["thought"]}, ensure_ascii=False)
        lines.append(line)
    return "\n".join(lines)


def render_node_history(history, max_steps=None, *, current_observation=None):
    if not history:
        return "No completed decisions for this node."
    retained = history if max_steps is None else history[-max_steps:]
    lines = []
    if len(retained) < len(history):
        lines.append(f"Earlier {len(history) - len(retained)} node decisions omitted. "
                     "Old coordinates are relative to the player at their observation time.")
    last_observation = None
    for entry in retained:
        before = entry["observation"]
        if before != last_observation and before != current_observation:
            lines.append(f"Observation at environment step {entry['step']}:\n{before}")
        last_observation = before
        lines.append("Decision: " + json.dumps(entry["decision"], ensure_ascii=False))
        feedback = dict(entry["feedback"])
        after = feedback.pop("observation", None)
        lines.append("Actual feedback: " + json.dumps(feedback, ensure_ascii=False))
        # 日志保留完整前后帧；输入只按时间呈现一次相同观测，末帧可由当前观测栏提供。
        if after is not None:
            if after != last_observation and after != current_observation:
                lines.append("Observation after decision:\n" + after)
            last_observation = after
    return "\n\n".join(lines)
