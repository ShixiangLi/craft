"""供智能体使用的 JSON 工具协议，不绑定模型供应商或执行环境。

调用：{"name": "get_node", "arguments": {"id": "wood"}}
结果：{"ok": true, "name": "add_node", "version": 1, "result": {...}, "error": null}
"""
from collections.abc import Mapping
from copy import deepcopy
import json

from .edges import TaskEdge
from .graph import GraphError, TaskGraph
from .nodes import TaskNode


def _object(properties: dict, required=(), **constraints) -> dict:
    return {"type": "object", "properties": properties, "required": list(required),
            "additionalProperties": False, **constraints}


def tool_definitions(*, condition_schema=None, recipe_action_schema=None) -> list[dict]:
    """供应商无关的工具定义；Planner 可注入环境支持的条件格式。"""
    identifier = {"type": "string", "minLength": 1, "pattern": "\\S"}
    task = {"type": "string", "minLength": 1, "pattern": "\\S",
            "description": "Describe a complete task in English, e.g. collect stone, craft a wood pickaxe, build a table or restore drink. Searching, moving, approaching and turning are internal actions, not separate nodes."}
    logic = {"type": "string", "enum": ["and", "or"]}
    recipe_action = recipe_action_schema if recipe_action_schema is not None else {
        "type": ["string", "null"], "minLength": 1, "pattern": "\\S"}
    condition = condition_schema if condition_schema is not None else {"oneOf": [
        _object({"source": identifier, "key": identifier,
            "op": {"type": "string", "enum": ["==", "!="]},
            "value": {"type": ["number", "string", "boolean", "null"]}},
            ("source", "key", "op", "value")),
        _object({"source": identifier, "key": identifier,
            "op": {"type": "string", "enum": [">", ">=", "<", "<="]},
            "value": {"type": "number"}}, ("source", "key", "op", "value")),
    ]}
    conditions = {"type": "array", "items": condition, "minItems": 1}
    kind = {"type": "string", "enum": ["dependency", "interrupt"]}
    specs = [
        ("add_node", "Add a reusable task with status=pending and incoming_logic defaulting to and. Bind crafting/placement tasks with recipe_action; other tasks may omit it. Nodes do not contain conditions.",
         _object({"id": identifier, "task": task, "incoming_logic": logic, "recipe_action": recipe_action}, ("id", "task"))),
        ("update_node", "Update the task description, recipe action binding or incoming AND/OR logic. ID and execution status cannot be changed.",
         _object({"id": identifier, "changes": _object({"task": task, "incoming_logic": logic,
                     "recipe_action": recipe_action}, minProperties=1)},
                 ("id", "changes"))),
        ("delete_node", "Delete a node. Reject nodes with incident edges unless cascade=true, which also deletes those edges.",
         _object({"id": identifier, "cascade": {"type": "boolean", "default": False}}, ("id",))),
        ("add_edge", "Add a directed edge with nonempty conditions that must all hold now; source status is irrelevant. Ordinary edges combine by target AND/OR. The scheduler owns interrupt edges.",
         _object({"id": identifier, "source": identifier, "target": identifier, "conditions": conditions, "kind": kind},
                 ("id", "source", "target", "conditions"))),
        ("update_edge", "Update source, target or kind, or replace the entire conditions list. The edge ID cannot be changed.",
         _object({"id": identifier, "changes": _object({"source": identifier, "target": identifier,
                     "conditions": conditions, "kind": kind}, minProperties=1)}, ("id", "changes"))),
        ("delete_edge", "Delete an edge while keeping both endpoint tasks.", _object({"id": identifier}, ("id",))),
        ("get_node", "Get a node with its direct predecessors, successors and incident edges. Does not recurse or evaluate conditions.",
         _object({"id": identifier}, ("id",))),
        ("get_edge", "Get an edge by ID, including endpoints and complete condition definitions.",
         _object({"id": identifier}, ("id",))),
        ("get_graph", "Read the complete graph, task execution states and current version without modifying it.", _object({})),
    ]
    return [{"name": name, "description": description, "parameters": deepcopy(parameters)}
            for name, description, parameters in specs]


def tool_call_schema(*, condition_schema=None, recipe_action_schema=None) -> dict:
    """可直接作为现有 LLMClient.generate 的 response_schema 使用。"""
    return {"oneOf": [_object({"name": {"type": "string", "enum": [definition["name"]]},
                                "arguments": definition["parameters"]}, ("name", "arguments"))
                       for definition in tool_definitions(condition_schema=condition_schema,
                                                          recipe_action_schema=recipe_action_schema)]}


def _check_fields(arguments: Mapping, allowed: set[str], required: set[str]) -> None:
    if not isinstance(arguments, Mapping) or not required <= set(arguments) <= allowed:
        raise ValueError(f"Required argument fields: {sorted(required)}; allowed fields: {sorted(allowed)}")


def _reject_constant(value: str):
    raise ValueError(f"JSON does not allow non-finite numbers: {value}")


def _unique_object(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate field in JSON object: {key}")
        result[key] = value
    return result


class GraphTools:
    """白名单分派工具调用，修改失败时返回错误且保持原图及版本不变。"""

    def __init__(self, graph: TaskGraph):
        self.graph = graph

    def execute(self, call: Mapping | str) -> dict:
        name = None
        try:
            if isinstance(call, str):
                call = json.loads(call, parse_constant=_reject_constant, object_pairs_hook=_unique_object)
            if not isinstance(call, Mapping) or set(call) != {"name", "arguments"}:
                raise ValueError("A tool call must contain exactly name and arguments")
            if not isinstance(call["name"], str):
                raise ValueError("name must be a string")
            name = call["name"]
            definitions = {item["name"]: item for item in tool_definitions()}
            if name not in definitions:
                raise GraphError("unknown_tool", f"Unknown graph tool: {name}")
            parameters = definitions[name]["parameters"]
            args = call["arguments"]
            _check_fields(args, set(parameters["properties"]), set(parameters["required"]))
            # No getattr/eval: tool names can reach only these explicit operations.
            if name == "add_node":
                result = {"node": self.graph.add_node(TaskNode.from_dict(args)).to_dict()}
            elif name == "update_node":
                _check_fields(args["changes"], {"task", "incoming_logic", "recipe_action"}, set())
                result = {"node": self.graph.update_node(args["id"], args["changes"]).to_dict()}
            elif name == "delete_node":
                result = self.graph.delete_node(args["id"], cascade=args.get("cascade", False))
            elif name == "add_edge":
                result = {"edge": self.graph.add_edge(TaskEdge.from_dict(args)).to_dict()}
            elif name == "update_edge":
                changes = args["changes"]
                _check_fields(changes, {"source", "target", "conditions", "kind"}, set())
                result = {"edge": self.graph.update_edge(args["id"], changes).to_dict()}
            elif name == "delete_edge":
                result = {"edge": self.graph.delete_edge(args["id"]).to_dict()}
            elif name == "get_node":
                result = self.graph.get_node_context(args["id"])
            elif name == "get_edge":
                result = {"edge": self.graph.get_edge(args["id"]).to_dict()}
            else:
                result = self.graph.snapshot()
            return {"ok": True, "name": name, "version": self.graph.version, "result": result, "error": None}
        except (ValueError, TypeError) as exc:
            error = {"code": exc.code if isinstance(exc, GraphError) else "invalid_arguments", "message": str(exc)}
            return {"ok": False, "name": name, "version": self.graph.version, "result": None, "error": error}
