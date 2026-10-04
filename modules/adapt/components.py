"""ADaPT 配置和任务组合解析；执行与短路由智能体控制。"""
from __future__ import annotations

import ast
import json
import re


PlanExpression = int | tuple[str, list["PlanExpression"]]


def _positive_int(value: object, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} 必须为正整数")


def validate_params(params: dict | None) -> dict:
    defaults = {
        "max_depth": 3,
        "max_executor_calls": 20,
        "max_subtasks": 5,
        "max_history_steps": 16,
    }
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise ValueError("ADaPT agent.params 必须是映射")
    unknown = set(params) - defaults.keys()
    if unknown:
        raise ValueError(f"未知 ADaPT 参数: {sorted(map(str, unknown))}")
    result = defaults | params
    for name, value in result.items():
        if name == "max_history_steps" and value is None:
            continue
        _positive_int(value, f"agent.params.{name}")
    return result


def planner_schema(max_subtasks: int) -> dict:
    _positive_int(max_subtasks, "max_subtasks")
    return {
        "type": "object",
        "properties": {
            "thought": {"type": "string"},
            "subtasks": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": 1,
                "maxItems": max_subtasks,
            },
            "logic": {"type": "string", "minLength": 1},
        },
        "required": ["thought", "subtasks", "logic"],
        "additionalProperties": False,
    }


def parse_plan(raw: str, max_subtasks: int) -> tuple[list[str], PlanExpression]:
    """解析 1-based 引用与 AND/OR；绝不执行模型生成的代码。"""
    _positive_int(max_subtasks, "max_subtasks")
    plan = json.loads(raw)
    if not isinstance(plan, dict) or set(plan) != {"thought", "subtasks", "logic"}:
        raise ValueError("ADaPT 规划只允许 thought、subtasks、logic 三个字段")
    if not isinstance(plan["thought"], str):
        raise ValueError("ADaPT thought 必须是字符串")
    subtasks = plan["subtasks"]
    if (not isinstance(subtasks, list) or not 1 <= len(subtasks) <= max_subtasks
            or any(not isinstance(task, str) or not task.strip() for task in subtasks)):
        raise ValueError(f"ADaPT subtasks 必须包含 1 至 {max_subtasks} 个非空任务")
    logic = plan["logic"]
    if not isinstance(logic, str) or not logic.strip():
        raise ValueError("ADaPT logic 必须是非空字符串")
    logic = re.sub(r"\b(?:AND|OR)\b", lambda match: match[0].lower(),
                   logic.strip(), flags=re.IGNORECASE)
    try:
        root = ast.parse(logic, mode="eval").body
    except (SyntaxError, RecursionError) as exc:
        raise ValueError("ADaPT logic 必须只使用整数任务编号、AND、OR 和括号") from exc

    references = []

    def convert(node: ast.AST) -> PlanExpression:
        if isinstance(node, ast.Constant) and type(node.value) is int:
            references.append(node.value)
            return node.value
        if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            operator = "AND" if isinstance(node.op, ast.And) else "OR"
            return operator, [convert(child) for child in node.values]
        raise ValueError("ADaPT logic 只允许整数任务编号、AND、OR 和括号")

    expression = convert(root)
    if sorted(references) != list(range(1, len(subtasks) + 1)):
        raise ValueError("ADaPT logic 必须且只能引用每个子任务一次，编号从 1 开始")
    return [task.strip() for task in subtasks], expression
