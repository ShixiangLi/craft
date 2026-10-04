"""统一原生 Crafter 动作的结构化输出契约。"""
import json


def action_schema(actions: list[str], extra_properties: dict | None = None) -> dict:
    properties = dict(extra_properties or {})
    properties["action"] = {"type": "string", "enum": actions}
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def parse_action(raw: str, actions: list[str]) -> int:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"模型输出不是 JSON: {raw[:500]}") from exc
    action = data.get("action") if isinstance(data, dict) else None
    if not isinstance(action, str) or action not in actions:
        raise ValueError(f"模型返回非法动作: {action!r}")
    return actions.index(action)
