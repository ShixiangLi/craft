"""ReAct 输出解析和轨迹序列化；不包含独立规划器或反思学习。"""
import json

from modules.common.actions import parse_action


def parse_react(raw: str, actions: list[str]) -> tuple[str, int]:
    action_id = parse_action(raw, actions)
    decision = json.loads(raw)
    thought = decision.get("thought")
    if not isinstance(thought, str):
        raise ValueError("ReAct 输出必须包含字符串 thought；无需新思考时使用空字符串")
    if set(decision) != {"thought", "action"}:
        raise ValueError("ReAct 输出只允许 thought 和 action，Observation 必须来自环境")
    return thought.strip(), action_id


def render_history(history: list[dict], max_steps: int | None = None) -> str:
    if not history:
        return "No completed interactions (episode start)."
    retained = history if max_steps is None else history[-max_steps:]
    omitted = len(history) - len(retained)
    lines = []
    if omitted:
        lines.append(f"Earlier {omitted} interactions omitted by max_history_steps. "
                     "Locations in old observations are relative to the player at that time.")
    for entry in retained:
        lines.append(f"Observation {entry['step']}:\n{entry['observation']}")
        if entry["thought"]:
            lines.append(f"Thought: {entry['thought']}")
        lines.append(f"Action: {entry['action']}")
        lines.append("Environment feedback: " + json.dumps(entry["feedback"], ensure_ascii=False))
    return "\n\n".join(lines)
