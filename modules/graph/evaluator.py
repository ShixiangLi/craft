"""根据允许的 Crafter 局部观测求值，不读取环境私有状态。"""
from collections.abc import Mapping
import math
import operator

from crafter import constants

from .edges import EdgeCondition


OPERATORS = {"==": operator.eq, "!=": operator.ne, ">": operator.gt,
             ">=": operator.ge, "<": operator.lt, "<=": operator.le}
LABELS = set(constants.materials) | {"cow", "zombie", "skeleton", "plant", "player", "boundary", "arrow"}
SUFFIXES = ("_in_range", "_visible", "_in_front")
CONDITION_KEYS = {
    "inventory": sorted(constants.items),
    "achievements": sorted(constants.achievements),
    "local_map": sorted(label + suffix for label in LABELS for suffix in SUFFIXES),
    "safety": ["no_immediate_threat"],
}


def condition_schema() -> dict:
    """Planner 的条件语言与 evaluator 使用同一套字段白名单。"""
    branches = []
    for source, keys in CONDITION_KEYS.items():
        numeric = source in ("inventory", "achievements")
        branches.append({"type": "object", "properties": {
            "source": {"type": "string", "enum": [source], "description": "Observation source, not a task ID"},
            "key": {"type": "string", "enum": keys, "description": "Field name without the source prefix"},
            "op": {"type": "string", "enum": list(OPERATORS) if numeric else ["==", "!="]},
            "value": {"type": "number" if numeric else "boolean"}},
            "required": ["source", "key", "op", "value"], "additionalProperties": False})
    return {"oneOf": branches}


def supported(condition: EdgeCondition) -> bool:
    if condition.key not in CONDITION_KEYS.get(condition.source, ()):
        return False
    if condition.source in ("inventory", "achievements"):
        return type(condition.value) in (int, float)
    return condition.op in ("==", "!=") and type(condition.value) is bool


def valid_grid(grid) -> bool:
    return (isinstance(grid, list) and bool(grid) and isinstance(grid[0], list) and bool(grid[0])
            and all(isinstance(row, list) and len(row) == len(grid[0])
                    and all(isinstance(cell, str) for cell in row) for row in grid))


def movement_feedback(action: str, before: Mapping, after: Mapping) -> dict:
    """只报告可确认的静态障碍；地图未变化不证明受阻，动态对象暂不判断。"""
    result = {"action": action, "execution_state": "unknown"}
    directions = {"move_left": (-1, 0), "move_right": (1, 0),
                  "move_up": (0, -1), "move_down": (0, 1)}
    direction = directions.get(action)
    grid = before.get("local_map")
    if (direction is None or before.get("sleeping") is not False or not valid_grid(grid)
            or tuple(after.get("facing", ())) != direction):
        return result
    x, y = len(grid[0]) // 2 + direction[0], len(grid) // 2 + direction[1]
    if 0 <= y < len(grid) and 0 <= x < len(grid[0]):
        tile = grid[y][x]
        # 原生 Player 可走 lava（会死亡），不能误报为受阻。
        obstacles = (set(constants.materials) - set(constants.walkable) - {"lava"}) | {"boundary"}
        if tile in obstacles:
            result.update(execution_state="blocked", reason="obstacle", evidence={"front_tile": tile})
    return result


def evaluate(condition: EdgeCondition, observation: Mapping) -> str:
    """返回 true/false/unknown；省略的库存/成就项按现有观测契约计为零。"""
    if not supported(condition):
        return "unknown"
    if condition.source in ("inventory", "achievements"):
        values = observation.get(condition.source)
        if not isinstance(values, Mapping):
            return "unknown"
        actual = values.get(condition.key, 0)
        if type(actual) not in (int, float) or not math.isfinite(actual):
            return "unknown"
    elif condition.source == "safety":
        values = observation.get("safety", {})
        actual = values.get(condition.key) if isinstance(values, Mapping) else None
        if type(actual) is not bool:
            return "unknown"
    else:
        grid = observation.get("local_map")
        if not valid_grid(grid):
            return "unknown"
        cx, cy = len(grid[0]) // 2, len(grid) // 2
        suffix = next(suffix for suffix in SUFFIXES if condition.key.endswith(suffix))
        label = condition.key[:-len(suffix)]
        if suffix == "_in_front":
            facing = observation.get("facing")
            if not isinstance(facing, (list, tuple)) or tuple(facing) not in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                return "unknown"
            x, y = cx + facing[0], cy + facing[1]
            if not (0 <= y < len(grid) and 0 <= x < len(grid[0])):
                return "unknown"
            actual = grid[y][x] == label
        else:
            actual = any(cell == label and (suffix == "_visible" or max(abs(x-cx), abs(y-cy)) <= 1)
                         for y, row in enumerate(grid) for x, cell in enumerate(row))
    return "true" if OPERATORS[condition.op](actual, condition.value) else "false"


def readiness(graph, task_id: str, observation: Mapping) -> dict:
    """普通边按 AND/OR，中断边全部必须满足。无普通入边的任务作为入口。"""
    context = graph.get_node_context(task_id)
    reports = []
    for edge in context["incoming_edges"]:
        values = [evaluate(EdgeCondition.from_dict(item), observation) for item in edge["conditions"]]
        source_status = graph.get_node(edge["source"]).status.value
        reports.append({"edge_id": edge["id"], "source": edge["source"],
                        "kind": edge.get("kind", "dependency"), "source_status": source_status,
                        "conditions": [{"predicate": predicate, "value": value}
                                       for predicate, value in zip(edge["conditions"], values)],
                        "satisfied": all(value == "true" for value in values)})
    normal = [item["satisfied"] for item in reports if item["kind"] == "dependency"]
    interrupts = [item["satisfied"] for item in reports if item["kind"] == "interrupt"]
    ordinary_ready = (all(normal) if context["node"]["incoming_logic"] == "and" else any(normal)) if normal else True
    return {"task_id": task_id, "ready": ordinary_ready and all(interrupts),
            "incoming_edges": reports}
