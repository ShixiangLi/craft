"""按原生规则校验结构化输入/输出契约；不猜测任务文字，也不修改图。"""
from crafter import constants

from .edges import EdgeCondition
from .evaluator import OPERATORS, supported
from .graph import GraphError


def _bindings(snapshot):
    actions = {f"{kind}_{item}": (kind, item, recipe)
               for kind, recipes in (("make", constants.make), ("place", constants.place))
               for item, recipe in recipes.items()}
    result = {}
    for identifier, node in snapshot["nodes"].items():
        action = node.get("recipe_action")
        if action is None:
            continue
        if not isinstance(action, str) or action not in actions:
            raise GraphError("invalid_recipe_action", f"Task {identifier} has recipe_action {action!r}, "
                             "which is not a native make_*/place_* recipe action")
        result[identifier] = actions[action]
    return result


def _positive_boolean(condition):
    return (condition["op"] == "==" and condition["value"] is True
            or condition["op"] == "!=" and condition["value"] is False)


def _positive_number(condition):
    value, op = condition["value"], condition["op"]
    return type(value) in (int, float) and (
        op in (">=", "==") and value > 0 or op == ">" and value >= 0
        or op == "!=" and value == 0)


def _validate_producers(snapshot):
    bindings = _bindings(snapshot)
    for edge in snapshot["edges"].values():
        if edge.get("kind", "dependency") != "dependency" or edge["source"] not in bindings:
            continue
        kind, item, _ = bindings[edge["source"]]
        action = f"{kind}_{item}"
        for condition in edge["conditions"]:
            source, key = condition["source"], condition["key"]
            valid = (source == "achievements" and key == action and _positive_number(condition)
                     or kind == "make" and source == "inventory" and key == item
                     and _positive_number(condition)
                     or kind == "place" and source == "local_map"
                     and key in {f"{item}_visible", f"{item}_in_range", f"{item}_in_front"}
                     and _positive_boolean(condition))
            if not valid:
                raise GraphError("invalid_producer", f"Edge {edge['id']} has source task {edge['source']} "
                                 f"bound to {action}, which cannot establish {condition}. "
                                 "Connect this condition to its actual producer; collection tasks must supply consumer materials directly.")
    return bindings


def validate_recipe_producers(graph):
    """拒绝把其他生产者的条件交给明确绑定的制作/放置行为。"""
    _validate_producers(graph.snapshot())


def _inventory_values(conditions, item):
    requirements = [condition for condition in conditions
                    if condition["source"] == "inventory" and condition["key"] == item]
    # 原生库存为有上限的非负整数，避免浮点门槛和不可满足谓词的空真。
    return [value for value in range(constants.items[item]["max"] + 1)
            if all(OPERATORS[condition["op"]](value, condition["value"])
                   for condition in requirements)]


def _resource_guaranteed(conditions, item, amount):
    feasible = _inventory_values(conditions, item)
    return bool(feasible) and min(feasible) >= amount


def _check_feasible(conditions, *, task_id, edge_ids):
    for item in sorted({c["key"] for c in conditions if c["source"] == "inventory"}):
        if not _inventory_values(conditions, item):
            raise GraphError("unsatisfiable_condition", f"Task {task_id}, edges {edge_ids}: "
                             f"conditions on inventory.{item} cannot hold simultaneously. "
                             f"Native inventory values are integers in 0..{constants.items[item]['max']}",
                             details={"task_id": task_id, "edge_ids": edge_ids, "item": item})


def _add_recipe_requirements(requirements, recipe):
    for item, amount in recipe["uses"].items():
        key = ("inventory", item)
        requirements[key] = max(requirements.get(key, 0), amount)
    for station in recipe.get("nearby", []):
        requirements[("local_map", station + "_in_range")] = True


def _inferred_requirements(effects, binding):
    """出边是源行为承诺建立的事实；终点的外部需求也作为输出校验。"""
    native = {}
    if binding:
        _add_recipe_requirements(native, binding[2])
    for condition in effects:
        source, key = condition["source"], condition["key"]
        if source in ("inventory", "achievements") and _positive_number(condition):
            for material, recipe in constants.collect.items():
                outputs = recipe["receive"]
                if (source == "inventory" and key in outputs
                        or source == "achievements" and key in {"collect_" + item for item in outputs}):
                    for tool, count in recipe["require"].items():
                        native[("inventory", tool)] = count
            for kind, recipes in (("make", constants.make), ("place", constants.place)):
                for item, recipe in recipes.items():
                    if (source == "inventory" and kind == "make" and key == item
                            or source == "achievements" and key == f"{kind}_{item}"):
                        _add_recipe_requirements(native, recipe)
    return native


def _check_requirements(task_id, branches, requirements):
    for edge_ids, conditions in branches:
        missing = []
        for (source, key), value in requirements.items():
            if source == "inventory":
                guaranteed = _resource_guaranteed(conditions, key, value)
            else:
                station = key.removesuffix("_in_range")
                guaranteed = any(c["source"] == source
                                 and c["key"] in (key, station + "_in_front")
                                 and _positive_boolean(c) for c in conditions)
            if not guaranteed:
                missing.append({"source": source, "key": key,
                                "op": ">=" if source == "inventory" else "==", "value": value})
        if missing:
            explanation = ", ".join(f"{c['source']}.{c['key']} {c['op']} {c['value']}" for c in missing)
            raise GraphError("invalid_task_dependencies",
                             f"Task {task_id} violates native Crafter prerequisites; incoming branch {edge_ids} lacks: {explanation}. "
                             "Add direct edges from the corresponding producers. Combine necessary inputs with AND; every OR branch must independently guarantee all prerequisites.",
                             details={"task_id": task_id, "edge_ids": edge_ids,
                                      "rule": "native_requirements",
                                      "missing_conditions": missing})


def validate_graph_contracts(graph, *, target_task=None, goal_condition=None):
    """统一提交检查；暂存过程中允许尚未补齐依赖，所有建图模式共用。"""
    snapshot = graph.snapshot()
    if goal_condition is not None:
        graph.get_node(target_task)
        condition = (goal_condition if isinstance(goal_condition, EdgeCondition)
                     else EdgeCondition.from_dict(goal_condition))
        snapshot["edges"]["__validation_goal"] = {
            "id": "__validation_goal", "source": target_task, "target": None,
            "conditions": [condition.to_dict()]}
    edges = list(snapshot["edges"].values())
    for edge in edges:
        for condition in edge["conditions"]:
            if not supported(EdgeCondition.from_dict(condition)):
                raise GraphError("unsupported_condition", f"Cannot evaluate condition on edge {edge['id']}: {condition}")
        _check_feasible(edge["conditions"], task_id=edge["source"], edge_ids=[edge["id"]])
    bindings = _validate_producers(snapshot)
    for identifier, node in snapshot["nodes"].items():
        incoming = [edge for edge in edges if edge["target"] == identifier
                    and edge.get("kind", "dependency") == "dependency"]
        branches = [([edge["id"]], edge["conditions"]) for edge in incoming]
        if node["incoming_logic"] == "and" or not branches:
            branches = [([edge["id"] for edge in incoming],
                         [c for edge in incoming for c in edge["conditions"]])]
        for edge_ids, conditions in branches:
            _check_feasible(conditions, task_id=identifier, edge_ids=edge_ids)
        effects = [c for edge in edges if edge["source"] == identifier
                   and edge.get("kind", "dependency") == "dependency" for c in edge["conditions"]]
        native = _inferred_requirements(effects, bindings.get(identifier))
        _check_requirements(identifier, branches, native)
