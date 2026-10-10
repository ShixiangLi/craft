"""Verify native action effects from before/after observations, without model claims."""
from .evaluator import movement_feedback, valid_grid


def _changes(before, after, source):
    previous, current = before.get(source, {}), after.get(source, {})
    return {key: value - previous.get(key, 0) for key, value in current.items()
            if type(value) in (int, float) and type(previous.get(key, 0)) in (int, float)
            and value != previous.get(key, 0)}


def verify_action(action, before, after):
    """succeeded proves an observed effect, not that the current demand is fulfilled."""
    result = movement_feedback(action, before, after)
    inventory = _changes(before, after, "inventory")
    achievements = _changes(before, after, "achievements")
    result["inventory_changes"], result["achievement_changes"] = inventory, achievements
    if before.get("sleeping") is True and action != "sleep":
        return {**result, "execution_state": "unknown", "reason": "sleep_overrides_action"}
    if action.startswith("make_"):
        item = action.removeprefix("make_")
        if "inventory" not in before or "inventory" not in after:
            return result
        increase = inventory.get(item, 0)
        result.update(execution_state="succeeded" if increase > 0 else "blocked",
                      reason="crafted_item_increased" if increase > 0 else "crafted_item_not_increased",
                      evidence={"item": item, "before": before["inventory"].get(item, 0),
                                "after": after["inventory"].get(item, 0)})
    elif action.startswith("place_"):
        label = action.removeprefix("place_")
        grid = after.get("local_map")
        if valid_grid(grid):
            facing = before.get("facing", ())
            if tuple(facing) in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                x, y = len(grid[0]) // 2 + facing[0], len(grid) // 2 + facing[1]
                if not (0 <= y < len(grid) and 0 <= x < len(grid[0])):
                    return result
                prior = before.get("local_map")
                if valid_grid(prior) and len(prior) == len(grid) and len(prior[0]) == len(grid[0]):
                    placed = grid[y][x] == label and prior[y][x] != label
                    result.update(execution_state="succeeded" if placed else "blocked",
                                  reason="placement_observed" if placed else "placement_not_observed",
                                  evidence={"label": label, "before": prior[y][x], "after": grid[y][x]})
    elif action == "do":
        increased = {key: value for key, value in inventory.items()
                     if value > 0 and key not in ("health", "energy")}
        if increased or any(value > 0 for value in achievements.values()):
            result.update(execution_state="succeeded", reason="interaction_effect_observed",
                          evidence={"inventory_increases": increased, "achievement_changes": achievements})
    elif action == "sleep" and (after.get("sleeping") is True or inventory.get("energy", 0) > 0):
        result.update(execution_state="succeeded", reason="sleep_observed")
    return result
