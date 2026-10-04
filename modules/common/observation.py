"""把已有局部观测整理为文本；不访问环境，也不增加可见范围。"""


def describe_observation(observation: dict) -> str:
    grid = observation["local_map"]
    cx, cy = len(grid[0]) // 2, len(grid) // 2
    dx, dy = observation["facing"]
    directions = {(-1, 0): "west", (1, 0): "east", (0, -1): "north", (0, 1): "south"}
    x, y = cx + dx, cy + dy
    target = grid[y][x] if 0 <= y < len(grid) and 0 <= x < len(grid[0]) else "outside the local view"
    inventory = observation["inventory"]
    vitals = ("health", "food", "drink", "energy")
    items = ", ".join(f"{k}={v}" for k, v in inventory.items() if k not in vitals and v) or "empty"
    achievements = ", ".join(f"{k}={v}" for k, v in observation["achievements"].items() if v) or "none"
    lines = [
        f"Step: {observation['step']}",
        "Vitals: " + ", ".join(f"{k}={inventory[k]}/9" for k in vitals),
        f"Inventory: {items}",
        f"Achievement counts: {achievements}",
        f"Sleeping: {observation['sleeping']}",
        f"Facing: {directions[(dx, dy)]}; adjacent target: {target} at (dx={dx:+d}, dy={dy:+d})",
        "Local map columns dx: " + " | ".join(f"{i - cx:+d}" for i in range(len(grid[0]))),
    ]
    lines.extend(f"dy={i - cy:+d}: " + " | ".join(row) for i, row in enumerate(grid))
    lines.append("Action names (requirements may be unmet): " + ", ".join(observation["actions"]))
    return "\n".join(lines)
