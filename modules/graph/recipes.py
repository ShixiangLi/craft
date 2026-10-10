"""只读 Crafter 制作/放置规则；不检查世界状态，也不创建任务图。"""
from copy import deepcopy

from crafter import constants

from .graph import GraphError


def recipe_tool_definition() -> dict:
    return {
        "name": "get_recipe",
        "description": "Query a native crafting/placement recipe: material costs, required facilities, outputs and placement rules. "
                       "Read-only; does not advance the environment, modify the graph or prove current availability. stone means placing stone.",
        "parameters": {"type": "object", "properties": {
            "item": {"type": "string", "enum": sorted(set(constants.make) | set(constants.place))}},
            "required": ["item"], "additionalProperties": False},
    }


def get_recipe(item: str) -> dict:
    """返回单次动作的原生规则副本；设施要求为 AND，材料数不含后续消耗。"""
    if not isinstance(item, str) or not item.strip():
        raise GraphError("invalid_arguments", "get_recipe.item must be a nonempty string")
    if item in constants.make:
        recipe = constants.make[item]
        return {"item": item, "action": f"make_{item}", "consumes": deepcopy(recipe["uses"]),
                "requires_nearby": list(recipe["nearby"]), "nearby_range": 1,
                "produces": {item: recipe["gives"]}, "output_location": "inventory",
                "inventory_max": constants.items[item]["max"]}
    if item in constants.place:
        recipe = constants.place[item]
        return {"item": item, "action": f"place_{item}", "consumes": deepcopy(recipe["uses"]),
                "requires_nearby": [], "produces": {item: 1}, "output_location": "world",
                "placement": {"target": "in_front", "allowed_terrain": list(recipe["where"]),
                              "requires_empty": True, "type": recipe["type"]}}
    supported_items = recipe_tool_definition()["parameters"]["properties"]["item"]["enum"]
    raise GraphError("unknown_recipe", f"No crafting/placement recipe for {item}; supported items: {supported_items}")
