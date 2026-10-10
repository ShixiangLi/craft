"""统一主任务建图、事件建图和局部修图；模型调用由外部注入。"""
from copy import deepcopy
from dataclasses import dataclass
import json

from .edges import EdgeCondition
from .contracts import validate_graph_contracts, validate_recipe_producers
from .evaluator import condition_schema, supported
from .graph import GraphError
from .recipes import get_recipe, recipe_tool_definition
from .tools import GraphTools, _reject_constant, _unique_object, tool_definitions


DEFAULT_PROMPT = """Build or revise a task graph using the supplied JSON tool schema.
Use English for all task descriptions and other natural-language output. Use English
snake_case identifiers and preserve the exact native action, item and predicate names.
Return ONE tool call per response, or finish_plan with arguments {"target_task": "id"}.
Nodes are reusable task-level behaviors: collect a resource, craft a tool, build a
workstation, or recover from a survival event. Finding, approaching, facing a target,
and ordinary movement are internal execution steps, not separate graph nodes.
For example, use Collect Stone, not Find Stone -> Move to Stone -> Collect Stone.
Nodes contain id, task,
incoming_logic and optional recipe_action; status belongs to execution code.
All world requirements are edge predicates. An edge passes only when all of its
predicates are true NOW; its source's historical status is irrelevant. Normal incoming
edges combine using the target's AND/OR. Every edge must contain at least one condition.
Allowed predicates: inventory.<native item/vital> and achievements.<native achievement>
with numeric comparisons; local_map.<native material/object>_visible/_in_range/_in_front
with boolean ==/!=; safety.no_immediate_threat with boolean ==/!=. In_range is within
one tile, including diagonals. Only observed local absence may be asserted.
INITIAL: work backward from the externally supplied final goal and return its final
concrete behavior. Include producers for necessary materials, tools and workstations.
Connect each necessary producer DIRECTLY to the behavior that consumes its result.
Each edge states facts that its source behavior can establish; do not place unrelated
producers' requirements together on one edge or hide material dependencies behind
movement tasks. Use AND for all necessary inputs; OR only for independently sufficient
alternatives, never to bypass missing materials. Tools/stations must be available NOW.
Budget materials across remaining tasks, accounting for earlier consumption and the
inventory cap. Reuse the same collection behavior when consumed resources are needed
again; a finish status or past achievement does not prove current inventory.
A false edge activates its producer; an unknown edge needs information. The executor
handles exploration, navigation and interaction within the selected task. Collection
tasks need the required tools, but must not require the resource to be visible or in
front before they can start. Keep workstation-range requirements on crafting edges;
reaching the workstation is an execution step within fulfilling the range demand,
not a standalone movement node. Emergency tasks should express recovery objectives
such as escape danger or restore drink, not individual movement actions.
For iron crafting, table and furnace must both be in range from the same player tile.
The executor's code verifies action results; do not add completion predicates to nodes.
Only the currently requested outgoing edge needs to be established by a producer,
not every outgoing edge simultaneously. Final goal and event recovery are checked by
code outside the graph; do not create fake goal or condition-only nodes.
finish_plan checks native prerequisites inferred from outgoing predicates and the final
goal, even for unbound behaviors: stone/coal collection requires wood_pickaxe inventory,
iron requires stone_pickaxe, and diamond requires iron_pickaxe. Tool production requires
its full recipe.
After a validation error, repair the staged graph using the reported missing_conditions
and call finish_plan again; previous successful edits remain staged.
EVENT: preserve the existing graph; add a focused response subgraph with the preparation
needed for the chosen response, reusing useful producers. Never depend on protected
suspended behaviors. Return the response target.
Use the supplied RECOVERY threshold rather than the lower trigger threshold. Choose
responses using current health, equipment and threats; recovery may need food, water
and safe rest. One attack does not prove safety, nor one food action health recovery.
For threat events and their revisions, consider both escape and combat before choosing.
Compare health, enemy type/count/distance, available swords, escape routes, and the
time and safety of obtaining materials and reaching a table. Use current observations;
do not assume an unseen route or workstation is safe. Do not default to endless flight.
If equipped and able to fight safely, consider eliminating the threat. If a sword can
be made safely with little preparation, use get_recipe and include crafting plus its
necessary dependencies before combat, with a sword-inventory edge into the combat task.
Prefer an affordable effective sword; do not insist on the highest tier or craft another
when the existing sword suffices. If health is low, enemies overwhelm you, or preparation
would expose you to attack, prioritize escape or cover rather than gathering under fire.
Choose one feasible response; do not add both alternatives merely to show consideration.
When revising an ineffective response, use the execution evidence to reconsider escape
versus combat instead of only renaming the same behavior. Keep the required tool-call format.
REVISION: repair only the relevant plan using the block report and execution feedback.
For an event, modify only editable_nodes. Otherwise preserve final-goal reachability
and return the final behavior, not merely a prerequisite. Reuse inactive finished
behaviors; new nodes are needed only for different behaviors or genuine alternatives.
Budget-exhausted behaviors remain failed; provide an alternative behavior instead of
resubmitting the unchanged exhausted path. A finished behavior is reusable; a failed one is not.
Interrupt edges and IDs starting __ belong to the scheduler; do not edit/create them.
Preserve protected behaviors. Avoid cycles and unrelated new nodes. Changes are staged
until a validated finish_plan. Tools do not advance the environment.
"""


class PlanningError(RuntimeError):
    pass


@dataclass
class PlanResult:
    target_task: str
    members: set[str]
    new_nodes: set[str]


def dependency_members(graph, target):
    """普通依赖闭包；中断关系由调度器单独管理。"""
    members, visiting = set(), set()

    def visit(task_id):
        if task_id in visiting:
            raise GraphError("dependency_cycle", f"Dependency cycle through {task_id}")
        if task_id in members:
            return
        graph.get_node(task_id)
        visiting.add(task_id)
        for edge in graph.get_node_context(task_id)["incoming_edges"]:
            if edge.get("kind", "dependency") == "dependency":
                visit(edge["source"])
        visiting.remove(task_id)
        members.add(task_id)

    visit(target)
    return members


class Planner:
    def __init__(self, graph, query, system_prompt=DEFAULT_PROMPT, *, max_calls=48, on_record=None):
        """query(messages, response_schema=..., label=...) -> JSON 字符串。"""
        if type(max_calls) is not int or max_calls < 1:
            raise ValueError("max_calls 必须是正整数")
        self.graph, self.query, self.system_prompt = graph, query, system_prompt
        self.max_calls, self.on_record = max_calls, on_record

    def plan(self, *, goal: str, observation: dict, goal_condition=None, mode="initial", event=None,
             protected=(), block_report=None, execution_feedback=None) -> PlanResult:
        if mode not in ("initial", "event", "revision"):
            raise ValueError("mode 必须是 initial、event 或 revision")
        if mode == "event" and event is None:
            raise ValueError("事件建图必须提供 event")
        protected = set(protected)
        version, before = self.graph.version, self.graph.snapshot()
        staged = deepcopy(self.graph)
        predicates = condition_schema()
        recipe_tool = recipe_tool_definition()
        recipe_actions = {"type": ["string", "null"], "enum": [None] + [
            get_recipe(item)["action"] for item in recipe_tool["parameters"]["properties"]["item"]["enum"]]}
        context = {"mode": mode, "goal": goal,
                   "goal_condition": (goal_condition.to_dict() if isinstance(goal_condition, EdgeCondition)
                                      else goal_condition), "observation": observation,
                   "event": event.to_dict() if hasattr(event, "to_dict") else event,
                   "protected_tasks": sorted(protected), "block_report": block_report,
                   "execution_feedback": execution_feedback, "graph": before,
                   "remaining_calls": self.max_calls}
        definitions = [*tool_definitions(condition_schema=predicates, recipe_action_schema=recipe_actions), recipe_tool, {
            "name": "finish_plan", "description": "Validate and commit the staged graph; return the final behavior ID.",
            "parameters": {"type": "object", "properties": {"target_task": {"type": "string", "minLength": 1}},
                           "required": ["target_task"], "additionalProperties": False}}]
        # Require an explicit choice from the planner; null denotes a non-recipe behavior.
        next(item for item in definitions if item["name"] == "add_node")["parameters"]["required"].append("recipe_action")
        schema = {"oneOf": [{"type": "object", "properties": {
            "name": {"type": "string", "enum": [item["name"]]}, "arguments": item["parameters"]},
            "required": ["name", "arguments"], "additionalProperties": False} for item in definitions]}
        tool_prompt = ("\n\nCreate missing nodes with add_node before adding edges. "
                       "update_node/update_edge only modify existing objects; they never create them. "
                       "Every response uses one call, including queries and failed calls; reserve one for finish_plan. "
                       "After an error, correct the failed arguments instead of repeating the same call. "
                       "Use get_recipe(item) to check native rules before adding or revising crafting/placement dependencies. "
                       "Every crafting/placement node must declare recipe_action using the recipe's action, e.g. place_table or make_wood_pickaxe. "
                       "For other behaviors explicitly set recipe_action=null. Bound nodes are checked for correct output ownership and complete direct recipe inputs. "
                       "For a wood pickaxe: collect_wood -> make_wood_pickaxe carries wood>=1; "
                       "place_table -> make_wood_pickaxe carries table_in_range only. Use AND. "
                       "consumes lists material costs; ALL requires_nearby facilities must be within one tile, including diagonals. "
                       "output_location distinguishes inventory products from objects/materials placed in the world. "
                       "Placement requires an empty front tile with allowed terrain. Inventory products are capped at inventory_max. "
                       "Recipe facts do not prove current availability; choose producers and add their dependency edges yourself. "
                       "Edge source/target are TASK IDs; conditions[].source is an OBSERVATION source. "
                       'For a tree in front use {"source":"local_map","key":"tree_in_front","op":"==","value":true}. '
                       'For wood inventory use {"source":"inventory","key":"wood","op":">=","value":2}. '
                       "Return a JSON object with name and arguments. The response schema specifies all argument fields.\n"
                       "TOOL PURPOSES:\n"
                       # LLMClient 已把完整 response_schema 写入提示词；这里只补工具用途。
                       + "\n".join(f"{item['name']}: {item['description']}" for item in definitions))
        messages = [{"role": "system", "content": self.system_prompt + tool_prompt},
                    {"role": "user", "content": json.dumps(context, ensure_ascii=False, allow_nan=False)}]
        last_error = None
        for index in range(self.max_calls):
            raw = self.query(messages, response_schema=schema, label=f"graph_planner_{mode}")
            messages.append({"role": "assistant", "content": raw})
            try:
                call = json.loads(raw, parse_constant=_reject_constant, object_pairs_hook=_unique_object)
                if isinstance(call, dict) and call.get("name") == "add_node":
                    if not isinstance(call.get("arguments"), dict) or "recipe_action" not in call["arguments"]:
                        raise GraphError("invalid_arguments", "add_node must declare recipe_action: use the native recipe action for crafting/placement, or null for other tasks")
                if isinstance(call, dict) and call.get("name") == "finish_plan":
                    if (set(call) != {"name", "arguments"} or not isinstance(call["arguments"], dict)
                            or set(call["arguments"]) != {"target_task"}):
                        raise ValueError("finish_plan accepts only arguments.target_task")
                    target = call["arguments"]["target_task"]
                    members = self._validate(staged, before, target, mode, event, protected,
                                             block_report or {}, goal_condition)
                    result = PlanResult(target, members, set(staged.snapshot()["nodes"]) - set(before["nodes"]))
                    self.graph.commit(staged, expected_version=version)
                    self._record({"type": "plan_committed", "mode": mode, "target_task": target,
                                  "new_nodes": sorted(result.new_nodes), "graph": self.graph.snapshot()})
                    return result
                # Report unsupported predicates before they enter the staged graph.
                if isinstance(call, dict) and call.get("name") in ("add_edge", "update_edge"):
                    arguments = call.get("arguments")
                    if isinstance(arguments, dict):
                        changes = arguments if call["name"] == "add_edge" else arguments.get("changes")
                        if isinstance(changes, dict) and isinstance(changes.get("conditions"), list):
                            for item in changes["conditions"]:
                                condition = EdgeCondition.from_dict(item)
                                if not supported(condition):
                                    raise GraphError("unsupported_condition", f"Cannot evaluate condition: {condition.to_dict()}. "
                                        "conditions[].source must be inventory/achievements/local_map/safety, "
                                        "not a task ID. key must be a supported field without its source prefix. "
                                        "Correct the condition using the response JSON Schema and retry; the failed call did not modify the graph.")
                if isinstance(call, dict) and call.get("name") == "get_recipe":
                    if (set(call) != {"name", "arguments"} or not isinstance(call["arguments"], dict)
                            or set(call["arguments"]) != {"item"}):
                        raise GraphError("invalid_arguments", "get_recipe accepts only arguments.item")
                    response = {"ok": True, "name": "get_recipe", "version": staged.version,
                                "result": get_recipe(call["arguments"]["item"]), "error": None}
                else:
                    candidate = deepcopy(staged)
                    response = GraphTools(candidate).execute(call)
                    if response["ok"]:
                        # Reject a wrong producer before it enters the staged graph.
                        validate_recipe_producers(candidate)
                        staged = candidate
            except (ValueError, TypeError) as exc:
                response = {"ok": False, "error": {"code": getattr(exc, "code", "invalid_plan"), "message": str(exc)}}
                if getattr(exc, "details", None) is not None:
                    response["error"]["details"] = exc.details
            last_error = response.get("error") if not response["ok"] else None
            response["remaining_calls"] = self.max_calls - index - 1
            self._record({"type": "staged_tool", "mode": mode, "request": raw, "response": response})
            messages.append({"role": "user", "content": json.dumps(response, ensure_ascii=False)})
        self._record({"type": "plan_rejected", "mode": mode, "error": last_error})
        detail = f"最后错误：{last_error}" if last_error else "最后工具调用成功，但尚未提交 finish_plan"
        raise PlanningError(f"Planner 调用额度耗尽，暂存修改未提交；{detail}")

    @staticmethod
    def _validate(graph, before, target, mode, event, protected, report, goal_condition=None):
        snapshot = graph.snapshot()
        editable = set(report.get("editable_nodes", []))
        for identifier, node in before["nodes"].items():
            locked = (identifier in protected or mode == "event"
                      or event is not None and mode == "revision" and identifier not in editable)
            if locked and snapshot["nodes"].get(identifier) != node:
                raise ValueError(f"Cannot delete or modify protected task: {identifier}")
        for identifier, edge in before["edges"].items():
            locked = (mode == "event" or edge.get("kind") == "interrupt"
                      or event is not None and mode == "revision" and edge["target"] not in editable)
            if locked and snapshot["edges"].get(identifier) != edge:
                raise ValueError(f"Cannot delete or modify protected edge: {identifier}")
        for identifier, node in snapshot["nodes"].items():
            if identifier not in before["nodes"] and identifier.startswith("__"):
                raise ValueError("The __ prefix is reserved for the scheduler")
        for identifier, edge in snapshot["edges"].items():
            if (edge.get("kind") == "interrupt" and before["edges"].get(identifier) != edge
                    or identifier not in before["edges"] and identifier.startswith("__")):
                raise ValueError("Interrupt edges are managed by the scheduler")
            if any(not supported(EdgeCondition.from_dict(item)) for item in edge["conditions"]):
                raise GraphError("unsupported_condition", f"Cannot evaluate conditions on edge {identifier}")
        members = dependency_members(graph, target)
        # 事件子图的终点承担恢复需求，不承担整个回合的最终目标。
        if event is not None:
            event_data = event.to_dict() if hasattr(event, "to_dict") else event
            goal_condition = (EdgeCondition("safety", "no_immediate_threat", "==", True)
                              if event_data["kind"] == "threat" else
                              EdgeCondition("inventory", event_data["kind"], ">=", event_data["evidence"]["recover"]))
        validate_graph_contracts(graph, target_task=target, goal_condition=goal_condition)
        # 同一事件的修图可以给其暂停目标补前置任务；不能依赖其他被暂停任务。
        resumable_target = {target} if mode == "revision" and target in editable else set()
        if event is not None and members & (protected - resumable_target):
            raise ValueError("An emergency subgraph cannot depend on protected suspended tasks")
        new_nodes = set(snapshot["nodes"]) - set(before["nodes"])
        if not new_nodes <= members:
            raise ValueError(f"New tasks must connect to the current target {target}; disconnected nodes: {sorted(new_nodes - members)}")
        if event is not None and any(edge["target"] not in members for identifier, edge in snapshot["edges"].items()
                                     if identifier not in before["edges"]):
            raise ValueError("An emergency subgraph cannot add ordinary edges into external tasks")
        return members

    def _record(self, record):
        if self.on_record:
            self.on_record(record)
