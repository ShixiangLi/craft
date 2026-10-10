"""按选定依赖路径推进：先完成前序，再返回消费者；维护中断与执行预算。"""
from copy import deepcopy
from dataclasses import asdict, dataclass, field

from .edges import EdgeCondition, TaskEdge
from .evaluator import evaluate, readiness, supported
from .graph import GraphError
from .planner import dependency_members


STOP_REASONS = ("goal_satisfied", "death", "environment_terminated", "truncated", "step_limit", "cancelled")


@dataclass
class Decision:
    kind: str
    task_id: str | None = None
    report: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class EmergencyPlan:
    target_task: str
    members: set[str]
    new_nodes: set[str]
    interrupted_task: str | None
    interrupt_edge: str | None
    demand: dict


class Scheduler:
    def __init__(self, graph, *, goal_condition, max_task_steps=80, blocked_threshold=3, on_record=None):
        if type(max_task_steps) is not int or max_task_steps < 1:
            raise ValueError("max_task_steps 必须是正整数")
        if type(blocked_threshold) is not int or blocked_threshold < 1:
            raise ValueError("blocked_threshold 必须是正整数")
        if not isinstance(goal_condition, EdgeCondition):
            raise ValueError("goal_condition 必须是明确的 EdgeCondition")
        self.graph, self.max_task_steps = graph, max_task_steps
        self.goal_condition, self.on_record = goal_condition, on_record
        self.target_task = self.current = None
        # 每个主线/事件保存选定的依赖路径；前序完成后返回直接消费者。
        self.progression = {}
        self.suspended = []
        self.demands, self.activation_steps = {}, {}
        self.emergency_plans = {}
        self.events = []
        self.failed, self.archived = set(), set()
        self.steps_used = {}
        self.last_feedback_step = None
        self.blocked_threshold = blocked_threshold
        self.blocked_feedback = {}
        self.termination = None

    @property
    def protected(self):
        return set(self.suspended) | ({self.current} if self.current else set())

    @property
    def current_demand(self):
        return deepcopy(self.demands.get(self.current))

    @staticmethod
    def _demand(conditions, *, consumer=None, edge_id=None):
        return {"conditions": [condition.to_dict() for condition in conditions],
                "consumer": consumer, "edge_id": edge_id}

    @staticmethod
    def _fulfilled(demand, observation):
        return all(evaluate(EdgeCondition.from_dict(item), observation) == "true"
                   for item in demand["conditions"])

    @classmethod
    def _event_demand(cls, event):
        condition = (EdgeCondition("safety", "no_immediate_threat", "==", True) if event.kind == "threat"
                     else EdgeCondition("inventory", event.kind, ">=", event.evidence["recover"]))
        return cls._demand((condition,))

    def install_plan(self, result, *, event_id=None):
        if self.termination is not None:
            raise RuntimeError("回合已结束，不能安装计划")
        members = dependency_members(self.graph, result.target_task)
        if result.members != members or not result.new_nodes <= members:
            raise ValueError("PlanResult 与当前图不一致")
        if event_id is None:
            self.target_task = result.target_task
            self.progression.pop("main", None)
            self.archived.difference_update(members)
            self._clear_blocks(members)
            for task_id in set(self.demands) - members:
                self._cancel(task_id, "plan_replaced")
            return
        event = next((event for event in self.events if event.id == event_id), None)
        if event is None:
            raise ValueError("只能安装当前活跃事件的应对计划")
        old = self.emergency_plans.get(event_id)
        resumable = {result.target_task} if old and result.target_task == old.target_task else set()
        if members & (self.protected - resumable):
            raise ValueError("应对子图不能依赖其他被暂停任务")
        interrupted = old.interrupted_task if old else (self.suspended[-1] if self.suspended else None)
        edge_id = old.interrupt_edge if old else None
        demand = self._event_demand(event)
        staged, version = deepcopy(self.graph), self.graph.version
        if edge_id:
            staged.delete_edge(edge_id)
        if interrupted:
            edge_id = f"__interrupt_{event_id}"
            conditions = tuple(EdgeCondition.from_dict(item) for item in demand["conditions"])
            staged.add_edge(TaskEdge(edge_id, result.target_task, interrupted, conditions, kind="interrupt"))
        self.graph.commit(staged, expected_version=version)
        self.emergency_plans[event_id] = EmergencyPlan(
            result.target_task, members, result.new_nodes | (old.new_nodes if old else set()),
            interrupted, edge_id, demand)
        self.progression.pop(event_id, None)
        self.archived.difference_update(members)
        self._clear_blocks(members)
        self._record("install_emergency_plan", event_id=event_id, target_task=result.target_task,
                     interrupt_edge=self.graph.get_edge(edge_id).to_dict() if edge_id else None)

    def update(self, observation, events):
        if self.termination is not None:
            return
        # 修图后需求跟随原边；OR 的替代路径已成立时不再补无用资源。
        for task_id, demand in list(self.demands.items()):
            if demand["edge_id"] is None:
                continue
            try:
                edge = self.graph.get_edge(demand["edge_id"])
            except GraphError:
                self._cancel(task_id, "demand_removed")
                continue
            if edge.source != task_id or edge.target != demand["consumer"]:
                self._cancel(task_id, "demand_replaced")
                continue
            demand["conditions"] = [condition.to_dict() for condition in edge.conditions]
            if not self._fulfilled(demand, observation) and readiness(self.graph, edge.target, observation)["ready"]:
                self._cancel(task_id, "alternative_satisfied")
        # 先验证本次需求，再检查消耗后的输入、预算和突发事件。
        for task_id in list(self.demands):
            if self._fulfilled(self.demands[task_id], observation):
                self._finish(task_id)
        self.events = sorted(events, key=lambda event: (event.priority, event.detected_at, event.id))
        active = {event.id for event in self.events}
        for identifier, plan in list(self.emergency_plans.items()):
            if identifier in active:
                continue
            if plan.interrupt_edge:
                self.graph.delete_edge(plan.interrupt_edge)
            self.archived.update(plan.new_nodes)
            self._clear_blocks(plan.new_nodes)
            for task_id in plan.new_nodes:
                if task_id in self.demands:
                    self._cancel(task_id, "event_resolved")
            self.suspended = [task for task in self.suspended if task not in plan.new_nodes]
            del self.emergency_plans[identifier]
            self.progression.pop(identifier, None)
            self._record("release_emergency", event_id=identifier)

    def feedback(self, task_id, observation, events, *, execution_feedback=None, termination_reason=None):
        """动作结果记录与需求达成分开；一次采集成功不等于数量需求满足。"""
        if self.termination is not None:
            raise RuntimeError("回合已结束，不能提交动作反馈")
        if task_id != self.current:
            raise ValueError("反馈必须属于当前运行任务")
        step = observation["step"]
        if self.last_feedback_step is not None and step <= self.last_feedback_step:
            raise ValueError("动作反馈 step 必须递增，不能重复记账")
        if execution_feedback is not None and (not isinstance(execution_feedback, dict)
                or execution_feedback.get("execution_state") not in ("succeeded", "blocked", "unknown")):
            raise ValueError("execution_feedback.execution_state 必须是 succeeded、blocked 或 unknown")
        if termination_reason is not None and termination_reason not in STOP_REASONS:
            raise ValueError(f"termination_reason 必须属于 {STOP_REASONS}")
        self.last_feedback_step = step
        self.steps_used[task_id] = self.steps_used.get(task_id, 0) + 1
        self.activation_steps[task_id] = self.activation_steps.get(task_id, 0) + 1
        self.update(observation, events)
        if termination_reason is not None:
            return self.stop(termination_reason, observation)
        if self.current == task_id:
            if execution_feedback and execution_feedback["execution_state"] == "blocked":
                self.blocked_feedback.setdefault(task_id, []).append(
                    {**deepcopy(execution_feedback), "step": step})
                self.blocked_feedback[task_id] = self.blocked_feedback[task_id][-self.blocked_threshold:]
            else:
                self.blocked_feedback.pop(task_id, None)
            if self.activation_steps[task_id] >= self.max_task_steps:
                self.failed.add(task_id)
                self.blocked_feedback.pop(task_id, None)
                self.graph.update_node(task_id, {"status": "failed"})
                demand = self.demands.pop(task_id)
                self._leave_progression(task_id, demand)
                self.current = None
                self._record("task_failed", task_id=task_id, reason="step_budget_exhausted")

    def decide(self, observation, events) -> Decision:
        if self.termination is not None:
            return Decision("FINISH", report=deepcopy(self.termination))
        self.update(observation, events)
        if evaluate(self.goal_condition, observation) == "true":
            return self.stop("goal_satisfied", observation)
        event = self.events[0] if self.events else None
        plan = self.emergency_plans.get(event.id) if event else None
        if event and (plan is None or self.current not in plan.members):
            if self.current:
                return self._suspend("emergency", event_id=event.id)
            if plan is None:
                return Decision("REQUEST_PLAN", report={"mode": "event", "event": event.to_dict()})
        if self.current:
            if len(self.blocked_feedback.get(self.current, [])) >= self.blocked_threshold:
                return self._suspend("execution_blocked")
            state = readiness(self.graph, self.current, observation)
            if state["ready"]:
                return Decision("CONTINUE", self.current, {"demand": self.current_demand})
            return self._suspend("dependency_invalidated", readiness=state)
        target = plan.target_task if plan else self.target_task
        if target is None:
            return Decision("REQUEST_PLAN", report={"mode": "initial"})
        demand = deepcopy(plan.demand) if plan else self._demand((self.goal_condition,))
        scope = event.id if event else "main"
        frames = self._progression_frames(scope, observation)
        try:
            relevant = dependency_members(self.graph, target)
        except GraphError:
            relevant = {target}
        paused = next((task for task in reversed(self.suspended) if task in relevant
                       and task not in self.failed and task not in self.archived), None)
        if frames:
            target, demand = frames[-1]["task_id"], deepcopy(frames[-1]["demand"])
        elif (paused and len(self.blocked_feedback.get(paused, [])) < self.blocked_threshold
                and readiness(self.graph, paused, observation)["ready"]):
            target, demand = paused, deepcopy(self.demands[paused])
        report = {"mode": "revision", "target_task": target, "main_target": self.target_task,
                  "demand": deepcopy(demand), "reason": "no_executable_path", "trace": [],
                  "blocked_tasks": [], "failed_tasks": sorted(self.failed)}
        if event:
            report.update(event=event.to_dict(), editable_nodes=sorted(plan.new_nodes))
        candidate = self._find(target, demand, observation, [], report, self._dependency_depths())
        if candidate:
            self.progression[scope] = frames[:-1] + candidate if frames else candidate
            task_id, requested = candidate[-1]["task_id"], candidate[-1]["demand"]
            resumed = task_id in self.suspended
            reactivated = self.graph.get_node(task_id).status == "finish"
            if resumed:
                self.suspended.remove(task_id)
            else:
                self.activation_steps[task_id] = 0
            self.current = task_id
            self.demands[task_id] = requested
            self.graph.update_node(task_id, {"status": "doing"})
            self._record("resume" if resumed else "start", task_id=task_id, demand=deepcopy(requested),
                         reactivated=reactivated)
            return Decision("RESUME" if resumed else "START", task_id, {"demand": self.current_demand})
        return Decision("REQUEST_REVISION", report=report)

    def _find(self, task_id, demand, observation, path, report, depths):
        if self._fulfilled(demand, observation):
            return None
        if task_id in path:
            report.update(reason="dependency_cycle", trace=path + [task_id])
            return None
        if any(not supported(EdgeCondition.from_dict(item)) for item in demand["conditions"]):
            report.update(reason="unsupported_condition", trace=path + [task_id])
            return None
        if task_id in self.failed or task_id in self.archived:
            report.update(reason="attempts_exhausted", trace=path + [task_id])
            return None
        state = readiness(self.graph, task_id, observation)
        feedback = self.blocked_feedback.get(task_id, [])
        if len(feedback) >= self.blocked_threshold:
            report.update(reason="execution_blocked", blocked_task=task_id, trace=path + [task_id],
                          execution_feedback=deepcopy(feedback))
            report["blocked_tasks"].append(state)
            return None
        if state["ready"]:
            return [{"task_id": task_id, "demand": deepcopy(demand)}]
        report["blocked_tasks"].append(state)
        edges = state["incoming_edges"]
        if self.graph.get_node(task_id).incoming_logic == "and":
            # 先展开尚需构建的前置链；不让消费者的直接材料抢在其前置任务之前。
            # 只用图结构排序，不写死资源/配方优先级。OR 保留原有候选顺序。
            edges = sorted(edges, key=lambda edge: -depths.get(edge["source"], 0))
        for edge in edges:
            if edge["satisfied"]:
                continue
            conditions = tuple(EdgeCondition.from_dict(item["predicate"]) for item in edge["conditions"])
            if any(not supported(condition) for condition in conditions):
                report.update(reason="unsupported_condition", trace=path + [task_id, edge["source"]])
                continue
            requested = self._demand(conditions, consumer=task_id, edge_id=edge["edge_id"])
            candidate = self._find(edge["source"], requested, observation, path + [task_id], report, depths)
            if candidate:
                return [{"task_id": task_id, "demand": deepcopy(demand)}] + candidate
        report["trace"] = report["trace"] or path + [task_id]
        return None

    def _dependency_depths(self):
        """入口深度为零；前置链深度仅用于首次展开 AND 的分支。"""
        snapshot = self.graph.snapshot()
        incoming = {task: [] for task in snapshot["nodes"]}
        for edge in snapshot["edges"].values():
            if edge.get("kind", "dependency") == "dependency":
                incoming[edge["target"]].append(edge["source"])
        depths = {}

        def visit(task, path):
            if task in path:
                return 0  # 实际路径中的依赖环由 _find 报告。
            if task not in depths:
                depths[task] = max((visit(source, path | {task}) + 1
                                    for source in incoming[task]), default=0)
            return depths[task]

        for task in incoming:
            visit(task, set())
        return depths

    def _progression_frames(self, scope, observation):
        frames = self.progression.get(scope, [])
        for index, frame in enumerate(frames):
            demand = frame["demand"]
            try:
                self.graph.get_node(frame["task_id"])
                if demand["edge_id"] is not None:
                    edge = self.graph.get_edge(demand["edge_id"])
                    if edge.source != frame["task_id"] or edge.target != demand["consumer"]:
                        break
                    demand["conditions"] = [c.to_dict() for c in edge.conditions]
            except GraphError:
                break
            if self._fulfilled(demand, observation):
                break
        else:
            return frames
        self.progression[scope] = frames[:index]
        return self.progression[scope]

    def _leave_progression(self, task_id, demand):
        for scope, frames in self.progression.items():
            for index, frame in enumerate(frames):
                if (frame["task_id"] == task_id
                        and frame["demand"]["edge_id"] == demand["edge_id"]
                        and frame["demand"]["consumer"] == demand["consumer"]):
                    self.progression[scope] = frames[:index]
                    break

    def _cancel(self, task_id, reason):
        demand = self.demands.pop(task_id)
        self._leave_progression(task_id, demand)
        self.blocked_feedback.pop(task_id, None)
        self.graph.update_node(task_id, {"status": "pending"})
        if self.current == task_id:
            self.current = None
        self.suspended = [task for task in self.suspended if task != task_id]
        self._record("cancel_unneeded_task", task_id=task_id, reason=reason, demand=demand)

    def _finish(self, task_id):
        demand = self.demands.pop(task_id)
        self._leave_progression(task_id, demand)
        self.blocked_feedback.pop(task_id, None)
        self.graph.update_node(task_id, {"status": "finish"})
        if self.current == task_id:
            self.current = None
        self.suspended = [task for task in self.suspended if task != task_id]
        self._record("finish", task_id=task_id, demand=demand)

    def _suspend(self, reason, **details):
        task_id, self.current = self.current, None
        if len(self.blocked_feedback.get(task_id, [])) < self.blocked_threshold:
            self.blocked_feedback.pop(task_id, None)
        if task_id not in self.suspended:
            self.suspended.append(task_id)
        self.graph.update_node(task_id, {"status": "suspended"})
        self._record("suspend", task_id=task_id, reason=reason, demand=deepcopy(self.demands[task_id]), **details)
        return Decision("SUSPEND", task_id, {"reason": reason, **details})

    def stop(self, reason, observation) -> Decision:
        if self.termination is not None:
            return Decision("FINISH", report=deepcopy(self.termination))
        if reason not in STOP_REASONS:
            raise ValueError(f"reason 必须属于 {STOP_REASONS}")
        self.update(observation, self.events)
        achieved = evaluate(self.goal_condition, observation) == "true"
        if reason == "goal_satisfied" and not achieved:
            raise ValueError("目标尚未满足，不能按成功结束")
        if reason == "step_limit" and achieved:
            reason = "goal_satisfied"
        ended_task = None
        if self.current is not None:
            ended_task = {"task_id": self.current, "steps_used": self.steps_used.get(self.current, 0),
                          "activation_steps": self.activation_steps.get(self.current, 0),
                          "demand": self.current_demand, "demand_satisfied": False}
        self.termination = {"reason": reason, "step": observation["step"], "goal_satisfied": achieved,
                            "ended_task": ended_task, "suspended_tasks": list(self.suspended)}
        for task_id in self.demands:
            self.graph.update_node(task_id, {"status": "pending"})
        self.current = None
        self.progression.clear()
        self.suspended.clear()
        self.demands.clear()
        self.blocked_feedback.clear()
        if ended_task is not None:
            self._record("execution_ended", reason=reason, **ended_task)
        self._record("episode_ended", **deepcopy(self.termination))
        return Decision("FINISH", report=deepcopy(self.termination))

    def snapshot(self):
        return {"target_task": self.target_task, "current_task": self.current, "demand": self.current_demand,
                "progression": deepcopy(self.progression),
                "suspended_stack": list(self.suspended),
                "suspended_demands": {task: deepcopy(self.demands[task]) for task in self.suspended},
                "steps_used": dict(self.steps_used), "activation_steps": dict(self.activation_steps),
                "failed_tasks": sorted(self.failed), "archived_tasks": sorted(self.archived),
                "blocked_feedback": deepcopy(self.blocked_feedback), "termination": deepcopy(self.termination),
                "emergency_plans": {identifier: {**asdict(plan), "members": sorted(plan.members),
                                                "new_nodes": sorted(plan.new_nodes)}
                                    for identifier, plan in self.emergency_plans.items()}}

    def _clear_blocks(self, members):
        for task_id in members:
            self.blocked_feedback.pop(task_id, None)

    def _record(self, kind, **details):
        if self.on_record:
            self.on_record({"type": kind, **details})
