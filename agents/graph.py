"""任务图智能体：模型建图/选动作，代码调度、验证与处理生存事件。"""
from copy import deepcopy
import json
from pathlib import Path
from string import Template

from agents.base import AgentFinished
from agents.llm_agent import LLMBaseAgent
from modules.common.actions import action_schema
from modules.graph import EdgeCondition, EventMonitor, Planner, PlanningError, Scheduler, TaskGraph
from modules.graph.evaluator import evaluate, readiness
from modules.graph.execution import verify_action
from modules.graph.planner import DEFAULT_PROMPT
from modules.react.components import parse_react
from utils.io import write_json


def validate_params(params=None):
    defaults = {"max_task_steps": 150, "blocked_threshold": 3, "max_plan_calls": 48,
                "max_control_rounds": 16, "max_history_steps": 20,
                "thresholds": None, "threat_radius": 5, "clear_steps": 2}
    params = {} if params is None else dict(params)
    if set(params) - set(defaults):
        raise ValueError(f"未知 graph 参数: {sorted(set(params) - set(defaults))}")
    values = {**defaults, **params}
    for key in set(defaults) - {"thresholds", "max_history_steps"}:
        if type(values[key]) is not int or values[key] < 1:
            raise ValueError(f"graph.{key} 必须是正整数")
    window = values["max_history_steps"]
    if window is not None and (type(window) is not int or window < 1):
        raise ValueError("graph.max_history_steps 必须是正整数或 null")
    EventMonitor(values["thresholds"], threat_radius=values["threat_radius"], clear_steps=values["clear_steps"])
    return values


class GraphAgent(LLMBaseAgent):
    def __init__(self, config, llm):
        self.params = validate_params(config.get("params"))
        super().__init__({**config, "params": self.params}, llm)
        self.system_prompt = Template(self.prompts["system"]).substitute(rules=self.prompts["rules"])
        self.output_dir = None

    def reset(self, task, *, seed):
        super().reset(task, seed=seed)
        self.output_dir = None
        self.graph = TaskGraph()
        self.monitor = EventMonitor(self.params["thresholds"], threat_radius=self.params["threat_radius"],
                                    clear_steps=self.params["clear_steps"])
        condition = self.task.get("success_condition")
        if condition is None:
            raise ValueError("GraphAgent 必须配置 success_condition，最终目标不再定义在节点上")
        goal = EdgeCondition("achievements", condition["achievement"], ">=", condition.get("count", 1))
        self.scheduler = Scheduler(self.graph, max_task_steps=self.params["max_task_steps"],
                                   blocked_threshold=self.params["blocked_threshold"], goal_condition=goal,
                                   on_record=lambda row: self._record("graph_events.jsonl", row))
        self.planner = Planner(self.graph, self.query,
                               system_prompt=self.prompts.get("planner", DEFAULT_PROMPT) + "\n" + self.prompts["rules"],
                               max_calls=self.params["max_plan_calls"],
                               on_record=lambda row: self._record("plans.jsonl", row))
        self.pending = self.observation = None
        self.events, self.history = [], []

    def bind_episode(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._flush()

    def _record(self, filename, record):
        row = {"step": self.observation["step"] if self.observation else 0, **deepcopy(record)}
        if self.output_dir is not None:
            with (self.output_dir / filename).open("a", encoding="utf-8") as file:
                file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    def _flush(self):
        if self.output_dir is not None:
            write_json(self.graph.snapshot(), self.output_dir / "graph.json")
            write_json(self.scheduler.snapshot(), self.output_dir / "runtime.json")
            write_json({"pending": self.pending, "steps_executed": len(self.history),
                        "events": [event.to_dict() for event in self.events]}, self.output_dir / "executor.json")

    def _update_observation(self, observation):
        self.observation = deepcopy(dict(observation))
        self.events = self.monitor.update(self.observation)
        self.observation["safety"] = dict(self.monitor.safety)
        for change in self.monitor.changes:
            self._record("condition_events.jsonl", change)

    def _recent_history(self):
        window = self.params["max_history_steps"]
        return self.history if window is None else self.history[-window:]

    def act(self, observation):
        if self.pending is not None:
            raise RuntimeError("GraphAgent 必须先 observe 环境反馈，再调用 act")
        self.begin_decision()
        self._update_observation(observation)
        try:
            for _ in range(self.params["max_control_rounds"]):
                decision = self.scheduler.decide(self.observation, self.events)
                self.last_decision["scheduler"] = decision.to_dict()
                self._record("decisions.jsonl", decision.to_dict())
                if decision.kind == "FINISH":
                    raise AgentFinished(decision.report["goal_satisfied"])
                if decision.kind == "SUSPEND":
                    continue
                if decision.kind in ("REQUEST_PLAN", "REQUEST_REVISION"):
                    report = decision.report
                    # 仅事件请求绑定事件；不能把主任务修图误装为应对计划。
                    event = next((event for event in self.events
                                  if event.id == report.get("event", {}).get("id")), None)
                    result = self.planner.plan(
                        goal=self.task["description"], observation=self.observation,
                        mode=report["mode"], event=event, protected=self.scheduler.protected,
                        goal_condition=self.scheduler.goal_condition,
                        block_report=report, execution_feedback={"recent_steps": self._recent_history()})
                    self.scheduler.install_plan(result, event_id=event.id if event else None)
                    self._flush()
                    continue
                if decision.kind not in ("START", "RESUME", "CONTINUE"):
                    raise RuntimeError(f"未知调度决策: {decision.kind}")
                return self._execute(decision.task_id)
            raise PlanningError("调度/修图轮数达到 max_control_rounds，未产生可执行动作")
        finally:
            self._flush()

    def _execute(self, task_id):
        context = self.context(self.observation)
        context.update(task=json.dumps(self.graph.get_node_context(task_id), ensure_ascii=False),
                       demand=json.dumps(self.scheduler.current_demand, ensure_ascii=False),
                       readiness=json.dumps(readiness(self.graph, task_id, self.observation), ensure_ascii=False),
                       events=json.dumps([event.to_dict() for event in self.events], ensure_ascii=False),
                       history=json.dumps(self._recent_history(), ensure_ascii=False))
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": Template(self.prompts["step"]).substitute(context)}]
        actions = self.observation["actions"]
        raw = self.query(messages, response_schema=action_schema(actions, {"thought": {"type": "string"}}),
                         label="graph_executor")
        thought, action_id = parse_react(raw, actions)
        self.pending = {"task_id": task_id, "action": actions[action_id], "thought": thought,
                        "before": deepcopy(self.observation)}
        self.last_decision.update(task_id=task_id, thought=thought, action=actions[action_id])
        return action_id

    def observe(self, transition):
        if self.pending is None:
            raise RuntimeError("GraphAgent observe 之前必须有待执行的 act")
        if transition["action"] != self.pending["action"]:
            raise ValueError("环境反馈动作与待执行动作不一致")
        if transition["observation"]["step"] != self.pending["before"]["step"] + 1:
            raise ValueError("每次 act 必须对应一个递增的环境 step")
        super().observe(transition)
        self._update_observation(transition["observation"])
        execution = verify_action(self.pending["action"], self.pending["before"], self.observation)
        achieved = evaluate(self.scheduler.goal_condition, self.observation) == "true"
        reason = ("death" if self.observation["inventory"]["health"] <= 0 else "environment_terminated"
                  ) if transition["terminated"] else "goal_satisfied" if achieved else (
                      "truncated" if transition["truncated"] else None)
        record = {**self.pending, "demand": deepcopy(self.scheduler.current_demand),
                  "step": self.observation["step"], "after": deepcopy(self.observation),
                  "execution_feedback": execution, "events": deepcopy(self.monitor.changes),
                  "reward": transition["reward"], "terminated": transition["terminated"],
                  "truncated": transition["truncated"]}
        self.history.append(record)
        self._record("task_steps.jsonl", record)
        self.scheduler.feedback(self.pending["task_id"], self.observation, self.events,
                                execution_feedback=execution, termination_reason=reason)
        self.pending = None
        self._flush()

    def finish_episode(self, result):
        if self.scheduler.termination is None:
            # Runner 可在第一次 act 前发现成功或用尽预算；无须额外模型调用。
            observation = self.observation or {"step": result["steps"], "inventory": result["inventory"],
                                               "achievements": result["achievements"]}
            reason = ("death" if result["inventory"]["health"] <= 0 else "goal_satisfied" if result["success"]
                      else "step_limit" if result["stop_reason"] == "max_steps" else "truncated")
            self.scheduler.stop(reason, observation)
        self.pending = None
        self._flush()
