"""ADaPT：先执行，失败后按需分解，递归执行 AND/OR 子任务。

递归生成器只在真实动作处挂起，继续使用统一 act/observe 环境闭环。
采用官方 TextCraft 的连续环境语义，不回滚失败动作、不重执行父任务。
"""
import json
from string import Template
from typing import Any, Mapping

from agents.base import AgentFinished
from agents.react import ReActAgent
from modules.adapt.components import parse_plan, planner_schema, validate_params
from modules.react.components import render_history


class AdaptAgent(ReActAgent):
    def __init__(self, config: Mapping[str, Any], llm):
        self.params = validate_params(config.get("params"))
        super().__init__({**config, "params": self.params}, llm)
        self._controller = None

    def reset(self, task: Mapping[str, Any], *, seed: int) -> None:
        if self._controller is not None:
            self._controller.close()
        super().reset(task, seed=seed)
        self.task_path = []
        self._observation = None
        self._finished = None
        self._controller = self._solve(self.task["description"], depth=1)

    def act(self, observation: Mapping[str, Any]) -> int:
        if self.pending is not None:
            raise RuntimeError("ADaPT 必须先 observe 实际环境反馈，再进行下一次 act")
        self.begin_decision()
        self.last_decision["events"] = []
        self._observation = observation
        if self._finished is not None:
            raise AgentFinished(self._finished)
        try:
            action_id = next(self._controller)
        except StopIteration as result:
            self._finished = bool(result.value)
            self.last_decision["completed"] = self._finished
            raise AgentFinished(self._finished) from None
        self.last_decision["task_path"] = list(self.task_path)
        return action_id

    def _event(self, event: str, **details) -> None:
        self.last_decision["events"].append({"event": event,
                                              "task_path": list(self.task_path), **details})

    def _task_context(self, task: str, depth: int, remaining: int) -> dict:
        return {"subtask": task, "task_path": json.dumps(self.task_path, ensure_ascii=False),
                "depth": str(depth), "max_depth": str(self.params["max_depth"]),
                "can_decompose": "yes" if depth < self.params["max_depth"] else "no",
                "remaining_calls": str(remaining)}

    def _execute(self, task: str, depth: int):
        # 每次 executor 只保留本次尝试的真实交互，最终目标与父链单独提供。
        self.history = []
        for call in range(self.params["max_executor_calls"]):
            remaining = self.params["max_executor_calls"] - call
            action = self._react_decision(
                self._observation, extra_context=self._task_context(task, depth, remaining),
                extra_actions=("task_completed", "task_failed"), label="adapt_execute")
            if action == "task_completed":
                return True, self.last_decision["thought"]
            if action == "task_failed":
                return False, self.last_decision["thought"]
            yield self._observation["actions"].index(action)
        return False, "Executor call budget exhausted without a completion report."

    def _solve(self, task: str, depth: int):
        self.task_path.append(task)
        self._event("task_started", depth=depth)
        try:
            completed, reason = yield from self._execute(task, depth)
            self._event("executor_finished", completed=completed, reason=reason, depth=depth)
            if not completed and depth < self.params["max_depth"]:
                context = self.context(self._observation)
                context.update(self._task_context(task, depth, 0))
                context.update(history=render_history(self.history, self.max_history_steps),
                               failure=reason, max_subtasks=str(self.params["max_subtasks"]))
                messages = [
                    {"role": "system", "content": "You plan Crafter tasks. Game rules:\n" + self.prompts["rules"]},
                    {"role": "user", "content": Template(self.prompts["planner"]).substitute(context)},
                ]
                raw = self.query(messages, response_schema=planner_schema(self.params["max_subtasks"]),
                                 label="adapt_plan")
                subtasks, expression = parse_plan(raw, self.params["max_subtasks"])
                self._event("plan", depth=depth, subtasks=subtasks, logic=json.loads(raw)["logic"])
                completed = yield from self._run_plan(expression, subtasks, depth + 1)
            self._event("task_finished", completed=completed, depth=depth)
            return completed
        finally:
            self.task_path.pop()

    def _run_plan(self, expression, subtasks: list[str], depth: int):
        if isinstance(expression, int):
            return (yield from self._solve(subtasks[expression - 1], depth))
        operator, children = expression
        for child in children:
            completed = yield from self._run_plan(child, subtasks, depth)
            if operator == "AND" and not completed:
                return False
            if operator == "OR" and completed:
                return True
        return operator == "AND"
