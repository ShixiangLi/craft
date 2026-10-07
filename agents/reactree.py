"""ReAcTree 的 Crafter 机制适配；递归生成器只在真实环境动作处挂起。"""
import json
from pathlib import Path
from string import Template
from typing import Any, Mapping

from agents.base import AgentFinished
from agents.llm_agent import LLMBaseAgent
from modules.common.llm import ModelCallBudgetExceeded
from modules.common.observation import describe_observation
from modules.reactree.components import (
    DECISION_PROTOCOL, Node, decision_schema, parse_decision, render_game_examples,
    render_node_history, validate_params,
)
from modules.reactree.memory import EpisodicMemory, WorkingMemory
from utils.io import write_json


class ReAcTreeAgent(LLMBaseAgent):
    def __init__(self, config: Mapping[str, Any], llm):
        self.params = validate_params(config.get("params"))
        super().__init__({**config, "params": self.params}, llm)
        self.system_prompt = Template(self.prompts["system"]).substitute(
            rules=self.prompts["rules"], examples=render_game_examples(self.prompts["examples"]))
        self.episodic = EpisodicMemory(
            self.params["episodic_memory_path"], enabled=self.params["episodic_memory"],
            embedding_model=self.params["embedding_model"],
            embedding_device=self.params["embedding_device"],
            max_examples=self.params["max_examples"],
            max_example_chars=self.params["max_example_chars"])
        self.working = WorkingMemory(mode=self.params["working_memory_mode"],
                                     max_locations=self.params["max_recall_locations"])
        self._controller = None
        self.output_dir = None

    def reset(self, task: Mapping[str, Any], *, seed: int) -> None:
        if self._controller is not None:
            self._controller.close()
        super().reset(task, seed=seed)
        self.episodic.validate_evaluation_seeds([seed])
        self.output_dir = None
        self.working.reset()
        self.nodes = {}
        self.task_path = []
        self.pending = None
        self._events = []
        self._observation = None
        self._finished = None
        self._episode_finished = False
        self.root = self._new_node("agent", self.task["description"], 1)
        self._controller = self._run_agent(self.root)

    def bind_episode(self, output_dir) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for event in self._events:
            self._append("tree_events.jsonl", event)
        self._flush()

    def _append(self, filename, record):
        if self.output_dir is not None:
            with (self.output_dir / filename).open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    def _flush(self):
        if self.output_dir is not None:
            write_json({"root": self.root.id, "decision_protocol": DECISION_PROTOCOL,
                        "parallel_policy": self.params["parallel_policy"],
                        "planning_error_policy": self.params["planning_error_policy"],
                        "working_memory_enabled": self.params["working_memory"],
                        "episodic_memory": self.episodic.metadata(),
                        "task_path": list(self.task_path),
                        "nodes": [node.snapshot() for node in self.nodes.values()]},
                       self.output_dir / "tree.json")
            write_json(self.working.snapshot(), self.output_dir / "working_memory.json")

    def _event(self, event, **details):
        row = {"event": event, "step": self._observation["step"] if self._observation else 0,
               "call": self.llm.calls, "task_path": list(self.task_path), **details}
        self._events.append(row)
        self.last_decision.setdefault("events", []).append(row)
        self._append("tree_events.jsonl", row)
        self._flush()

    def _new_node(self, kind, content, depth, parent=None):
        if len(self.nodes) >= self.params["max_nodes"]:
            raise ValueError("ReAcTree 达到 max_nodes，不能继续创建节点")
        node = Node(f"node_{len(self.nodes):04d}", kind, content, depth,
                    parent.id if parent else None)
        self.nodes[node.id] = node
        if parent is not None:
            parent.children.append(node.id)
        # reset 中 root 尚未赋值，初始化事件在 bind_episode 时统一落盘。
        if parent is not None:
            self._event("node_created", node=node.snapshot())
        else:
            row = {"event": "node_created", "step": 0, "call": 0,
                   "task_path": [], "node": node.snapshot()}
            self._events.append(row)
        return node

    def act(self, observation: Mapping[str, Any]) -> int:
        if self.pending is not None:
            raise RuntimeError("ReAcTree 必须先 observe 实际环境反馈，再进行下一次 act")
        self.begin_decision()
        self.last_decision["events"] = []
        self._observation = observation
        if self.params["working_memory"]:
            self.working.update(observation)
        if self._finished is not None:
            raise AgentFinished(self._finished)
        try:
            action_id = next(self._controller)
        except StopIteration as result:
            self._finished = bool(result.value)
            self.last_decision["completed"] = self._finished
            self._flush()
            raise AgentFinished(self._finished) from None
        self.last_decision["task_path"] = list(self.task_path)
        self._flush()
        return action_id

    def _node_context(self, node, can_expand, max_subgoals, examples):
        context = self.context(self._observation)
        family = "Root node; no sibling tasks."
        if node.parent is not None:
            control = self.nodes[node.parent]
            parent = self.nodes[control.parent]
            family = json.dumps({"parent_goal": parent.content, "control_flow": control.content,
                                 "sibling_goals": [self.nodes[n].content for n in control.children]},
                                ensure_ascii=False)
        context.update(subgoal=node.content, task_path=json.dumps(self.task_path, ensure_ascii=False),
                       family=family, history=render_node_history(
                           node.history, self.params["max_history_steps"],
                           current_observation=context["observation"]),
                       episodic_examples=json.dumps(examples, ensure_ascii=False)
                       if examples else "No retrieved episodic examples.",
                       can_expand="yes" if can_expand else "no",
                       max_subgoals=str(max_subgoals),
                       depth=str(node.depth), max_depth=str(self.params["max_depth"]),
                       remaining_nodes=str(self.params["max_nodes"] - len(self.nodes)),
                       parallel_policy=self.params["parallel_policy"],
                       recall_targets=json.dumps(self.working.targets(), ensure_ascii=False)
                       if self.params["working_memory"] else "Disabled.")
        return context

    def _commit(self, node, entry, feedback):
        row = {**entry, "feedback": feedback}
        node.history.append(row)
        self._append("node_traces.jsonl", {"node_id": node.id, "goal": node.content, **row})
        self._flush()

    def _run_agent(self, node):
        self.task_path.append(node.content)
        node.status = "running"
        self._event("agent_started", node_id=node.id, goal=node.content, depth=node.depth)
        try:
            examples = self.episodic.retrieve(node.content)
            node.examples = [{k: e[k] for k in ("goal", "state", "similarity")} for e in examples]
            self._event("examples_retrieved", node_id=node.id, examples=node.examples)
            while True:
                capacity = self.params["max_nodes"] - len(self.nodes) - 1
                max_subgoals = min(self.params["max_subgoals"], max(0, capacity))
                # 与官方一致：上限检查 control 节点；子 agent 的深度再加 1。
                can_expand = node.depth + 1 <= self.params["max_depth"] and max_subgoals > 0
                targets = self.working.targets() if self.params["working_memory"] else []
                context = self._node_context(node, can_expand, max_subgoals, examples)
                messages = [{"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": Template(self.prompts["step"]).substitute(context)}]
                schema = decision_schema(self._observation["actions"], can_expand=can_expand,
                                         max_subgoals=max_subgoals, recall_targets=targets)
                calls_before = self.llm.calls
                raw = None
                planning_error = None
                try:
                    raw = self.query(messages, response_schema=schema, label="reactree")
                except ModelCallBudgetExceeded:
                    raise
                except (ValueError, RuntimeError) as exc:
                    planning_error = exc
                finally:
                    # finally 的落盘异常不进入同一个 try 的 except，避免将记录故障当规划失败。
                    node.decisions += self.llm.calls - calls_before
                    self._flush()
                if planning_error is None:
                    try:
                        decision = parse_decision(raw, self._observation["actions"], can_expand=can_expand,
                                                  max_subgoals=max_subgoals, recall_targets=targets)
                    except ValueError as exc:
                        planning_error = exc
                if planning_error is not None:
                    if self.params["planning_error_policy"] == "abort":
                        raise planning_error
                    # 仅模型调用/解析边界的预期错误成为节点失败。预算、用户中断、
                    # 程序错误和落盘 IO 错误仍终止运行；不重试该节点、不替代环境动作。
                    node.status = node.termination = "failure"
                    node.failure_reason = f"{type(planning_error).__name__}: {planning_error}"
                    self.last_decision.update(node_id=node.id, subgoal=node.content,
                                              planning_error=node.failure_reason)
                    self._commit(node, {"step": self._observation["step"],
                                       "observation": context["observation"],
                                       "decision": {"action": "planning_error", "thought": "",
                                                    "call": self.llm.calls,
                                                    "response_excerpt": raw[:2000] if raw else None}},
                                 {"source": "controller", "self_reported": False,
                                  "completed": False, "planning_error": node.failure_reason})
                    self._event("planning_failed", node_id=node.id, error=node.failure_reason)
                    self._event("agent_finished", node_id=node.id, completed=False,
                                self_reported=False, reason="planning_error")
                    return False
                kind = decision["type"]
                action = decision["action"] if kind == "Act" else {"Think": "think", "Expand": "expand"}[kind]
                self.last_decision.update(node_id=node.id, subgoal=node.content, type=kind,
                                          thought=decision["thought"], action=action)
                self._event("decision", node_id=node.id, decision=decision)
                entry = {"step": self._observation["step"],
                         "observation": context["observation"], "decision": decision}
                if action in ("done", "failure"):
                    completed = action == "done"
                    node.status = node.termination = "success" if completed else "failure"
                    self._commit(node, entry, {"source": "controller", "self_reported": True,
                                               "completed": completed})
                    self._event("agent_finished", node_id=node.id, completed=completed,
                                self_reported=True)
                    return completed
                if action == "think":
                    self._commit(node, entry, {"source": "controller", "environment_advanced": False})
                    continue
                if action == "recall_observation":
                    result = self.working.recall(decision["target"], self._observation["step"])
                    self._commit(node, entry, {"source": "working_memory", "result": result,
                                              "environment_advanced": False})
                    self._event("recall", node_id=node.id, target=decision["target"], result=result)
                    continue
                if action == "expand":
                    node.status = "expanded"
                    node.termination = "expand"
                    control = self._new_node("control", decision["control_flow"], node.depth + 1, node)
                    for goal in decision["subgoals"]:
                        self._new_node("agent", goal, node.depth + 2, control)
                    self._commit(node, entry, {"source": "controller", "control_node": control.id,
                                              "children": list(control.children)})
                    self._event("expanded", node_id=node.id, control_node=control.id,
                                control_flow=control.content, subgoals=decision["subgoals"])
                    completed = yield from self._run_control(control)
                    node.status = "success" if completed else "failure"
                    self._event("agent_finished", node_id=node.id, completed=completed,
                                from_control_flow=True)
                    return completed
                self.pending = {"node_id": node.id, "entry": entry}
                yield self._observation["actions"].index(action)
        except ModelCallBudgetExceeded:
            node.status = "interrupted"
            if node.termination is None:
                node.termination = "interrupted"
            self._event("node_budget_exhausted", node_id=node.id)
            raise
        except (Exception, KeyboardInterrupt) as exc:
            node.status = "error"
            if node.termination is None:
                node.termination = "error"
            self._event("node_error", node_id=node.id, error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            self.task_path.pop()

    def _run_control(self, node):
        node.status = "running"
        self._event("control_started", node_id=node.id, control_flow=node.content)
        outcomes = []
        for child in node.children:
            completed = yield from self._run_agent(self.nodes[child])
            outcomes.append(completed)
            if node.content == "sequence" and not completed:
                break
            if node.content == "fallback" and completed:
                break
        if node.content == "sequence":
            result = all(outcomes)
        elif node.content == "fallback":
            result = any(outcomes)
        elif self.params["parallel_policy"] == "all":
            result = all(outcomes)
        else:
            result = sum(outcomes) * 2 > len(outcomes)
        node.status = node.termination = "success" if result else "failure"
        self._event("control_finished", node_id=node.id, completed=result, child_outcomes=outcomes)
        return result

    def observe(self, transition: Mapping[str, Any]) -> None:
        if self.pending is None:
            raise RuntimeError("ReAcTree observe 之前必须有待执行的 act")
        super().observe(transition)
        self._observation = transition["observation"]
        if self.params["working_memory"]:
            self.working.update(self._observation, action=transition["action"])
        node = self.nodes[self.pending["node_id"]]
        self._commit(node, self.pending["entry"], {
            "source": "environment", **self.previous,
            "observation": describe_observation(dict(self._observation)),
            "terminated": transition["terminated"], "truncated": transition["truncated"],
        })
        self.pending = None
        self._event("environment_feedback", node_id=node.id, **self.previous)

    def finish_episode(self, result: Mapping[str, Any]) -> None:
        if self._episode_finished:
            return
        self._episode_finished = True
        for node in self.nodes.values():
            if node.status in ("running", "expanded"):
                node.status = "interrupted"
                if node.termination is None:
                    node.termination = "interrupted"
        self._event("episode_finished", success=result["success"], stop_reason=result["stop_reason"])
        if result["success"] is True and self.output_dir is not None:
            for node in self.nodes.values():
                if node.kind != "agent" or not node.history:
                    continue
                self._append("episodic_candidates.jsonl", {
                    "goal": node.content, "state": node.termination,
                    "trajectory": render_node_history(node.history), "episode_success": True,
                    "source": {"seed": self.seed, "episode_id": self.output_dir.name,
                               "run_id": self.output_dir.parent.name, "node_id": node.id,
                               "method": "reactree", "task": self.task},
                })
        if self._controller is not None:
            self._controller.close()
        self._flush()
