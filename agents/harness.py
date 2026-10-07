"""ReAct + 按需文件读写；文件和工具对话只属于当前回合。"""
import json
from pathlib import Path
from string import Template

from agents.react import ReActAgent
from modules.harness.components import (
    TOOLS, EpisodeWorkspace, decision_schema, parse_decision, validate_params,
)
from modules.react.components import render_history


class HarnessAgent(ReActAgent):
    def __init__(self, config, llm):
        config = {**config, "params": validate_params(config.get("params"))}
        super().__init__(config, llm)
        self.params = config["params"]

    def reset(self, task, *, seed):
        super().reset(task, seed=seed)
        self.workspace = None
        self.tool_log = None

    def bind_episode(self, output_dir):
        if self.workspace is not None:
            raise RuntimeError("当前 Harness 回合已经绑定工作区")
        output_dir = Path(output_dir)
        self.workspace = EpisodeWorkspace(output_dir / "workspace", self.params)
        self.tool_log = output_dir / "tool_calls.jsonl"
        templates = {
            "plan.md": (f"# Plan\n\nFinal goal: {self.task['description']}\n\n"
                        "## Current subgoal\nNot yet selected.\n\n"
                        "## Progress and evidence\nNo actions completed yet.\n\n"
                        "## Interrupted task and resumption\nNone.\n"),
            "memory.md": ("# Memory\n\nRecord important observations, resource locations with "
                          "their reference frame, failed attempts and their evidence.\n"
                          "No observations recorded yet.\n"),
        }
        for name, content in templates.items():
            self.workspace.write_file(name, content)
        self._record_tool({"type": "initialize", "files": templates})

    def _record_tool(self, record):
        with self.tool_log.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    def act(self, observation):
        if self.pending is not None:
            raise RuntimeError("Harness 必须先 observe 实际反馈")
        if self.workspace is None:
            raise RuntimeError("Harness 必须先绑定回合工作区")
        self.begin_decision()
        self.last_decision["tool_calls"] = []
        self.last_decision["protocol_errors"] = []
        context = self.context(observation)
        context.update(history=render_history(self.history, self.max_history_steps),
                       files=json.dumps(self.workspace.list_files(), ensure_ascii=False),
                       tools_remaining=str(self.params["max_tool_calls_per_step"]),
                       max_file_bytes=str(self.params["max_file_bytes"]),
                       max_write_chars=str(self.params["max_write_chars"]))
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": Template(self.prompts["step"]).substitute(context)}]
        limit = self.params["max_tool_calls_per_step"]
        for index in range(limit + 1):
            raw = self.query(messages, response_schema=decision_schema(
                observation["actions"], allow_tools=index < limit,
                max_write_chars=self.params["max_write_chars"]), label="harness")
            try:
                decision = parse_decision(raw, observation["actions"], allow_tools=index < limit,
                                          max_write_chars=self.params["max_write_chars"])
            except ValueError as exc:
                result = {"ok": False, "error": str(exc), "files_changed": False}
                event = {"type": "protocol_error", "step": observation["step"],
                         "call": self.llm.calls, "response_excerpt": raw[:2000], "response": result}
                self._record_tool(event)
                self.last_decision["protocol_errors"].append(event)
                if index == limit:
                    raise ValueError("Harness 输出协议错误且本步纠错预算耗尽: " + str(exc)) from exc
                # 不静默丢弃写入意图、不自动改成 write_file/noop。模型必须明确重选。
                # 不把异常长的无效回复重新塞入上下文；完整原文已在调用日志中保存。
                messages = [*messages, {"role": "user", "content": "OUTPUT PROTOCOL ERROR:\n"
                            + json.dumps(result, ensure_ascii=False)
                            + f"\nRemaining auxiliary calls: {limit - index - 1}. "
                            + ("Choose an environment action now."
                               if index + 1 == limit else "Submit one valid tool or environment action.")}]
                continue
            action = decision["action"]
            if action not in TOOLS:
                self.last_decision.update(thought=decision["thought"], action=action,
                                          history_steps=min(len(self.history), self.max_history_steps)
                                          if self.max_history_steps is not None else len(self.history))
                self.pending = {"step": observation["step"], "observation": context["observation"],
                                "thought": decision["thought"], "action": action}
                return observation["actions"].index(action)
            result = self.workspace.execute(decision)
            event = {"type": "tool", "step": observation["step"],
                     "call": self.llm.calls, "request": decision, "response": result}
            self._record_tool(event)
            self.last_decision["tool_calls"].append(event)
            messages = [*messages, {"role": "assistant", "content": raw}, {
                "role": "user", "content": "FILE TOOL RESULT (game has not advanced):\n"
                + json.dumps(result, ensure_ascii=False)
                + f"\nRemaining file tool calls: {limit - index - 1}. "
                + ("Choose an environment action now." if index + 1 == limit
                   else "Read/write another file if needed, or choose an environment action.")}]
