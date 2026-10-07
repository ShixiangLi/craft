"""有界文本文件工作区及统一 JSON 工具协议，不提供 shell 或全局地图。"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile

TOOLS = ("list_files", "read_file", "write_file")
MAX_THOUGHT_CHARS = 1024
MAX_PATH_CHARS = 240
DEFAULTS = {"max_history_steps": 16, "max_tool_calls_per_step": 4,
            "max_file_bytes": 16384, "max_workspace_bytes": 131072,
            "max_files": 32, "max_write_chars": 4096}


def validate_params(params=None):
    explicit_write_limit = "max_write_chars" in (params or {})
    params = {**DEFAULTS, **(params or {})}
    if set(params) != set(DEFAULTS):
        raise ValueError(f"未知 harness 参数: {sorted(set(params) - set(DEFAULTS))}")
    for key, value in params.items():
        if key == "max_history_steps" and value is None:
            continue
        minimum = 0 if key == "max_tool_calls_per_step" else 1
        if type(value) is not int or value < minimum:
            raise ValueError(f"harness.{key} 必须为整数且 >= {minimum}")
    if not explicit_write_limit:
        params["max_write_chars"] = min(params["max_write_chars"], params["max_file_bytes"])
    if params["max_files"] < 2:
        raise ValueError("harness.max_files 至少为 2，以容纳计划与记忆模板")
    if params["max_workspace_bytes"] < params["max_file_bytes"]:
        raise ValueError("max_workspace_bytes 不能小于 max_file_bytes")
    if params["max_write_chars"] > params["max_file_bytes"]:
        raise ValueError("max_write_chars 不能大于 max_file_bytes；UTF-8 字节数仍另行校验")
    return params


def decision_schema(actions, *, allow_tools, max_write_chars=4096):
    def branch(names, **arguments):
        # 动作判别字段在前，避免先生成长文件内容再决定是否写入。
        properties = {"action": {"type": "string", "enum": list(names)},
                      "thought": {"type": "string", "maxLength": MAX_THOUGHT_CHARS},
                      **arguments}
        return {"type": "object", "properties": properties,
                "required": list(properties), "additionalProperties": False}

    environment = branch(actions)
    if not allow_tools:
        return environment
    path = {"type": "string", "minLength": 1, "maxLength": MAX_PATH_CHARS}
    return {"oneOf": [environment, branch(["list_files"]),
                      branch(["read_file"], path=path),
                      branch(["write_file"], path=path, content={
                          "type": "string", "maxLength": max_write_chars})]}


def parse_decision(raw, actions, *, allow_tools, max_write_chars=4096):
    data = json.loads(raw)
    if not isinstance(data, dict) or not isinstance(data.get("action"), str):
        raise ValueError("Harness 输出必须是包含字符串 action 的 JSON 对象")
    allowed = list(actions) + (list(TOOLS) if allow_tools else [])
    if data["action"] not in allowed:
        raise ValueError(f"非法 Harness 动作: {data['action'][:100]!r}")
    keys = {"action", "thought"}
    if data["action"] in ("read_file", "write_file"):
        keys.add("path")
    if data["action"] == "write_file":
        keys.add("content")
    if set(data) != keys or any(not isinstance(data[key], str) for key in keys):
        raise ValueError(f"{data['action']} 必须且仅包含字符串字段 {sorted(keys)}；"
                         "修改文件请单独选择 write_file，不能与游戏动作混合")
    if len(data["thought"]) > MAX_THOUGHT_CHARS:
        raise ValueError(f"thought 超过 {MAX_THOUGHT_CHARS} 字符")
    if "path" in keys and not 1 <= len(data["path"]) <= MAX_PATH_CHARS:
        raise ValueError(f"path 必须为 1–{MAX_PATH_CHARS} 字符")
    if "content" in keys and len(data["content"]) > max_write_chars:
        raise ValueError(f"content 超过 max_write_chars={max_write_chars}；精简后重新提交")
    return data


class EpisodeWorkspace:
    def __init__(self, root, params):
        self.root = Path(root)
        self.params = params
        self.root.mkdir(parents=True, exist_ok=False)

    def _path(self, name):
        if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
            raise ValueError("文件名必须为非空 POSIX 相对路径")
        parts = name.split("/")
        if PurePosixPath(name).is_absolute() or any(part in ("", ".", "..") for part in parts):
            raise ValueError("禁止绝对路径或路径穿越")
        path = self.root
        if path.is_symlink():
            raise ValueError("工作区不能为符号链接")
        for part in parts:
            path = path / part
            if path.is_symlink():
                raise ValueError("禁止访问符号链接")
        if not path.resolve().is_relative_to(self.root.resolve()):
            raise ValueError("文件必须位于当前回合工作区")
        return path

    def list_files(self):
        if self.root.is_symlink():
            raise ValueError("工作区不能为符号链接")
        files = []
        for path in sorted(self.root.rglob("*")):
            if path.is_symlink():
                raise ValueError("工作区包含符号链接")
            if path.is_file():
                files.append({"path": path.relative_to(self.root).as_posix(),
                              "bytes": path.stat().st_size})
        return files

    def read_file(self, name):
        path = self._path(name)
        if not path.is_file():
            raise ValueError(f"文件不存在: {name}")
        if path.stat().st_size > self.params["max_file_bytes"]:
            raise ValueError("文件超过 max_file_bytes，不能读取")
        return path.read_text(encoding="utf-8")

    def write_file(self, name, content):
        path = self._path(name)
        encoded = content.encode("utf-8")
        if len(encoded) > self.params["max_file_bytes"]:
            raise ValueError("文件超过 max_file_bytes")
        files = self.list_files()
        old = path.read_bytes() if path.is_file() else b""
        if not path.exists() and len(files) >= self.params["max_files"]:
            raise ValueError("工作区达到 max_files")
        total = sum(file["bytes"] for file in files) - len(old) + len(encoded)
        if total > self.params["max_workspace_bytes"]:
            raise ValueError("工作区超过 max_workspace_bytes")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:
                temporary = Path(file.name)
                file.write(encoded)
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        return {"path": name, "bytes": len(encoded),
                "previous_sha256": hashlib.sha256(old).hexdigest(),
                "sha256": hashlib.sha256(encoded).hexdigest()}

    def execute(self, decision):
        try:
            action = decision["action"]
            if action == "list_files":
                result = self.list_files()
            elif action == "read_file":
                result = {"path": decision["path"], "content": self.read_file(decision["path"])}
            elif action == "write_file":
                result = self.write_file(decision["path"], decision["content"])
            else:
                raise ValueError(f"未知文件工具: {action}")
            return {"ok": True, "result": result}
        except (OSError, ValueError) as exc:
            # 返回真实错误供模型纠错，不执行替代动作。
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
