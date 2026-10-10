"""统一文本生成接口：Ollama 原生协议 / DeepSeek / OpenAI 兼容协议。"""
import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from modules.common.model_config import (
    endpoint_url, normalize_model_config, redact_secrets, resolve_provider,
)

# 沿用直接连接，避免本地模型请求被 shell 中的 HTTP_PROXY 转发。
opener = build_opener(ProxyHandler({}))


class ModelCallBudgetExceeded(RuntimeError):
    """在发出 HTTP 请求前耗尽预算；不作为模型错误处理。"""


class LLMClient:
    def __init__(self, config: dict):
        config = normalize_model_config(config)
        self.provider = resolve_provider(config)
        self.display_name = {"ollama": "Ollama", "deepseek": "DeepSeek", "openai": "OpenAI-compatible"}[self.provider]
        self.model = config["name"]
        self.url = endpoint_url(config, self.provider)
        self.timeout = config["timeout"]
        self.options = dict(config["params"])
        self.think = config.get("think")
        self.stream = config["stream"]
        self._api_key = config.get("api_key") or ""
        if not self._api_key and config.get("api_key_env"):
            self._api_key = os.environ.get(config["api_key_env"], "")
            if not self._api_key:
                raise ValueError("model.api_key_env 指定的环境变量未设置或为空")
        if "\n" in self._api_key or "\r" in self._api_key:
            raise ValueError("模型 API Key 必须是单行字符串")
        if self.provider == "deepseek" and not self._api_key:
            raise ValueError("DeepSeek 需要配置 model.api_key 或 model.api_key_env")
        self.max_calls = None
        self.on_call = None
        self.reset_stats()

    def reset_stats(self) -> None:
        self.calls = self.input_tokens = self.output_tokens = 0
        self.last_response = None

    def generate(self, messages: list[dict[str, str]], *, seed: int,
                 actions: list[str] | None = None,
                 response_schema: dict | None = None, label: str = "decision") -> str:
        if self.max_calls is not None and self.calls >= self.max_calls:
            raise ModelCallBudgetExceeded(f"已达到 {self.max_calls} 次模型调用预算")
        if response_schema is None and actions is not None:
            from modules.common.actions import action_schema
            response_schema = action_schema(actions)
        payload = self._payload(messages, seed, response_schema)
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        request = Request(self.url, data=json.dumps(payload).encode(),
                          headers=headers, method="POST")
        self.calls += 1
        self.last_response = None
        start = time.monotonic()
        record = {"call": self.calls, "label": label, "provider": self.provider,
                  "url": self.url, "request": payload}
        try:
            content = self._send(request)
            record["status"] = "ok"
            return content
        except (Exception, KeyboardInterrupt) as exc:
            record.update(status="error", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            record["seconds"] = time.monotonic() - start
            record["response"] = self.last_response
            if self.last_response is not None:
                self.last_response["client_seconds"] = record["seconds"]
            if self.on_call is not None:
                self.on_call(self._redact(record))

    def _redact(self, value):
        return redact_secrets(value, (self._api_key,))

    def _payload(self, messages: list[dict], seed: int, schema: dict | None) -> dict:
        if schema is not None:
            # A native output constraint is not a model-visible tool/field contract.
            instruction = ("Return only a JSON object matching this JSON schema:\n"
                           + json.dumps(schema, ensure_ascii=False))
            messages = [dict(message) for message in messages]
            if messages and messages[0].get("role") == "system":
                messages[0]["content"] += "\n\n" + instruction
            else:
                messages.insert(0, {"role": "system", "content": instruction})
        payload = {"model": self.model, "messages": messages, "stream": self.stream}
        if self.provider == "ollama":
            options = dict(self.options)
            # 两种命名都可使用，保留原有 Ollama 配置的优先级。
            limit = options.pop("max_tokens", options.pop("max_completion_tokens", None))
            if limit is not None:
                options.setdefault("num_predict", limit)
            payload["options"] = {**options, "seed": seed}
            if schema is not None:
                payload["format"] = schema
            if self.think is not None:
                payload["think"] = self.think
            return payload

        options = dict(self.options)
        if self.provider == "deepseek":
            limit = options.pop("max_completion_tokens", None)
            if limit is not None:
                options.setdefault("max_tokens", limit)
        limit = options.pop("num_predict", None)
        if limit is not None and "max_tokens" not in options and "max_completion_tokens" not in options:
            options["max_tokens"] = limit
        # 托管服务无法设置 Ollama 的模型加载/采样选项；不发送这些字段。
        for name in ("num_ctx", "num_batch", "num_gpu", "main_gpu", "num_thread",
                     "use_mmap", "use_mlock", "num_keep", "top_k", "min_p", "typical_p",
                     "repeat_last_n", "repeat_penalty", "mirostat", "mirostat_tau",
                     "mirostat_eta", "penalize_newline"):
            options.pop(name, None)
        # 环境 seed 仅用于本地后端。DeepSeek 文档未声明支持 seed。
        if self.provider == "deepseek":
            options.pop("seed", None)
        payload.update(options)
        if self.provider == "deepseek" and self.think is not None:
            payload["thinking"] = {"type": "enabled" if self.think else "disabled"}
        if schema is not None:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _send(self, request: Request) -> str:
        try:
            with opener.open(request, timeout=self.timeout) as response:
                data = self._read_stream(response) if self.stream else json.load(response)
        except HTTPError as exc:
            detail = self._redact(exc.read().decode(errors="replace"))[:2000]
            raise RuntimeError(f"{self.display_name} HTTP {exc.code}: {detail}") from None
        except (URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"无法调用 {self.display_name} {self.url}: {self._redact(str(exc))}") from None
        if not isinstance(data, dict):
            raise ValueError(f"{self.display_name} 响应必须是 JSON 对象")
        data = self._redact(data)
        self.last_response = data
        if data.get("error"):
            raise RuntimeError(f"{self.display_name}: {data['error']}")
        if self.provider != "ollama":
            usage = data.get("usage") or {}
            if not isinstance(usage, dict):
                raise ValueError(f"{self.display_name} 未返回有效 usage")
            # 兼容原有轨迹/绘图字段；choices 和 usage 原文同时保留。
            data["prompt_eval_count"] = usage.get("prompt_tokens", 0)
            data["eval_count"] = usage.get("completion_tokens", 0)
            choices = data.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                raise ValueError(f"{self.display_name} 未返回有效 choices")
            choice = choices[0]
            message = choice.get("message")
            data["message"] = dict(message) if isinstance(message, dict) else {}
            data["done_reason"] = choice.get("finish_reason")
            if data["message"].get("reasoning_content") is not None:
                data["message"]["thinking"] = data["message"]["reasoning_content"]
        self.input_tokens += int(data.get("prompt_eval_count", 0))
        self.output_tokens += int(data.get("eval_count", 0))
        if data.get("done_reason") == "length":
            raise ValueError(
                f"{self.display_name} 生成被截断（done_reason=length，"
                f"eval_count={data.get('eval_count')}，"
                f"输出上限={self.options.get('max_tokens', self.options.get('num_predict', '默认'))}）。"
                "请增大 model.params.num_predict 或 max_tokens；开启思考时需为思考和最终动作预留预算。"
                "若不需要思考，可设置 model.think: false。"
            )
        if self.provider != "ollama" and data.get("done_reason") != "stop":
            raise ValueError(f"{self.display_name} 生成未正常完成（finish_reason={data.get('done_reason')}）")
        if self.provider == "ollama" and (data.get("done") is not True
                or data.get("done_reason") not in (None, "stop")):
            raise ValueError(f"Ollama 生成未正常完成（done={data.get('done')}，done_reason={data.get('done_reason')}）")
        message = data.get("message") or {}
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise ValueError(
                f"{self.display_name} 未返回有效 message.content "
                f"（done_reason={data.get('done_reason')}，"
                f"has_thinking={isinstance(message, dict) and bool(message.get('thinking'))}）。"
                "原始响应已保存到轨迹；思考内容不能作为最终动作执行。"
            )
        return content

    def _read_stream(self, response):
        """合并 SSE 正文与推理，沿用非流式响应校验；中断时保留已收到的证据。"""
        message = {"role": "assistant", "content": "", "reasoning_content": ""}
        choice = {"index": 0, "message": message, "finish_reason": None}
        data = {"choices": [choice], "stream": True}
        self.last_response = data
        last_notice = time.monotonic()
        for line in response:
            line = line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue  # 空行、SSE 心跳和事件名都不是正文。
            raw = line[5:].strip()
            if raw == "[DONE]":
                break
            chunk = self._redact(json.loads(raw))
            if not isinstance(chunk, dict):
                raise ValueError("模型流式响应必须是 JSON 对象")
            if chunk.get("error"):
                data["error"] = chunk["error"]
                raise RuntimeError(f"{self.display_name}: {chunk['error']}")
            for key in ("id", "model", "created", "usage", "system_fingerprint"):
                if chunk.get(key) is not None:
                    data[key] = chunk[key]
            for part in chunk.get("choices", []):
                if part.get("index", 0) != 0:
                    continue
                delta = part.get("delta") or {}
                for key in ("content", "reasoning_content"):
                    value = delta.get(key)
                    if value is not None:
                        if not isinstance(value, str):
                            raise ValueError(f"流式 {key} 必须是字符串")
                        message[key] += value
                if part.get("finish_reason") is not None:
                    choice["finish_reason"] = part["finish_reason"]
            if time.monotonic() - last_notice >= 30:
                print(f"  模型调用{self.calls}接收中：正文{len(message['content'])}字，"
                      f"推理{len(message['reasoning_content'])}字", flush=True)
                last_notice = time.monotonic()
        return data
