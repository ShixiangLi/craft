"""Ollama 原生 /api/chat 客户端，仅使用标准库，不自动重试。"""
import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

# 本地模型请求直接连接，避免被 shell 中的 HTTP_PROXY 转发。
opener = build_opener(ProxyHandler({}))


class LLMClient:
    def __init__(self, config: dict):
        self.model = config["name"]
        self.url = config.get("base_url", "http://localhost:11434").rstrip("/") + "/api/chat"
        self.timeout = config.get("timeout", 120)
        self.options = dict(config.get("params", {}))
        self.think = config.get("think")
        self.reset_stats()

    def reset_stats(self) -> None:
        self.calls = self.input_tokens = self.output_tokens = 0
        self.last_response = None

    def generate(self, messages: list[dict[str, str]], *, seed: int,
                 actions: list[str]) -> str:
        payload = {
            "model": self.model, "messages": messages, "stream": False,
            "options": {**self.options, "seed": seed},
            "format": {"type": "object", "properties": {
                "action": {"type": "string", "enum": actions}},
                "required": ["action"], "additionalProperties": False},
        }
        if self.think is not None:
            payload["think"] = self.think
        request = Request(self.url, data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"}, method="POST")
        self.calls += 1
        self.last_response = None
        start = time.monotonic()
        try:
            with opener.open(request, timeout=self.timeout) as response:
                data = json.load(response)
        except HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:2000]
            raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"无法调用 Ollama {self.url}: {exc}") from exc
        if data.get("error"):
            raise RuntimeError(f"Ollama: {data['error']}")
        self.last_response = data
        self.input_tokens += int(data.get("prompt_eval_count", 0))
        self.output_tokens += int(data.get("eval_count", 0))
        self.last_response["client_seconds"] = time.monotonic() - start
        if data.get("done_reason") == "length":
            raise ValueError(
                "Ollama 生成被截断（done_reason=length，"
                f"eval_count={data.get('eval_count')}，"
                f"num_predict={self.options.get('num_predict', '默认')}）。"
                "请增大 model.params.num_predict；开启思考时需为思考和最终动作预留预算。"
                "若不需要思考，可设置 model.think: false。"
            )
        content = data.get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(
                "Ollama 未返回有效 message.content "
                f"（done_reason={data.get('done_reason')}，"
                f"has_thinking={bool(data.get('message', {}).get('thinking'))}）。"
                "原始响应已保存到轨迹；思考内容不能作为最终动作执行。"
            )
        return content
