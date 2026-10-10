"""共享模型配置：协议识别、端点和密钥脱敏，不依赖具体智能体。"""
from copy import deepcopy
import math
from urllib.parse import urlsplit


def normalize_model_config(config: dict) -> dict:
    if not isinstance(config, dict):
        raise ValueError("model 必须是配置映射")
    config = deepcopy(config)
    provider = config.setdefault("provider", "auto")
    if provider not in ("auto", "ollama", "deepseek", "openai"):
        raise ValueError("model.provider 必须是 auto、ollama、deepseek 或 openai")
    if not isinstance(config.get("name"), str) or not config["name"].strip():
        raise ValueError("请指定 model.name（服务端可用的模型名）")
    defaults = {"deepseek": "https://api.deepseek.com", "openai": "https://api.openai.com/v1"}
    url = config.setdefault("base_url", defaults.get(provider, "http://localhost:11434"))
    if not isinstance(url, str):
        raise ValueError("model.base_url 必须是 HTTP(S) 地址")
    try:
        parsed = urlsplit(url)
        valid = (parsed.scheme in ("http", "https") and parsed.hostname
                 and (parsed.port is None or parsed.port > 0)
                 and not parsed.username and not parsed.password
                 and not parsed.query and not parsed.fragment
                 and not any(char.isspace() for char in url))
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("model.base_url 必须是有效 HTTP(S) 地址，不能包含凭据、查询参数或片段")
    config["base_url"] = url.rstrip("/")
    timeout = config.setdefault("timeout", 120)
    if (not isinstance(timeout, (int, float)) or isinstance(timeout, bool)
            or not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("model.timeout 必须大于零")
    think = config.get("think")
    if think is not None and type(think) is not bool:
        raise ValueError("model.think 必须是 true、false 或 null")
    stream = config.setdefault("stream", False)
    if type(stream) is not bool:
        raise ValueError("model.stream 必须是 true 或 false")
    if stream and resolve_provider(config) == "ollama":
        raise ValueError("model.stream 当前仅支持 OpenAI 兼容及 DeepSeek 接口")
    for field in ("api_key", "api_key_env"):
        value = config.get(field)
        if value is not None and (not isinstance(value, str) or "\n" in value or "\r" in value):
            raise ValueError(f"model.{field} 必须是单行字符串或 null")
    params = config.setdefault("params", {})
    if not isinstance(params, dict):
        raise ValueError("model.params 必须是配置映射")
    for name in ("num_ctx", "num_predict", "max_tokens", "max_completion_tokens"):
        value = params.get(name)
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError(f"model.params.{name} 必须是正整数")
    reserved = {"model", "messages", "stream", "stream_options", "format", "response_format", "options",
                "api_key", "api_key_env", "authorization", "headers", "thinking", "think"}
    if reserved & params.keys():
        raise ValueError("model.params 只能包含生成参数；模型、密钥、思考及输出格式由统一接口管理")
    effort = params.get("reasoning_effort")
    if resolve_provider(config) == "deepseek" and think is not None and effort is not None:
        if (think and effort == "none") or (not think and effort != "none"):
            raise ValueError("model.think 与 model.params.reasoning_effort 的思考开关冲突")
    return config


def resolve_provider(config: dict) -> str:
    """非标准地址可显式设置 provider，避免根据模型名称猜测协议。"""
    provider = config.get("provider", "auto")
    if provider != "auto":
        return provider
    parsed = urlsplit(config.get("base_url", "http://localhost:11434"))
    host = parsed.hostname or ""
    path = parsed.path.rstrip("/")
    if host == "deepseek.com" or host.endswith(".deepseek.com"):
        return "deepseek"
    # Ollama 也有 /v1 兼容接口；显式给出该路径时使用对应协议。
    if path.endswith("/chat/completions") or path.endswith("/v1"):
        return "openai"
    if path.endswith("/api/chat") or parsed.port == 11434:
        return "ollama"
    return "openai"


def endpoint_url(config: dict, provider: str) -> str:
    base = config["base_url"].rstrip("/")
    path = urlsplit(base).path
    if provider == "ollama":
        if path.endswith("/api/chat"):
            return base
        return base + ("/chat" if path.endswith("/api") else "/api/chat")
    if path.endswith("/chat/completions"):
        return base
    if provider == "openai" and not path:
        base += "/v1"
    return base + "/chat/completions"


def redact_secrets(value, secrets: tuple[str, ...]):
    """脱敏诊断响应及错误中的密钥，避免服务端回显凭据进入日志。"""
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[REDACTED]")
        return value
    if isinstance(value, dict):
        return {redact_secrets(k, secrets): redact_secrets(v, secrets) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_secrets(v, secrets) for v in value]
    return value


def redact_config(config: dict) -> dict:
    config = deepcopy(config)
    key = config.get("model", {}).get("api_key")
    if key:
        config = redact_secrets(config, (key,))
    return config
