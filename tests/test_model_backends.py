"""模型协议与真实回合的回归；所有 HTTP 均使用 mock，不访问付费接口。"""
from contextlib import redirect_stdout
from copy import deepcopy
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from engine.experiment import run_experiment
from modules.common.actions import action_schema
from modules.common.llm import LLMClient, ModelCallBudgetExceeded
from modules.common.model_config import normalize_model_config, redact_config, resolve_provider
from utils.config import PROJECT_ROOT, load_config


FAKE_KEY = "sk-test-never-send-this-key"


def completion(content='{"action":"noop"}', *, finish_reason="stop", reasoning=None):
    message = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return {
        "model": "test-model", "choices": [{"index": 0, "message": message,
                                               "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }


def response(data):
    return BytesIO(json.dumps(data).encode())


class BackendConfigurationTest(unittest.TestCase):
    def test_auto_routing_and_endpoint_construction(self):
        cases = [
            ("http://localhost:11434", "ollama", "http://localhost:11434/api/chat"),
            ("http://127.0.0.1:11434/", "ollama", "http://127.0.0.1:11434/api/chat"),
            ("http://model.internal:8181/api/chat", "ollama", "http://model.internal:8181/api/chat"),
            ("https://api.deepseek.com", "deepseek", "https://api.deepseek.com/chat/completions"),
            ("https://api.deepseek.com/v1", "deepseek", "https://api.deepseek.com/v1/chat/completions"),
            ("https://api.deepseek.com/chat/completions", "deepseek", "https://api.deepseek.com/chat/completions"),
            ("https://models.example", "openai", "https://models.example/v1/chat/completions"),
            ("https://models.example/v1/", "openai", "https://models.example/v1/chat/completions"),
            ("https://models.example/proxy/v2", "openai", "https://models.example/proxy/v2/chat/completions"),
            ("https://models.example/v1/chat/completions", "openai", "https://models.example/v1/chat/completions"),
        ]
        for base_url, provider, url in cases:
            with self.subTest(base_url=base_url):
                config = {"name": "test", "base_url": base_url, "api_key": FAKE_KEY}
                normalized = normalize_model_config(config)
                self.assertEqual(normalized["provider"], "auto")
                self.assertEqual(resolve_provider(normalized), provider)
                self.assertEqual(LLMClient(normalized).url, url)

    def test_explicit_provider_supports_proxy_and_custom_ollama_port(self):
        for provider, base_url, url in [
            ("ollama", "http://localhost:12345", "http://localhost:12345/api/chat"),
            ("deepseek", "https://proxy.example/v1", "https://proxy.example/v1/chat/completions"),
            ("openai", "http://localhost:11434/v1", "http://localhost:11434/v1/chat/completions"),
        ]:
            with self.subTest(provider=provider):
                config = {"name": "test", "provider": provider, "base_url": base_url, "api_key": FAKE_KEY}
                self.assertEqual(resolve_provider(normalize_model_config(config)), provider)
                self.assertEqual(LLMClient(config).url, url)

    def test_redaction_does_not_mutate_user_configuration(self):
        config = {"model": {"name": "test", "api_key": FAKE_KEY, "api_key_env": "TEST_MODEL_KEY"}}
        original = deepcopy(config)
        saved = redact_config(config)
        self.assertEqual(config, original)
        self.assertNotIn(FAKE_KEY, json.dumps(saved))
        self.assertEqual(saved["model"]["api_key_env"], "TEST_MODEL_KEY")

    def test_unknown_provider_is_rejected(self):
        with self.assertRaises(ValueError):
            normalize_model_config({"name": "test", "provider": "unsupported"})

    def test_credentials_in_url_are_rejected_without_echoing_them(self):
        for url in [f"https://user:{FAKE_KEY}@models.example", f"https://models.example?api_key={FAKE_KEY}"]:
            with self.subTest(url_type="query" if "?" in url else "userinfo"):
                with self.assertRaises(ValueError) as raised:
                    normalize_model_config({"name": "test", "base_url": url})
                self.assertNotIn(FAKE_KEY, str(raised.exception))

    def test_stream_options_cannot_enable_unsupported_streaming(self):
        with self.assertRaises(ValueError):
            normalize_model_config({"name": "test", "params": {"stream_options": {"include_usage": True}}})

    def test_deepseek_conflicting_thinking_switches_fail_validation(self):
        for think, effort in [(True, "none"), (False, "high")]:
            with self.subTest(think=think, effort=effort):
                with self.assertRaises(ValueError):
                    normalize_model_config({"name": "test", "provider": "deepseek", "think": think,
                                            "params": {"reasoning_effort": effort}})
        for think, effort in [(True, "high"), (False, "none")]:
            with self.subTest(think=think, effort=effort):
                config = normalize_model_config({"name": "test", "provider": "deepseek", "think": think,
                                                 "params": {"reasoning_effort": effort}})
                self.assertEqual(config["params"]["reasoning_effort"], effort)


class RemoteModelClientTest(unittest.TestCase):
    def client(self, **changes):
        config = {"name": "deepseek-flash", "base_url": "https://api.deepseek.com", "api_key": FAKE_KEY}
        config.update(changes)
        return LLMClient(config)

    def test_auth_options_json_and_usage_are_adapted(self):
        client = self.client(think=False, params={"temperature": 0, "num_predict": 128, "num_ctx": 65536})
        messages = [{"role": "system", "content": "Choose an action."}, {"role": "user", "content": "Wait."}]
        original = deepcopy(messages)
        records = []
        client.on_call = records.append
        with patch("modules.common.llm.opener.open", return_value=response(completion())) as http:
            self.assertEqual(client.generate(messages, seed=123, actions=["noop", "move_left"]), '{"action":"noop"}')
        request = http.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(request.get_header("Authorization"), "Bearer " + FAKE_KEY)
        self.assertEqual(payload["model"], "deepseek-flash")
        self.assertEqual(payload["max_tokens"], 128)
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertFalse(payload["stream"])
        for name in ("options", "num_ctx", "num_predict", "seed", "think", "api_key"):
            self.assertNotIn(name, payload)
        self.assertEqual(messages, original)
        system = "\n".join(m["content"] for m in payload["messages"] if m["role"] == "system")
        self.assertIn("json", system.lower())
        self.assertIn("move_left", system)
        self.assertEqual((client.calls, client.input_tokens, client.output_tokens), (1, 11, 7))
        self.assertEqual(client.last_response["prompt_eval_count"], 11)
        self.assertEqual(client.last_response["eval_count"], 7)
        self.assertEqual(client.last_response["done_reason"], "stop")
        self.assertEqual(records[0]["provider"], "deepseek")
        self.assertEqual(records[0]["url"], client.url)
        self.assertEqual(records[0]["status"], "ok")
        self.assertGreaterEqual(records[0]["seconds"], 0)
        self.assertNotIn("headers", records[0])
        self.assertNotIn(FAKE_KEY, json.dumps(records))

    def test_native_max_tokens_wins_over_ollama_alias(self):
        client = self.client(params={"num_predict": 128, "max_tokens": 256})
        with patch("modules.common.llm.opener.open", return_value=response(completion())) as http:
            client.generate([], seed=0)
        self.assertEqual(json.loads(http.call_args.args[0].data)["max_tokens"], 256)

    def test_deepseek_maps_max_completion_tokens_without_overwriting_max_tokens(self):
        for params, expected in [({"max_completion_tokens": 128}, 128),
                                 ({"max_completion_tokens": 128, "max_tokens": 256, "num_predict": 64}, 256)]:
            with self.subTest(params=params):
                client = self.client(params=params)
                with patch("modules.common.llm.opener.open", return_value=response(completion())) as http:
                    client.generate([], seed=0)
                payload = json.loads(http.call_args.args[0].data)
                self.assertEqual(payload["max_tokens"], expected)
                self.assertNotIn("max_completion_tokens", payload)
                self.assertNotIn("num_predict", payload)

    def test_schema_retains_react_thought_contract(self):
        schema = action_schema(["noop"], {"thought": {"type": "string"}})
        client = self.client()
        with patch("modules.common.llm.opener.open", return_value=response(completion('{"thought":"Wait.","action":"noop"}'))) as http:
            client.generate([], seed=0, response_schema=schema)
        payload = json.loads(http.call_args.args[0].data)
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        system = "\n".join(m["content"] for m in payload["messages"] if m["role"] == "system")
        self.assertIn('"thought"', system)

    def test_spring_text_nodes_do_not_request_json(self):
        client = self.client(think=True)
        with patch("modules.common.llm.opener.open", return_value=response(completion("Look for nearby wood."))) as http:
            self.assertEqual(client.generate([], seed=0, label="q1"), "Look for nearby wood.")
        payload = json.loads(http.call_args.args[0].data)
        self.assertNotIn("response_format", payload)
        self.assertEqual(payload["thinking"], {"type": "enabled"})

    def test_generic_compatible_api_does_not_receive_deepseek_thinking(self):
        client = self.client(base_url="http://localhost:8000/v1", api_key="", think=True)
        with patch("modules.common.llm.opener.open", return_value=response(completion())) as http:
            client.generate([], seed=0)
        request = http.call_args.args[0]
        payload = json.loads(request.data)
        self.assertNotIn("thinking", payload)
        self.assertNotIn("think", payload)
        self.assertIsNone(request.get_header("Authorization"))

    def test_key_from_environment_and_explicit_key_precedence(self):
        for configured_key, expected_key in [("", "environment-secret"), (FAKE_KEY, FAKE_KEY)]:
            with self.subTest(configured_key=configured_key), patch.dict(os.environ, {"TEST_MODEL_KEY": "environment-secret"}):
                client = self.client(api_key=configured_key, api_key_env="TEST_MODEL_KEY")
                with patch("modules.common.llm.opener.open", return_value=response(completion())) as http:
                    client.generate([], seed=0)
                self.assertEqual(http.call_args.args[0].get_header("Authorization"), "Bearer " + expected_key)

    def test_missing_deepseek_key_fails_before_http(self):
        with patch.dict(os.environ, {}, clear=True), patch("modules.common.llm.opener.open") as http:
            with self.assertRaises(ValueError):
                self.client(api_key="", api_key_env="MISSING_TEST_KEY").generate([], seed=0)
            http.assert_not_called()

    def test_truncation_and_other_finish_reasons_keep_usage_and_error_records(self):
        for finish_reason in ("length", "content_filter", "tool_calls"):
            with self.subTest(finish_reason=finish_reason):
                client = self.client()
                records = []
                client.on_call = records.append
                with patch("modules.common.llm.opener.open", return_value=response(completion(finish_reason=finish_reason))) as http:
                    with self.assertRaises(ValueError):
                        client.generate([], seed=0, actions=["noop"])
                    self.assertEqual(http.call_count, 1)
                self.assertEqual((client.input_tokens, client.output_tokens), (11, 7))
                self.assertEqual(client.last_response["done_reason"], finish_reason)
                self.assertEqual(records[0]["status"], "error")

    def test_reasoning_content_is_not_a_final_answer(self):
        client = self.client()
        with patch("modules.common.llm.opener.open", return_value=response(completion("", reasoning='{"action":"noop"}'))):
            with self.assertRaises(ValueError):
                client.generate([], seed=0, actions=["noop"])

    def test_error_body_cannot_leak_api_key_and_is_not_retried(self):
        client = self.client()
        records = []
        client.on_call = records.append
        error = HTTPError(client.url, 401, "Unauthorized", {}, BytesIO(f"Invalid key: {FAKE_KEY}".encode()))
        with patch("modules.common.llm.opener.open", side_effect=error) as http:
            with self.assertRaises(RuntimeError) as raised:
                client.generate([], seed=0)
        self.assertEqual(http.call_count, 1)
        self.assertIn("401", str(raised.exception))
        self.assertNotIn(FAKE_KEY, str(raised.exception))
        self.assertNotIn(FAKE_KEY, json.dumps(records))

    def test_budget_is_checked_before_remote_http(self):
        client = self.client()
        client.max_calls = 0
        with patch("modules.common.llm.opener.open") as http:
            with self.assertRaises(ModelCallBudgetExceeded):
                client.generate([], seed=0)
            http.assert_not_called()
        self.assertEqual(client.calls, 0)


class RemoteAgentFlowTest(unittest.TestCase):
    def test_failed_request_does_not_leak_environment_key_into_saved_files(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"TEST_MODEL_KEY": FAKE_KEY}):
            config = load_config(PROJECT_ROOT / "configs/naive.yaml")
            config["model"] = {"name": "deepseek-flash", "base_url": "https://api.deepseek.com", "api_key_env": "TEST_MODEL_KEY"}
            config["output_dir"] = temp
            config["task"]["success_condition"] = None
            config["experiment"].update(seeds=[0], episodes_per_seed=1, max_steps=1, max_model_calls=1)
            error = HTTPError("https://api.deepseek.com/chat/completions", 401, "Unauthorized", {},
                              BytesIO(f"Invalid key: {FAKE_KEY}".encode()))
            captured = StringIO()
            with redirect_stdout(captured), patch("modules.common.llm.opener.open", side_effect=error) as http:
                with self.assertRaises(RuntimeError) as raised:
                    run_experiment(config)
            self.assertEqual(http.call_count, 1)
            self.assertNotIn(FAKE_KEY, str(raised.exception))
            self.assertNotIn(FAKE_KEY, captured.getvalue())
            output = next(Path(temp).iterdir())
            self.assertTrue((output / "error.json").exists())
            episode = output / "episode_0000_seed_0_repeat_0"
            calls = [json.loads(line) for line in (episode / "model_calls.jsonl").read_text().splitlines()]
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["status"], "error")
            for path in output.rglob("*"):
                if path.is_file():
                    self.assertNotIn(FAKE_KEY.encode(), path.read_bytes(), str(path))

    def test_all_agents_complete_real_crafter_step_with_remote_protocol(self):
        with tempfile.TemporaryDirectory() as temp:
            for name, expected_calls in [("naive", 1), ("react", 1), ("spring", 9)]:
                with self.subTest(agent=name):
                    config = load_config(PROJECT_ROOT / f"configs/{name}.yaml")
                    config["model"] = {"name": "deepseek-flash", "base_url": "https://api.deepseek.com", "api_key": FAKE_KEY}
                    config["output_dir"] = temp
                    config["task"]["success_condition"] = None
                    config["experiment"].update(seeds=[0], episodes_per_seed=1, max_steps=1, max_model_calls=9)

                    def reply(request, **kwargs):
                        payload = json.loads(request.data)
                        if "response_format" in payload:
                            decision = {"action": "noop"}
                            if name == "react":
                                decision["thought"] = "Wait for environmental feedback."
                            content = json.dumps(decision)
                        else:
                            content = "Inspect the current observation before choosing a valid action."
                        return response(completion(content))

                    with redirect_stdout(StringIO()), patch("modules.common.llm.opener.open", side_effect=reply) as http, \
                            patch("utils.visualization.visualize_run", return_value=[]):
                        result = run_experiment(config)
                    self.assertEqual(http.call_count, expected_calls)
                    self.assertEqual(result["total_model_calls"], expected_calls)
                    self.assertEqual(result["total_input_tokens"], expected_calls * 11)
                    self.assertEqual(result["total_output_tokens"], expected_calls * 7)
                    episode = result["results"][0]
                    self.assertEqual(episode["steps"], 1)
                    self.assertEqual(episode["stop_reason"], "max_steps")
                    output = Path(result["output_dir"])
                    self.assertFalse((output / "error.json").exists())
                    calls_path = output / episode["episode_id"] / "model_calls.jsonl"
                    calls = [json.loads(line) for line in calls_path.read_text().splitlines()]
                    self.assertEqual([c["call"] for c in calls], list(range(1, expected_calls + 1)))
                    self.assertTrue(all(c["provider"] == "deepseek" and c["status"] == "ok" for c in calls))
                    if name == "spring":
                        self.assertEqual([c["label"] for c in calls], ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "qa"])
                        self.assertTrue(all("response_format" not in c["request"] for c in calls[:-1]))
                        self.assertEqual(calls[-1]["request"]["response_format"], {"type": "json_object"})
                    for path in output.rglob("*"):
                        if path.is_file():
                            self.assertNotIn(FAKE_KEY.encode(), path.read_bytes(), str(path))


if __name__ == "__main__":
    unittest.main()
