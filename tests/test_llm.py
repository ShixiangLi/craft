"""思考输出与生成截断的回归检查。"""
import json
from io import BytesIO
import unittest
from unittest.mock import patch
from modules.common.llm import LLMClient, ModelCallBudgetExceeded


class LLMResponseTest(unittest.TestCase):
    def test_budget_checked_before_http(self):
        client = LLMClient({'name': 'test'})
        client.max_calls = 0
        with patch('modules.common.llm.opener.open') as http:
            with self.assertRaises(ModelCallBudgetExceeded):
                client.generate([], seed=0)
            http.assert_not_called()
        self.assertEqual(client.calls, 0)

    def test_unstructured_nodes_have_no_action_schema_and_are_recorded(self):
        client = LLMClient({'name': 'test'})
        records = []
        client.on_call = records.append
        data = {'message': {'content': 'Node answer'}, 'done_reason': 'stop'}
        with patch('modules.common.llm.opener.open', return_value=BytesIO(json.dumps(data).encode())) as http:
            self.assertEqual(client.generate([], seed=0, label='q1'), 'Node answer')
        self.assertNotIn('format', json.loads(http.call_args.args[0].data))
        self.assertEqual(records[0]['label'], 'q1')
        self.assertEqual(records[0]['response']['message']['content'], 'Node answer')

    def test_truncation_keeps_response_and_usage(self):
        client = LLMClient({'name': 'qwen3:8b', 'think': True, 'params': {'num_predict': 128}})
        for content in ('', '{"action":'):
            data = {'message': {'content': content, 'thinking': 'unfinished'},
                    'done_reason': 'length', 'eval_count': 128, 'prompt_eval_count': 10}
            with self.subTest(content=content), patch('modules.common.llm.opener.open', return_value=BytesIO(json.dumps(data).encode())):
                with self.assertRaisesRegex(ValueError, '生成被截断'):
                    client.generate([], seed=0, actions=['noop'])
            self.assertEqual(client.last_response['done_reason'], 'length')
        self.assertEqual(client.calls, 2)
        self.assertEqual(client.output_tokens, 256)

    def test_thinking_is_not_an_action_fallback(self):
        client = LLMClient({'name': 'qwen3:8b'})
        data = {'message': {'content': '', 'thinking': '{"action":"noop"}'}, 'done_reason': 'stop'}
        with patch('modules.common.llm.opener.open', return_value=BytesIO(json.dumps(data).encode())):
            with self.assertRaisesRegex(ValueError, 'has_thinking=True'):
                client.generate([], seed=0, actions=['noop'])

    def test_finished_answer_with_thinking(self):
        client = LLMClient({'name': 'qwen3:8b'})
        data = {'message': {'content': '{"action":"noop"}', 'thinking': 'analysis'}, 'done_reason': 'stop'}
        with patch('modules.common.llm.opener.open', return_value=BytesIO(json.dumps(data).encode())):
            self.assertEqual(client.generate([], seed=0, actions=['noop']), '{"action":"noop"}')
