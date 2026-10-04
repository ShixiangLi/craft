"""思考输出与生成截断的回归检查。"""
import json
from io import BytesIO
import unittest
from unittest.mock import patch
from modules.common.llm import LLMClient


class LLMResponseTest(unittest.TestCase):
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
