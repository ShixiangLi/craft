"""各策略通过同一个实验入口运行，包含真实 Crafter 和多调用预算。"""
from contextlib import redirect_stdout
from io import BytesIO, StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from engine.experiment import run_experiment
from utils.config import PROJECT_ROOT, load_config


def reply(request, **kwargs):
    payload = json.loads(request.data)
    if 'format' in payload:
        decision = {'action': 'noop'}
        if 'thought' in payload['format']['properties']:
            decision['thought'] = 'Wait for environmental feedback.'
        content = json.dumps(decision)
    else:
        content = 'The player should check the current observation and requirements.'
    return BytesIO(json.dumps({'model': 'test', 'message': {'content': content},
                              'done': True, 'done_reason': 'stop',
                              'prompt_eval_count': 10, 'eval_count': 5}).encode())


class AgentFlowTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def config(self, name):
        config = load_config(PROJECT_ROOT / f'configs/{name}.yaml')
        config['output_dir'] = self.temp.name
        config['task']['success_condition'] = None
        config['experiment'].update(seeds=[0], episodes_per_seed=1, max_steps=2, max_model_calls=30)
        return config

    def run_config(self, config):
        with redirect_stdout(StringIO()), patch('modules.common.llm.opener.open', side_effect=reply) as http:
            result = run_experiment(config)
        return result, http.call_count

    def test_all_methods_share_runner_and_preserve_every_call(self):
        for name, per_step in [('naive', 1), ('react', 1), ('spring', 9)]:
            with self.subTest(agent=name):
                result, count = self.run_config(self.config(name))
                self.assertEqual(count, 2 * per_step)
                self.assertEqual(result['total_model_calls'], count)
                self.assertEqual(result['total_output_tokens'], count * 5)
                self.assertNotIn('visualization_error', result)
                self.assertEqual(len(result['visualizations']), 2)
                for figure in result['visualizations']:
                    self.assertGreater((Path(result['output_dir']) / figure).stat().st_size, 100)
                episode = result['results'][0]
                self.assertEqual(episode['steps'], 2)
                self.assertEqual(episode['stop_reason'], 'max_steps')
                path = Path(result['output_dir']) / episode['episode_id']
                calls = [json.loads(line) for line in (path/'model_calls.jsonl').read_text().splitlines()]
                self.assertEqual(len(calls), count)
                self.assertEqual([c['call'] for c in calls], list(range(1, count+1)))
                self.assertTrue(all(c['status'] == 'ok' for c in calls))
                if name == 'spring':
                    self.assertEqual([c['label'] for c in calls], ['q1','q2','q3','q4','q5','q6','q7','q8','qa'] * 2)

    def test_spring_insufficient_budget_does_not_execute_partial_decision(self):
        config = self.config('spring')
        config['experiment']['max_model_calls'] = 10
        result, count = self.run_config(config)
        self.assertEqual(count, 9)
        self.assertEqual(result['results'][0]['steps'], 1)
        self.assertEqual(result['results'][0]['stop_reason'], 'max_model_calls')
        self.assertFalse((Path(result['output_dir'])/'error.json').exists())

    def test_spring_budget_smaller_than_one_decision(self):
        config = self.config('spring')
        config['experiment']['max_model_calls'] = 8
        result, count = self.run_config(config)
        self.assertEqual(count, 0)
        self.assertEqual(result['results'][0]['steps'], 0)
        self.assertEqual(result['results'][0]['stop_reason'], 'max_model_calls')

    def test_multiple_episodes_reset_counts_and_agent_state(self):
        config = self.config('react')
        config['experiment'].update(episodes_per_seed=2, max_steps=1)
        result, count = self.run_config(config)
        self.assertEqual(count, 2)
        self.assertEqual([r['model_calls'] for r in result['results']], [1,1])
        prompts = []
        for episode in result['results']:
            path = Path(result['output_dir'])/episode['episode_id']/'model_calls.jsonl'
            prompts.append(json.loads(path.read_text())['request']['messages'])
        self.assertEqual(prompts[0], prompts[1])

    def test_plot_failure_preserves_completed_experiment(self):
        config = self.config('naive')
        config['experiment']['max_steps'] = 1
        with patch('utils.visualization.visualize_run', side_effect=RuntimeError('plot unavailable')):
            result, count = self.run_config(config)
        self.assertEqual(count, 1)
        self.assertIn('plot unavailable', result['visualization_error'])
        output = Path(result['output_dir'])
        self.assertTrue((output / 'summary.json').exists())
        self.assertFalse((output / 'error.json').exists())


if __name__ == '__main__':
    unittest.main()
