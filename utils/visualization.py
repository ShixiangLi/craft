"""从已保存的轨迹绘图，不调用模型或环境；支持无桌面的服务器。"""
import json
from collections import Counter
from pathlib import Path


def _pyplot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding='utf-8') as file:
        return [json.loads(line) for line in file if line.strip()]


def _save(figure, path: Path, plt) -> list[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    paths = [path.with_suffix('.png'), path.with_suffix('.svg')]
    try:
        for target in paths:
            figure.savefig(target, dpi=160, bbox_inches='tight')
    finally:
        plt.close(figure)
    return paths


def plot_results(results: list[dict], output_path: str | Path) -> list[Path]:
    """多个回合的基础指标；不把不同回合连成同一条状态曲线。"""
    plt = _pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    labels = [f"{i}: seed {r['seed']}" for i, r in enumerate(results)]
    for ax, key, title in zip(axes, ['steps', 'total_reward', 'achievement_count'],
                              ['Environment steps', 'Episode reward', 'Achievements unlocked']):
        ax.bar(labels, [r[key] for r in results], color='#3576b9')
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=.2)
    return _save(fig, Path(output_path), plt)


def plot_episode(records: list[dict], calls: list[dict], result: dict,
                 output_path: str | Path) -> list[Path]:
    plt = _pyplot()
    states = [r for r in records if r.get('type') in ('reset', 'step')]
    steps = [r for r in records if r.get('type') == 'step']
    if not states:
        raise ValueError('轨迹没有初始状态或环境动作')
    x = [r['observation']['step'] for r in states]
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), constrained_layout=True)
    ax = axes[0, 0]
    for key in ['health', 'food', 'drink', 'energy']:
        ax.plot(x, [r['observation']['inventory'][key] for r in states], label=key)
    ax.set(title='Survival state', xlabel='Environment step', ylim=(-.3, 9.5))
    ax.legend(ncol=4)

    ax = axes[0, 1]
    counts = [sum(v > 0 for v in r['observation']['achievements'].values()) for r in states]
    ax.step(x, counts, where='post', color='#20835c')
    for r in steps:
        if r.get('new_achievements'):
            count = sum(v > 0 for v in r['observation']['achievements'].values())
            ax.annotate('\n'.join(r['new_achievements']), (r['step'], count),
                        xytext=(4, 6), textcoords='offset points', fontsize=8)
    ax.set(title='Achievement milestones', xlabel='Environment step', ylabel='Unique achievements')
    ax.set_ylim(-.2, max(counts) + 1.5)

    ax = axes[1, 0]
    for key in ['wood', 'stone', 'coal', 'iron', 'diamond']:
        ax.step(x, [r['observation']['inventory'][key] for r in states], where='post', label=key)
    ax.set(title='Resources currently held', xlabel='Environment step', ylabel='Inventory count')
    ax.legend(ncol=5)

    ax = axes[1, 1]
    counts = Counter(r['action'] for r in steps)
    actions = sorted(counts, key=counts.get)
    bars = ax.barh(actions, [counts[a] for a in actions], color='#3576b9')
    ax.bar_label(bars, padding=3)
    ax.set(title='Actions requested (not proof of success)', xlabel='Count')
    ax.margins(x=.15)

    ax = axes[2, 0]
    for key, label in [('prompt_eval_count', 'Input tokens / call'), ('eval_count', 'Output tokens / call')]:
        points = [(c['call'], c['response'][key]) for c in calls
                  if isinstance(c.get('response'), dict) and key in c['response']]
        if points:
            ax.plot([p[0] for p in points], [p[1]/1000 for p in points], label=label)
    ax.set(title='Ollama-reported tokens', xlabel='Model call (not environment step)', ylabel='Thousands of tokens')
    if ax.lines:
        ax.legend()
    else:
        ax.text(.5, .5, 'No model call log', transform=ax.transAxes, ha='center')

    ax = axes[2, 1]
    if calls:
        ax.plot([c['call'] for c in calls], [c['seconds'] for c in calls], color='#be6d24')
    else:
        ax.text(.5, .5, 'No model call log', transform=ax.transAxes, ha='center')
    ax.set(title='Model request latency', xlabel='Model call (not environment step)', ylabel='Seconds')
    for ax in axes.flat:
        ax.grid(alpha=.2)
    success = result.get('success')
    fig.suptitle(f"{Path(output_path).stem} | steps={result['steps']} | success={success} | "
                 f"stop={result['stop_reason']} | time={result['seconds']/60:.1f} min", fontsize=14)
    return _save(fig, Path(output_path), plt)


def visualize_run(run_dir: str | Path) -> list[Path]:
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / 'summary.json').read_text(encoding='utf-8'))
    output = run_dir / 'visualizations'
    files = []
    for result in summary['results']:
        episode = run_dir / result['episode_id']
        files.extend(plot_episode(_read_jsonl(episode / 'trajectory.jsonl'),
                                  _read_jsonl(episode / 'model_calls.jsonl'), result,
                                  output / result['episode_id']))
    if len(summary['results']) > 1:
        files.extend(plot_results(summary['results'], output / 'episodes'))
    return files
