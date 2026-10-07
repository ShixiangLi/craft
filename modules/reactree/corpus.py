"""从显式选择的训练日志构建冻结经验库；不调用模型，也不修改源日志。"""

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import yaml

from engine.evaluator import task_succeeded
from modules.common.observation import describe_observation
from modules.reactree.components import render_node_history


def _json(raw):
    def reject(value):
        raise ValueError(f"日志中不允许非有限 JSON 数值: {value}")
    return json.loads(raw, parse_constant=reject)


def _read(path, hashes, *, jsonl=False, yaml_file=False):
    if not path.is_file():
        raise ValueError(f"经验来源缺少已完成日志: {path}")
    raw = path.read_bytes()
    hashes[str(path)] = hashlib.sha256(raw).hexdigest()
    try:
        if yaml_file:
            return yaml.safe_load(raw)
        if jsonl:
            return [_json(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
        return _json(raw)
    except (ValueError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"无法读取经验来源 {path}: {exc}") from exc


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _episodes(source):
    """返回显式 episode，或已完成 run 的完整回合集合，不扫描其他运行目录。"""
    source = Path(source).resolve()
    _require(source.is_dir(), f"经验来源必须是 run 或 episode 目录: {source}")
    if (source / "config.yaml").is_file():
        hashes = {}
        summary = _read(source / "summary.json", hashes)
        _require(isinstance(summary, dict) and isinstance(summary.get("results"), list),
                 f"经验 run 缺少完整 summary.results: {source}")
        config = _read(source / "config.yaml", hashes, yaml_file=True)
        experiment = config.get("experiment", {}) if isinstance(config, dict) else {}
        seeds, repeats = experiment.get("seeds"), experiment.get("episodes_per_seed")
        _require(isinstance(seeds, list) and type(repeats) is int and repeats > 0
                 and len(summary["results"]) == len(seeds) * repeats,
                 f"经验 run 尚未完成配置的全部回合: {source}")
        episodes = []
        for result in summary["results"]:
            episode_id = result.get("episode_id") if isinstance(result, dict) else None
            _require(isinstance(episode_id, str) and Path(episode_id).name == episode_id
                     and episode_id.startswith("episode_"), f"非法来源 episode_id: {episode_id}")
            episodes.append(source / episode_id)
        _require(episodes and len(set(episodes)) == len(episodes),
                 f"经验 run 回合列表为空或重复: {source}")
        _require(set(source.glob("episode_*")) == set(episodes),
                 f"经验 run 的 summary 与实际回合目录不一致: {source}")
        return episodes, True, hashes
    _require((source.parent / "config.yaml").is_file(),
             f"经验 episode 缺少父目录 config.yaml: {source}")
    return [source], False, {}


def _verify_episode(episode, evaluation_seeds, hashes):
    config = _read(episode.parent / "config.yaml", hashes, yaml_file=True)
    _require(isinstance(config, dict), f"经验来源 config 格式无效: {episode}")
    task = config.get("task")
    _require(isinstance(task, dict) and isinstance(task.get("description"), str)
             and isinstance(task.get("success_condition"), dict)
             and isinstance(task["success_condition"].get("achievement"), str)
             and type(task["success_condition"].get("count", 1)) is int
             and task["success_condition"].get("count", 1) > 0,
             f"经验来源必须有独立的根任务和真实成就条件: {episode}")
    _require(config.get("environment", {}).get("name") == "crafter",
             f"经验来源必须使用 Crafter: {episode}")
    method = config.get("agent", {}).get("name")
    _require(method in ("reactree", "react"), f"不支持此经验来源智能体: {method}")
    if method == "reactree":
        params = config["agent"].get("params", {})
        _require(isinstance(params, dict), f"经验来源 agent.params 格式无效: {episode}")
        inherited_memory = params.get("episodic_memory_path") or params.get("episodic_memory_sources")
        _require(params.get("episodic_memory", True) is False or not inherited_memory,
                 "当前构建器仅支持空经验库采集的训练轨迹：来源 ReAcTree 曾启用外部经验，"
                 f"无法排除间接评估种子泄漏: {episode}")
    result = _read(episode / "result.json", hashes)
    _require(isinstance(result, dict), f"非法 result.json: {episode}")
    _require(result.get("success") is True and result.get("stop_reason") == "success",
             f"经验来源未完成真实最终目标: {episode}")
    seed = result.get("seed")
    _require(type(seed) is int and seed in config.get("experiment", {}).get("seeds", []),
             f"经验来源 seed 与采集配置不一致: {episode}")
    _require(seed not in evaluation_seeds,
             f"经验来源训练 seed={seed} 与评估 seeds 重叠: {episode}")
    rows = _read(episode / "trajectory.jsonl", hashes, jsonl=True)
    _require(rows and all(isinstance(row, dict) for row in rows),
             f"经验来源 trajectory 为空或格式无效: {episode}")
    reset, steps = rows[0], rows[1:]
    _require(reset.get("type") == "reset" and reset.get("seed") == seed
             and reset.get("observation", {}).get("step") == 0,
             f"经验来源 reset 与 seed 不一致: {episode}")
    _require(type(result.get("steps")) is int and len(steps) == result["steps"],
             f"经验来源环境步数不完整: {episode}")
    observations = [reset["observation"]]
    for index, row in enumerate(steps, 1):
        obs = row.get("observation", {})
        _require(row.get("type") == "step" and row.get("step") == index
                 and obs.get("step") == index,
                 f"经验来源环境步序不完整: {episode}, step={index}")
        before = observations[-1]
        action_id = row.get("action_id")
        _require(type(action_id) is int and 0 <= action_id < len(before.get("actions", []))
                 and before["actions"][action_id] == row.get("action"),
                 f"经验来源动作与 action_id 不一致: {episode}, step={index}")
        _require(index == len(steps) or row.get("stop_reason") == "running",
                 f"经验来源成功/停止后仍有动作: {episode}, step={index}")
        observations.append(obs)
    final = observations[-1]
    _require(task_succeeded(final, task) is True,
             f"经验来源末帧真实成就未满足根任务: {episode}")
    _require(result.get("achievements") == final.get("achievements")
             and result.get("inventory") == final.get("inventory"),
             f"经验来源 result 与末帧环境状态不一致: {episode}")
    if steps:
        _require(steps[-1].get("stop_reason") == "success",
                 f"经验来源末步不是环境成功: {episode}")
    return method, seed, task, observations, steps


def _tree_records(episode, task, observations, steps, hashes):
    tree = _read(episode / "tree.json", hashes)
    traces = _read(episode / "node_traces.jsonl", hashes, jsonl=True)
    _require(isinstance(tree, dict) and isinstance(tree.get("nodes"), list),
             f"经验来源缺少任务树: {episode}")
    _require(all(isinstance(node, dict) and isinstance(node.get("id"), str)
                 and node["id"] and node.get("kind") in ("agent", "control")
                 and isinstance(node.get("content"), str) and node["content"]
                 and isinstance(node.get("status"), str)
                 and type(node.get("history_entries")) is int and node["history_entries"] >= 0
                 for node in tree["nodes"]), f"经验来源任务树节点字段无效: {episode}")
    nodes = {node["id"]: node for node in tree["nodes"]}
    _require(len(nodes) == len(tree["nodes"]) and isinstance(tree.get("root"), str)
             and tree["root"] in nodes,
             f"经验来源任务树节点无效: {episode}")
    root = nodes[tree["root"]]
    _require(root.get("kind") == "agent" and root.get("content") == task["description"],
             f"经验来源任务树根目标不一致: {episode}")
    _require(all(node.get("status") not in ("running", "expanded") for node in nodes.values()),
             f"经验来源任务树尚未完成 finish_episode: {episode}")
    histories, covered_steps = defaultdict(list), set()
    for trace in traces:
        _require(isinstance(trace, dict) and isinstance(trace.get("node_id"), str)
                 and trace["node_id"] in nodes,
                 f"经验来源节点轨迹归属不明: {episode}")
        node = nodes[trace["node_id"]]
        index = trace.get("step")
        _require(node.get("kind") == "agent" and trace.get("goal") == node.get("content")
                 and type(index) is int and 0 <= index < len(observations),
                 f"经验来源节点目标或步数不一致: {episode}")
        _require(trace.get("observation") == describe_observation(observations[index]),
                 f"经验来源节点观测与真实轨迹不一致: {episode}, step={index}")
        decision, feedback = trace.get("decision"), trace.get("feedback")
        _require(isinstance(decision, dict) and isinstance(feedback, dict),
                 f"经验来源节点缺少决策或反馈: {episode}")
        kind = decision.get("type")
        _require((kind in ("Think", "Expand") and "action" not in decision)
                 or (kind in (None, "Act") and isinstance(decision.get("action"), str)),
                 f"经验来源节点决策类别或动作无效: {episode}")
        _require(feedback.get("source") in ("environment", "controller", "working_memory"),
                 f"经验来源节点反馈 source 无效: {episode}")
        if feedback.get("source") == "environment":
            _require(index < len(steps), f"经验来源节点环境反馈越界: {episode}")
            actual = steps[index]
            _require(actual.get("decision", {}).get("node_id") == node["id"]
                     and decision.get("action") == actual["action"]
                     and all(feedback.get(k) == actual.get(k)
                             for k in ("action", "reward", "new_achievements")),
                     f"经验来源节点动作/反馈与真实轨迹不一致: {episode}, step={index}")
            _require(feedback.get("observation") == describe_observation(observations[index + 1])
                     and all(type(feedback.get(k)) is bool and feedback[k] == actual.get(k)
                             for k in ("terminated", "truncated")),
                     f"经验来源节点后帧观测/终止标志与真实轨迹不一致: {episode}, step={index}")
            _require(index not in covered_steps,
                     f"经验来源环境 step 被节点轨迹重复覆盖: {episode}, step={index + 1}")
            covered_steps.add(index)
        histories[node["id"]].append({key: trace[key] for key in
                                       ("step", "observation", "decision", "feedback")})
    _require(covered_steps == set(range(len(steps))),
             f"经验来源节点轨迹未完整覆盖所有环境 step: {episode}")
    records = []
    for node in nodes.values():
        if node.get("kind") != "agent":
            continue
        history = histories[node["id"]]
        _require(len(history) == node.get("history_entries"),
                 f"经验来源节点历史不完整: {episode}, node={node['id']}")
        if not history:
            continue
        state = node.get("termination")
        _require(state in ("expand", "success", "failure", "interrupted"),
                 f"经验来源节点终态不可导入: {state}")
        records.append({"goal": node["content"], "state": state,
                        "trajectory": render_node_history(history),
                        "episode_success": True,
                        "source": {"node_id": node["id"], "format": "reactree_node",
                                   "node_state_evidence": "model_or_controller"}})
    _require(records, f"经验来源没有完整节点轨迹: {episode}")
    return records


def _react_record(task, observations, steps):
    """只复用完整根任务；不从动作猜测子目标，不导入 probe 的私有状态。"""
    history = []
    for index, row in enumerate(steps):
        thought = row.get("decision", {}).get("thought")
        _require(isinstance(thought, str), "ReAct 经验来源缺少原始 thought")
        history.append({"step": index, "observation": describe_observation(observations[index]),
                        "decision": {"thought": thought, "action": row["action"]},
                        "feedback": {"source": "environment",
                                     **{k: row[k] for k in
                                        ("action", "reward", "new_achievements")},
                                     "observation": describe_observation(observations[index + 1])}})
    _require(history, "ReAct 根任务经验需要至少一个真实环境动作")
    return [{"goal": task["description"], "state": "success",
             "trajectory": render_node_history(history), "episode_success": True,
             "source": {"node_id": None, "format": "react_root_trajectory",
                        "adaptation": "whole_root_trajectory",
                        "node_state_evidence": "environment_final_goal",
                        "subgoals_inferred": False}}]


def build_frozen_corpus(sources, output_path, *, evaluation_seeds, task):
    """验证并冻结显式训练来源，返回审计 manifest（包括输出 path 与 sha256）。

    episode 来源必须成功；完整 run 中失败回合会列入 excluded_episodes。
    输入与评估 seeds 严格隔离。整条经验不总结、不截断；检索器决定是否放入预算。
    仅写新的 output_path 和同名 .manifest.json，绝不覆盖源或现有库。
    """
    _require(isinstance(sources, (list, tuple)), "episodic_memory_sources 必须是路径列表")
    _require(isinstance(evaluation_seeds, (list, tuple, set))
             and all(type(seed) is int for seed in evaluation_seeds), "评估 seeds 必须是整数列表")
    _require(isinstance(task, dict) and isinstance(task.get("description"), str)
             and isinstance(task.get("success_condition"), dict),
             "经验库需要明确根任务 description 和 success_condition")
    output_path = Path(output_path).resolve()
    manifest_path = output_path.with_suffix(".manifest.json")
    _require(not output_path.exists() and not manifest_path.exists(),
             "冻结库或 manifest 已存在，拒绝覆盖")
    records, episodes, excluded, hashes, fingerprints = [], [], [], {}, set()
    for source in sources:
        _require(isinstance(source, (str, Path)) and str(source).strip(), "经验来源路径不能为空")
        selected, is_run, run_hashes = _episodes(source)
        hashes.update(run_hashes)
        for episode in selected:
            source_hashes = {}
            if is_run:
                result = _read(episode / "result.json", source_hashes)
                _require(isinstance(result, dict), f"非法 result.json: {episode}")
                if result.get("success") is not True:
                    excluded.append({"path": str(episode), "reason": "environment_goal_not_successful"})
                    hashes.update(source_hashes)
                    continue
            method, seed, training_task, observations, steps = _verify_episode(
                episode, set(evaluation_seeds), source_hashes)
            fingerprint = (seed, source_hashes[str(episode / "trajectory.jsonl")])
            if fingerprint in fingerprints:
                excluded.append({"path": str(episode), "reason": "duplicate_trajectory"})
                continue
            fingerprints.add(fingerprint)
            examples = (_tree_records(episode, training_task, observations, steps, source_hashes)
                        if method == "reactree" else _react_record(training_task, observations, steps))
            provenance = {"run_id": episode.parent.name, "run_path": str(episode.parent),
                          "episode_id": episode.name, "episode_path": str(episode),
                          "seed": seed, "method": method, "root_task": training_task,
                          "files_sha256": source_hashes}
            for example in examples:
                example["source"] = {**provenance, **example["source"]}
                records.append(example)
            episodes.append({**provenance, "records": len(examples)})
            hashes.update(source_hashes)
    _require(not sources or records, "显式经验来源中没有真实成功且合格的训练回合")
    raw = "".join(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                  for record in records).encode("utf-8")
    manifest = {"version": 1, "path": str(output_path), "sha256": hashlib.sha256(raw).hexdigest(),
                "records": len(records), "evaluation_task": task,
                "evaluation_seeds": sorted(set(evaluation_seeds)),
                "training_seeds": sorted({episode["seed"] for episode in episodes}),
                "sources": [str(Path(source).resolve()) for source in sources],
                "episodes": episodes, "excluded_episodes": excluded, "files_sha256": hashes,
                "frozen_during_evaluation": True,
                "validation": "logged_environment_result_and_complete_trajectory_not_simulator_replay"}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as handle:
        handle.write(raw)
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return manifest
