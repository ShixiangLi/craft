"""统一实验组装；方法通过相同环境、调用预算和记录流程运行。"""
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4
import yaml

from agents.naive import NaiveAgent
from agents.react import ReActAgent
from agents.spring import SpringAgent
from agents.adapt import AdaptAgent
from agents.harness import HarnessAgent
from agents.reactree import ReAcTreeAgent
from agents.graph import GraphAgent
from agents.llm_agent import LLMBaseAgent
from engine.environment import create_environment
from engine.evaluator import aggregate_results
from engine.recorder import ExperimentRecorder
from engine.runner import run_episode
from modules.common.llm import LLMClient
from modules.common.model_config import redact_config
from utils.config import validate_config
from utils.io import write_json


def create_agent(config: dict) -> LLMBaseAgent:
    methods = {"naive": NaiveAgent, "react": ReActAgent, "spring": SpringAgent,
               "adapt": AdaptAgent, "harness": HarnessAgent, "reactree": ReAcTreeAgent,
               "graph": GraphAgent}
    name = config["agent"]["name"]
    if name not in methods:
        raise ValueError(f"未知智能体: {name}")
    agent = methods[name](config["agent"], LLMClient(config["model"]))
    if name == "reactree":
        agent.episodic.validate_evaluation_seeds(config["experiment"]["seeds"])
    return agent


def run_experiment(config: dict) -> dict:
    config = validate_config(config)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    output = Path(config["output_dir"]) / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.yaml").write_text(yaml.safe_dump(redact_config(config), allow_unicode=True, sort_keys=False), encoding="utf-8")
    metadata = {"run_id": run_id, "versions": {name: version(name) for name in ("crafter", "numpy", "PyYAML")},
                "prompts": {key: Path(path).read_text(encoding="utf-8") for key, path in config["agent"]["prompts"].items()}}
    write_json(metadata, output / "metadata.json")
    print(f"输出目录: {output}", flush=True)
    environment = create_environment(config["environment"])
    results = []
    try:
        if config["agent"]["name"] == "reactree" and config["agent"]["params"]["episodic_memory_sources"]:
            from modules.reactree.corpus import build_frozen_corpus
            params = config["agent"]["params"]
            library = output / "episodic_memory.jsonl"
            manifest = build_frozen_corpus(params["episodic_memory_sources"], library,
                                           evaluation_seeds=config["experiment"]["seeds"],
                                           task=config["task"])
            # 保存可直接重放的有效配置；原始训练来源和校验摘要由 manifest/metadata 留存。
            params["episodic_memory_path"] = str(library.resolve())
            params["episodic_memory_sources"] = []
            metadata["reactree_memory_build"] = manifest
            write_json(metadata, output / "metadata.json")
            (output / "config.yaml").write_text(yaml.safe_dump(
                redact_config(config), allow_unicode=True, sort_keys=False), encoding="utf-8")
        agent = create_agent(config)
        for seed in config["experiment"]["seeds"]:
            for repeat in range(config["experiment"]["episodes_per_seed"]):
                episode_id = f"episode_{len(results):04d}_seed_{seed}_repeat_{repeat}"
                recorder = ExperimentRecorder(output / episode_id)
                result = run_episode(agent, environment, config, seed=seed, recorder=recorder)
                result["episode_id"] = episode_id
                results.append(result)
        summary = {"run_id": run_id, "output_dir": str(output),
                   **aggregate_results(results), "results": results}
        write_json(summary, output / "summary.json")
        # 绘图失败不使已完成的实验丢失；可用分析入口离线重试。
        try:
            from utils.visualization import visualize_run
            figures = visualize_run(output)
            summary["visualizations"] = [str(path.relative_to(output)) for path in figures]
        except Exception as exc:
            summary["visualization_error"] = f"{type(exc).__name__}: {exc}"
            print(f"实验已完成，但绘图失败：{exc}。可稍后运行 scripts.analyze_results。", flush=True)
        write_json(summary, output / "summary.json")
        return summary
    except (Exception, KeyboardInterrupt) as exc:
        write_json({"error": f"{type(exc).__name__}: {exc}", "completed_episodes": results}, output / "error.json")
        raise
    finally:
        environment.close()
