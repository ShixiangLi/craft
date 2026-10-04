"""统一实验组装；方法通过相同环境、调用预算和记录流程运行。"""
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4
import yaml

from agents.naive import NaiveAgent
from agents.react import ReActAgent
from agents.spring import SpringAgent
from agents.llm_agent import LLMBaseAgent
from engine.environment import create_environment
from engine.evaluator import aggregate_results
from engine.recorder import ExperimentRecorder
from engine.runner import run_episode
from modules.common.llm import LLMClient
from utils.config import validate_config
from utils.io import write_json


def create_agent(config: dict) -> LLMBaseAgent:
    methods = {"naive": NaiveAgent, "react": ReActAgent, "spring": SpringAgent}
    name = config["agent"]["name"]
    if name not in methods:
        raise ValueError(f"未知智能体: {name}")
    return methods[name](config["agent"], LLMClient(config["model"]))


def run_experiment(config: dict) -> dict:
    config = validate_config(config)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    output = Path(config["output_dir"]) / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    write_json({"run_id": run_id, "versions": {name: version(name) for name in ("crafter", "numpy", "PyYAML")},
                "prompts": {key: Path(path).read_text(encoding="utf-8") for key, path in config["agent"]["prompts"].items()}}, output / "metadata.json")
    print(f"输出目录: {output}", flush=True)
    environment = create_environment(config["environment"])
    results = []
    try:
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
        return summary
    except (Exception, KeyboardInterrupt) as exc:
        write_json({"error": f"{type(exc).__name__}: {exc}", "completed_episodes": results}, output / "error.json")
        raise
    finally:
        environment.close()
