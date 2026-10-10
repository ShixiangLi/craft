"""读取并校验实验 YAML；相对路径以项目根目录为准。"""
from copy import deepcopy
from pathlib import Path
import yaml
from crafter import constants

from modules.common.model_config import normalize_model_config
from modules.adapt.components import validate_params as validate_adapt_params
from modules.harness.components import validate_params as validate_harness_params
from modules.reactree.components import validate_params as validate_reactree_params

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def validate_config(config: dict) -> dict:
    config = deepcopy(config)
    agent_name = config.get("agent", {}).get("name")
    if agent_name not in ("naive", "react", "spring", "adapt", "harness", "reactree", "graph"):
        raise ValueError("agent.name 必须是 naive、react、spring、adapt、harness、reactree 或 graph")
    config["model"] = normalize_model_config(config.get("model", {}))
    env = config.get("environment", {})
    if env.get("name") != "crafter" or env.get("observation") != "local_semantic":
        raise ValueError("仅支持 crafter / local_semantic 观测")
    if {"seed", "length"} & env.get("params", {}).keys():
        raise ValueError("环境 seed 和 length 由 experiment 统一控制")
    experiment = config.get("experiment", {})
    for name in ("max_steps", "max_model_calls", "episodes_per_seed"):
        value = experiment.get(name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"experiment.{name} 必须是正整数")
    seeds = experiment.get("seeds")
    if not isinstance(seeds, list) or not seeds or any(type(s) is not int or not 0 <= s < 2**31 for s in seeds):
        raise ValueError("experiment.seeds 必须是非空整数列表，范围 [0, 2**31)")
    task = config.get("task", {})
    if not isinstance(task.get("description"), str) or not task["description"].strip():
        raise ValueError("请填写 task.description")
    condition = task.get("success_condition")
    if condition is not None:
        if not isinstance(condition, dict) or condition.get("achievement") not in constants.achievements:
            raise ValueError("success_condition.achievement 必须是 Crafter 成就名")
        count = condition.get("count", 1)
        if type(count) is not int or count < 1:
            raise ValueError("success_condition.count 必须是正整数")
    prompts = config["agent"]["prompts"]
    required_prompts = {"system", "step"}
    if agent_name in ("react", "adapt", "harness", "reactree"):
        required_prompts |= {"rules", "examples"}
        if agent_name == "adapt":
            required_prompts.add("planner")
            config["agent"]["params"] = validate_adapt_params(config["agent"].get("params"))
        elif agent_name == "harness":
            config["agent"]["params"] = validate_harness_params(config["agent"].get("params"))
        elif agent_name == "reactree":
            config["agent"]["params"] = validate_reactree_params(config["agent"].get("params"))
            memory_path = config["agent"]["params"]["episodic_memory_path"]
            if memory_path is not None:
                path = Path(memory_path)
                path = path if path.is_absolute() else PROJECT_ROOT / path
                if not path.is_file():
                    raise ValueError(f"ReAcTree 经验库不存在: {path}")
                config["agent"]["params"]["episodic_memory_path"] = str(path.resolve())
            sources = []
            for source in config["agent"]["params"]["episodic_memory_sources"]:
                path = Path(source)
                path = path if path.is_absolute() else PROJECT_ROOT / path
                if not path.is_dir():
                    raise ValueError(f"ReAcTree 训练源必须是已有运行/回合目录: {path}")
                sources.append(str(path.resolve()))
            config["agent"]["params"]["episodic_memory_sources"] = sources
        window = config["agent"].get("params", {}).get("max_history_steps")
        if window is not None and (type(window) is not int or window <= 0):
            raise ValueError("agent.params.max_history_steps 必须为 null 或正整数")
    elif agent_name == "graph":
        from agents.graph import validate_params as validate_graph_params
        required_prompts.add("rules")
        config["agent"]["params"] = validate_graph_params(config["agent"].get("params"))
    elif agent_name == "spring":
        required_prompts |= {"questions", "knowledge"}
    if not required_prompts <= prompts.keys():
        raise ValueError(f"缺少提示词: {sorted(required_prompts - prompts.keys())}")
    for key in prompts:
        path = Path(prompts[key])
        path = path if path.is_absolute() else PROJECT_ROOT / path
        if not path.is_file():
            raise ValueError(f"提示词不存在: {path}")
        prompts[key] = str(path.resolve())
    output = Path(config["output_dir"])
    config["output_dir"] = str(output if output.is_absolute() else PROJECT_ROOT / output)
    if agent_name == "harness" and Path(config["output_dir"]).resolve() != PROJECT_ROOT / "outputs/harness":
        raise ValueError("Harness 产物必须保存在 outputs/harness，请勿使用其他 output_dir")
    return config


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("配置必须为 YAML 映射")
    return validate_config(config)
