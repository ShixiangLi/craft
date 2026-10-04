"""读取并校验第一版 naive 实验 YAML；相对路径以项目根目录为准。"""
from copy import deepcopy
from pathlib import Path
import yaml
from crafter import constants

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def validate_config(config: dict) -> dict:
    config = deepcopy(config)
    if config.get("agent", {}).get("name") != "naive":
        raise ValueError("第一版仅实现 agent.name: naive")
    if config.get("model", {}).get("provider") != "ollama":
        raise ValueError("model.provider 必须是 ollama")
    if not isinstance(config["model"].get("name"), str) or not config["model"]["name"].strip():
        raise ValueError("请指定 model.name（ollama list 中的模型名）")
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
    for key in ("system", "step"):
        path = Path(prompts[key])
        path = path if path.is_absolute() else PROJECT_ROOT / path
        if not path.is_file():
            raise ValueError(f"提示词不存在: {path}")
        prompts[key] = str(path.resolve())
    output = Path(config["output_dir"])
    config["output_dir"] = str(output if output.is_absolute() else PROJECT_ROOT / output)
    timeout = config["model"].get("timeout", 120)
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
        raise ValueError("model.timeout 必须大于零")
    return config


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("配置必须为 YAML 映射")
    return validate_config(config)
