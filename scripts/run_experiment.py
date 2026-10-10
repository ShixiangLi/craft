"""运行方式：python -m scripts.run_experiment --config configs/naive.yaml。"""
import argparse
import json
from engine.experiment import run_experiment
from utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Crafter 智能体实验")
    parser.add_argument("--config", default="configs/naive.yaml")
    parser.add_argument("--model", help="覆盖 model.name")
    parser.add_argument("--base-url", help="覆盖模型 API 地址")
    parser.add_argument("--provider", choices=("auto", "ollama", "deepseek", "openai"),
                        help="覆盖接口类型；auto 根据 API 地址识别")
    parser.add_argument("--max-steps", type=int, help="覆盖每回合步数，用于短程验证")
    parser.add_argument("--seed", type=int, help="仅运行指定 seed，覆盖 experiment.seeds")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.model:
            config["model"]["name"] = args.model
        if args.base_url:
            config["model"]["base_url"] = args.base_url
        if args.provider:
            config["model"]["provider"] = args.provider
        if args.max_steps is not None:
            config["experiment"]["max_steps"] = args.max_steps
        if args.seed is not None:
            config["experiment"]["seeds"] = [args.seed]
        summary = run_experiment(config)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"实验失败: {exc}\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
