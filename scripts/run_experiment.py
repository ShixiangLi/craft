"""运行方式：python -m scripts.run_experiment --config configs/naive.yaml。"""
import argparse
import json
from engine.experiment import run_experiment
from utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Crafter naive + Ollama 实验")
    parser.add_argument("--config", default="configs/naive.yaml")
    parser.add_argument("--model", help="覆盖 model.name")
    parser.add_argument("--base-url", help="覆盖 Ollama 服务地址")
    parser.add_argument("--max-steps", type=int, help="覆盖每回合步数，用于短程验证")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.model:
        config["model"]["name"] = args.model
    if args.base_url:
        config["model"]["base_url"] = args.base_url
    if args.max_steps is not None:
        config["experiment"]["max_steps"] = args.max_steps
    try:
        summary = run_experiment(config)
    except (ValueError, RuntimeError) as exc:
        parser.exit(1, f"实验失败: {exc}\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
