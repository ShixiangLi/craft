"""基于环境成就判定目标；汇总基础指标，不计算官方 Crafter score。"""
from statistics import mean


def task_succeeded(observation: dict, task: dict) -> bool | None:
    condition = task.get("success_condition")
    if condition is None:
        return None
    return observation["achievements"].get(condition["achievement"], 0) >= condition.get("count", 1)


def aggregate_results(results: list[dict]) -> dict:
    if not results:
        raise ValueError("没有可汇总的回合")
    judged = [r["success"] for r in results if r["success"] is not None]
    return {
        "episodes": len(results),
        "success_rate": mean(judged) if judged else None,
        "mean_reward": mean(r["total_reward"] for r in results),
        "mean_steps": mean(r["steps"] for r in results),
        "mean_achievements": mean(r["achievement_count"] for r in results),
        "total_model_calls": sum(r["model_calls"] for r in results),
        "total_input_tokens": sum(r["input_tokens"] for r in results),
        "total_output_tokens": sum(r["output_tokens"] for r in results),
    }
