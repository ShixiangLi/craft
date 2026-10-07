"""统一回合闭环；每步决策可包含多次模型调用。"""
import time
from agents.base import AgentFinished
from agents.llm_agent import LLMBaseAgent
from modules.common.llm import ModelCallBudgetExceeded
from engine.evaluator import task_succeeded
from engine.recorder import ExperimentRecorder


def run_episode(agent: LLMBaseAgent, environment, config: dict, *, seed: int,
                recorder: ExperimentRecorder) -> dict:
    start = time.monotonic()
    limits = config["experiment"]
    agent.reset(config["task"], seed=seed)
    agent.bind_episode(recorder.output_dir)
    agent.llm.max_calls = limits["max_model_calls"]
    agent.llm.on_call = recorder.record_model_call
    observation = environment.reset(seed=seed)
    recorder.record_step({"type": "reset", "seed": seed, "observation": observation})
    total_reward = 0.0
    steps = 0
    reason = "max_steps"
    while steps < limits["max_steps"]:
        if task_succeeded(observation, config["task"]):
            reason = "success"
            break
        if agent.llm.calls >= limits["max_model_calls"]:
            reason = "max_model_calls"
            break
        try:
            action = agent.act(observation)
        except AgentFinished as finished:
            reason = "agent_completed" if finished.completed else "agent_failed"
            recorder.record_step({"type": "agent_stopped", "step": steps,
                                  "stop_reason": reason, "decision": agent.last_decision})
            break
        except ModelCallBudgetExceeded:
            reason = "max_model_calls"
            recorder.record_step({"type": "budget_exhausted", "step": steps,
                                  "stop_reason": reason, "decision": agent.last_decision})
            break
        except (Exception, KeyboardInterrupt) as exc:
            recorder.record_step({"type": "error", "step": steps,
                                  "error": f"{type(exc).__name__}: {exc}",
                                  "decision": agent.last_decision,
                                  "model_response": agent.llm.last_response})
            raise
        transition = environment.step(action)
        next_observation = transition["observation"]
        steps += 1
        total_reward += transition["reward"]
        new_achievements = [name for name, count in next_observation["achievements"].items()
                            if count > 0 and observation["achievements"].get(name, 0) == 0]
        transition.update(action=observation["actions"][action], new_achievements=new_achievements)
        success = task_succeeded(next_observation, config["task"])
        if success:
            reason = "success"
        elif transition["terminated"]:
            reason = "terminated"
        elif steps >= limits["max_steps"]:
            reason = "max_steps"
        elif agent.llm.calls >= limits["max_model_calls"]:
            reason = "max_model_calls"
        else:
            reason = "running"
        transition["truncated"] = reason in ("max_steps", "max_model_calls") and not transition["terminated"]
        agent.observe(transition)
        recorder.record_step({"type": "step", "step": steps,
                              "action_id": action, **transition,
                              "stop_reason": reason,
                              "decision": agent.last_decision,
                              "model_response": agent.llm.last_response})
        observation = next_observation
        if steps == 1 or steps % 10 == 0 or reason != "running":
            print(f"  seed={seed} step={steps} action={transition['action']} reward={total_reward:.2f} state={reason}", flush=True)
        if reason != "running":
            break
    result = {
        "seed": seed, "steps": steps, "stop_reason": reason,
        "success": task_succeeded(observation, config["task"]),
        "total_reward": total_reward,
        "achievement_count": sum(count > 0 for count in observation["achievements"].values()),
        "achievements": observation["achievements"], "inventory": observation["inventory"],
        "model_calls": agent.llm.calls, "input_tokens": agent.llm.input_tokens,
        "output_tokens": agent.llm.output_tokens, "seconds": time.monotonic() - start,
    }
    recorder.save_result(result)
    agent.finish_episode(result)
    return result
