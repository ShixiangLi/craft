"""SPRING：论文知识 + 最近两帧观测 + 九问有向无环图。

每个环境动作执行完整 DAG，每个节点单独调用模型；不加入规则规划器、
长时记忆或节点跳过优化。模型、观测及输出协议差异见 prompts/spring/SOURCES.txt。
"""
from collections import deque
from string import Template
from typing import Any, Mapping

import yaml

from agents.llm_agent import LLMBaseAgent
from modules.common.actions import parse_action
from modules.common.llm import LLMClient, ModelCallBudgetExceeded
from modules.spring.components import QUESTION_DEPENDENCIES, compose_messages


class SpringAgent(LLMBaseAgent):
    def __init__(self, config: Mapping[str, Any], llm: LLMClient):
        super().__init__(config, llm)
        self.questions = yaml.safe_load(self.prompts["questions"])
        if (not isinstance(self.questions, dict)
                or set(self.questions) != set(QUESTION_DEPENDENCIES)
                or any(not isinstance(q, str) or not q.strip()
                       for q in self.questions.values())):
            raise ValueError("SPRING questions 必须包含 q1–q8、qa 九个非空问题")
        if not self.prompts["knowledge"].strip():
            raise ValueError("SPRING 必须配置非空的论文知识上下文")
        self.observations = deque(maxlen=2)

    def reset(self, task: Mapping[str, Any], *, seed: int) -> None:
        super().reset(task, seed=seed)
        self.observations.clear()

    def act(self, observation: Mapping[str, Any]) -> int:
        self.begin_decision()
        if (self.llm.max_calls is not None
                and self.llm.max_calls - self.llm.calls < len(QUESTION_DEPENDENCIES)):
            raise ModelCallBudgetExceeded("SPRING 剩余预算不足以完成九问决策")
        context = self.context(observation)
        # 官方 text_obs 自带产生当前帧的动作；不提供额外 reward 信号。
        previous_action = self.previous["action"] if self.previous is not None else "None (episode start)"
        self.observations.append(
            f"Player Observation Step {observation['step']}:\n"
            f"{context['observation']}\nLast player action: {previous_action}"
        )
        context["observations"] = "\n\n".join(self.observations)
        observations = Template(self.prompts["step"]).substitute(context)
        answers = {}
        self.last_decision["nodes"] = []
        for node, parents in QUESTION_DEPENDENCIES.items():
            messages = compose_messages(self.prompts["system"], self.prompts["knowledge"],
                                        observations, self.questions, answers, node)
            answer = self.query(messages, label=node,
                                actions=observation["actions"] if node == "qa" else None)
            answers[node] = answer
            self.last_decision["nodes"].append({
                "id": node, "question": self.questions[node],
                "parents": list(parents), "answer": answer,
            })
        return parse_action(answers["qa"], observation["actions"])
