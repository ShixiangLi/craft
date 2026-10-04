"""SPRING 的固定九问 DAG；问题文字独立保存在 prompts/spring。

依赖关系来自官方 GPT_Actor_Chat.ipynb，见 prompts/spring/SOURCES.txt。
固定拓扑顺序消除官方 set 遍历的顺序不确定性，不改变节点和依赖。
"""
from typing import Mapping


QUESTION_DEPENDENCIES = {
    "q1": (),
    "q2": (),
    "q3": ("q1",),
    "q4": ("q2",),
    "q5": ("q1", "q3"),
    "q6": ("q5",),
    "q7": ("q6",),
    "q8": ("q7",),
    "qa": ("q2", "q4", "q7", "q8"),
}


def compose_messages(system: str, knowledge: str, observations: str,
                     questions: Mapping[str, str], answers: Mapping[str, str],
                     node: str) -> list[dict[str, str]]:
    """每问只接收本轮直接父节点问答，不继承所有先前节点或上轮答案。"""
    messages = [
        {"role": "system", "content": system},
        {"role": "system", "content": knowledge},
        {"role": "system", "content": observations},
    ]
    for parent in QUESTION_DEPENDENCIES[node]:
        messages.append({"role": "user", "content": questions[parent]})
        messages.append({"role": "assistant", "content": answers[parent]})
    question = questions[node]
    if node == "qa":
        # 只变更环境动作的序列化协议；八个分析节点仍输出自由文本。
        question += '\nReturn only {"action": "<one exact action name from the current observation>"}.'
    messages.append({"role": "user", "content": question})
    return messages
