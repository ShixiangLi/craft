# ReAct 提示词与移植边界

方法依据：Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models*，
[原论文 v3，第 2、4 节](https://arxiv.org/html/2210.03629v3)，
[作者官方实现](https://github.com/ysymyth/ReAct/tree/6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9)。
核对的官方提交为 `6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9`。

- 保留显式语言思考、环境动作、真实观测的闭环，以及回合轨迹上下文。
- 按交互任务的稀疏思考设定，允许 `thought` 为空；没有新增反思学习或外部规划器。
- 本项目将一次可选思考与一个环境动作合并到一次模型调用，用 JSON 作为传输格式。
  官方 HotpotQA 同样合并 Thought/Action 生成；官方 ALFWorld 则允许单独的 `think:` 输出。
  这里不实现 ALFWorld 的独立思考动作协议，也不将 Ollama 的内部 `thinking` 字段当作 ReAct 轨迹。
- 默认保留完整轨迹。`agent.params.max_history_steps` 为正整数时，仅把最近若干个完整交互放进提示词，
  并明确标记截断；这是资源受限的配置选项，不是论文中的记忆压缩机制。
- 原论文未在 Crafter 上评测。这里沿用本项目的局部语义观测、任务终止条件和 Ollama 模型，
  因此复现的是算法机制，不能声称重现原论文分数。

`examples.txt` 中的两段 Crafter 示例为本项目手写，并非复制论文或官方轨迹。
树木采集、转向、工作台和石镐的材料规则核对自已安装的 Crafter 1.8.3 的
`crafter.objects.Player` 与 `crafter.constants`；`tests/test_react.py` 用真实环境的受控场景验证对应动作序列。
两个示例用于说明交互模式，不是自动规划代码，也不为智能体提供隐藏地图信息。

官方实现参考：
[alfworld.ipynb](https://github.com/ysymyth/ReAct/blob/6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9/alfworld.ipynb)、
[hotpotqa.ipynb](https://github.com/ysymyth/ReAct/blob/6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9/hotpotqa.ipynb)、
[ALFWorld few-shot 提示词](https://github.com/ysymyth/ReAct/blob/6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9/prompts/alfworld_3prompts.json)。
