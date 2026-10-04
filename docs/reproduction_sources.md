# ReAct 与 SPRING：来源、实现边界和对比口径

本项目复现两种策略在 Crafter 中的决策机制，并接入现有 Ollama、观测和实验流程。模型、任务、提示词与评测设置均存在下述适配，因此运行成功不代表复现了原论文分数。

## 固定来源

| 方法 / 资源 | 原始来源 | 核对版本 |
| --- | --- | --- |
| ReAct 论文 | [Yao et al., ICLR 2023，第 2、4 节](https://arxiv.org/abs/2210.03629v3) | arXiv v3 |
| ReAct 官方代码 | [ysymyth/ReAct](https://github.com/ysymyth/ReAct/tree/6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9) | `6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9` |
| SPRING 论文 | [Wu et al., NeurIPS 2023，第 2 节及表 1](https://arxiv.org/abs/2305.15486v3) | arXiv v3 |
| SPRING 官方代码 | [GPT_Actor_Chat.ipynb](https://github.com/Holmeswww/SPRING/blob/c0369b9127ab9ec63797797a3952be9e119334e1/GPT_Actor_Chat.ipynb) | `c0369b9127ab9ec63797797a3952be9e119334e1` |
| SPRING 发布的知识与环境描述器 | [SmartPlay Crafter](https://github.com/microsoft/SmartPlay/tree/9dec5f2247da0dc88a47fe3daf3fb4fa02629b89/src/smartplay/crafter) | `9dec5f2247da0dc88a47fe3daf3fb4fa02629b89` |

## ReAct

原方法把显式语言思考、环境动作和真实环境观测交织到同一回合轨迹。交互任务允许在需要时思考，不要求每一步都有新的长篇推理。原论文评测 HotpotQA、FEVER、ALFWorld 和 WebShop，没有 Crafter 实验。[原论文](https://arxiv.org/abs/2210.03629v3)

本项目的对应实现为 [agents/react.py](../agents/react.py) 和 [modules/react/components.py](../modules/react/components.py)：

- 每次调用生成可选的显式 `thought` 与一个 `action`；只有动作进入环境。下一次调用看到已执行的交互轨迹与当前真实观测。
- `thought` 被保存在后续提示词和日志中；Ollama 返回的内部 `thinking` 不作为 ReAct 的显式轨迹。
- 默认完整历史；`agent.params.max_history_steps` 为正整数时保留最近若干个完整交互并标记截断。这是上下文资源选项，没有加入自动总结、反思学习或其他记忆算法。
- JSON 是本项目的传输协议。官方 [HotpotQA notebook](https://github.com/ysymyth/ReAct/blob/6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9/hotpotqa.ipynb) 同样可在一次生成中输出思考与动作；本项目没有复制 [ALFWorld notebook](https://github.com/ysymyth/ReAct/blob/6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9/alfworld.ipynb) 中可单独执行 `think:` 的协议。
- [Crafter few-shot 示例](../prompts/react/examples.txt) 为本项目原创，并以真实 Crafter 受控场景校验。它们展示交互格式与游戏规则，不是原论文演示数据，也不是运行时硬编码计划。

## SPRING

SPRING 包含离线知识准备和在线 QA-DAG 两个阶段，不是任意形式的分层规划器。[原论文第 2 节](https://arxiv.org/abs/2305.15486v3)

### 论文知识 C

原论文先按段处理 Crafter 的 LaTeX 源码：使用两个相关性问题筛选段落，只要任一问题判断相关就保留；随后分别提取有用信息、交互对象、目标要求和动作要求，再按问题汇总去重，形成固定知识 C。这些相关性判断属于离线阅读阶段，不是在线 DAG 节点的跳过条件。

发布的官方 notebook 从 SmartPlay `info['manual']` 直接读取已生成的 C，而不是每步重新阅读论文。本项目同样使用 [SmartPlay 发布的 crafter_ctxt.pkl](https://github.com/microsoft/SmartPlay/blob/9dec5f2247da0dc88a47fe3daf3fb4fa02629b89/src/smartplay/crafter/assets/crafter_ctxt.pkl) 对应的纯文本 [paper_context.txt](../prompts/spring/paper_context.txt)。SmartPlay 源码注明该资产由 `text-davinci-003` 按论文流程生成；本项目没有重新实现或重新执行 LaTeX 提取流程。本项目原样保存字符串，未沿用 SmartPlay 运行时删除 LaTeX 输出提示和替换四方向词的处理；API 动作命名映射在系统提示词中说明。

知识资产本身含有不准确的信息，例如提到蜘蛛、把吃牛表述成直接恢复生命值、把醒来描述成开始回合。保留这些内容是为了明确固定知识来源，不应将它当成当前 Crafter 规则的权威说明。若后续修正，必须另存版本并作为知识消融记录，不能与原始 C 的结果混称。

### 每步九问与依赖

问题和依赖来自官方 notebook 的 `question_dependencies`；原始英文问题存于 [questions.yaml](../prompts/spring/questions.yaml)。每步按拓扑顺序回答全部九问，每节点只附带**直接父节点**的当前步问答。

| 节点 | 问题内容概要 | 直接父节点 |
| --- | --- | --- |
| q1 | 当前对象、可提供资源、交互要求 | 无 |
| q2 | 上一次执行的动作 | 无 |
| q3 | 对象交互要求是否满足 | q1 |
| q4 | 上一步成功与否及原因 | q2 |
| q5 | 三个优先子任务与优先级 | q1、q3 |
| q6 | 最高优先任务的要求与第一步 | q5 |
| q7 | 五个候选动作、要求和优先级 | q6 |
| q8 | 候选动作要求是否满足 | q7 |
| qa | 最佳可执行动作 | q2、q4、q7、q8 |

每节点还接收固定知识 C，以及按旧到新排列的最近两帧观测。当前帧的上一动作是产生该帧的实际环境动作。第一步只有 reset 观测时不虚构第二帧；回合 reset 会清空窗口。每个游戏步重新回答九问，不把上一步节点答案当成本步缓存，也不使用持久任务树。

对应实现位于 [agents/spring.py](../agents/spring.py) 和 [modules/spring/components.py](../modules/spring/components.py)。一个正常环境动作需要九次模型调用，无法用一条包含九问的长提示词替代；论文专门比较过移除 DAG 依赖限制的变体。

官方最终动作以名称子串匹配，未匹配时使用 `Do`，并没有动作编号输出协议。本项目 q1–q8 仍生成自由文本，qa 生成 JSON 动作并校验合法名称，非法输出明确报错，取消默认 `Do`。这些适配记录在 [SPRING 来源说明](../prompts/spring/SOURCES.txt)。动作必须在整次决策完成后才进入环境，不能把中间节点中的建议直接执行；剩余调用预算少于九次时，在开始本步 QA 前结束。

## 共同适配与实验解释

- **环境观测**：本项目统一使用局部 semantic 地图、朝向、背包和成就，不模拟夜间遮挡。SPRING 论文使用截图模板匹配描述器，夜间停止检测；后来的官方 SmartPlay 实现改用了 semantic 信息并描述每类最近对象。当前观测与论文、SmartPlay 的文字格式都不完全相同。
- **动作时间**：一次返回动作对应一次 Crafter `step`。模型查询期间游戏不会自动推进。ReAct 的思考或 SPRING 的中间 QA 均不消耗环境步，但消耗真实模型调用与时间。
- **任务**：当前 YAML 使用指定目标和成功终止。SPRING 论文主要评估开放生存中的 22 项成就、reward 和 Crafter score。`success_condition: null` 可取消指定成就的提前终止，但这本身不足以重现论文评测协议。
- **模型**：使用本地 Ollama 模型。原论文的 GPT-4、PaLM 或 GPT-3 设置不能用当前模型成绩代替。模型内部 `think` 开关与两种策略显式组织的推理不同；对比时应统一该开关及采样、上下文和生成限制。
- **知识与演示**：ReAct 使用共同游戏规则与原创演示，SPRING 使用发布的论文知识 C。默认配置同时包含策略和先验知识差异，不能把差异全部归因于推理结构；若研究结构贡献，应另设控制知识的消融。
- **预算**：相同环境步数不意味着相同推理成本。ReAct 正常每步一次调用，SPRING 每步九次；同时报告环境步、模型调用、输入/输出 token 和耗时。硬调用上限逐次执行，不把一次 `act` 计成一次调用。
- **成绩**：当前结果中的目标成功率、平均成就数和累计 reward 不是官方 Crafter score。后者跨回合汇总 22 项成就成功率。单回合、单种子或短程连通性测试不能支持论文分数复现或策略优劣结论。

日志中的 `trajectory.jsonl` 记录环境动作与结果，`model_calls.jsonl` 逐次保存模型请求、响应、调用标签和耗时；可据此检查多调用策略的预算与证据链。

## 来源许可

ReAct 与 SPRING 代码分别采用 [ReAct MIT](https://github.com/ysymyth/ReAct/blob/6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9/LICENSE)、[SPRING MIT](https://github.com/Holmeswww/SPRING/blob/c0369b9127ab9ec63797797a3952be9e119334e1/LICENSE)。SmartPlay 区分 [代码 MIT](https://github.com/microsoft/SmartPlay/blob/9dec5f2247da0dc88a47fe3daf3fb4fa02629b89/LICENSE-CODE) 和 [数据 CC BY 4.0](https://github.com/microsoft/SmartPlay/blob/9dec5f2247da0dc88a47fe3daf3fb4fa02629b89/LICENSE)。随项目保留的知识和提示词资产需保留其来源、许可和变更说明；论文正文仅作方法依据，本文件没有搬运长篇论文内容。
