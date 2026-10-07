# ReAct、SPRING、ADaPT 与 ReAcTree：来源、实现边界和对比口径

本项目在 naive 基线之外复现这些策略在 Crafter 中的决策机制，并接入统一模型客户端、观测和实验流程。模型、任务、提示词与评测设置均存在下述适配，因此运行成功不代表复现了原论文分数。

## 固定来源

| 方法 / 资源 | 原始来源 | 核对版本 |
| --- | --- | --- |
| ReAct 论文 | [Yao et al., ICLR 2023，第 2、4 节](https://arxiv.org/abs/2210.03629v3) | arXiv v3 |
| ReAct 官方代码 | [ysymyth/ReAct](https://github.com/ysymyth/ReAct/tree/6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9) | `6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9` |
| SPRING 论文 | [Wu et al., NeurIPS 2023，第 2 节及表 1](https://arxiv.org/abs/2305.15486v3) | arXiv v3 |
| SPRING 官方代码 | [GPT_Actor_Chat.ipynb](https://github.com/Holmeswww/SPRING/blob/c0369b9127ab9ec63797797a3952be9e119334e1/GPT_Actor_Chat.ipynb) | `c0369b9127ab9ec63797797a3952be9e119334e1` |
| SPRING 发布的知识与环境描述器 | [SmartPlay Crafter](https://github.com/microsoft/SmartPlay/tree/9dec5f2247da0dc88a47fe3daf3fb4fa02629b89/src/smartplay/crafter) | `9dec5f2247da0dc88a47fe3daf3fb4fa02629b89` |
| ADaPT 论文 | [Prasad et al., Findings of NAACL 2024，第 3 节、附录 A/B](https://arxiv.org/abs/2311.05772v2) | arXiv v2，camera-ready |
| ADaPT 官方代码 | [archiki/ADaPT](https://github.com/archiki/ADaPT/tree/ecdc4ab0030b4be9be122622d8ea78f8c59c44c4) | `ecdc4ab0030b4be9be122622d8ea78f8c59c44c4` |
| ReAcTree 论文 | [Choi et al., AAMAS 2026，第 4 节和附录 A](https://arxiv.org/html/2511.02424v2) | arXiv v2 |
| ReAcTree 官方代码 | [Choi-JaeWoo/ReAcTree](https://github.com/Choi-JaeWoo/ReAcTree/tree/88b737f1eb8f2b68d3b01bdb7ad46e4c00b3ff3b) | `88b737f1eb8f2b68d3b01bdb7ad46e4c00b3ff3b` |

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

## ADaPT

ADaPT 的核心是执行失败后按需分解，而不是先生成固定计划。它使用 ReAct 风格执行器、自生成的成功判断、LLM 规划器和递归控制器。原实验包含 ALFWorld、WebShop 和 TextCraft，没有 Crafter；方法与逻辑组合见 [论文第 3 节、算法 1 和附录 B](https://arxiv.org/html/2311.05772v2)。

对应实现为 [agents/adapt.py](../agents/adapt.py) 和 [modules/adapt/components.py](../modules/adapt/components.py)，与现有方法共享模型接口、ReAct 历史格式、游戏规则和示例。

- **执行优先**：每个任务先由执行器直接尝试；只有返回 `task_failed` 或耗尽本次执行调用预算时才请求规划器。执行器输出 `task_completed` 即认为该子任务完成，不调用额外验证模型，也不硬编码子任务成就判据。
- **递归与逻辑**：规划器输出自然语言子任务和 AND/OR 表达式。AND 按序执行且失败短路；OR 按序尝试且成功短路；支持括号组合。纯逻辑分组不额外调用执行器或增加分解深度，只有失败任务的新分解才增加深度。组合结果直接返回父任务，没有自动重规划、父任务重执行或整回合重试。
- **深度边界**：根任务深度为 1，`max_depth=1` 只执行任务。达到最大深度且执行失败时直接返回，不进行没有可执行子层的规划，遵循官方 [TextCraft 控制器](https://github.com/archiki/ADaPT/blob/ecdc4ab0030b4be9be122622d8ea78f8c59c44c4/run_textcraft.py#L394) 的 `succ or depth >= max_depth` 处理。
- **持续环境**：Crafter 失败执行的真实状态继续保留，沿用官方 TextCraft 的行为。官方 [ALFWorld](https://github.com/archiki/ADaPT/blob/ecdc4ab0030b4be9be122622d8ea78f8c59c44c4/run_alfworld.py#L476) 与 [WebShop](https://github.com/archiki/ADaPT/blob/ecdc4ab0030b4be9be122622d8ea78f8c59c44c4/run_webshop.py#L881) 则在递归入口重置环境并回放成功动作 checkpoint。本项目不实现这类回滚，也不会返还失败尝试消耗的生命、时间或资源。
- **上下文适配**：一次执行尝试有独立 ReAct 历史，可用 `max_history_steps` 限制。每次调用提供当前真实局部观测、最终目标、当前子任务及祖先路径；规划器还获得失败原因与本次历史。官方按环境传递最近成功动作、当前网页或库存；本项目这些字段是 Crafter 的状态传递适配。
- **预算适配**：`max_executor_calls` 按单次执行尝试的模型调用计数，动作和完成/失败声明都占用调用。`max_subtasks` 限制一次规划的子任务数。规划、执行和状态声明共享全局 `max_model_calls`。官方 executor 还带有连续无效动作的环境专属耐心阈值，本项目未加入该启发式。
- **协议适配**：执行器一次 JSON 生成显式 `thought` 和环境动作或终结信号；`task_completed`、`task_failed` 仅交给控制器，不作为 Crafter 动作。规划器 JSON 使用 1-based 子任务索引和 AND/OR 表达式，每个索引恰好出现一次；不使用原代码的文本 `Step` 标签解析。非法输出报错，不猜测替代动作。
- **评分隔离**：子任务的自判结果用于控制流程，最终 `success` 仍由现有环境成就条件判定。根控制器返回记录为 `agent_completed` 或 `agent_failed`，不会把自称完成直接记作实验目标成功。真实环境目标满足、死亡或全局预算耗尽可提前结束控制器。

ADaPT 的 Crafter 提示词为本项目原创，复用 [ReAct 游戏示例](../prompts/react/examples.txt)，没有复制原论文演示，也不预设钻石科技树。完整资产来源与差异见 [SOURCES.txt](../prompts/adapt/SOURCES.txt)。当前配置使用一个模型同时承担规划与执行，单次执行最多 20 次调用，最大深度 3，单计划最多 5 个子任务，历史窗口 16；这些参数不代表论文 Crafter 设置。

## 共同适配与实验解释

- **环境观测**：本项目统一使用局部 semantic 地图、朝向、背包和成就，不模拟夜间遮挡。SPRING 论文使用截图模板匹配描述器，夜间停止检测；后来的官方 SmartPlay 实现改用了 semantic 信息并描述每类最近对象。当前观测与论文、SmartPlay 的文字格式都不完全相同。
- **动作时间**：一次返回环境动作对应一次 Crafter `step`。模型查询期间游戏不会自动推进。ReAct 的思考、SPRING 的中间 QA、ADaPT 的规划和终结声明均不消耗环境步，但消耗真实模型调用与时间。
- **任务**：当前 YAML 使用指定目标和成功终止。SPRING 论文主要评估开放生存中的 22 项成就、reward 和 Crafter score。`success_condition: null` 可取消指定成就的提前终止，但这本身不足以重现论文评测协议。
- **模型**：可在各方法配置中选择本地 Ollama、DeepSeek 或 OpenAI 兼容接口。原论文模型设置不能用当前模型成绩代替。模型内部 `think` 开关与各策略显式组织的推理不同；对比时应统一该开关及采样、上下文和生成限制。
- **知识与演示**：ReAct 和 ADaPT 使用共同游戏规则与原创演示，SPRING 使用发布的论文知识 C。默认配置同时包含策略和先验知识差异，不能把差异全部归因于推理结构；若研究结构贡献，应另设控制知识的消融。
- **预算**：相同环境步数不意味着相同推理成本。ReAct 正常每步一次调用，SPRING 每步九次，ADaPT 还会有不执行环境动作的状态判断和规划调用；同时报告环境步、模型调用、输入/输出 token 和耗时。硬调用上限逐次执行，不把一次 `act` 计成一次调用。
- **成绩**：当前结果中的目标成功率、平均成就数和累计 reward 不是官方 Crafter score。后者跨回合汇总 22 项成就成功率。单回合、单种子或短程连通性测试不能支持论文分数复现或策略优劣结论。

日志中的 `trajectory.jsonl` 记录环境动作与结果，`model_calls.jsonl` 逐次保存模型请求、响应、调用标签和耗时；可据此检查多调用策略的预算与证据链。

## 来源许可

ReAct 与 SPRING 代码分别采用 [ReAct MIT](https://github.com/ysymyth/ReAct/blob/6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9/LICENSE)、[SPRING MIT](https://github.com/Holmeswww/SPRING/blob/c0369b9127ab9ec63797797a3952be9e119334e1/LICENSE)。SmartPlay 区分 [代码 MIT](https://github.com/microsoft/SmartPlay/blob/9dec5f2247da0dc88a47fe3daf3fb4fa02629b89/LICENSE-CODE) 和 [数据 CC BY 4.0](https://github.com/microsoft/SmartPlay/blob/9dec5f2247da0dc88a47fe3daf3fb4fa02629b89/LICENSE)。随项目保留的知识和提示词资产需保留其来源、许可和变更说明；论文正文仅作方法依据，本文件没有搬运长篇论文内容。

ADaPT 官方代码采用 [MIT 许可](https://github.com/archiki/ADaPT/blob/ecdc4ab0030b4be9be122622d8ea78f8c59c44c4/LICENSE)。本项目依照其控制机制重新实现 Crafter 适配，提示词和文档为原创，不复制原论文长段正文或原环境 demonstrations。

## ReAcTree

ReAcTree 原论文实验为 WAH-NL 和 ALFRED，没有 Crafter。项目独立实现动态
子目标节点与控制流，接入既有局部语义观测、模型客户端和评测。每个节点有
独立上下文，可主动选择动作、思考、扩展、回忆或完成／失败声明；子节点结果
直接返回父节点，不增加父节点自动重试。没有手写钻石任务树。

固定官方代码中 sequence/fallback 短路，parallel 串行执行所有子节点并要求
全部成功；论文描述多数投票。配置默认 `parallel_policy: all` 按代码复现，
`majority` 显式采用论文描述，严格多数平票失败。深度按 agent 与 control 都计数，
限制 control-node 深度；窗口、总节点数、子目标数与 JSON 契约是工程适配。
当前协议首字段为 `type: Think|Act|Expand`，先选择决策类别，再生成对应内容，
具体游戏动作只属于 Act；一次模型请求完成，不额外增加分类调用。官方使用
Guidance 逐段约束生成；本项目类别优先的 JSON 不是其候选评分的严格等价实现。
不强制根节点展开。默认 `max_history_steps: null` 保留应用侧完整节点历史，
正整数为显式窗口，实际容量仍受后端上下文限制。正式对照须统一历史设置。

默认 `planning_error_policy: node_failure` 将模型调用边界的预期 ValueError／
RuntimeError 和输出解析 ValueError 转成当前节点失败，再按控制流传播；
不重试该调用、不补伪造环境动作。`abort` 可立即上抛，预算耗尽、用户中断、
其他程序和 IO 错误仍全局传播。原始响应与错误保留。节点模型输入去重相邻
before／after 和当前观测，落盘及经验导出仍保留完整观测时序。

工作记忆默认 `spatial`，使用实际动作与相邻公开局部地图的静态地形配准，
区分成功移动和受阻；不确定时开启不合并的坐标段。同段保留多地点、换算
当前位置相对关系并按新可见地形更新旧地标，动态物体只记录最后观测。
查询默认最多 8 个位置并报告总数和截断；`last_seen` 为初版语义消融。
没有读取隐藏环境坐标、全图或 oracle 子目标进度，也没有额外导航器。

情景记忆使用冻结 JSONL、目标句向量和余弦检索。官方从最终真实成功的任务
收集所有节点经验，包括失败和扩展节点；本项目真实成功回合仅导出候选，
不在评估中更新检索库。可在同一配置用独立训练 seeds 收集短目标，再改回
钻石及独立评估 seeds，通过 `episodic_memory_sources` 选择已完成训练
run／episode，统一启动时核验日志末帧真实成就、步序及节点与环境对应关系，
将冻结库、来源目标、seed、文件哈希和排除记录保存在新 run。验证依赖日志
一致性，不是重放模拟器。训练根目标可不同于评估目标；ReAct 来源只作为
明确标记的完整根轨迹，不推断子目标。

直接 `episodic_memory_path` 与 sources 互斥，非空库必须有每条 `source.seed`；
启动前拒绝未知 seed 或与评估重叠的经验，但外部库的 success 标记不等于
构建器独立核验了原始日志。超出字符预算的完整示例会跳过，不自动总结。
项目当前没有可默认导入的完成 ReAcTree 训练库，空库运行属于没有 episodic
示例的机制适配，不能称为复现了原论文完整经验设置或真实长程成绩。

原代码和论文还在同分经验选择、Recall 步数计量方面存在差异；完整对应、
许可边界、参数与产物见 [ReAcTree 适配说明](reactree.md) 和
[来源清单](../prompts/reactree/SOURCES.txt)。统一结果不是官方 Crafter Score，
行为测试和模拟 HTTP 的成功不支持真实模型长程效果结论。
