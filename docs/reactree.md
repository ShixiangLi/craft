# ReAcTree 的 Crafter 适配

本项目复现 ReAcTree 的动态子目标树、控制流和节点间记忆机制，接入已有
Crafter 环境、共享模型客户端和实验记录。原论文实验为 WAH-NL 与 ALFRED，
没有 Crafter；这里是核心机制的环境适配，不是原论文任务、模型或成绩复现。
工程测试通过、短程连通和长程性能提升是三个不同结论。

## 固定来源

| 资源 | 固定版本与用途 |
| --- | --- |
| [ReAcTree 原论文](https://arxiv.org/html/2511.02424v2) | arXiv `2511.02424v2`，重点核对第 4 节和附录算法；官方仓库标为 AAMAS 2026 |
| [官方代码](https://github.com/Choi-JaeWoo/ReAcTree/tree/88b737f1eb8f2b68d3b01bdb7ad46e4c00b3ff3b) | commit `88b737f1eb8f2b68d3b01bdb7ad46e4c00b3ff3b`，控制器、节点生命周期和记忆作为机制依据 |
| [官方控制流实现](https://github.com/Choi-JaeWoo/ReAcTree/blob/88b737f1eb8f2b68d3b01bdb7ad46e4c00b3ff3b/src/reactree.py) | 核对 sequence、fallback 和 parallel；论文与代码的差异见下文 |
| [WAH 节点实现](https://github.com/Choi-JaeWoo/ReAcTree/blob/88b737f1eb8f2b68d3b01bdb7ad46e4c00b3ff3b/src/wah/wah_reactree.py) | 核对节点上下文和 expand 后直接返回子计划结果 |
| [WAH 经验预处理](https://github.com/Choi-JaeWoo/ReAcTree/blob/88b737f1eb8f2b68d3b01bdb7ad46e4c00b3ff3b/src/wah/wah_embedder.py) | 核对最终任务真实成功筛选与节点级经验导出 |

本项目按机制重新实现 Crafter 接口，提示词为本项目原创，不复制家居环境
演示或把钻石科技树写成运行时计划。游戏知识复用
[共同规则](../prompts/common/crafter_rules.txt)，新增策略说明只描述任务树与记忆协议。

## 执行机制

从一个最终目标节点开始。每个智能体节点围绕自己的自然语言子目标生成决策，
可以执行原生 Crafter 动作、继续思考、检索共享工作记忆、报告完成／失败，
或生成若干子目标并指定控制流。扩展增加一个控制节点和相应的子目标节点，
由控制器执行子节点并向上传递结果；不搜索可回滚的动作树。

| 控制类型 | 执行与返回 |
| --- | --- |
| `sequence` | 按顺序执行，任一子节点失败则立即失败；全部成功才成功 |
| `fallback` | 按顺序尝试，第一个成功即成功；全部失败才失败 |
| `parallel` | 在同一持续环境中按顺序执行所有子节点，不因某个子节点的结果提前结束；执行完成后汇总 |

`parallel` 不表示并发操作 Crafter。固定官方代码使用全部成功的汇总规则，
论文第 4.1 节则描述 majority voting。本项目默认沿用官方代码的 `all`；
显式选择 `majority` 时采用严格多数，偶数平票视为失败。两种设置必须在结果中
区分，不能把论文和代码当作一致的算法版本。

扩展后的子计划结果直接作为该智能体节点的结果返回。节点不会在子计划结束后
自动恢复执行、核验父目标或重新规划；这是固定官方节点实现的生命周期。
所有真实移动、伤害、采集与资源消耗持续保留，失败分支不回滚环境。

各节点的局部交互历史相互独立，只在获得真实反馈后记录环境结果。历史窗口
只限制当前节点输入，不把兄弟节点完整轨迹自动拼到一起，也不自动总结。
固定官方实现没有历史窗口；本项目默认 `max_history_steps: null`，应用侧保留
节点完整历史。正整数窗口仅作为显式的上下文资源适配，不自动总结。实际可用
上下文仍受后端 `num_ctx` 等限制，完整历史不代表服务端不会截断超长输入。
模型输入按时间顺序呈现相邻前后观测，相同的 after／下一次 before 只展示一次，
当前观测也只在当前观测栏出现一次。此项仅去重输入：落盘节点轨迹仍保留完整
before／after，导出的经验保留完整观测时序，不以删除原始证据节省上下文。
思考、扩展、记忆查询和完成／失败声明不推进游戏，但每次实际模型请求均计入
统一调用预算。深度、节点数和决策额度是工程资源边界，不是新的规划启发式。

`planning_error_policy: node_failure` 将模型调用边界的预期
`ValueError`／`RuntimeError` 和输出解析 `ValueError` 作为当前节点失败，
交由 sequence／fallback／parallel 传播。例如 fallback 可以继续下一个已有
分支；这不会重试原调用、重新生成整棵树或补一个伪造环境动作。原始模型响应、
失败原因和 `planning_failed` 事件保留。`abort` 会立即上抛这类错误并结束实验；
调用预算、用户中断、其他程序错误和落盘 IO 错误仍全局传播。

决策先选择 Think／Act／Expand，再生成对应内容。统一 JSON 的首字段必须为
`type`；具体动作只在 Act 内选择，Think 与 Expand 不包含 `action`。一次模型
请求同时生成类别和内容，不额外增加分类调用。`thought` 为显式策略说明，
不是模型内部 thinking；Act／Expand 的说明可以为空。

```json
{"type":"Think","thought":"Check the unmet requirements before continuing."}
{"type":"Act","action":"move_right","thought":""}
{"type":"Expand","control_flow":"sequence","subgoals":["First natural-language subgoal","Second natural-language subgoal"],"thought":""}
{"type":"Act","action":"recall_observation","target":"tree","thought":""}
{"type":"Act","action":"done","thought":"The current subgoal appears completed."}
{"type":"Act","action":"failure","thought":"The current subgoal cannot be completed in this attempt."}
```

官方 `src/llm_agent.py` 使用 Guidance 先约束选择三类决策，再选择具体内容。
这里通过 Ollama JSON Schema／兼容接口 JSON mode 保留类别优先的结构，
不是重现 Guidance 的候选评分或完全相同的生成概率。新增类型层不保证模型
一定选择 Expand；根节点和子节点均可直接行动，不强制分解、不设停滞触发器。
旧 action-first 响应不再作为在线决策接受，类型混用和首字段不是 type 均明确
报协议错误。旧输出不改写，历史经验仍可导入；提示词明确其旧格式只作经历
参考，新的响应必须使用当前协议。
共享的两条原生动作示例只在注入时转换 Response 为 type-first 格式，观测、
动作和说明保持原内容，不复制或修改 ReAct 示例文件，不新增分解演示。

只有原生动作进入游戏。`think`、`expand`、`recall_observation`、`done` 和
`failure` 交给控制器；它们在 schema 和本地解析中分别约束，不将文件读写或
其他方法的完成信号混入协议。

## 记忆及观测边界

工作记忆在同一回合内共享，默认 `working_memory_mode: spatial`。它只使用
实际已执行的动作和相邻公开 `local_map` 的静态地形配准，区分成功移动和受阻；
不能仅根据“选择了向东移动”累加位置。位移无法唯一确定时开启新的坐标段，
旧段不自动合并，也不伪造跨段相对距离。

同一坐标段保留多个已见地标，查询时可以换算为当前玩家相对位置。重见某地点
并观察到静态标签变化时更新旧地标；动态物体只保留 last-seen，不能当成固定
位置。查询默认最多返回 `max_recall_locations: 8` 个位置，附总数、返回数和
`truncated`，不把截断后的列表说成全部已知地点。旧观测和定位不确定性都在
结果中标记，已定位但暂时不可见的地标也不保证仍存在。

`working_memory_mode: last_seen` 保留初版标签最后观测的语义，历史相对位置
只适用于当时玩家，不能当成当前位置，可作位置管理消融。两种模式都不读取
环境隐藏坐标、未见全图或 oracle 子任务进度，不提供导航器；回合重置清空记忆。

情景记忆以节点子目标及其完整局部轨迹为经验单元，使用 SentenceTransformer
生成目标向量，以余弦相似度选择输入示例。这是上下文检索，不更新决策 LLM
权重。首次没有人工演示库时，节点以空情景记忆运行；需要使用既有经验时显式
配置 `episodic_memory_path` 或 `episodic_memory_sources`，两者互斥。编码模型
是独立依赖，不等于切换基础决策模型。

只有真实环境最终目标成功的回合才能导出情景候选，导出其中节点的轨迹时
保留其各自终结状态；失败节点也可能属于最终成功回合，不能把候选库等同于
所有子任务都验证成功。默认候选只写入当前回合目录，不在同一实验后续回合
自动并入检索库，以避免种子执行顺序引入跨回合学习和评测泄漏。正式训练／
评测需明确经验库来源并隔离评测种子。非空输入库每条经验必须提供有效
`source.seed`，启动时拒绝来源 seed 不明或与任一评估 seed 重叠的经验。

固定官方实现对相似度相同的经验按 `expand → success → failure` 状态轮转，
论文则描述按状态均匀采样；这与 `parallel` 汇总规则一样，属于需要固定版本
并说明的差异，而不是自动认为论文与官方实现完全一致。

原官方经验检索使用模型 tokenizer 控制示例 token；本项目用 `max_examples`
和 `max_example_chars` 控制输入大小，属于显式字符预算适配，不声称精确复现
官方 token 截断。模型服务的实际上下文上限仍由配置和后端决定。

## 运行与配置

沿用统一入口，不增加独立启动器：

```bash
.venv/bin/python -m scripts.run_experiment --config configs/reactree.yaml
```

默认配置是一回合的机制适配，使用与当前 ReAct 相同的游戏规则和示例。
`episodic_memory_path: null` 且 `episodic_memory_sources: []` 表示空经验库，
不加载句向量模型，也不要求安装额外依赖；因此首次运行准确地说是
**没有成功经验库的 ReAcTree + 工作记忆**。
它没有原论文用于启动经验库的人工轨迹，不能称为完整复现原始经验设置。

使用显式经验库时，先安装可选依赖，再修改同一 YAML：

```bash
.venv/bin/python -m pip install -r modules/reactree/requirements.txt
```

```yaml
agent:
  name: reactree
  params:
    max_depth: 6
    max_subgoals: 5
    max_nodes: 128
    max_history_steps: null
    parallel_policy: all
    planning_error_policy: node_failure
    working_memory: true
    working_memory_mode: spatial
    max_recall_locations: 8
    episodic_memory: true
    episodic_memory_path: null
    episodic_memory_sources: []
    embedding_model: sentence-transformers/all-MiniLM-L6-v2
    embedding_device: cpu
    max_examples: 3
    max_example_chars: 20000
```

`episodic_memory_path` 设为已有 JSONL 候选库路径，或使用下述显式训练来源流程，
才会加载非空经验。句向量
模型首次加载可能需要下载权重，配置本地模型目录可用于离线环境；决策 LLM
仍通过共享客户端访问原服务，不要求本地可训练 LLM 权重。

| 参数 | 默认值与含义 |
| --- | --- |
| `max_depth` | `6`；沿用官方智能体节点和控制节点都计深度的方式，约束新控制节点深度，其直接子智能体可处于第 7 层；不是 ADaPT 的任务递归层数 |
| `max_subgoals` | `5`；一次 expand 可生成的子目标上限 |
| `max_nodes` | `128`；限制动态树的总节点数 |
| `max_history_steps` | `null`；默认保留完整节点历史，包括思考和记忆查询；正整数为显式滑动窗口 |
| `parallel_policy` | `all` 沿用固定官方代码；`majority` 为论文严格多数版本 |
| `planning_error_policy` | `node_failure` 将预期模型调用／解析错误作为节点失败交给控制流；`abort` 立即上抛 |
| `working_memory` | `true`；关闭可做共享观测记忆消融 |
| `working_memory_mode` | `spatial` 使用公开相邻局部观测配准；`last_seen` 保留标签最后观测语义 |
| `max_recall_locations` | `8`；spatial 查询返回位置数量上限，同时报告总数与截断状态 |
| `episodic_memory` | `true`；显式开关，与输入库路径共同决定是否检索 |
| `episodic_memory_path` | `null`；不提供先验成功轨迹、不跨回合在线学习 |
| `episodic_memory_sources` | `[]`；显式选择已完成训练 run／episode 目录，启动时校验并构建冻结库；与 path 互斥 |
| `embedding_model` / `embedding_device` | `sentence-transformers/all-MiniLM-L6-v2` / `cpu`；仅非空经验库加载 |
| `max_examples` / `max_example_chars` | `3` / `20000`；节点输入示例数量和总字符预算 |

每回合模型调用预算仍由 `experiment.max_model_calls` 硬性控制，环境步由
`experiment.max_steps` 控制。模型名称、服务地址和 think 配置以当前 YAML
为准；修改 `model` 即可切换项目支持的接口，不需复制配置或改变执行脚本。

### 从独立训练日志构建经验库

这里的“训练”是收集成功轨迹和构建检索库，不是更新 LLM 权重。推荐先用
独立训练 seeds 执行短目标，获得可放入预算的节点级成功经验，再评估钻石
任务。当前没有已经完成、可自动导入的 ReAcTree 训练库；默认不扫描旧输出。

1. 修改同一个 `configs/reactree.yaml`，保持两个经验输入项为空，选择与评估
   不交叠的训练 seeds，并设置一个短目标。例如：

   ```yaml
   task:
     description: Make a wood pickaxe while staying alive.
     success_condition:
       achievement: make_wood_pickaxe
       count: 1
   experiment:
     seeds: [100, 101, 102]
     episodes_per_seed: 1
     max_steps: 150
     max_model_calls: 1000
   ```

   使用相同统一入口采集，保留该次输出 run 路径。只有实际成功的回合可成为
   经验；没有成功则不能把自报完成的节点充作已验证的训练数据。

2. 仍修改同一个 YAML：将目标改回 `collect_diamond`，选择不交叠的评估 seeds，
   恢复计划采用的评估预算，设置 `episodic_memory_path: null`，并把已完成训练
   run／episode 路径填入 `episodic_memory_sources`，例如：

   ```yaml
   agent:
     params:
       episodic_memory_path: null
       episodic_memory_sources:
       - outputs/reactree/<已完成训练run>
   task:
     description: Collect at least one unit of diamond while staying alive.
     success_condition:
       achievement: collect_diamond
       count: 1
   experiment:
     seeds: [0, 2, 3]
     episodes_per_seed: 1
     max_steps: 500
     max_model_calls: 4500
   ```

   再运行同一条命令即可；无需复制配置文件或手动执行经验转换脚本。以上
   seeds 与预算仅为流程示例，不是已执行实验或性能结论。

统一引擎在第一个模型调用前执行 `modules/reactree/corpus.py` 的
`build_frozen_corpus`。来源可以是已完成整个 run，或明确选择的已完成 episode：
run 必须有与全部配置回合相符的 `summary.json`，其中未成功回合明确排除；
单个 episode 不要求其所属 run 已全部结束，但该 episode 必须有完整成功结果。

构建器重新核对日志末帧真实成就、动作和步序、节点目标及环境反馈对应关系，
不只信任候选文件的 success 标签。它验证的是保存的日志一致性，不重新运行
模拟器。训练根目标可以是采木或木镐，不必等同评估的钻石目标；来源根目标
和成就条件保留在 manifest，训练／评估 seeds 必须隔离。

为防止间接种子泄漏，构建器目前只接受采集时未使用已有情景经验的 ReAcTree
训练来源：若当时启用 EM 且配置了非空 `episodic_memory_path` 或 sources，
即使来源自身 seed 独立也拒绝导入，因为其已有经验可能包含评估 seed。
采集时禁用 EM 不受此限制。上述先空库采集、再冻结评估的流程符合这一边界。

支持显式选择 ReAct 成功日志作为适配来源，但只导出完整根任务轨迹并标记
`format: react_root_trajectory`、`subgoals_inferred: false`，不会从动作猜测子目标。
这与原始 ReAcTree 节点库并不相同，应作为经验来源变量记录。现存成功钻石
ReAct 整例约 52 万字符，超过默认 20000 字符示例预算，会被完整跳过；它不是
可直接支持默认 bootstrap 的短节点经验，不能通过偷偷总结或截断使其适配。

新运行根目录保存 `episodic_memory.jsonl` 和 `episodic_memory.manifest.json`。
有效 `config.yaml` 改为该冻结库的绝对路径并清空 sources，原始来源、训练目标、
训练／评估 seeds、输入文件哈希和被排除回合保留在 manifest／metadata。
源日志不会修改，评估期间也不回灌新经验。

直接提供 `episodic_memory_path` 时仍校验每条 `source.seed` 和 seed 隔离，
但只检查所声明的来源 seed，不独立验证完整经验谱系；外部库自带
`episode_success: true` 也不等于构建器已核验原始环境日志。
报告时应区分这两种来源验证程度，并核对 `examples_retrieved` 实际检索到的
示例；库非空不意味着示例一定能放入字符预算。

## 产物与审计

标准输出保留有效配置、依赖与提示词信息、模型调用、真实环境轨迹、结果、
汇总和图表。ReAcTree 额外文件均在自己的回合目录内：

```text
outputs/reactree/<run_id>/
  config.yaml
  metadata.json
  episodic_memory.jsonl          # 配置显式训练 sources 时构建
  episodic_memory.manifest.json  # 来源验证、seed 隔离、文件哈希
  episode_<index>_seed_<seed>_repeat_<repeat>/
    trajectory.jsonl
    model_calls.jsonl
    result.json
    tree.json
    tree_events.jsonl
    node_traces.jsonl
    working_memory.json
    episodic_candidates.jsonl  # 仅真实环境最终目标成功时导出
  summary.json
  error.json                  # 仅异常时
```

`tree.json` 保存树结构和状态，并记录 `decision_protocol: type_first_v1`，
`tree_events.jsonl` 保存扩展与控制事件，
树快照还记录经验库条目数与实际冻结输入的 SHA-256，便于确认使用的是哪份库；
默认预期协议／接口错误记录为 `planning_failed` 和节点 failure；只有需要
全局传播的错误或 `planning_error_policy: abort` 记录 `node_error` 并上抛。
`node_traces.jsonl` 保存节点局部轨迹，`working_memory.json` 保存回合内已见
观测。`episodic_candidates.jsonl` 是可审查的经验候选，每行包含 `goal`、
`state`、`trajectory`、`episode_success: true` 和来源
`source: {seed, episode_id, node_id}`；节点终态可为 `success`、`failure`、
`expand` 或 `interrupted`。它不会自动回灌；必须主动配置为后续实验的输入库。

正在运行的旧进程不会热更新这套实现。修改配置和代码后启动的新 run 才使用
新行为，不改写旧输出，也不自动重启用户进程。

树节点状态来自模型自报或子计划聚合，真实环境轨迹独立记录，环境成就才支持最终任务成功。
分析树日志时需把两者一起查看，特别注意完成信号过早、分解漏掉条件、失败
分支消耗资源，以及祖先子目标与实际观测不一致等情形。

## 与现有策略的区别

| 策略 | 任务组织 | 历史和记忆 |
| --- | --- | --- |
| ReAct | 单条交互链；模型可在 thought 中讨论子任务，控制器不执行子任务树 | 一份回合历史，可配置窗口 |
| ADaPT | 先尝试执行，失败或执行预算耗尽后分解；递归 AND/OR 组合 | 每次执行尝试的独立历史与祖先路径 |
| Harness | 给单个智能体文件读写工具，计划内容和落实均由模型维护 | 回合内文件与最近交互历史 |
| ReAcTree | 节点可主动扩展动态子目标树；控制器执行 sequence/fallback/parallel | 节点独立历史、共享工作记忆和显式经验库检索 |

ReAcTree 的工作记忆和情景记忆均为显式软件状态，不是 MFC/HPC latent token，
也没有学习不同时间尺度的内部神经动力学。它不保证模型能正确分解任务、
遵守规则或判断子任务成功。

## 评估口径

模型自报完成只是控制信号。最终实验 `success` 始终由已有 Crafter 环境成就
条件独立判定；`agent_completed` 与真实 `success` 可以不一致。环境死亡、真实
目标完成和预算耗尽沿用统一停止规则。日志应同时保留节点目标、扩展结构、
控制返回、实际动作与环境反馈，以区分错误规划、错误成功判断和动作执行失败。

公平对照需控制决策模型及版本、think 设置、采样参数、观测、游戏知识、最终
目标、种子和预算。经验库带来的额外示例也是实验变量；有经验库的 ReAcTree
与零示例 ReAct 的差异不能全部解释为任务树收益。建议先比较空经验库，随后
分别关闭工作记忆、扩展和经验检索，并记录 `parallel` 汇总设置。
当前 ReAcTree 示例默认完整历史，ReAct 示例 YAML 的窗口为 16；正式对照应
显式统一历史设置，不能把上下文容量的差异都归因于任务树。

同时报告目标成功率、各成就、深层成就、死亡／生存、环境步数、实际调用、
输入输出 token 和耗时。相同环境步数并不代表相同成本，完成／失败判断和
扩展也消耗调用。当前统一 summary 不是官方 Crafter Score；需要完整的多种子
22 项成就汇总及相应协议，才能报告该指标。单种子或短程测试不能支持性能
优劣或原论文成绩复现结论。

当前实现验证侧重行为与集成测试；没有真实模型长程实验前，不声称 ReAcTree
的 Crafter 适配有效或优于现有基线。

## 许可边界

固定官方仓库根目录没有独立的 LICENSE、COPYING 或 NOTICE，不能将其中
VirtualHome／ALFRED 子目录的许可证直接视为 ReAcTree 整体许可。本项目独立
实现控制机制，不搬运官方源码或演示资产；论文正文只作方法依据。来源与
适配应与代码和提示词一起保留，不能无依据标记官方代码为 MIT。
