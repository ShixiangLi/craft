# Crafter 长程任务实验

项目实现 naive、ReAct、SPRING、ADaPT、ReAcTree 与初版 Harness，复用同一 Crafter 环境、局部语义
观测、统一模型客户端、预算检查和实验记录。各方法的论文来源、固定
代码版本、许可与复现边界见 [复现说明](docs/reproduction_sources.md)。

## 运行

在项目根目录使用已配置的 Python 3.10 虚拟环境。使用本地 Ollama 时，先确保
服务已运行且配置指定的模型已安装；代码不会自动启动服务或下载模型。
使用远程 API 时，直接在对应智能体的配置中填写地址、模型名和密钥。

```bash
ollama list

.venv/bin/python -m scripts.run_experiment --config configs/naive.yaml
.venv/bin/python -m scripts.run_experiment --config configs/react.yaml
.venv/bin/python -m scripts.run_experiment --config configs/spring.yaml
.venv/bin/python -m scripts.run_experiment --config configs/adapt.yaml
.venv/bin/python -m scripts.run_experiment --config configs/harness.yaml
.venv/bin/python -m scripts.run_experiment --config configs/reactree.yaml

# 短程验证；也可用 --model / --base-url / --provider 覆盖模型配置
.venv/bin/python -m scripts.run_experiment --config configs/react.yaml --max-steps 2
.venv/bin/python -m scripts.run_experiment --config configs/spring.yaml --max-steps 2
.venv/bin/python -m scripts.run_experiment --config configs/adapt.yaml --max-steps 2
```

新建 Python 环境时，可用 `python -m pip install -r requirements.txt` 安装已
验证的依赖。客户端支持 [Ollama /api/chat](https://docs.ollama.com/api/chat)、
[DeepSeek API](https://api-docs.deepseek.com/) 和兼容 OpenAI Chat Completions
的接口，直接连接配置地址，不使用 shell 的 HTTP 代理；每次调用均不自动重试。

## 模型接口配置

配置仍按智能体划分：`configs/naive.yaml`、`configs/react.yaml`、
`configs/spring.yaml`、`configs/adapt.yaml`、`configs/harness.yaml`、`configs/reactree.yaml`。修改选定文件中的 `model` 即可切换模型后端，运行命令
和智能体实现无需改变。默认配置连接本地 Ollama：

```yaml
model:
  provider: auto
  base_url: http://localhost:11434
  name: qwen3.8:latest
  api_key: ""
  api_key_env: null
  timeout: 120
  think: true
  params:
    temperature: 0
    num_predict: 32768
```

例如，在同一个智能体配置中改用 DeepSeek：

```yaml
model:
  provider: auto
  base_url: https://api.deepseek.com
  name: deepseek-flash
  api_key: "填写你的 API Key"
  api_key_env: null
  timeout: 120
  think: true
  params:
    temperature: 0
    num_predict: 32768
```

也可以令 `api_key: ""`、`api_key_env: DEEPSEEK_API_KEY`，从同名环境变量读取
密钥。实验目录中保存的配置会去除明文密钥，请求日志不保存鉴权头；填写了
明文密钥的原始配置文件仍包含密钥。示例模型名依据
[DeepSeek 官方模型更新](https://api-docs.deepseek.com/news/news260424/)；
实际使用时填写账号可用的模型名。

`provider: auto` 根据 URL 识别协议：默认端口 `11434` 或 `/api/chat` 路径
识别为 Ollama，官方 DeepSeek 域名识别为 DeepSeek，其他地址采用 OpenAI
兼容协议。Ollama 使用非标准端口时请显式设置 `provider: ollama`；DeepSeek
通过其他域名的代理访问时请设置 `provider: deepseek`，以保留其思考模式适配。
显式 `provider: openai` 可连接其他兼容服务，是否需要密钥由该服务决定。

模型参数由后端适配：

- Ollama 保持原生 `options` 和 `think`，回合 seed 传入其 `options.seed`。
- DeepSeek 和通用 OpenAI 兼容接口将 `num_predict` 映射为 `max_tokens`，
  忽略仅供 Ollama 使用的 `num_ctx`；也可直接配置 `max_tokens`。
- DeepSeek 将 `think` 转为 `thinking` 参数；通用 OpenAI 兼容接口不发送
  `think`，避免将 Ollama 或 DeepSeek 的扩展参数传给其他服务。
- 结构化动作在 Ollama 中使用原生 schema；DeepSeek 和通用 OpenAI 兼容
  接口使用 JSON object 模式，并在提示词中描述 schema，随后执行项目已有的
  本地解析校验。这不代表服务端强制执行该 schema，兼容服务需支持 JSON 模式。

## 策略

| 策略 | 每步模型调用 | 保留的策略状态 |
| --- | --- | --- |
| naive | 1 | 当前观测、最终目标、上一步反馈 |
| ReAct | 1 | 可选显式 thought、动作和真实观测组成的交互历史 |
| SPRING | 9 | 固定论文知识 C、最近两帧观测、本轮九问 DAG 答案 |
| ADaPT | 可变：执行动作通常 1 次，另有状态判定和失败后的规划 | 递归任务路径、AND/OR 计划、当前执行尝试的 ReAct 历史 |
| Harness | 1–5（默认最多 4 次文件工具选择，再选择游戏动作） | ReAct 最近历史、回合内持久计划与记忆文件 |
| ReAcTree | 可变：真实动作、思考、扩展、记忆查询与完成判断均计调用 | 动态子目标树、节点独立历史、共享可见事实和冻结经验库 |

ReAct 将显式思考与一个动作合并为一次结构化生成；默认保留完整回合历史。
`agent.params.max_history_steps` 为正整数时，只向模型提供最近若干个完整交互，
并明确标记截断；这属于上下文受限的变体。模型的 `num_ctx` 仍限制实际上下文，
完整历史会随回合增长，应按回合长度和显存配置容量，或显式选择历史窗口。
当前客户端没有模型 tokenizer，不保证长回合的完整历史都能被 Ollama 接收；
服务端可能截断超长上下文，长程正式实验需额外核对实际输入长度。

SPRING 每步按原始依赖图分别查询 q1–q8 和 qa；每问只获得直接父节点的问答。
只有 qa 选择的最终动作进入环境；不跨步复用节点答案。论文知识 C 使用官方
发布的离线提取资产，不另写人工规则取代；其中原有错误也保留并注明。完整
来源与差异见 [SPRING 资产说明](prompts/spring/SOURCES.txt)。

ADaPT 先用 ReAct 执行器直接尝试当前任务，只有执行器报告失败或用尽本次调用
预算时才分解。规划器生成子任务及 AND/OR 表达式：AND 按顺序执行并在失败时
停止；OR 按顺序尝试并在成功时停止；两者可以嵌套。子任务沿用同一规则递归。
完成子计划后直接返回组合结果，不自动重新执行父任务或重规划。

`agent.params.max_depth` 包含根层（根为 1），设为 1 时不会调用规划器；
`max_executor_calls` 限制一次执行尝试的模型调用数，包含状态判定调用；
`max_subtasks` 限制一次分解的子任务数。各尝试的历史独立，
`max_history_steps` 只裁剪当前尝试的真实交互，不删除父任务关系。
示例分别为 3 层、20 次、5 个子任务和 16 步历史。

子任务由模型返回 `task_completed` / `task_failed` 自判；这些信号不进入游戏，
根任务声明完成也不会替代环境成就评分。Crafter 中失败后的移动、消耗和伤害
继续保留，沿用官方 TextCraft 连续状态行为，不使用 ALFWorld/WebShop 的
重置与成功动作回放。提示词共享 ReAct 的游戏规则与原创示例，未硬编码钻石
子任务树。具体来源和适配见 [ADaPT 资产说明](prompts/adapt/SOURCES.txt)。

Harness 复用 ReAct 的历史、真实反馈和共享模型客户端，增加 `list_files`、
`read_file`、`write_file` 三个文本文件工具。每步先提供当前观测、最近历史和文件
目录，模型按需读取或更新文件，再选择一个原生 Crafter 动作；工具操作不推进
游戏，工具选择和动作选择均计入模型调用预算。默认 `max_tool_calls_per_step=4`，
文件工具选择及输出格式纠错共用这四次额度；达到上限后下一次调用只允许
选择游戏动作，不自动补一个 noop。最后一次仍输出错误时保留日志并报错。

每个 episode 在自己的 `workspace/` 下初始化 `plan.md` 和 `memory.md`，分别提供
计划进展／中断恢复和重要经历的空模板。任务分解、记忆内容、读写时机与计划
修订都由模型决定；没有硬编码钻石路径、自动规划器、地图或坐标记忆。文件
内容不自动注入下一轮输入，需要按需读取，具体协议见
[Harness 提示词](prompts/harness/system.txt)。文件会跨环境 step 保留，不跨回合共享，
也不训练模型或实现 latent dynamics。

Harness 的所有实验产物固定在 `outputs/harness/<run_id>/`，配置校验会拒绝其他
输出根目录。每回合除原有轨迹、模型日志、结果外，还有独立的 `workspace/`
与 `tool_calls.jsonl`；初始化模板、每次实际工具请求、结果／错误和写入前后
哈希均落盘；输出协议错误单独标为 `protocol_error`，不算文件工具执行成功。
工具不能读写工作区外的文件或日志，不提供 shell。默认单文件
16 KiB、工作区128 KiB、最多32个文件（含两个初始模板），均可通过 agent 参数
配置。`write_file` 完整替换 UTF-8 文件，使用原子替换；重写时需要模型自行
保留重要内容。工具返回错误后模型可在剩余预算内纠错，模型接口本身仍不重试。
输出使用四个互斥 JSON Schema 分支：游戏动作／列目录只允许 `action/thought`，
读文件另带 `path`，写文件另带 `path/content`。动作字段优先；混合输出不再
静默忽略，而是拒绝执行并返回格式反馈，模型需重新明确选择。只有正确的
`write_file` 才执行写入，不把移动动作自动转换为写入。
`max_write_chars=4096` 在生成 Schema 和本地解析中同时限制写入内容字符数，
实际落盘还需满足 UTF-8 字节限制；显式 thought 最多1024字符、路径最多240字符。
Ollama 原生 `format` 接收这些约束，OpenAI 兼容 JSON object 模式则通过提示词
和本地校验约束，不保证服务端强制限制字符串长度。协议依据
[Ollama 结构化输出接口](https://docs.ollama.com/capabilities/structured-outputs)。
Harness 配置的 `num_predict=8192` 限制单次思考与输出成本；约束文件内容不保证
内部思考必然终止，服务截断和 HTTP 超时仍按异常记录，不使用半成品动作。
Harness 默认 `model.timeout=600` 秒，给开启思考的长请求留出等待时间；仍可能
因服务异常或更长生成超时。超时保留已写文件、轨迹和失败调用，不自动重试
或续跑；重新启动会创建新的运行目录。

设置 `max_tool_calls_per_step=0` 可做工具关闭消融。比较效果时应同时报告实际
调用、token 与耗时；文件中的完成声明不替代环境成就。初版只验证工程行为，
尚无真实模型实验支持其性能收益。

当前四种配置保留本地 `qwen3.8:latest`、`think: true` 及已有预算参数。
对于支持此设置的服务，`think: false` 关闭的是模型内部思考，ReAct 的显式
thought、SPRING 的节点问答和 ADaPT 的规划仍正常执行。各方法知识、演示和调用成本不同，不应仅凭一次
运行将成绩差异归因于推理结构。短程连通性测试也不代表复现原论文分数。

## 配置与停止条件

每个 YAML 包含 `agent`、`environment`、`task`、`model`、`experiment` 和
`output_dir`。提示词及输出路径相对项目根目录解析；`--config` 相对工作目录。

- `task.description` 为最终目标；`success_condition` 使用环境成就名和计数。
  当前 ReAct / SPRING / ADaPT 示例目标为收集钻石，成功即结束。设为 `null` 可取消
  目标提前终止，此时 `success` 为 `null`，不能当作成功率。
- `experiment.max_steps` 是每回合环境步数上限；`max_model_calls` 是实际 HTTP
  模型调用次数上限。SPRING 不足九次余额时停止，不执行半成品决策。其示例
  500 步 / 4500 次调用刚好容纳全部九问决策。
- 角色死亡、目标完成、达到任一预算都会结束回合。相同 seed 的回合会重建
  相同初始世界，并重置智能体状态和调用计数；不同模型/硬件不保证输出一致。
- ADaPT 控制器返回时也会正常结束，`stop_reason` 为 `agent_completed` 或
  `agent_failed`。其中 `agent_completed` 只是模型和子计划的判断，`success`
  仍取决于环境真实成就；两者可以不一致。规划与判定调用计入全局模型预算，
  但不会增加环境步数。
- `model.params` 按所选服务适配，具体映射见上文。Ollama 的 `num_predict`
  限制思考与最终输出，`num_ctx` 控制上下文容量。开启模型思考可能显著增加
  token 和耗时。模型返回 `done_reason: length` 或 `finish_reason: length`
  会明确报截断，不从内部思考猜测动作，也不自动重试或更换动作。

## 目录和复用

```text
agents/
  base.py                环境交互接口
  llm_agent.py           共享生命周期、观测文本、模型调用和反馈
  naive.py / react.py / spring.py / adapt.py / harness.py
modules/
  common/llm.py          自由文本 / JSON 请求、计数、硬预算、逐调用记录
  common/model_config.py  接口识别、模型配置校验和配置密钥脱敏
  common/actions.py      统一动作 schema 和解析
  common/observation.py  统一局部观测描述
  react/components.py   ReAct 输出与轨迹上下文
  spring/components.py  SPRING 固定 DAG 和直接父问答构造
  adapt/components.py   ADaPT 输出协议和 AND/OR 计划解析
  harness/components.py 回合文件工作区、读写限制和工具输出协议
configs/                 每种策略的 YAML
prompts/                 公用规则和各策略提示词、示例、论文知识
scripts/run_experiment.py  统一启动入口
utils/                   配置、提示词、JSON 工具
engine/                  环境、运行、记录和基础评估
outputs/                 独立实验产物
tests/                  策略、预算及真实 Crafter 集成测试
```

公用记忆、`run_batch.py` 和 `comparison.yaml`
仍是预留内容，不参与当前运行；多个 seed 可直接在单个方法配置中指定。

## 观测与评估边界

`local_semantic` 截取 Crafter LocalView 范围（默认宽 9、高 7），提供背包、
朝向、睡眠状态和成就。不提供全图或绝对坐标，但绕过视觉识别和夜间遮挡，
属于结构化状态辅助观测。与纯像素基线、原论文的描述器不能直接混为一谈。
环境适配使用 Crafter 1.8.3 的部分私有字段，升级时需重新验证。

每个动作执行一次原生 `step`；模型推理期间游戏不推进。成功依据环境成就
判定；记录奖励、成就数、步数、停止原因、调用数、token 和耗时。当前汇总
不是官方 Crafter score；原始图像暂不保存。

## 输出与检查

```text
outputs/<method>/<UTC时间戳_唯一编号>/
  config.yaml          有效配置（含命令行覆盖，移除明文 API Key）
  metadata.json        依赖版本及实际提示词/知识/示例内容
  episode_0000_seed_0_repeat_0/
    trajectory.jsonl   初始状态、实际动作、反馈、策略诊断
    model_calls.jsonl  每次模型请求和响应、节点标签、状态、耗时
    result.json        单回合指标
    tool_calls.jsonl   仅 Harness：初始化和逐工具请求／实际反馈
    workspace/         仅 Harness：plan.md、memory.md 及模型创建的文件
  summary.json         所有回合完成后的汇总
  error.json           仅异常时生成，保留已完成回合和错误
```

每次模型调用与环境动作均独立落盘。SPRING 中间节点即使出错也可审计，日志
中 `q1`–`qa` 对应当前 DAG 节点；`trajectory.jsonl` 的 `decision.nodes` 保存
成功完成的节点答案。预算耗尽是正常结束，不冒充成功或错误回合。

```bash
.venv/bin/python -m unittest discover -s tests -v
```

自动测试使用真实 Crafter、受控场景和模拟模型接口响应验证策略与边界，不调用
模型。真实模型联调使用上面的短程命令，完整策略效果需要多 seed 实验。

## ReAcTree

ReAcTree 可由节点主动扩展子目标树，控制器执行 sequence、fallback 和 parallel。
决策首字段 `type` 先选择 Think／Act／Expand，再填写对应内容，具体游戏动作
只属于 Act；一次决策仍只请求一次模型，不强制根节点分解。默认
`max_history_steps: null` 保留节点完整历史，正整数才启用窗口；后端上下文
上限仍生效，与 ReAct 比较时需显式统一历史设置。
默认 parallel 按官方代码依次执行所有子节点并要求全部成功；`parallel_policy: majority`
选择论文描述的严格多数汇总。子计划完成后直接返回父节点，不额外重执行父任务。
默认 `planning_error_policy: node_failure` 将预期模型调用或解析错误交给控制流
作为当前节点失败，不隐式重试或伪造游戏动作；`abort` 可改为立即结束实验。
节点输入去重相邻观测，原始节点日志保留完整前后帧。

工作记忆默认 `working_memory_mode: spatial`，通过实际动作和相邻公开局部地图
保守定位已见地标；无法确定移动时开启独立坐标段，保留多地点、可见变化和
定位不确定性。动态物体仅记录最后观测；不读取隐藏坐标、全图或提供导航器。
查询最多返回 `max_recall_locations: 8` 个位置，同时报告总数和截断状态；
`last_seen` 保留初版标签最后观测语义，可做消融。

`configs/reactree.yaml` 使用当前 ReAct 的模型、规则、原生动作示例、目标和预算，
不修改现有基线。产物写入 `outputs/reactree`，每回合另存树结构、节点轨迹和工作记忆。
默认没有 Crafter 情景经验库，不加载额外模型；真实环境成功后只导出候选经验，
不在评估中自动跨回合学习。需要经验时先用独立训练 seeds 执行采木／木镐等
短目标，完成后将同一配置改回钻石和不交叠的评估 seeds，并通过
`episodic_memory_sources` 指定已完成训练 run／episode；同一启动命令会核验
末帧真实成就、步序和节点轨迹，在新 run 内生成冻结库与来源 manifest。
也可直接配置 `episodic_memory_path`，与 sources 互斥；非空库必须提供每条
`source.seed`，seed 不明或与评估重叠会在模型调用前拒绝。直接外部库的 success
标签不等于构建器已验证原始日志。输入库保持冻结，不自动扫描或改动旧结果。
当前没有已完成的 ReAcTree 训练经验可默认导入，空库运行仍不能称为完整复现
原论文经验设置。运行中的旧进程不热更新，也不会自动重启。
配置、可选依赖、产物格式及原论文/代码差异见 [ReAcTree 说明](docs/reactree.md)。

## 可视化

实验正常结束后自动生成 `visualizations/` 下每个回合的 PNG 和 SVG 图表，包含
生存状态、成就节点、资源数量、动作分布、模型 token 与调用耗时。多回合
运行还生成回合指标对比图。绘图失败不会使已完成的实验失效，原因记录在
`summary.json` 的 `visualization_error` 字段。

已有实验可离线补图，无需重新调用模型：

```bash
python -m scripts.analyze_results --run-dir outputs/react/<运行目录>
```

图表使用英文标签避免服务器缺少中文字库。它们是轨迹指标图，不是游戏
录像：当前没有保存原始 RGB 帧，因此不能仅凭旧日志生成真实画面录像。
