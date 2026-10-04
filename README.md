# Crafter 长程任务实验

项目实现 naive、ReAct 与 SPRING 三种策略，复用同一 Crafter 环境、局部语义
观测、统一模型客户端、预算检查和实验记录。ReAct / SPRING 的论文来源、固定
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

# 短程验证；也可用 --model / --base-url / --provider 覆盖模型配置
.venv/bin/python -m scripts.run_experiment --config configs/react.yaml --max-steps 2
.venv/bin/python -m scripts.run_experiment --config configs/spring.yaml --max-steps 2
```

新建 Python 环境时，可用 `python -m pip install -r requirements.txt` 安装已
验证的依赖。客户端支持 [Ollama /api/chat](https://docs.ollama.com/api/chat)、
[DeepSeek API](https://api-docs.deepseek.com/) 和兼容 OpenAI Chat Completions
的接口，直接连接配置地址，不使用 shell 的 HTTP 代理；每次调用均不自动重试。

## 模型接口配置

配置仍按智能体划分：`configs/naive.yaml`、`configs/react.yaml`、
`configs/spring.yaml`。修改选定文件中的 `model` 即可切换模型后端，运行命令
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

## 三种策略

| 策略 | 每步模型调用 | 保留的策略状态 |
| --- | --- | --- |
| naive | 1 | 当前观测、最终目标、上一步反馈 |
| ReAct | 1 | 可选显式 thought、动作和真实观测组成的交互历史 |
| SPRING | 9 | 固定论文知识 C、最近两帧观测、本轮九问 DAG 答案 |

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

当前三种配置保留本地 `qwen3.8:latest`、`think: true` 及已有预算参数。
对于支持此设置的服务，`think: false` 关闭的是模型内部思考，ReAct 的显式
thought 和 SPRING 的节点问答仍正常执行。三者知识、演示和调用成本不同，不应仅凭一次
运行将成绩差异归因于推理结构。短程连通性测试也不代表复现原论文分数。

## 配置与停止条件

每个 YAML 包含 `agent`、`environment`、`task`、`model`、`experiment` 和
`output_dir`。提示词及输出路径相对项目根目录解析；`--config` 相对工作目录。

- `task.description` 为最终目标；`success_condition` 使用环境成就名和计数。
  当前 ReAct / SPRING 示例目标为收集钻石，成功即结束。设为 `null` 可取消
  目标提前终止，此时 `success` 为 `null`，不能当作成功率。
- `experiment.max_steps` 是每回合环境步数上限；`max_model_calls` 是实际 HTTP
  模型调用次数上限。SPRING 不足九次余额时停止，不执行半成品决策。其示例
  500 步 / 4500 次调用刚好容纳全部九问决策。
- 角色死亡、目标完成、达到任一预算都会结束回合。相同 seed 的回合会重建
  相同初始世界，并重置智能体状态和调用计数；不同模型/硬件不保证输出一致。
- `model.params` 按所选服务适配，具体映射见上文。Ollama 的 `num_predict`
  限制思考与最终输出，`num_ctx` 控制上下文容量。开启模型思考可能显著增加
  token 和耗时。模型返回 `done_reason: length` 或 `finish_reason: length`
  会明确报截断，不从内部思考猜测动作，也不自动重试或更换动作。

## 目录和复用

```text
agents/
  base.py                环境交互接口
  llm_agent.py           共享生命周期、观测文本、模型调用和反馈
  naive.py / react.py / spring.py
modules/
  common/llm.py          自由文本 / JSON 请求、计数、硬预算、逐调用记录
  common/model_config.py  接口识别、模型配置校验和配置密钥脱敏
  common/actions.py      统一动作 schema 和解析
  common/observation.py  统一局部观测描述
  react/components.py   ReAct 输出与轨迹上下文
  spring/components.py  SPRING 固定 DAG 和直接父问答构造
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
