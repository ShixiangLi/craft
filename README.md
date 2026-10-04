# Crafter 长程任务实验

第一版实现 naive 智能体与本地 Ollama 的实验闭环。每步根据最终目标、当前
局部状态和上一步反馈调用一次模型，返回一个动作；不包含任务分解、规划器
或长期记忆。ReAct、Spring 和批量对比入口仍是占位，不能运行。

## 启动

在项目根目录使用已配置的虚拟环境：

```bash
# 查看本地模型；如服务尚未启动，可在另一个终端执行 ollama serve
ollama list

# 先验证 3 步，再运行配置中的完整预算
.venv/bin/python -m scripts.run_experiment --config configs/naive.yaml --max-steps 3
.venv/bin/python -m scripts.run_experiment --config configs/naive.yaml

# 临时覆盖模型和服务地址
.venv/bin/python -m scripts.run_experiment --config configs/naive.yaml \
  --model qwen3:8b --base-url http://localhost:11434
```

默认使用已在当前本地模型列表中确认存在的 `qwen3:8b`，不会自动下载模型或
启动服务。新建环境时可用 `python -m pip install -r requirements.txt` 安装
本次验证的依赖版本；当前 Python 版本为 3.10。

客户端使用 [Ollama 原生 Chat API](https://docs.ollama.com/api/chat)，关闭流式
返回，并通过 JSON schema 约束动作名。请求直接连接配置地址，不使用 shell
中的 HTTP 代理。`model.params` 传给 Ollama 的 `options`，随机种子由回合 seed
设置。若换用其他模型，可调整 `think` 和生成参数；设 `think: null` 则使用
服务默认行为。非法动作、服务错误和超时会终止实验，保留错误及已有轨迹，
不会自动重试或用随机动作替代。

## 配置

`configs/naive.yaml` 包含：

- `agent`：智能体名、系统与每步提示词文件。
- `environment`：Crafter 参数和观测方式；`seed` 与 `length` 由实验统一管理。
- `task`：自然语言目标与基于环境成就计数的成功条件。
- `model`：Ollama 地址、模型名、请求超时、生成参数。
- `experiment`：种子列表、每个种子的回合数、每回合步数和模型调用上限。
- `output_dir`：结果根目录；每次运行创建独立子目录。

默认任务是收集木头，预算为 100 步、100 次模型调用，目标达成即结束。
`success_condition` 支持 `achievement` 和 `count`；设为 `null` 时不按目标提前
停止，结果中的 `success` 为 `null`，不能据此报告任务成功率。

提示词和输出路径相对项目根目录解析；`--config` 相对当前工作目录解析。
同一 seed 的重复回合会重建相同初始世界，并重置智能体和模型调用计数。
这不保证不同硬件或 Ollama 版本产生完全相同的模型输出。

## 观测与评估边界

本版采用 `local_semantic`：从 Crafter 状态截取与 LocalView 相同范围的局部
地图（默认宽 9、高 7），再提供背包、朝向、睡眠状态和已完成成就。
不提供全图或绝对坐标；但语义信息绕过视觉识别和夜间遮挡，因此属于
**结构化状态辅助观测**，不能直接当作纯像素基线进行对比。

地图按从上到下的行、从左到右的列排列，玩家在中心。每次模型决策对应一次
原生环境动作。原始图像暂不保存。当前适配器依赖 Crafter 1.8.3 的部分私有
字段，访问集中在 `engine/environment.py`，升级 Crafter 时需要重新验证。

任务成功依据环境成就计数判定。记录总奖励、成就数、步数、终止原因、模型
调用次数、输入/输出 token 和耗时；汇总成功率及基础均值，不计算官方 Crafter
score。到达步数或调用上限按截断记录，死亡按环境终止记录。

## 目录

```text
agents/naive.py           naive 决策；base.py 保留统一交互接口
modules/common/llm.py    最小 Ollama HTTP 客户端
configs/naive.yaml       可运行的第一版配置
prompts/naive/           system.txt 与 step.txt（使用 $变量 占位符）
scripts/run_experiment.py  单次配置启动入口，支持多个 seed
utils/                  配置校验、提示词读取、JSON 读写等
engine/                 环境适配、实验组装、运行、记录与汇总
outputs/                每次实验的产物（不纳入版本控制）
tests/                  流程与停止条件测试
```

公用记忆、可视化、ReAct、Spring、批量对比等预留文件保持占位，不参与当前
执行流程。`configs/react.yaml`、`spring.yaml` 和 `comparison.yaml` 是草案。

## 输出

```text
outputs/naive/<UTC时间戳_唯一编号>/
  config.yaml           本次有效配置（含命令行覆盖）
  metadata.json         依赖版本和实际提示词内容
  episode_0000_seed_0_repeat_0/
    trajectory.jsonl    初始状态、每步动作/反馈、请求提示词、模型原始响应
    result.json         单回合指标
  summary.json          完成后的整体汇总和各回合结果
  error.json            仅异常时产生，记录错误及已完成回合
```

每步立即写入轨迹。没有 `summary.json` 的失败运行不会作为完整实验汇总。
轨迹包含完整提示词与模型响应，后续可用于分析主线保持和子任务行为。

## 检查

```bash
.venv/bin/python -m unittest discover -s tests -v
```

自动化测试使用真实 Crafter 和模拟的 Ollama HTTP 响应，覆盖回合重置、预算
上限、成功/死亡停止、非法动作、连接失败及轨迹落盘。真实模型调用用上面的
短程启动命令单独验证。

## naive 提示词依据

提示词参考了 BALROG 的 [Crafter 动作说明](https://github.com/balrog-ai/BALROG/blob/b7afe79e3e4265811cfa985ed7c95c4d1a11e3f5/balrog/environments/crafter/__init__.py)、
[文本观测组织](https://github.com/balrog-ai/BALROG/blob/b7afe79e3e4265811cfa985ed7c95c4d1a11e3f5/balrog/environments/crafter/env.py)
和 [naive 单动作输出](https://github.com/balrog-ai/BALROG/blob/b7afe79e3e4265811cfa985ed7c95c4d1a11e3f5/balrog/agents/naive.py)的设计。
动作规则和资源数量按本地 Crafter 1.8.3 的 `Player` 实现及 `constants` 核对，
提示词为适配本项目重新编写，并非 BALROG 基线的严格复现。

本项目保留配置指定的最终目标、单步 JSON 输出和上一步反馈，不引入 BALROG
的多步历史窗口，也不把目标改成完成全部成就。局部地图仍保留所有可见格子，
通过 `modules/common/observation.py` 标记相对坐标和正前方目标，并分开显示生存
状态、当前物品和历史成就。零值物品与成就仅在提示词文本中省略，原始轨迹不变。
提示词更完整不等于已经证明成功率提升；正式比较应固定模型、目标、种子和预算。

开启 `model.think: true` 时，`num_predict` 必须容纳思考及最终动作输出。
当前配置使用 4096；128 的预算曾导致 qwen3:8b 只返回思考、最终内容为空。
若返回 `done_reason: length`，客户端会明确报告截断并保留原始响应和用量，
不从思考文本中猜测动作，也不自动重试。4096 并不保证所有请求都足够；
后续仍发生截断时应调整预算，或在不需要思考的实验中关闭 `think`。
