# Think Less, Think Safely

**中文题目：少想一点，但安全退出：面向循环语言模型的风险校准动态停止机制**  
**English Title: Risk-Calibrated Early Halting for Looped Language Models**

## 项目简介 / Overview

SafeLoop-DAR 是一个面向循环语言模型的实验研究框架，用来研究一个核心问题：

> 对于当前 token，语言模型什么时候可以安全停止继续循环计算？

SafeLoop-DAR is an experimental research framework for studying early halting in
looped / recurrent-depth language models. It is designed around one central
question:

> When can a language model safely stop recurrent computation for the current
> token?

本项目不是简单地调节 Ouro 的 `early_exit_threshold`，而是把动态退出建模为一个
**深度状态是否收敛、继续计算是否仍有边际收益、当前 token 风险是否可接受** 的联合决策问题。

This project is not merely a sweep over Ouro's `early_exit_threshold`. It treats
dynamic halting as a joint decision over **depth-state convergence**, **marginal
value of additional computation**, and **token/stage-specific risk**.

## 核心信号 / Core Signals

- **风险校准动态退出 / Risk-calibrated halting**  
  只有当预测的提前退出错误风险低于对应风险组的校准阈值时，才允许退出。  
  Exit only when the predicted premature-exit risk is below a calibrated
  group-specific threshold.

- **深度注意力稳定性 / Depth attention stability**  
  衡量当前 token 对历史循环状态的深度聚合是否已经稳定。  
  Measure whether the model's depth-state aggregation has stabilized.

- **残差新颖性 / Residual novelty**  
  衡量下一轮循环是否仍然带来新的隐藏状态信息。  
  Measure whether the next loop step still contributes new information.

- **Token / 阶段风险控制 / Token-stage risk control**  
  对数学答案、代码标识符、JSON 值、工具参数和检索实体采用更保守的退出策略。  
  Make math answers, code identifiers, JSON values, tool arguments, and
  retrieved entities more conservative than ordinary text.

## 验证层级 / Validation Levels

1. **主验证：真实循环语言模型 / Primary looped LMs**  
   Ouro-1.4B / Ouro-2.6B adapters.

2. **辅助验证：非 Ouro 循环 Transformer / Secondary looped Transformer checks**  
   LoopFormer / looped early-exit adapters.

3. **机制验证：可控循环深度任务 / Controlled mechanism checks**  
   Deterministic mock adapter and synthetic LoopTiny-style tasks.

The repository intentionally separates trace collection, feature extraction,
risk prediction, calibration, and evaluation so each claim can be tested
independently.

本仓库有意将 trace 采集、特征提取、风险预测、校准和评估分开实现，便于分别验证每一项论文主张。

## 项目结构 / Project Layout

```text
SafeLoop-DAR/
├── configs/                  # 模型、benchmark、实验配置 / model, benchmark, experiment configs
├── docs/                     # 论文计划与实验协议 / paper plan and experiment protocol
├── experiments/              # 命令行实验入口 / command-line experiment entrypoints
├── safeloop/                 # 核心库 / library package
│   ├── adapters/             # Ouro / LoopFormer / LayerSkip / mock adapters
│   ├── controlled/           # 可控循环深度任务 / controlled recurrent-depth tasks
│   ├── evaluation/           # 指标与报告 / metrics and reports
│   ├── features/             # confidence、novelty、depth attention 特征 / feature extractors
│   ├── halting/              # 动态退出策略 / halting policies
│   ├── risk/                 # 风险标签、预测器、校准器 / labels, predictors, calibration
│   ├── tracing/              # teacher-forced 与 free-generation trace / trace builders
│   ├── utils/                # config、IO、seed 工具 / config, IO, seeding
│   └── workloads/            # benchmark 加载与合成任务 / benchmark loaders
└── tests/                    # 后续静态与单元测试 / static and unit tests
```

## 第一阶段目标 / First Milestone

第一阶段只建议使用 deterministic mock adapter 跑通最小闭环，先不要加载真实模型权重：

The first milestone should run only on the deterministic mock adapter. This
validates the experimental pipeline before loading real model weights:

```bash
python experiments/collect_traces.py --config configs/experiments/collect_traces_mock.json
python experiments/run_signal_prediction.py --trace runs/mock_traces/teacher_forced.jsonl
python experiments/run_frontier.py --trace runs/mock_traces/teacher_forced.jsonl
python experiments/run_calibration.py --trace runs/mock_traces/teacher_forced.jsonl
python experiments/run_residual_convergence.py --trace runs/mock_traces/teacher_forced.jsonl
python experiments/run_budget_scaling.py --trace runs/mock_traces/teacher_forced.jsonl
```

真实模型适配器采用懒加载设计：只有当实验显式请求真实模型时，才会加载 `torch`、`transformers`
和模型权重。

Real model adapters use lazy imports. They do not load `torch`, `transformers`,
or model weights unless an experiment explicitly asks for them.

