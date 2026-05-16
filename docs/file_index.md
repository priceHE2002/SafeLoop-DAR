# 文件索引 / File Index

本文档解释 SafeLoop-DAR 中主要文件的作用，便于复试讲解、论文复现和后续工程扩展。

This document explains the purpose of major files in SafeLoop-DAR for interview
discussion, paper reproducibility, and future engineering extensions.

## 配置文件 / Configuration Files

- `configs/models/*.json`
  模型适配器配置，包括 Ouro、LoopFormer、LayerSkip 和 mock / LoopTiny。
  Model adapter configs for Ouro, LoopFormer, LayerSkip, and mock / LoopTiny.

- `configs/benchmarks/*.json`
  Benchmark 样本配置，用于构造 prompt、任务类型、生成阶段和期望输出。
  Benchmark sample configs for prompts, task types, generation stages, and
  expected outputs.

- `configs/experiments/*.json`
  实验入口配置，用于指定模型、benchmark、输出目录和 trace 模式。
  Experiment entry configs specifying model, benchmark, output directory, and
  trace mode.

## 核心库 / Core Library

- `safeloop/adapters/`
  模型适配层。真实模型采用懒加载，mock adapter 用于不依赖权重的管线验证。
  Model adapters. Real models use lazy loading; the mock adapter validates the
  pipeline without model weights.

- `safeloop/tracing/`
  Trace 采集逻辑，支持 teacher-forced 和 free-generation 两种模式。
  Trace collection logic for teacher-forced and free-generation modes.

- `safeloop/features/`
  特征提取模块，包括 confidence、depth attention stability、residual novelty 和 token/stage 特征。
  Feature extraction modules for confidence, depth attention stability, residual
  novelty, and token/stage features.

- `safeloop/risk/`
  风险标签、风险预测器和 group-wise calibration。
  Risk labels, risk predictors, and group-wise calibration.

- `safeloop/halting/`
  动态退出策略，包括 fixed depth、score threshold 和 group calibrated policy。
  Halting policies, including fixed-depth, score-threshold, and group-calibrated
  policies.

- `safeloop/evaluation/`
  指标计算与 risk-compute frontier 汇总。
  Metric computation and risk-compute frontier summarization.

## 实验入口 / Experiment Entrypoints

- `experiments/collect_traces.py`
  收集 loop-depth trace。
  Collect loop-depth traces.

- `experiments/run_signal_prediction.py`
  训练并评估 safe-exit risk predictor。
  Train and evaluate the safe-exit risk predictor.

- `experiments/run_frontier.py`
  构建 risk-compute frontier，对比 fixed depth、oracle、confidence baseline 和 SafeLoop-DAR。
  Build the risk-compute frontier across fixed depth, oracle, confidence
  baselines, and SafeLoop-DAR.

- `experiments/run_calibration.py`
  验证 in-domain calibration validity。
  Evaluate in-domain calibration validity.

- `experiments/run_token_stage_ablation.py`
  消融 token/stage 与 DAR 特征。
  Ablate token/stage and DAR features.

- `experiments/run_residual_convergence.py`
  分析 residual novelty 与下一深度收益的相关性。
  Analyze the correlation between residual novelty and next-depth gain.

- `experiments/run_budget_scaling.py`
  分析固定 loop budget 下的风险变化。
  Analyze risk under fixed loop budgets.

- `docs/experiment_plan.md`
  详细实验任务书，包含每组实验的目的、命令、输出、指标和预期结论。
  Detailed experiment plan with goals, commands, outputs, metrics, and expected
  conclusions for each experiment.
