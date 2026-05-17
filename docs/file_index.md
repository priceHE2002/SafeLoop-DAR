# 文件索引 / File Index

本文档解释 SafeLoop-DAR 中主要文件的作用，便于复试讲解、论文复现和后续工程扩展。

This document explains the purpose of major files in SafeLoop-DAR for interview
discussion, paper reproducibility, and future engineering extensions.

## 配置文件 / Configuration Files

- `configs/models/*.json`
  模型适配器配置，包括 Ouro、LoopFormer、LayerSkip、mock / LoopTiny 和 TinyLoopLM。
  Model adapter configs for Ouro, LoopFormer, LayerSkip, mock / LoopTiny, and
  TinyLoopLM.

- `configs/benchmarks/*.json`
  Benchmark 样本配置，用于构造 prompt、任务类型、生成阶段和期望输出。
  Benchmark sample configs for prompts, task types, generation stages, and
  expected outputs.
  支持内联 `samples`，也支持指向本地 JSONL / JSON / CSV 的 `source_path`
  manifest，用于可复现接入真实 benchmark 子集。
  They support inline `samples` and local JSONL / JSON / CSV `source_path`
  manifests for reproducible real-benchmark subsets.
  `synthetic_mixed_control_60.json` 用于 halt-aware pretraining 的可控机制验证。
  `synthetic_mixed_control_60.json` is used by the controlled halt-aware
  pretraining study.

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
  风险标签、风险预测器、empirical calibration 和 UCB calibration。
  Risk labels, risk predictors, empirical calibration, and UCB calibration.

- `safeloop/halting/`
  动态退出策略，包括 fixed depth、score threshold 和 group calibrated policy。
  Halting policies, including fixed-depth, score-threshold, and group-calibrated
  policies.

- `safeloop/evaluation/`
  指标计算、risk-compute frontier 汇总与开销感知 frontier 估算。
  Metric computation, risk-compute frontier summarization, and overhead-aware
  frontier estimation.

- `safeloop/evaluation/tasks/`
  GSM8K、MATH500、HumanEval、JSON、tool call、RAG/entity 的任务 evaluator。
  Task evaluators for GSM8K, MATH500, HumanEval, JSON, tool calls, and
  RAG/entity tasks.

- `safeloop/controlled/halt_aware.py`
  可控 LoopTiny 的 halt-aware continued-pretraining 状态、proxy loss 与训练曲线。
  Halt-aware continued-pretraining state, proxy losses, and training curves for
  controlled LoopTiny studies.

- `safeloop/controlled/tiny_loop_lm.py`
  自训练小型 LoopLM：构建词表、训练 token prototype、学习 group-wise recurrent-depth
  offset，并生成与主 trace pipeline 兼容的 depth states。
  Self-trained tiny LoopLM: builds vocabulary, trains token prototypes, learns
  group-wise recurrent-depth offsets, and emits depth states compatible with
  the main trace pipeline.

- `safeloop/adapters/tiny_loop_lm_adapter.py`
  TinyLoopLM checkpoint 的模型适配器。
  Model adapter for TinyLoopLM checkpoints.

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

- `experiments/run_task_evaluation.py`
  为 trace 增加 task degradation 与 high-cost semantic risk metadata。
  Attach task degradation and high-cost semantic risk metadata to traces.

- `experiments/run_token_stage_ablation.py`
  消融 token/stage 与 DAR 特征。
  Ablate token/stage and DAR features.

- `experiments/run_residual_convergence.py`
  分析 residual novelty 与下一深度收益的相关性。
  Analyze the correlation between residual novelty and next-depth gain.

- `experiments/run_budget_scaling.py`
  分析固定 loop budget 下的风险变化。
  Analyze risk under fixed loop budgets.

- `experiments/run_overhead_frontier.py`
  读取 `frontier.json`，估算不同刹车系统实现 profile 下的有效 speedup 与 break-even。
  Read `frontier.json` and estimate effective speedup and break-even behavior
  under different controller implementation profiles.

- `experiments/validate_depth_control.py`
  验证真实模型的 depth-control 字段是否改变 hidden/logit/latency。
  Validate whether real-model depth-control fields change hidden states, logits,
  and latency.

- `experiments/profile_latency.py`
  从 trace 或真实 profiling 入口生成延迟 profile，供 overhead frontier 使用。
  Produce latency profiles for overhead-aware frontier analysis.

- `experiments/run_builtin_early_exit.py`
  评估模型原生 early-exit threshold grid，主要用于 Ouro built-in baseline。
  Evaluate a model's built-in early-exit threshold grid, mainly for the Ouro
  built-in baseline.

- `experiments/run_free_generation.py`
  收集自回归生成 trace，用于后续端到端任务指标评估。
  Collect autoregressive generation traces for later end-to-end task metrics.

- `experiments/run_transfer.py`
  在一个 trace 上训练风险预测器，并在另一个 trace 上评估跨模型或跨任务迁移。
  Train a risk predictor on one trace and evaluate cross-model or cross-task
  transfer on another trace.

- `experiments/run_halt_aware_pretraining.py`
  生成 standard continued-pretraining 与 halt-aware continued-pretraining 的
  LoopTiny 状态和模型配置。
  Generate LoopTiny states and model configs for standard continued pretraining
  and halt-aware continued pretraining.

- `experiments/train_tiny_loop_lm.py`
  从 synthetic mixed benchmark 自训练 TinyLoopLM，并导出 untrained、standard、
  halt-aware 三组 checkpoint / model config。
  Train TinyLoopLM from the synthetic mixed benchmark and export untrained,
  standard, and halt-aware checkpoints / model configs.

- `experiments/run_pretraining_frontier_comparison.py`
  对比 baseline、standard continued-pretraining、halt-aware continued-pretraining
  的 risk-compute frontier。
  Compare risk-compute frontiers for baseline, standard continued pretraining,
  and halt-aware continued pretraining.

- `scripts/plot_frontier.py`
  将 `frontier.json` 打印成便于阅读的 risk-compute 表格。
  Print `frontier.json` as a readable risk-compute table.

- `scripts/plot_overhead_frontier.py`
  将 `overhead_frontier.json` 打印成便于比较的开销感知 frontier 表格。
  Print `overhead_frontier.json` as a readable overhead-aware frontier table.

- `docs/experiment_plan.md`
  详细实验任务书，包含每组实验的目的、命令、输出、指标和预期结论。
  Detailed experiment plan with goals, commands, outputs, metrics, and expected
  conclusions for each experiment.
