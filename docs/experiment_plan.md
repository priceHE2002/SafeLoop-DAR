# 实验计划 / Experiment Plan

本文档对应 SafeLoop-DAR 的最新研究方向：

This document describes the latest SafeLoop-DAR research direction:

```text
Think Less, Think Safely:
Risk-Calibrated Halting and Halt-Aware Training for Looped Language Models
```

研究不再只做 post-hoc early halting，而是形成三层证据：

The study is no longer only post-hoc early halting. It builds three evidence
layers:

```text
Layer 1: Post-hoc SafeLoop-DAR
  不改模型，证明 depth-state signals 可以预测 task degradation risk。

Layer 2: Self-trained TinyLoopLM
  自己训练一个小型 LoopLM，证明 halt-aware objective 能改善 risk-compute frontier。

Layer 3: Real LoopLM validation
  在 Ouro-1.4B / Ouro-2.6B / LoopFormer 等模型上验证真实任务和真实延迟。
```

---

## 0. 主问题 / Main Question

```text
Can looped language models learn or infer when to stop recurrent computation
while controlling task degradation risk?
```

中文表述：

```text
循环语言模型能否在控制任务退化风险的前提下，学会或推断何时停止继续循环计算？
```

核心假设：

Core hypotheses:

```text
H1: Depth-state trajectories contain useful halting-risk signals.
H2: UCB / conformal-style calibration can control in-domain premature-halting risk.
H3: Halt-aware training improves the risk-compute frontier beyond post-hoc control.
H4: The saved recurrent-depth compute can translate into positive wall-clock speedup.
```

---

## 1. 研究对象 / Research Objects

### 1.1 模型 / Models

| 层级 | 模型 | 目的 |
|---|---|---|
| Mock | MockLoop | pipeline smoke test |
| Controlled | LoopTiny controlled state | 快速机制验证 |
| Self-trained | TinyLoopLM | 自己训练小型 LoopLM，验证 halt-aware objective |
| Real primary | Ouro-1.4B | 真实主实验 |
| Real scaling | Ouro-2.6B | scaling 验证 |
| Generalization | LoopFormer / LayerSkip | 跨模型验证 |

| Level | Model | Purpose |
|---|---|---|
| Mock | MockLoop | pipeline smoke test |
| Controlled | LoopTiny controlled state | quick mechanism validation |
| Self-trained | TinyLoopLM | train a small LoopLM and test halt-aware objective |
| Real primary | Ouro-1.4B | main real-model experiments |
| Real scaling | Ouro-2.6B | scaling validation |
| Generalization | LoopFormer / LayerSkip | cross-model validation |

### 1.2 任务 / Tasks

| 类别 | Benchmark | 主指标 |
|---|---|---|
| Controlled | synthetic_mixed_control_60 | known group risk, task degradation |
| Math | GSM8K | final answer exact match |
| Math | MATH500 | boxed answer exact match |
| Code | HumanEval-lite / HumanEval | pass@1 |
| Structured | JSON schema subset | schema validity |
| Tool | executable tool-call subset | tool execution success |
| RAG | entity QA subset | entity correctness |

---

## 2. 风险标签 / Risk Labels

### 2.1 Token Consistency Risk

```text
early-depth token != full-depth token
```

用途：debug、机制分析、residual convergence。不能作为主结论。

Use: debugging, mechanism analysis, residual convergence. It is not a main
paper claim.

### 2.2 Task Degradation Risk

主风险标签：

Main risk label:

```text
task_degradation(depth) = 1
if early-depth output is worse than full-depth output under a task evaluator
```

实现入口：

Implementation:

```text
experiments/run_task_evaluation.py
safeloop/evaluation/task_degradation.py
safeloop/evaluation/tasks/
```

### 2.3 High-Cost Semantic Risk

高风险错误：

High-cost errors:

```text
math final number wrong
code identifier / unit-test failure
JSON field or schema invalid
tool argument wrong
retrieved entity missing or wrong
```

---

## 3. 数据划分与校准 / Splits And Calibration

所有正式结果使用 request-level 三分：

All formal results use request-level three-way splits:

```text
train        50%  train risk predictor
calibration 25%  fit group threshold / UCB threshold
test         25%  final evaluation only
```

校准规则：

Calibration rule:

```text
For each group g and threshold tau:
  accept tokens with score <= tau
  compute Wilson upper confidence bound of empirical risk
  choose largest tau whose risk upper bound <= epsilon
```

实现：

Implementation:

```text
safeloop/evaluation/splits.py
safeloop/risk/ucb_calibration.py
```

---

## 4. 方法 / Methods

### 4.1 Post-hoc SafeLoop-DAR

特征：

Features:

```text
confidence:
  entropy, top1 probability, top1/top2 margin

state trajectory:
  hidden_delta, logit_delta, residual_novelty, depth_attention_stability

risk metadata:
  relative_depth, token_type, stage, group
```

对照：

Baselines:

```text
fixed depth
oracle halting
entropy only
margin only
top1 probability
confidence-only learned predictor
hidden-delta-only
DAR no token/stage
SafeLoop-DAR-Lite
SafeLoop-DAR-Full
Ouro built-in early_exit_threshold
```

### 4.2 Halt-Aware Training

目标：

Goal:

```text
让模型训练时感知 recurrent-depth compute 与 premature-halting risk 的 trade-off。
Make the model aware of the trade-off between recurrent-depth compute and
premature-halting risk during training.
```

训练目标：

Training objective:

```text
L = L_token
  + alpha * L_intermediate
  + beta  * risk_violation
  + gamma * compute_penalty
  + mu    * halting_margin
```

仓库中的两种实现：

Two implementations in this repository:

```text
safeloop/controlled/halt_aware.py
  controlled state training for quick mechanism tests

safeloop/controlled/tiny_loop_lm.py
  self-trained stdlib TinyLoopLM with token prototypes and learned depth offsets
```

---

## 5. 实验 E0：Mock Pipeline

目的：

Goal:

```text
验证 trace -> task evaluation -> features -> calibration -> frontier -> overhead 全链路。
Validate trace -> task evaluation -> features -> calibration -> frontier -> overhead.
```

命令：

Commands:

```bash
python experiments/collect_traces.py \
  --config configs/experiments/collect_traces_mock.json

python experiments/run_task_evaluation.py \
  --config configs/experiments/task_evaluation_mock.json

python experiments/run_signal_prediction.py \
  --config configs/experiments/signal_prediction.json

python experiments/run_frontier.py \
  --config configs/experiments/frontier.json
```

通过条件：

Pass criteria:

```text
all scripts finish
frontier.json contains task_degradation and ucb calibration
train/cal/test split exists in output
```

---

## 6. 实验 E1：自训练 TinyLoopLM

目的：

Goal:

```text
训练一个小型 LoopLM，并比较 untrained / standard pretraining / halt-aware pretraining。
Train a small LoopLM and compare untrained / standard pretraining / halt-aware pretraining.
```

模型：

Model:

```text
TinyLoopLM:
  stdlib-only
  token prototype vocabulary
  recurrent hidden-state interpolation
  learned group-wise depth offset
  optional halt-aware risk/compute loss
```

训练命令：

Training command:

```bash
python experiments/train_tiny_loop_lm.py \
  --config configs/experiments/train_tiny_loop_lm.json
```

输出：

Outputs:

```text
runs/tiny_loop_lm_training/untrained_checkpoint.json
runs/tiny_loop_lm_training/standard_checkpoint.json
runs/tiny_loop_lm_training/halt_aware_checkpoint.json
runs/tiny_loop_lm_training/training_summary.json
```

frontier 比较：

Frontier comparison:

```bash
python experiments/run_pretraining_frontier_comparison.py \
  --config configs/experiments/tiny_loop_lm_frontier_comparison.json
```

主要指标：

Main metrics:

```text
best avg depth at target risk
fixed-depth risk before/after training
standard_vs_halt_aware.claim_pass
baseline_vs_halt_aware.claim_pass
improved_targets / comparable_targets
fixed_depth_non_worse / fixed_depth_comparable
```

通过条件：

Pass criteria:

```text
halt-aware frontier improves over standard TinyLoopLM at all target risks
fixed-depth risk is not worse
improvement appears on multiple token/stage groups
```

---

## 7. 实验 E2：Controlled Halt-Aware State Training

目的：

Goal:

```text
快速验证 halt-aware depth adjustment 机制是否有效。
Quickly validate the halt-aware depth-adjustment mechanism.
```

命令：

Commands:

```bash
python experiments/run_halt_aware_pretraining.py \
  --config configs/experiments/halt_aware_pretraining.json

python experiments/run_pretraining_frontier_comparison.py \
  --config configs/experiments/pretraining_frontier_comparison.json
```

---

## 8. 实验 E3：Depth Signal Prediction

目的：

Goal:

```text
验证 depth-state signals 是否比 confidence-only 更能预测 task degradation。
Test whether depth-state signals predict task degradation better than confidence-only.
```

命令：

Command:

```bash
python experiments/run_signal_prediction.py \
  --trace runs/mock_task_evaluation/teacher_forced.evaluated.jsonl \
  --label-type task_degradation \
  --output-dir runs/signal_prediction
```

报告：

Report:

```text
train metrics
calibration metrics
test metrics
label rate
prediction loss
```

---

## 9. 实验 E4：Risk Calibration

目的：

Goal:

```text
证明方法不是普通阈值调参，而是 train/cal/test 风险校准。
Show this is calibrated risk control rather than threshold tuning.
```

命令：

Command:

```bash
python experiments/run_calibration.py \
  --trace runs/mock_task_evaluation/teacher_forced.evaluated.jsonl \
  --label-type task_degradation \
  --calibration-method ucb \
  --targets 0.005,0.01,0.02,0.05,0.1 \
  --output-dir runs/calibration
```

报告：

Report:

```text
target risk
calibration empirical risk
calibration upper bound
test empirical risk
coverage
avg depth
```

---

## 10. 实验 E5：Risk-Compute Frontier

目的：

Goal:

```text
比较 fixed depth、confidence-only、SafeLoop-DAR、halt-aware training 的风险-计算曲线。
Compare the risk-compute frontier of fixed depth, confidence-only, SafeLoop-DAR,
and halt-aware training.
```

命令：

Command:

```bash
python experiments/run_frontier.py \
  --trace runs/mock_task_evaluation/teacher_forced.evaluated.jsonl \
  --label-type task_degradation \
  --calibration-method ucb \
  --output-dir runs/frontier
```

主表：

Main table:

```text
method
target risk
empirical risk
avg depth
coverage
risk by group
avg depth by group
```

---

## 11. 实验 E6：Token / Stage Risk Control

目的：

Goal:

```text
证明 token/stage group 是 group-conditional calibration metadata，而不是 regex trick。
Show token/stage groups are calibration metadata, not a regex trick.
```

命令：

Command:

```bash
python experiments/run_token_stage_ablation.py \
  --trace runs/mock_task_evaluation/teacher_forced.evaluated.jsonl \
  --label-type task_degradation \
  --output-dir runs/token_stage_ablation
```

对照：

Comparisons:

```text
confidence
confidence_hidden
DAR implicit no group
explicit group only
random group
DAR hybrid full
```

---

## 12. 实验 E7：Residual Novelty And Depth Stability

目的：

Goal:

```text
分析 residual novelty 和 depth stability 是否对应下一轮循环的边际收益。
Analyze whether residual novelty and depth stability correspond to next-depth marginal gain.
```

命令：

Command:

```bash
python experiments/run_residual_convergence.py \
  --trace runs/mock_task_evaluation/teacher_forced.evaluated.jsonl \
  --label-type task_degradation \
  --output-dir runs/residual_convergence
```

---

## 13. 实验 E8：真实模型接入 / Real Model Validation

真实 Ouro 实验必须先验证 depth-control：

Real Ouro experiments must first validate depth control:

```bash
python experiments/validate_depth_control.py \
  --config configs/experiments/collect_traces_ouro_1_4b.json \
  --output-dir runs/ouro_1_4b_depth_control
```

通过条件：

Pass criteria:

```text
hidden states differ across depths
logits differ across depths
latency changes with depth
full-depth output matches normal model behavior
```

然后再跑：

Then run:

```bash
python experiments/collect_traces.py \
  --config configs/experiments/collect_traces_ouro_1_4b.json

python experiments/run_task_evaluation.py \
  --trace runs/ouro_1_4b_traces/teacher_forced.jsonl \
  --output-dir runs/ouro_1_4b_task_eval

python experiments/run_frontier.py \
  --trace runs/ouro_1_4b_task_eval/teacher_forced.evaluated.jsonl \
  --label-type task_degradation \
  --calibration-method ucb \
  --output-dir runs/ouro_1_4b_frontier
```

---

## 14. 实验 E9：Ouro Built-In Early Exit Baseline

目的：

Goal:

```text
正面对比 Ouro 原生 early_exit_threshold。
Directly compare against Ouro's built-in early_exit_threshold.
```

命令：

Command:

```bash
python experiments/run_builtin_early_exit.py \
  --config configs/experiments/collect_traces_ouro_1_4b.json \
  --output-dir runs/ouro_1_4b_builtin_exit
```

报告：

Report:

```text
threshold
task accuracy
task degradation risk
avg depth
latency
```

---

## 15. 实验 E10：Latency And Overhead

目的：

Goal:

```text
证明省掉的 recurrent loop 大于刹车系统开销。
Show saved recurrent-loop compute exceeds controller overhead.
```

命令：

Command:

```bash
python experiments/profile_latency.py \
  --trace runs/mock_task_evaluation/teacher_forced.evaluated.jsonl \
  --output-dir runs/latency_profile

python experiments/run_overhead_frontier.py \
  --frontier runs/frontier/frontier.json \
  --profile runs/latency_profile/profile.json \
  --output-dir runs/overhead_frontier
```

真实论文实验需要 GPU wall-clock profiler 替换 trace-based profile。

Paper experiments must replace the trace-based profile with GPU wall-clock
profiling.

---

## 16. 论文级主表 / Paper-Level Main Tables

```text
Table 1: SafeLoop-DAR post-hoc frontier on Ouro-1.4B
Table 2: Calibration validity under target epsilon
Table 3: Ablation of confidence / residual novelty / depth stability / token group
Table 4: TinyLoopLM standard vs halt-aware training frontier
Table 5: Ouro-2.6B scaling
Table 6: Latency and overhead-aware speedup
Table 7: Failure cases and high-cost semantic risk
```

---

## 17. 最低投稿门槛 / Minimum Submission Gate

若目标是 ICML / NeurIPS / ICLR，至少需要完成：

For ICML / NeurIPS / ICLR, at least complete:

```text
TinyLoopLM halt-aware training improves frontier
Ouro-1.4B depth-control validation passes
Ouro-1.4B runs on GSM8K + MATH500 + JSON/tool + RAG/entity
task_degradation is the main label
train/cal/test and UCB calibration are used
Ouro built-in early_exit_threshold baseline is included
measured latency shows positive speedup for SafeLoop-DAR-Lite
failure analysis is reported
```

---

## 18. 当前仓库覆盖 / Current Code Coverage

已经实现：

Implemented:

```text
mock trace collection
task degradation annotation
request-level train/cal/test split
UCB group calibration
post-hoc frontier
token/stage ablation
latency profile interface
overhead-aware frontier
controlled halt-aware state training
self-trained TinyLoopLM
TinyLoopLM before/after frontier comparison
local benchmark manifest loader
```

仍需服务器完成：

Needs server-side completion:

```text
real Ouro trace collection
real benchmark data ingestion
HumanEval sandboxed execution
GPU wall-clock profiling
Ouro continued pretraining or LoRA halting-head training
```
