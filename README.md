# SafeLoop-DAR

**少想一点，但安全退出：面向循环语言模型的风险校准动态停止与 Halt-Aware 训练**

**Think Less, Think Safely: Risk-Calibrated Halting and Halt-Aware Training for Looped Language Models**

SafeLoop-DAR 是一个面向循环语言模型（Looped Language Models, LoopLMs）的研究项目。它研究的问题不是简单调一个 early-exit threshold，而是：

SafeLoop-DAR is a research codebase for Looped Language Models (LoopLMs). It does
not merely tune an early-exit threshold. It asks:

```text
给定一个 token 的 depth-state trajectory，模型什么时候可以在风险可控的前提下停止继续循环？
Given a token-level depth-state trajectory, when can the model safely stop recurrent computation?
```

本仓库覆盖两条互补路线：

This repository supports two complementary tracks:

```text
Track A: Post-hoc SafeLoop-DAR
  不改模型，只从 depth-state signals 学习风险校准早停策略。
  Learn a calibrated halting policy from depth-state signals without changing the base model.

Track B: Halt-Aware Pretraining / Continued Pretraining
  在训练阶段加入 compute penalty、risk violation penalty 和 halting signal，
  让 LoopLM 本身学会更好的 recurrent-depth frontier。
  Add compute penalty, risk violation penalty, and halting signals during training
  so the LoopLM itself learns a better recurrent-depth frontier.
```

***

## 1. 研究主张 / Research Claim

核心主张：

Main claim:

```text
LoopLM 的 recurrent depth 不应由固定 T 或 confidence-only threshold 决定。
Depth-state residual novelty、depth stability、token/stage risk 与校准风险控制
可以共同决定更安全、更省计算的动态停止路径。

Recurrent depth in LoopLMs should not be controlled by a fixed T or a
confidence-only threshold. Depth-state residual novelty, depth stability,
token/stage risk, and calibrated risk control can jointly choose safer and
cheaper dynamic halting paths.
```

进一步主张：

Extended claim:

```text
如果在训练阶段加入 halt-aware objective，模型可以系统性改善 risk-compute frontier；
也就是说，在相同风险预算下更早退出，或在相同平均 depth 下更少任务退化。

With a halt-aware objective during pretraining or continued pretraining, the
model can systematically improve the risk-compute frontier: lower average depth
at the same risk budget, or lower task degradation at the same compute budget.
```

***

## 2. 方法概览 / Method Overview

### 2.1 Post-hoc SafeLoop-DAR

对每个 token-depth pair 提取：

For each token-depth pair, SafeLoop-DAR extracts:

```text
confidence features:
  entropy, top-1 probability, top-1/top-2 margin

depth-state features:
  hidden delta, logit delta, residual novelty, depth attention stability

risk metadata:
  token type, generation stage, high-risk group

calibration:
  train/cal/test split, group-wise UCB calibration
```

最终选择最早满足风险约束的 depth：

The policy selects the earliest depth satisfying the risk constraint:

```text
halt at min d such that risk_upper_bound(group, score_d) <= epsilon
```

### 2.2 Halt-Aware Training

训练目标由两部分组成：

The training objective has two parts:

```text
语言建模 / LM target:
  保证 full-depth 能力和中间 depth 可预测性。
  Preserve full-depth capability and intermediate-depth predictability.

停止控制 / Halting control:
  compute penalty
  risk violation penalty
  halting margin
  group-aware depth adjustment
```

概念性目标：

Conceptual objective:

```text
L = L_full_depth_LM
  + alpha * L_intermediate_depth
  + beta  * L_risk_prediction
  + gamma * compute_penalty
  + mu    * risk_violation_penalty
```

当前仓库提供两种训练验证：

The repository provides two training studies:

```text
Controlled LoopTiny state training:
  轻量机制验证，不依赖 torch。
  Lightweight controlled mechanism validation, no torch dependency.

Self-trained TinyLoopLM:
  标准库实现的小型 LoopLM，从 synthetic mixed benchmark 训练 token prototype、
  recurrent-depth schedule 和 halt-aware depth offsets。
  A stdlib-only tiny LoopLM that trains token prototypes, recurrent-depth
  schedules, and halt-aware depth offsets from a synthetic mixed benchmark.
```

***

## 3. 项目结构 / Project Layout

```text
SafeLoop-DAR/
├── configs/
│   ├── benchmarks/
│   │   ├── looped_core.json
│   │   ├── synthetic_mixed_control_60.json
│   │   ├── gsm8k_local_template.json
│   │   ├── humaneval_local_template.json
│   │   └── rag_entity_local_template.json
│   ├── experiments/
│   │   ├── collect_traces_mock.json
│   │   ├── task_evaluation_mock.json
│   │   ├── frontier.json
│   │   ├── halt_aware_pretraining.json
│   │   ├── train_tiny_loop_lm.json
│   │   └── tiny_loop_lm_frontier_comparison.json
│   └── models/
├── experiments/
│   ├── collect_traces.py
│   ├── run_task_evaluation.py
│   ├── run_signal_prediction.py
│   ├── run_calibration.py
│   ├── run_frontier.py
│   ├── run_overhead_frontier.py
│   ├── train_tiny_loop_lm.py
│   ├── run_halt_aware_pretraining.py
│   └── run_pretraining_frontier_comparison.py
├── safeloop/
│   ├── adapters/
│   │   ├── mock_adapter.py
│   │   ├── looptiny_adapter.py
│   │   ├── tiny_loop_lm_adapter.py
│   │   └── ouro_adapter.py
│   ├── controlled/
│   │   ├── halt_aware.py
│   │   └── tiny_loop_lm.py
│   ├── evaluation/
│   │   ├── splits.py
│   │   ├── task_degradation.py
│   │   └── tasks/
│   ├── features/
│   ├── risk/
│   │   ├── calibration.py
│   │   └── ucb_calibration.py
│   └── workloads/
└── docs/
```

***

## 4. 安装 / Installation

### 4.1 最小环境 / Minimal Environment

Mock、controlled training、TinyLoopLM 都不依赖 torch。

Mock experiments, controlled training, and TinyLoopLM do not require torch.

```bash
cd /Users/heyuhang/Documents/New\ project/SafeLoop-DAR
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[analysis]"
```

### 4.2 真实模型环境 / Real Model Environment

Ouro / LoopFormer / LayerSkip 需要模型依赖：

Ouro / LoopFormer / LayerSkip require model dependencies:

```bash
python -m pip install -e ".[model,analysis]"
```

正式实验请记录：

For formal experiments, record:

```text
git commit hash
model revision SHA
dataset version
split salt
CUDA / torch / transformers version
GPU type
batch size
exact command
output directory
```

***

## 5. Benchmark 输入 / Benchmark Input

实验读取 `configs/benchmarks/*.json`。支持三种形式：

Experiment scripts read `configs/benchmarks/*.json`. Three formats are
supported:

```text
inline samples
local JSONL / JSON / CSV manifest
synthetic generator
```

本地 JSONL 示例：

Local JSONL example:

```json
{
  "name": "gsm8k_local",
  "task": "gsm8k",
  "stage": "final_answer",
  "source_path": "../../data/gsm8k_test.jsonl",
  "format": "jsonl",
  "limit": 256,
  "field_map": {
    "request_id": "id",
    "expected": "answer"
  },
  "prompt_template": "Question: {question}\nAnswer:",
  "metadata_fields": ["question", "answer"]
}
```

controlled benchmark 示例：

Controlled benchmark example:

```json
{
  "name": "synthetic_mixed_control_60",
  "generator": "synthetic_mixed_control",
  "num_repeats": 12
}
```

***

## 6. 最小 Mock Pipeline / Minimal Mock Pipeline

这条链路验证 post-hoc SafeLoop-DAR，不加载真实模型。

This pipeline validates post-hoc SafeLoop-DAR without loading real models.

```bash
python experiments/collect_traces.py \
  --config configs/experiments/collect_traces_mock.json

python experiments/run_task_evaluation.py \
  --config configs/experiments/task_evaluation_mock.json

python experiments/run_signal_prediction.py \
  --config configs/experiments/signal_prediction.json

python experiments/run_calibration.py \
  --trace runs/mock_task_evaluation/teacher_forced.evaluated.jsonl \
  --label-type task_degradation \
  --calibration-method ucb \
  --output-dir runs/calibration

python experiments/run_frontier.py \
  --config configs/experiments/frontier.json

python experiments/run_overhead_frontier.py \
  --config configs/experiments/overhead_frontier.json
```

关键输出：

Key outputs:

```text
runs/mock_task_evaluation/teacher_forced.evaluated.jsonl
runs/signal_prediction/metrics.json
runs/calibration/calibration.json
runs/frontier/frontier.json
runs/overhead_frontier/overhead_frontier.json
```

***

## 7. 自训练小型 LoopLM / Self-Trained Tiny LoopLM

这一步从 synthetic mixed benchmark 训练一个小型 LoopLM。它不是大模型替代品，而是用于机制验证：

This step trains a tiny LoopLM on the synthetic mixed benchmark. It is not a
replacement for large models; it is a controlled mechanism study:

```text
untrained TinyLoopLM
standard TinyLoopLM pretraining
halt-aware TinyLoopLM pretraining
```

运行：

Run:

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
runs/tiny_loop_lm_training/untrained_model_config.json
runs/tiny_loop_lm_training/standard_model_config.json
runs/tiny_loop_lm_training/halt_aware_model_config.json
runs/tiny_loop_lm_training/training_summary.json
```

比较训练前后 frontier：

Compare frontiers before and after training:

```bash
python experiments/run_pretraining_frontier_comparison.py \
  --config configs/experiments/tiny_loop_lm_frontier_comparison.json
```

输出：

Outputs:

```text
runs/tiny_loop_lm_frontier_comparison/baseline_frontier.json
runs/tiny_loop_lm_frontier_comparison/standard_frontier.json
runs/tiny_loop_lm_frontier_comparison/halt_aware_frontier.json
runs/tiny_loop_lm_frontier_comparison/comparison.json
```

判断标准：

Pass criteria:

```text
baseline_vs_halt_aware.claim_pass == true
standard_vs_halt_aware.claim_pass == true
improved_targets == comparable_targets
fixed_depth_non_worse == fixed_depth_comparable
```

***

## 8. Halt-Aware Continued Pretraining Controlled Study

这条路径保留早期 LoopTiny controlled state 训练，用于快速检查 halt-aware depth adjustment 是否能改善 frontier。

This path keeps the earlier LoopTiny controlled-state training for quick checks
of whether halt-aware depth adjustment improves the frontier.

```bash
python experiments/run_halt_aware_pretraining.py \
  --config configs/experiments/halt_aware_pretraining.json

python experiments/run_pretraining_frontier_comparison.py \
  --config configs/experiments/pretraining_frontier_comparison.json
```

***

## 9. 真实模型实验 / Real Model Experiments

真实模型必须先验证 depth-control 字段有效：

Real models must first pass depth-control validation:

```bash
python experiments/validate_depth_control.py \
  --config configs/experiments/collect_traces_ouro_1_4b.json \
  --output-dir runs/ouro_1_4b_depth_control
```

然后收集 trace：

Then collect traces:

```bash
python experiments/collect_traces.py \
  --config configs/experiments/collect_traces_ouro_1_4b.json

python experiments/run_task_evaluation.py \
  --trace runs/ouro_1_4b_traces/teacher_forced.jsonl \
  --output-dir runs/ouro_1_4b_task_eval

python experiments/run_frontier.py \
  --trace runs/ouro_1_4b_task_eval/teacher_forced.evaluated.jsonl \
  --output-dir runs/ouro_1_4b_frontier \
  --label-type task_degradation \
  --calibration-method ucb
```

Ouro built-in baseline：

```bash
python experiments/run_builtin_early_exit.py \
  --config configs/experiments/collect_traces_ouro_1_4b.json \
  --output-dir runs/ouro_1_4b_builtin_exit
```

真实 latency：

Measured latency:

```bash
python experiments/profile_latency.py \
  --trace runs/ouro_1_4b_task_eval/teacher_forced.evaluated.jsonl \
  --output-dir runs/ouro_1_4b_latency

python experiments/run_overhead_frontier.py \
  --frontier runs/ouro_1_4b_frontier/frontier.json \
  --profile runs/ouro_1_4b_latency/profile.json \
  --output-dir runs/ouro_1_4b_overhead_frontier
```

***

## 10. 主要实验表 / Main Tables

论文主表建议包括：

Recommended paper tables:

```text
Table 1: Post-hoc risk-compute frontier
Table 2: Calibration validity under train/cal/test
Table 3: Token/stage and depth-signal ablation
Table 4: TinyLoopLM halt-aware training frontier
Table 5: Ouro-1.4B / Ouro-2.6B measured latency
Table 6: Failure analysis and high-cost semantic risk
```

***

## 11. 当前边界 / Current Boundaries

已经支持：

Currently supported:

```text
mock pipeline
task degradation labels
train/cal/test split
UCB group calibration
post-hoc risk-compute frontier
controlled halt-aware training
self-trained TinyLoopLM
frontier comparison before/after halt-aware training
local benchmark manifest loader
```

仍需在服务器完成：

Still required on a GPU server:

```text
Ouro-1.4B / Ouro-2.6B real trace collection
real task benchmark runs
HumanEval sandboxed pass@1
measured GPU latency profiling
Ouro continued pretraining or LoRA-style halting-head training
```

***

## 12. 推荐阅读 / Recommended Docs

- [详细实验计划 / Detailed Experiment Plan](docs/experiment_plan.md)
- [文件索引 / File Index](docs/file_index.md)
- [风险控制 / Risk Mitigation](docs/risk_mitigation.md)

