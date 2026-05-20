# SafeLoop

**中文题目：面向循环语言模型的风险校准 Test-Time Scaling**

**English Title: SafeLoop: Risk-Calibrated Test-Time Scaling for Looped Language Models**

SafeLoop 的研究中心不是“给 looped LM 外接一个 early-exit 阈值器”，而是：

SafeLoop is framed as a risk-calibrated test-time scaling framework for looped
LMs. The research question is:

```text
Looped LM 的 recurrent depth 能否成为一个可预测、可校准、可扩展的 test-time compute 轴？
Can recurrent depth in looped LMs become a predictable, calibrated, and scalable
test-time compute axis?
```

当前仓库仍是研究平台，不是完整顶会实证包。它提供 mock / controlled / TinyLoopLM
机制验证和真实模型实验接口；AAAI / ICLR / NeurIPS 主轨级证据仍需要真实任务、真实
HumanEval sandbox、真实 agentic RAG、真实 GPU latency，以及 Ouro/LoopFormer 真实实验。

The current repository is still a research platform rather than a finished
top-conference empirical package. It provides mock / controlled / TinyLoopLM
mechanism studies and real-model experiment interfaces. AAAI / ICLR / NeurIPS
level evidence still requires real tasks, sandboxed HumanEval, real agentic RAG,
measured GPU latency, and real Ouro/LoopFormer experiments.

***

## 1. 新研究主线 / Updated Research Direction

本项目围绕三条主线组织：

The project is organized around three tracks:

```text
Track A: Hidden-Only SafeLoop
  主论文结果优先使用 hidden_lite / dynamics_only / dynamics_confidence。
  不把 token-stage / risk-group one-hot 当成主结果证据。

Track B: Sequence-Level Risk Control
  从 token-level early exit 升级到 request-level / sequence-level 风险控制。
  对 code/tool/RAG 这类高代价任务，以 request failure 为核心风险对象。

Track C: Safe Test-Time Scaling Law
  拟合 residual novelty / depth shift 随 recurrent depth 的衰减曲线，
  研究最优退出深度如何随训练预算、测试预算和风险容忍度变化。
```

```text
Track A: Hidden-Only SafeLoop
  Main paper results prioritize hidden_lite / dynamics_only / dynamics_confidence.
  Token-stage / risk-group one-hot features are auxiliary analyses only.

Track B: Sequence-Level Risk Control
  Move from token-level early exit to request-level / sequence-level control.
  For code/tool/RAG, request failure is the primary risk object.

Track C: Safe Test-Time Scaling Law
  Fit residual novelty / depth-shift decay across recurrent depth and study how
  optimal exit depth scales with training budget, test budget, and risk tolerance.
```

***

## 2. 核心原则 / Core Principles

```text
P1. 主结果先去泄漏
    hidden-only / dynamics-only 是主证据；full_with_groups_aux 只做辅助上界分析。

P2. 风险先按 request 定义
    token-level risk 可做分析，但 code/tool/RAG 的安全性必须看 request-level failure。

P3. mock 只做 smoke test
    synthetic_mixed_control_60 只能说明机制可跑，不能支撑顶会主张。

P4. 统计必须报告不确定性
    多 seed、bootstrap CI、配对检验和多重比较控制是论文级结果的必要条件。

P5. 速度必须实测
    avg_depth 不等于 speedup，正式实验必须包含 GPU wall-clock latency。
```

***

## 3. 强 Baseline 体系 / Strong Baseline Suite

SafeLoop 的实验必须和强 baseline 对齐，避免只和弱阈值方法比较。

SafeLoop must be evaluated against strong baselines, not only weak threshold
rules.

```text
Inference upper/lower bounds:
  full recurrent depth
  fixed depth 1..T
  oracle earliest correct depth

Confidence baselines:
  entropy threshold
  top-1 probability threshold
  top-1/top-2 margin threshold
  learned confidence-only predictor

Looped-LM native baselines:
  model built-in early_exit_threshold when available
  hidden_delta-only threshold
  recurrent-depth budget scheduling

Learned risk baselines:
  hidden_lite predictor
  dynamics_only predictor
  sequence-level risk controller
  empirical calibration vs UCB calibration

Training-time baselines:
  standard continued pretraining
  loop-depth supervision / exit-head training
  LayerSkip-style early-exit training when applicable

System reference baselines:
  speculative decoding as an orthogonal serving reference
  measured full-depth serving latency
```

主论文结论必须至少证明：

The main claims must show at least:

```text
SafeLoop hidden-only > confidence-only
SafeLoop sequence-level > token-only for high-cost tasks
SafeLoop measured latency > full-depth latency after controller overhead
SafeLoop training-time variant improves the risk-compute frontier over standard continued pretraining
```

***

## 4. 项目结构 / Project Layout

```text
SafeLoop/
├── configs/
│   ├── benchmarks/
│   │   ├── synthetic_mixed_control_60.json
│   │   ├── gsm8k_local_template.json
│   │   ├── humaneval_local_template.json
│   │   └── rag_entity_local_template.json
│   └── experiments/
│       ├── frontier.json
│       ├── sequence_frontier.json
│       ├── bootstrap_frontier.json
│       ├── depth_scaling_mock.json
│       ├── train_tiny_loop_lm.json
│       └── tiny_loop_lm_frontier_comparison.json
├── experiments/
│   ├── run_frontier.py
│   ├── run_sequence_frontier.py
│   ├── run_bootstrap_frontier.py
│   ├── run_depth_scaling_analysis.py
│   ├── train_tiny_loop_lm.py
│   └── run_pretraining_frontier_comparison.py
├── safeloop/
│   ├── features/sets.py
│   ├── evaluation/sequence_risk.py
│   ├── evaluation/bootstrap.py
│   ├── evaluation/scaling.py
│   ├── controlled/tiny_loop_lm.py
│   └── risk/ucb_calibration.py
└── docs/
```

***

## 5. 安装 / Installation

Mock、TinyLoopLM、bootstrap、scaling 分析都不依赖 torch。

Mock, TinyLoopLM, bootstrap, and scaling analyses do not require torch.

```bash
cd <project-root>
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[analysis]"
```

真实 Ouro / LoopFormer / LayerSkip 实验需要：

Real Ouro / LoopFormer / LayerSkip experiments require:

```bash
python -m pip install -e ".[model,analysis]"
```

***

## 6. Mock Smoke Test

这条链路只验证管线，不作为论文主结果。

This pipeline validates code paths only; it is not a paper-level result.

```bash
python experiments/collect_traces.py \
  --config configs/experiments/collect_traces_mock.json

python experiments/run_task_evaluation.py \
  --config configs/experiments/task_evaluation_mock.json

python experiments/run_frontier.py \
  --config configs/experiments/frontier.json
```

输出 / Outputs:

```text
runs/mock_task_evaluation/teacher_forced.evaluated.jsonl
runs/frontier/frontier.json
```

`frontier.json` 现在会显式区分：

`frontier.json` now explicitly separates:

```text
hidden_lite
dynamics_only
dynamics_confidence
full_with_groups_aux
```

其中 `full_with_groups_aux` 不能作为主论文结论，因为它使用显式 token/stage group metadata。

`full_with_groups_aux` must not be used as a main paper conclusion because it
uses explicit token/stage group metadata.

***

## 7. Sequence-Level Risk Control

request-level controller 将 token-depth rows 聚合成 request-depth rows：

The request-level controller aggregates token-depth rows into request-depth
rows:

```text
request label = any token at this depth causes task degradation
features      = mean/max/min of hidden-only token features
group         = request by default, no task/stage leakage
```

运行：

Run:

```bash
python experiments/run_sequence_frontier.py \
  --config configs/experiments/sequence_frontier.json
```

输出：

Output:

```text
runs/sequence_frontier/sequence_frontier.json
```

这一步是 code/tool/RAG 方向的主风险控制对象；token-level frontier 只作为分析视角。

This is the main risk-control object for code/tool/RAG; token-level frontier is
an analysis view only.

***

## 8. Bootstrap CI

论文级结果不能只报告单次 split 的点估计。

Paper-level results must not rely on a single split point estimate.

```bash
python experiments/run_bootstrap_frontier.py \
  --config configs/experiments/bootstrap_frontier.json
```

输出：

Output:

```text
runs/bootstrap_frontier/bootstrap_frontier.json
```

报告：

Report:

```text
risk mean / lower / upper
avg_depth mean / lower / upper
number of selected tokens
```

***

## 9. Depth-Dynamics Scaling Analysis

为了把论文从 feature engineering 提升为 safe test-time scaling，需要分析 recurrent depth
动力学。当前入口拟合：

To move beyond feature engineering, SafeLoop analyzes recurrent-depth dynamics.
The current script fits:

```text
residual_novelty(depth) = a * exp(-b * depth) + c
```

运行：

Run:

```bash
python experiments/run_depth_scaling_analysis.py \
  --config configs/experiments/depth_scaling_mock.json
```

输出：

Output:

```text
runs/depth_scaling/depth_scaling.json
```

论文级版本应在不同训练 loop budget、测试最大 depth、风险容忍度 epsilon 下拟合
`best_exit_depth(epsilon)` 的缩放规律。

The paper-level version should fit how `best_exit_depth(epsilon)` scales with
training loop budget, maximum test depth, and risk tolerance.

***

## 10. 自训练 TinyLoopLM / Self-Trained TinyLoopLM

TinyLoopLM 仍是 controlled mechanism model，不是大模型结果。它用于验证 halt-aware
training 是否可能改善 frontier。

TinyLoopLM is still a controlled mechanism model, not a large-model result. It
tests whether halt-aware training can improve the frontier.

```bash
python experiments/train_tiny_loop_lm.py \
  --config configs/experiments/train_tiny_loop_lm.json

python experiments/run_pretraining_frontier_comparison.py \
  --config configs/experiments/tiny_loop_lm_frontier_comparison.json
```

输出：

Outputs:

```text
runs/tiny_loop_lm_training/training_summary.json
runs/tiny_loop_lm_frontier_comparison/comparison.json
```

论文中只能把它作为机制验证，不可替代 Ouro / LoopFormer 真实实验。

In the paper, this is a mechanism study only and cannot replace real
Ouro/LoopFormer experiments.

***

## 11. 真实任务与真实模型 / Real Tasks And Real Models

仓库提供模板，但真实数据与真实 evaluator 需要在服务器补齐：

Templates are provided, but real data and evaluators must be completed on a
server:

```text
configs/benchmarks/gsm8k_local_template.json
configs/benchmarks/humaneval_local_template.json
configs/benchmarks/rag_entity_local_template.json
```

必须完成：

Required for paper-level claims:

```text
GSM8K / MATH500 final-answer correctness
HumanEval sandboxed pass@1
JSON schema validity and exact field accuracy
tool execution success
RAG/entity correctness
Ouro built-in early_exit_threshold baseline
measured GPU latency
```

***

## 12. 最低投稿门槛 / Minimum Submission Gate

要冲 AAAI / ICLR / NeurIPS，至少需要：

For AAAI / ICLR / NeurIPS, at minimum:

```text
hidden-only SafeLoop beats confidence-only on real task degradation
sequence-level controller improves request-level risk frontier
bootstrap CI over multiple seeds / split salts
Ouro-1.4B and Ouro-2.6B real results
real HumanEval / tool / RAG evaluation
GPU wall-clock speedup for SafeLoop-Lite
scaling-law analysis of recurrent-depth dynamics
failure analysis for early-exit mistakes
```

***

## 13. 当前边界 / Current Boundaries

已经实现：

Implemented:

```text
mock pipeline
hidden-only feature sets
request-level sequence frontier
bootstrap CI script
residual novelty depth-scaling fit
self-trained TinyLoopLM controlled study
UCB group calibration
local benchmark manifest loader
```

尚未完成：

Not yet complete:

```text
real Ouro trace collection
real benchmark runs
HumanEval sandbox
real agentic RAG
GPU wall-clock profiler
Ouro continued pretraining / LoRA halting head
multi-seed large-sample statistical study
```
