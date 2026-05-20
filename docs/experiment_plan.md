# SafeLoop 实验计划 / Experiment Plan

## 0. 研究定位 / Research Positioning

SafeLoop 的研究方向定义为：

SafeLoop is framed as:

```text
SafeLoop: Risk-Calibrated Test-Time Scaling for Looped Language Models.
```

核心目标不是证明一个 toy early-exit controller 有效，而是研究：

The goal is not to validate a toy early-exit controller, but to study:

```text
1. looped LM 的 recurrent-depth dynamics 是否包含可预测的剩余风险信号；
2. 这些信号能否在 request-level task failure 上被校准；
3. 最优退出深度是否随训练 loop budget、测试 depth budget、风险容忍度 epsilon 呈现可拟合规律；
4. avg-depth 节省能否转化为真实 GPU wall-clock speedup。
```

---

## 1. 研究风险与对应设计 / Research Risks And Design Responses

| 风险 | 设计响应 |
|---|---|
| synthetic benchmark 和 hand-written group 过度对齐 | 主结果使用 hidden-only / dynamics-only，不使用 group one-hot |
| token-level 风险太弱 | 增加 request-level / sequence-level 风险控制 |
| task_degradation 只相对 full-depth | 真实任务必须额外报告 gold correctness |
| claim_pass 是布尔拼接 | 增加 bootstrap CI、多 seed / split salt |
| avg_depth 不等于速度 | 增加 measured latency / overhead frontier |
| toy 机制缺少理论中心 | 增加 residual novelty 衰减与 safe test-time scaling 分析 |

---

## 2. 实验层级 / Evidence Layers

```text
Layer 0: Mock smoke test
  只验证代码路径，不写进主结论。

Layer 1: Hidden-only post-hoc SafeLoop
  主论文方法。只用 hidden_lite / dynamics_only / dynamics_confidence。

Layer 2: Sequence-level SafeLoop
  对 code/tool/RAG 用 request-level failure 做风险控制。

Layer 3: TinyLoopLM controlled training
  机制验证：halt-aware training 是否可能改善 frontier。

Layer 4: Real looped LM experiments
  Ouro-1.4B / Ouro-2.6B / LoopFormer，真实任务，真实 latency。

Layer 5: Scaling-law analysis
  拟合 depth dynamics 与 best_exit_depth(epsilon) 的缩放规律。
```

---

## 3. 强 Baseline 体系 / Strong Baseline Suite

SafeLoop 的实验对照分为六类。主文必须覆盖 A-D，E-F 至少作为补充或真实系统参考。

SafeLoop uses six baseline families. A-D are required in the main paper; E-F are
recommended for appendix or system-reference experiments.

### A. Compute Bounds

```text
full recurrent depth:
  quality and latency upper reference

fixed depth 1..T:
  non-adaptive compute budget frontier

oracle earliest correct depth:
  best possible post-hoc compute bound, not deployable
```

### B. Confidence Baselines

```text
entropy threshold
top-1 probability threshold
top-1/top-2 margin threshold
confidence-only learned predictor
```

这些 baseline 回答：“SafeLoop 是否真的超过普通 confidence 早停？”

These baselines answer whether SafeLoop improves over ordinary confidence-based
halting.

### C. Looped-LM Native Baselines

```text
model built-in early_exit_threshold when available
hidden_delta-only threshold
fixed recurrent-depth budget scheduler
```

这些 baseline 回答：“SafeLoop 是否超过模型自带退出机制和最简单的 loop-state heuristic？”

These baselines answer whether SafeLoop beats native looped-model exits and the
simplest loop-state heuristics.

### D. Learned Risk Controllers

```text
confidence_only predictor
hidden_lite predictor
dynamics_only predictor
dynamics_confidence predictor
request-level sequence controller
empirical calibration
UCB calibration
```

这部分是主方法的递进消融。

This family is the progressive ablation of the main method.

### E. Training-Time Baselines

```text
standard continued pretraining
loop-depth supervision / exit-head training
LayerSkip-style early-exit training when applicable
SafeLoop training-time variant
```

这部分回答：“训练阶段是否能系统性改善 risk-compute frontier？”

This family answers whether training-time intervention systematically improves
the risk-compute frontier.

### F. System Reference Baselines

```text
full-depth serving latency
speculative decoding reference
batch-size and sequence-length latency grid
```

Speculative decoding 是正交 serving 参照，不作为同类方法直接胜负判定。

Speculative decoding is an orthogonal serving reference rather than a direct
same-category competitor.

---

## 4. 主方法 / Main Method

### 4.1 Primary Feature Sets

主结果只允许使用：

Primary results should use only:

```text
hidden_lite:
  hidden_delta, residual_novelty, relative_depth

dynamics_only:
  hidden_delta, logit_delta, residual_novelty,
  depth_attention_stability, depth_attention_shift, relative_depth

dynamics_confidence:
  dynamics_only + entropy/top1/margin
```

辅助上界：

Auxiliary upper bound:

```text
full_with_groups_aux:
  all features including token/stage group metadata
```

`full_with_groups_aux` 只能说明“如果给显式任务组信息，上限能到哪里”，不能作为主论文贡献。

`full_with_groups_aux` only shows an upper bound with explicit task-group
metadata and must not be the main contribution.

### 4.2 Token-Level Controller

命令：

Command:

```bash
python experiments/run_frontier.py \
  --config configs/experiments/frontier.json
```

输出：

Output:

```text
runs/frontier/frontier.json
```

### 4.3 Request-Level Controller

命令：

Command:

```bash
python experiments/run_sequence_frontier.py \
  --config configs/experiments/sequence_frontier.json
```

输出：

Output:

```text
runs/sequence_frontier/sequence_frontier.json
```

request-level label：

```text
request_degradation(depth) = any token at this depth causes task degradation
```

---

## 5. 统计计划 / Statistical Plan

### 5.1 Bootstrap CI

命令：

Command:

```bash
python experiments/run_bootstrap_frontier.py \
  --config configs/experiments/bootstrap_frontier.json
```

输出：

Output:

```text
runs/bootstrap_frontier/bootstrap_frontier.json
```

正式实验要求：

Formal experiments require:

```text
at least 3 seeds or split salts
bootstrap 95% CI
paired comparison when comparing two controllers
report number of requests and tokens
```

### 5.2 Sample Size Guidance

```text
per high-risk group: hundreds to thousands of effective examples
request-level tasks: at least hundreds of paired requests
synthetic_mixed_control_60: smoke/mechanism only
```

---

## 6. Scaling-Law Analysis

### 6.1 Residual Novelty Decay

命令：

Command:

```bash
python experiments/run_depth_scaling_analysis.py \
  --config configs/experiments/depth_scaling_mock.json
```

拟合：

Fit:

```text
residual_novelty(depth) = a * exp(-b * depth) + c
```

报告：

Report:

```text
a, b, c, mse
c vs first_safe_depth correlation
decay rate b by task / risk group
```

### 6.2 Paper-Level Scaling Grid

真实论文版本需要网格：

Paper-level version needs a grid over:

```text
training loop budget
training token budget
test maximum recurrent depth
risk tolerance epsilon
task family
```

目标：

Goal:

```text
fit best_exit_depth(epsilon, train_loop_budget, task_family)
```

---

## 7. TinyLoopLM 机制验证 / TinyLoopLM Mechanism Study

TinyLoopLM 是 controlled mechanism model，不替代真实模型。

TinyLoopLM is a controlled mechanism model and does not replace real models.

命令：

Commands:

```bash
python experiments/train_tiny_loop_lm.py \
  --config configs/experiments/train_tiny_loop_lm.json

python experiments/run_pretraining_frontier_comparison.py \
  --config configs/experiments/tiny_loop_lm_frontier_comparison.json
```

报告：

Report:

```text
standard_vs_halt_aware frontier
fixed-depth risk before / after
training curve
which groups improve or fail
```

---

## 8. 真实任务计划 / Real Task Plan

### 8.1 Math

```text
GSM8K
MATH500
metric: exact final answer correctness
risk: correct -> incorrect degradation and absolute gold correctness
```

### 8.2 Code

```text
HumanEval / HumanEval-lite
metric: sandboxed pass@1
risk: pass -> fail and absolute pass/fail
```

当前仓库的 HumanEval evaluator 只是 smoke test，不是论文级 evaluator。

The current HumanEval evaluator is a smoke test, not a paper-level evaluator.

### 8.3 Tool / JSON

```text
schema validity
exact argument accuracy
mock and real executable tool success
```

### 8.4 RAG / Agentic RAG

```text
entity correctness
citation correctness
tool-call success
final answer correctness
```

---

## 9. 真实模型计划 / Real Model Plan

必须先跑：

Must run first:

```bash
python experiments/validate_depth_control.py \
  --config configs/experiments/collect_traces_ouro_1_4b.json \
  --output-dir runs/ouro_1_4b_depth_control
```

然后：

Then:

```bash
python experiments/collect_traces.py \
  --config configs/experiments/collect_traces_ouro_1_4b.json

python experiments/run_task_evaluation.py \
  --trace runs/ouro_1_4b_traces/teacher_forced.jsonl \
  --output-dir runs/ouro_1_4b_task_eval

python experiments/run_frontier.py \
  --trace runs/ouro_1_4b_task_eval/teacher_forced.evaluated.jsonl \
  --feature-sets hidden_lite,dynamics_only,dynamics_confidence,full_with_groups_aux \
  --label-type task_degradation \
  --calibration-method ucb \
  --output-dir runs/ouro_1_4b_frontier
```

必须对比：

Must compare against:

```text
fixed depth
entropy/top1/margin threshold
confidence-only learned predictor
hidden-only SafeLoop
full_with_groups_aux
Ouro built-in early_exit_threshold
```

---

## 10. Latency Plan

正式论文不能用 avg_depth 代替速度。

Paper results must not use avg_depth as a substitute for speed.

必须报告：

Must report:

```text
loop step latency
feature extraction overhead
risk-head overhead
LM-head overhead
CPU sync overhead
batch compaction overhead
tokens/sec
ms/token
memory footprint
```

---

## 11. 最低投稿门槛 / Minimum Submission Gate

```text
1. hidden-only SafeLoop improves over confidence-only on real tasks
2. request-level controller improves high-cost task risk frontier
3. bootstrap CI / multi-seed stability is reported
4. Ouro-1.4B and Ouro-2.6B real results are included
5. HumanEval uses sandboxed pass@1
6. real RAG/tool tasks use executable evaluators
7. measured latency shows net speedup
8. scaling analysis connects depth dynamics to safe exit depth
```

---

## 12. 当前代码覆盖 / Current Code Coverage

已实现：

Implemented:

```text
hidden-only feature sets
full_with_groups_aux as auxiliary feature set
token-level frontier
request-level sequence frontier
bootstrap CI for a frontier point
residual novelty decay fitting
TinyLoopLM controlled training
UCB calibration
mock task degradation pipeline
```

仍需完成：

Still required:

```text
real large-model traces
real benchmark datasets
sandboxed code evaluator
real agentic RAG evaluator
GPU wall-clock profiling
multi-seed large-sample study
Ouro continued pretraining / LoRA halting head
```
