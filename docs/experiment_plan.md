# 详细实验方案 / Detailed Experiment Plan

本文档给出 SafeLoop-DAR 的完整实验设计。每组实验都包含：目的、输入、命令、输出文件、指标和预期结论。

This document provides the full experimental plan for SafeLoop-DAR. Each
experiment includes its goal, inputs, commands, output files, metrics, and
expected conclusions.

---

## 0. 总体实验矩阵 / Overall Experiment Matrix

### 0.1 模型层级 / Model Levels

| 层级 | 模型 | 目的 |
|---|---|---|
| Primary | Ouro-1.4B, Ouro-2.6B | 主验证：真实 Looped LM 上的 dynamic halting |
| Secondary | LoopFormer, Base-Loop-EE, TMLT-EE | 非 Ouro 循环模型泛化 |
| Transfer | LayerSkip-Llama2-7B | layer-wise depth 的辅助泛化 |
| Controlled | MockLoop, LoopTiny-style synthetic | 机制验证与管线 smoke test |

| Level | Models | Purpose |
|---|---|---|
| Primary | Ouro-1.4B, Ouro-2.6B | Main validation on real looped LMs |
| Secondary | LoopFormer, Base-Loop-EE, TMLT-EE | Generalization beyond Ouro |
| Transfer | LayerSkip-Llama2-7B | Auxiliary layer-wise depth transfer |
| Controlled | MockLoop, LoopTiny-style synthetic | Mechanism study and smoke tests |

### 0.2 任务层级 / Task Levels

当前仓库内置轻量 benchmark：

Built-in lightweight benchmark:

```text
configs/benchmarks/looped_core.json
configs/benchmarks/synthetic_khop.json
```

论文级扩展建议：

Recommended paper-level extensions:

```text
Math: GSM8K, MATH500
Reasoning: BBH subset
Code: HumanEval, LiveCodeBench-lite
Tool/Agent: BFCL, Agentic RAG
RAG: HotpotQA / 2WikiMultihopQA optional
```

---

## 1. 实验 E0：Mock Smoke Test

### 目的 / Goal

验证整个实验管线是否可以不依赖真实模型权重运行：

Validate the full pipeline without loading real model weights:

```text
trace collection -> feature extraction -> risk prediction -> calibration -> frontier
```

### 输入 / Input

```text
configs/models/mock_loop.json
configs/benchmarks/looped_core.json
configs/experiments/collect_traces_mock.json
```

### 命令 / Commands

```bash
python experiments/collect_traces.py \
  --config configs/experiments/collect_traces_mock.json

python experiments/run_signal_prediction.py \
  --config configs/experiments/signal_prediction.json

python experiments/run_frontier.py \
  --config configs/experiments/frontier.json

python experiments/run_calibration.py \
  --trace runs/mock_traces/teacher_forced.jsonl \
  --output-dir runs/calibration
```

### 输出 / Outputs

```text
runs/mock_traces/teacher_forced.jsonl
runs/signal_prediction/metrics.json
runs/frontier/frontier.json
runs/calibration/calibration.json
```

### 检查点 / Checks

- `teacher_forced.jsonl` 行数应大于 0。
- `metrics.json` 应包含 `train` 和 `test`。
- `frontier.json` 应包含 `fixed_depth_*`、`oracle_halting`、`safeloop_dar`。
- `calibration.json` 应包含多个 `target_risk`。

---

## 2. 实验 E1：Loop Depth Characterization

### 目的 / Goal

证明不同任务、不同 token 类型确实需要不同循环深度。

Show that different tasks and token types require different loop depths.

### 关键问题 / Questions

- 普通文本、标点是否更早稳定？
- 数字、代码标识符、JSON 参数、实体是否需要更深计算？
- 高风险 token 的 premature-exit error 是否更高？

### 命令 / Commands

Mock:

```bash
python experiments/collect_traces.py \
  --config configs/experiments/collect_traces_mock.json

python experiments/run_budget_scaling.py \
  --trace runs/mock_traces/teacher_forced.jsonl \
  --output-dir runs/mock_budget_scaling
```

Ouro-1.4B:

```bash
python experiments/collect_traces.py \
  --config configs/experiments/collect_traces_ouro_1_4b.json

python experiments/run_budget_scaling.py \
  --trace runs/ouro_1_4b_traces/teacher_forced.jsonl \
  --output-dir runs/ouro_1_4b_budget_scaling
```

### 指标 / Metrics

```text
average required depth
fixed-depth teacher-consistency risk
high-risk token error
token_type -> avg depth
task -> depth distribution
```

### 预期结论 / Expected Conclusion

高风险 token 的稳定深度通常更高；固定浅层 depth 会在数字、代码、JSON 参数和实体上产生更高风险。

High-risk tokens usually require deeper states; fixed shallow depth should
produce higher errors on numbers, code, JSON values, and entities.

---

## 3. 实验 E2：Depth Signal Prediction

### 目的 / Goal

验证 residual novelty 和 depth attention stability 是否比 confidence-only 特征更能预测 safe exit。

Test whether residual novelty and depth attention stability predict safe exit
better than confidence-only features.

### 命令 / Commands

```bash
python experiments/run_signal_prediction.py \
  --trace runs/mock_traces/teacher_forced.jsonl \
  --output-dir runs/mock_signal_prediction

python experiments/run_token_stage_ablation.py \
  --trace runs/mock_traces/teacher_forced.jsonl \
  --output-dir runs/mock_token_stage_ablation
```

真实模型：

Real model:

```bash
python experiments/run_signal_prediction.py \
  --trace runs/ouro_1_4b_traces/teacher_forced.jsonl \
  --output-dir runs/ouro_1_4b_signal_prediction

python experiments/run_token_stage_ablation.py \
  --trace runs/ouro_1_4b_traces/teacher_forced.jsonl \
  --output-dir runs/ouro_1_4b_token_stage_ablation
```

### 对比特征 / Feature Sets

代码中 `run_token_stage_ablation.py` 目前比较：

The current ablation compares:

```text
confidence
confidence_hidden
dar
dar_token_stage
```

### 指标 / Metrics

```text
AUROC
Brier score
ECE
label rate
average risk score
```

### 预期结论 / Expected Conclusion

`dar` 和 `dar_token_stage` 应该优于 `confidence`，说明深度信息流特征具有额外预测价值。

`dar` and `dar_token_stage` should outperform `confidence`, showing that
depth-state signals add predictive value.

---

## 4. 实验 E3：Risk-Compute Frontier

### 目的 / Goal

验证 SafeLoop-DAR 是否能在相同风险下降低平均 depth，或在相同平均 depth 下提高质量。

Evaluate whether SafeLoop-DAR reduces average depth at the same risk, or improves
quality at the same average depth.

### 命令 / Commands

Mock:

```bash
python experiments/run_frontier.py \
  --trace runs/mock_traces/teacher_forced.jsonl \
  --output-dir runs/mock_frontier \
  --targets 0.005,0.01,0.02,0.05,0.1

python scripts/plot_frontier.py \
  --frontier runs/mock_frontier/frontier.json
```

Ouro:

```bash
python experiments/run_frontier.py \
  --trace runs/ouro_1_4b_traces/teacher_forced.jsonl \
  --output-dir runs/ouro_1_4b_frontier \
  --targets 0.005,0.01,0.02,0.05,0.1
```

### 对比方法 / Compared Methods

```text
fixed_depth_1 ... fixed_depth_T
oracle_halting
entropy_only
margin_only
confidence
dar_no_token_stage
safeloop_dar
```

### 指标 / Metrics

```text
avg_depth
teacher-consistency risk
target_risk
group thresholds
tokens
```

### 预期结论 / Expected Conclusion

在相同 `target_risk` 下，`safeloop_dar` 的 `avg_depth` 应低于 confidence baseline，且风险不显著超过目标。

At the same `target_risk`, `safeloop_dar` should use lower `avg_depth` than
confidence baselines without substantially exceeding the target risk.

---

## 5. 实验 E4：Risk Calibration Validity

### 目的 / Goal

验证目标风险与测试集 empirical risk 是否匹配。

Check whether target risk matches empirical test risk.

### 命令 / Commands

```bash
python experiments/run_calibration.py \
  --trace runs/mock_traces/teacher_forced.jsonl \
  --targets 0.005,0.01,0.02,0.05,0.1 \
  --output-dir runs/mock_calibration
```

Ouro:

```bash
python experiments/run_calibration.py \
  --trace runs/ouro_1_4b_traces/teacher_forced.jsonl \
  --targets 0.005,0.01,0.02,0.05,0.1 \
  --output-dir runs/ouro_1_4b_calibration
```

### 指标 / Metrics

```text
target_risk
calibration.coverage
calibration.risk
test.coverage
test.risk
thresholds by group
```

### 重要声明 / Important Claim Boundary

只在 in-domain split 中讨论 calibration validity。跨任务、跨模型只报告 empirical transfer。

Calibration validity is discussed only in in-domain splits. Cross-task and
cross-model results are empirical transfer results.

---

## 6. 实验 E5：Token / Stage Risk Control

### 目的 / Goal

验证 token/stage-aware 特征是否能降低高风险 token 的提前退出错误。

Evaluate whether token/stage-aware features reduce premature-exit errors for
high-risk tokens.

### 命令 / Commands

```bash
python experiments/run_token_stage_ablation.py \
  --trace runs/ouro_1_4b_traces/teacher_forced.jsonl \
  --output-dir runs/ouro_1_4b_token_stage_ablation
```

也可以专门评估高风险标签：

High-risk label evaluation:

```bash
python experiments/run_token_stage_ablation.py \
  --trace runs/ouro_1_4b_traces/teacher_forced.jsonl \
  --label-type high_risk_teacher \
  --output-dir runs/ouro_1_4b_high_risk_ablation
```

### 高风险组 / High-Risk Groups

```text
math_final_number
number
code_identifier
json_value
tool_argument
retrieved_entity
citation_entity
```

### 指标 / Metrics

```text
high-risk label AUROC
high-risk Brier
high-risk ECE
coverage under group thresholds
```

---

## 7. 实验 E6：Residual Convergence Bench

### 目的 / Goal

验证 residual novelty 是否和下一 depth 的收益相关。

Test whether residual novelty correlates with next-depth gain.

### 命令 / Commands

```bash
python experiments/run_residual_convergence.py \
  --trace runs/mock_traces/teacher_forced.jsonl \
  --output-dir runs/mock_residual_convergence

python experiments/run_residual_convergence.py \
  --trace runs/ouro_1_4b_traces/teacher_forced.jsonl \
  --output-dir runs/ouro_1_4b_residual_convergence
```

### 指标 / Metrics

```text
pairs
pearson_novelty_next_gain
avg_novelty
avg_next_gain
```

### 预期结论 / Expected Conclusion

如果 residual novelty 是有效信号，`pearson_novelty_next_gain` 应为正，低 novelty token 的继续计算收益应更低。

If residual novelty is useful, `pearson_novelty_next_gain` should be positive,
and low-novelty tokens should have lower next-step gain.

---

## 8. 实验 E7：LoopFormer / Base-Loop-EE / TMLT-EE 泛化

### 目的 / Goal

验证 SafeLoop-DAR 不是 Ouro-specific thresholding trick。

Verify that SafeLoop-DAR is not an Ouro-specific thresholding trick.

### 命令 / Commands

LoopFormer:

```bash
python experiments/collect_traces.py \
  --config configs/experiments/collect_traces_loopformer.json

python experiments/run_frontier.py \
  --trace runs/loopformer_traces/teacher_forced.jsonl \
  --output-dir runs/loopformer_frontier
```

Base-Loop-EE:

```bash
python experiments/collect_traces.py \
  --config configs/experiments/collect_traces_base_loop_ee.json

python experiments/run_frontier.py \
  --trace runs/base_loop_ee_traces/teacher_forced.jsonl \
  --output-dir runs/base_loop_ee_frontier
```

TMLT-EE:

```bash
python experiments/collect_traces.py \
  --config configs/experiments/collect_traces_tmlt_ee.json

python experiments/run_frontier.py \
  --trace runs/tmlt_ee_traces/teacher_forced.jsonl \
  --output-dir runs/tmlt_ee_frontier
```

### 指标 / Metrics

```text
avg_depth
risk
frontier shape
SafeLoop-DAR vs confidence baselines
```

### 解释边界 / Interpretation Boundary

LoopFormer / Base-Loop-EE 的 depth 机制不一定和 Ouro 完全相同；这里只验证 depth-state signal 是否跨循环模型族仍有预测价值。

LoopFormer / Base-Loop-EE depth is not necessarily mechanistically identical to
Ouro depth. This experiment only tests whether depth-state signals remain
predictive across looped model families.

---

## 9. 实验 E8：Controlled LoopTiny-style Mechanism Study

### 目的 / Goal

用可控 synthetic task 验证机制，而不是追求大模型性能。

Use controlled synthetic tasks for mechanism validation rather than headline
model performance.

### 命令 / Commands

```bash
python experiments/collect_traces.py \
  --config configs/experiments/collect_traces_looptiny.json

python experiments/run_residual_convergence.py \
  --trace runs/looptiny_traces/teacher_forced.jsonl \
  --output-dir runs/looptiny_residual_convergence

python experiments/run_frontier.py \
  --trace runs/looptiny_traces/teacher_forced.jsonl \
  --output-dir runs/looptiny_frontier
```

### 关注点 / Focus

```text
known reasoning depth
required depth vs predicted exit depth
residual novelty decreases after convergence
oracle gap
```

---

## 10. 实验 E9：Free-Generation End-to-End

### 目的 / Goal

验证真实自回归生成下的端到端行为。

Evaluate end-to-end behavior under real autoregressive generation.

### 命令 / Commands

Mock:

```bash
python experiments/run_free_generation.py \
  --config configs/experiments/free_generation_mock.json
```

真实模型可复制 collect config，把 `mode` 改为 `free_generation`。

For real models, copy a collect config and set `mode` to `free_generation`.

### 输出 / Outputs

```text
runs/*/free_generation.jsonl
runs/*/summary.json
```

### 指标 / Metrics

当前代码主要生成 trace；后续论文级评估应补充：

The current code emits traces. Paper-level evaluation should add:

```text
task accuracy
pass@1
JSON validity
tool success
answer EM/F1
output length
latency
```

---

## 11. 实验 E10：Overhead-Aware Risk-Compute Frontier

### 目的 / Goal

验证动态退出不只是减少平均循环深度，还要在考虑刹车系统自身开销后真正降低每 token 延迟。

Verify that dynamic halting reduces per-token latency after accounting for the
controller overhead, not merely average loop depth.

### 核心问题 / Key Questions

- hidden-only 刹车路径是否明显小于一轮 loop 计算？
- 每轮 full-vocab logits + CPU sync 是否会抵消 early halting 收益？
- SafeLoop-DAR 的 `effective_speedup` 是否在主要风险目标下仍大于 1？
- 哪些实现 profile 的 `min_saved_depth_to_break_even` 过高，不适合作为 serving fast path？

- Is the hidden-only controller much cheaper than one loop step?
- Does full-vocab logits plus CPU sync at every depth erase early-halting gains?
- Does SafeLoop-DAR keep `effective_speedup > 1` under major risk targets?
- Which implementation profiles have too high a `min_saved_depth_to_break_even`
  for serving?

### 输入 / Input

该实验读取 E3 产生的 `frontier.json`，不重新运行模型：

This experiment reads `frontier.json` from E3 and does not rerun the model:

```text
runs/frontier/frontier.json
configs/experiments/overhead_frontier.json
```

### 命令 / Commands

Mock:

```bash
python experiments/run_frontier.py \
  --config configs/experiments/frontier.json

python experiments/run_overhead_frontier.py \
  --config configs/experiments/overhead_frontier.json

python scripts/plot_overhead_frontier.py \
  --overhead-frontier runs/overhead_frontier/overhead_frontier.json
```

Ouro-1.4B:

```bash
python experiments/run_frontier.py \
  --trace runs/ouro_1_4b_traces/teacher_forced.jsonl \
  --output-dir runs/ouro_1_4b_frontier

python experiments/run_overhead_frontier.py \
  --frontier runs/ouro_1_4b_frontier/frontier.json \
  --output-dir runs/ouro_1_4b_overhead_frontier \
  --loop-step-ms <measured_loop_step_ms> \
  --full-depth 4
```

### 默认开销 profile / Default Overhead Profiles

```text
hidden_only_fast_path
  每轮只计算 hidden delta / residual novelty / depth stability，状态留在 GPU。
  Hidden-delta, residual-novelty, and depth-stability features only; states stay on GPU.

hybrid_exit_only_lm_head
  每轮用 hidden-only 判断，真正退出时才走一次 LM head。
  Hidden-only decisions at each depth; one LM head call only at exit.

logits_every_depth_cpu_sync
  每轮都算 full-vocab logits 并同步到 CPU，是应避免的反例路径。
  Full-vocab logits and CPU sync at every depth; this is the anti-pattern.
```

### 输出 / Outputs

```text
runs/overhead_frontier/overhead_frontier.json
```

### 指标 / Metrics

```text
baseline_full_depth_ms_per_token
saved_loop_steps
model_compute_ms_per_token
brake_overhead_ms_per_token
total_ms_per_token
gross_saved_ms_per_token
net_saved_ms_per_token
effective_speedup
overhead_loop_step_equivalent
min_saved_depth_to_break_even
is_break_even
overhead_exceeds_one_loop_step
overhead_share
```

### 判定标准 / Decision Rule

动态退出真正值得部署，需要同时满足：

For dynamic halting to be deployment-worthy:

```text
effective_speedup > 1
net_saved_ms_per_token > 0
overhead_loop_step_equivalent < saved_loop_steps
风险指标不显著超过目标风险
```

### 预期结论 / Expected Conclusion

如果实现为 GPU-resident hidden-only fast path，刹车开销应明显小于一轮 loop step，SafeLoop-DAR 的净收益应保持为正。
如果实现为每轮 full-vocab logits + CPU sync，则可能出现 `overhead_exceeds_one_loop_step = true`，说明该实现路径不适合 serving。

With a GPU-resident hidden-only fast path, controller overhead should be far
below one loop step and SafeLoop-DAR should retain positive net savings. With
full-vocab logits plus CPU sync at every depth, `overhead_exceeds_one_loop_step`
may become true, showing that this implementation path is unsuitable for
serving.

---

## 12. 推荐运行顺序 / Recommended Run Order

### P0: 管线验证 / Pipeline validation

```text
E0 mock smoke test
E2 mock signal prediction
E3 mock frontier
E10 mock overhead-aware frontier
E4 mock calibration
```

### P1: 主实验 / Main experiments

```text
Ouro-1.4B trace
Ouro-1.4B signal prediction
Ouro-1.4B frontier
Ouro-1.4B overhead-aware frontier
Ouro-1.4B calibration
Ouro-1.4B token/stage ablation
```

### P2: 扩展实验 / Extension experiments

```text
Ouro-2.6B
LoopFormer
Base-Loop-EE
TMLT-EE
LoopTiny-style controlled study
```

### P3: 论文补强 / Paper strengthening

```text
Free-generation end-to-end metrics
task-level degradation labels
larger benchmark integration
cross-model transfer
runtime overhead analysis
```

---

## 13. 复现记录模板 / Reproducibility Record Template

每次正式实验建议保存以下信息：

For every formal experiment, save:

```text
Date:
Git commit:
Command:
Model:
Model revision:
Dataset / benchmark config:
GPU:
CUDA:
Python:
torch:
transformers:
Seed:
Output directory:
Notes:
```

获取当前 commit：

Get current commit:

```bash
git rev-parse HEAD
```

记录 Python 包：

Record Python packages:

```bash
python -m pip freeze > runs/<run_name>/environment.txt
```

---

## 14. 当前限制 / Current Limitations

- 真实 Ouro / LoopFormer depth-control 字段依赖各自 remote code，需要小样本 smoke test 验证。
- 当前 free-generation 主要收集 trace，任务级 metric 还需要接入具体 benchmark evaluator。
- `task_degradation` label 当前支持 metadata 接口，真实任务需补充 evaluator 写入。
- Cross-domain calibration 不应声明 formal guarantee。
- Overhead-aware frontier 当前默认是估算模型；正式论文需要用目标 GPU 的实测 `loop_step_ms`、LM head 开销和调度开销替换默认 profile。

Current limitations:

- Real Ouro / LoopFormer depth-control fields depend on remote code and require
  smoke testing.
- Free-generation currently focuses on trace collection; task-level metrics need
  benchmark-specific evaluators.
- `task_degradation` labels are supported through metadata, but real tasks need
  evaluators to populate them.
- Cross-domain calibration should not be presented as a formal guarantee.
- The overhead-aware frontier is currently an estimation model; paper experiments
  should replace default profiles with measured `loop_step_ms`, LM-head overhead,
  and scheduler overhead on the target GPU.
