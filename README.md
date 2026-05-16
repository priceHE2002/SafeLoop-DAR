# Think Less, Think Safely

**中文题目：少想一点，但安全退出：面向循环语言模型的风险校准动态停止机制**
**English Title: Risk-Calibrated Early Halting for Looped Language Models**

SafeLoop-DAR 是一个面向循环语言模型与深度自适应语言模型的研究代码库。项目目标不是简单调节
Ouro 的 `early_exit_threshold`，而是系统研究：

> 当前 token 什么时候可以在风险可控的前提下停止继续循环计算？

SafeLoop-DAR is a research codebase for looped and depth-adaptive language
models. The goal is not to merely tune Ouro's `early_exit_threshold`, but to
study:

> When can the current token safely stop recurrent computation under calibrated
> risk?

---

## 1. 研究目标 / Research Goal

本项目将动态早停建模为一个联合决策问题：

```text
是否继续计算 =
    输出置信度是否足够？
  + 深度状态是否稳定？
  + 新一轮循环是否仍有残差新信息？
  + 当前 token / 生成阶段的错误风险是否可接受？
```

The project treats early halting as a joint decision:

```text
continue_or_halt =
    output confidence
  + depth-state stability
  + residual novelty of the next loop step
  + token/stage-specific calibrated risk
```

核心假设 / Core hypotheses:

- **Depth Attention Stability / 深度注意力稳定性**
  如果当前状态对历史 depth states 的聚合分布已经稳定，继续循环的边际收益可能降低。
  If the depth-state aggregation distribution becomes stable, the marginal
  benefit of additional computation may be low.

- **Residual Novelty / 残差新颖性**
  如果当前循环状态相对历史 depth readout 的新增信息很少，当前 token 可能已经收敛。
  If the current loop state adds little new information over the previous
  depth readout, the token may have converged.

- **Risk-Calibrated Halting / 风险校准动态停止**
  数字、代码标识符、JSON 参数、工具参数、引用实体等高风险 token 应该比普通文本更保守。
  High-risk tokens such as numbers, code identifiers, JSON values, tool
  arguments, and retrieved entities should halt more conservatively than
  ordinary text.

---

## 2. 当前代码覆盖的实验主张 / Claims Covered by the Code

1. **Safe-exit prediction / 安全退出预测**
   比较 entropy、margin、logit stability、hidden delta、residual novelty、
   depth attention stability 对 safe-exit label 的预测能力。

2. **Risk-compute frontier / 风险-计算前沿**
   比较 fixed depth、oracle halting、confidence-only、DAR no token-stage、
   SafeLoop-DAR 在不同目标风险下的平均 depth 和错误风险。

3. **Overhead-aware frontier / 开销感知风险-计算前沿**
   将平均 depth 节省进一步换算成估计延迟、刹车系统开销、有效吞吐收益和 break-even 判断。
   Convert average-depth savings into estimated latency, controller overhead,
   effective speedup, and break-even decisions.

4. **Calibration validity / 校准有效性**
   在 in-domain split 上验证目标风险 `ε` 与 empirical risk 的关系。

5. **Token/stage ablation / Token 与阶段消融**
   验证 token/stage features 是否能降低高风险 token 的提前退出错误。

6. **Residual convergence / 残差收敛分析**
   验证 residual novelty 与下一 depth 的 label 改善是否相关。

7. **Cross-model validation / 跨模型验证**
   提供 Ouro、LoopFormer、Base-Loop-EE、TMLT-EE、LayerSkip、LoopTiny/mock 的适配入口。

---

## 3. 项目结构 / Project Layout

```text
SafeLoop-DAR/
├── configs/
│   ├── models/                 # 模型配置 / model configs
│   ├── benchmarks/             # benchmark 样本 / benchmark samples
│   └── experiments/            # 实验配置 / experiment configs
├── docs/
│   ├── paper_plan.md           # 论文主张 / paper plan
│   ├── experiment_plan.md      # 详细实验方案 / detailed experiment plan
│   ├── experiment_protocol.md  # 实验协议 / experiment protocol
│   ├── risk_mitigation.md      # 风险控制 / risk mitigation
│   └── file_index.md           # 文件索引 / file index
├── experiments/
│   ├── collect_traces.py
│   ├── run_signal_prediction.py
│   ├── run_frontier.py
│   ├── run_calibration.py
│   ├── run_token_stage_ablation.py
│   ├── run_residual_convergence.py
│   ├── run_budget_scaling.py
│   ├── run_overhead_frontier.py
│   ├── run_free_generation.py
│   └── run_transfer.py
├── safeloop/
│   ├── adapters/               # mock / Ouro / LoopFormer / LayerSkip adapters
│   ├── tracing/                # teacher-forced 与 free-generation trace
│   ├── features/               # confidence、DAR、token-stage features
│   ├── risk/                   # labels、risk predictor、calibration
│   ├── halting/                # halting policies
│   ├── evaluation/             # metrics、splits、frontier、overhead model
│   ├── controlled/             # LoopTiny-style controlled tasks
│   └── workloads/              # benchmark loaders
└── tests/
```

---

## 4. 安装与环境 / Installation

### 4.1 最小环境：只跑 mock 实验 / Minimal setup for mock experiments

Mock 实验不需要下载模型权重，适合先验证实验管线。

Mock experiments do not download model weights and are suitable for validating
the pipeline first.

```bash
cd /Users/heyuhang/Documents/New\ project/SafeLoop-DAR
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[analysis]"
```

### 4.2 真实模型环境：Ouro / LoopFormer / LayerSkip

Real model experiments require model dependencies:

```bash
python -m pip install -e ".[model,analysis]"
```

建议真实论文实验固定以下信息：

For paper experiments, pin and record:

```text
git commit hash
model revision SHA
Python / torch / transformers version
CUDA version
GPU type
random seed
benchmark config
experiment command
```

---

## 5. 最小可复现实验 / Minimal Reproducible Pipeline

下面命令只使用 deterministic mock adapter，不加载真实模型。

The commands below use only the deterministic mock adapter and do not load real
models.

### Step 1: 收集 teacher-forced trace

```bash
python experiments/collect_traces.py \
  --config configs/experiments/collect_traces_mock.json
```

输出 / Outputs:

```text
runs/mock_traces/teacher_forced.jsonl
runs/mock_traces/run_config.json
runs/mock_traces/summary.json
```

### Step 2: 训练并评估 safe-exit risk predictor

```bash
python experiments/run_signal_prediction.py \
  --config configs/experiments/signal_prediction.json
```

输出 / Outputs:

```text
runs/signal_prediction/predictor.json
runs/signal_prediction/metrics.json
runs/signal_prediction/feature_rows.jsonl
```

### Step 3: 构建 risk-compute frontier

```bash
python experiments/run_frontier.py \
  --config configs/experiments/frontier.json
```

输出 / Outputs:

```text
runs/frontier/frontier.json
```

可读表格 / Human-readable table:

```bash
python scripts/plot_frontier.py \
  --frontier runs/frontier/frontier.json
```

### Step 4: 校准有效性

```bash
python experiments/run_calibration.py \
  --trace runs/mock_traces/teacher_forced.jsonl \
  --targets 0.005,0.01,0.02,0.05,0.1 \
  --output-dir runs/calibration
```

输出 / Outputs:

```text
runs/calibration/calibration.json
```

### Step 5: Token / stage 特征消融

```bash
python experiments/run_token_stage_ablation.py \
  --trace runs/mock_traces/teacher_forced.jsonl \
  --output-dir runs/token_stage_ablation
```

输出 / Outputs:

```text
runs/token_stage_ablation/ablation.json
```

### Step 6: Residual novelty 收敛分析

```bash
python experiments/run_residual_convergence.py \
  --trace runs/mock_traces/teacher_forced.jsonl \
  --output-dir runs/residual_convergence
```

输出 / Outputs:

```text
runs/residual_convergence/residual_convergence.json
```

### Step 7: 固定预算 scaling

```bash
python experiments/run_budget_scaling.py \
  --trace runs/mock_traces/teacher_forced.jsonl \
  --output-dir runs/budget_scaling
```

输出 / Outputs:

```text
runs/budget_scaling/budget_scaling.json
```

### Step 8: 开销感知 frontier / Overhead-aware frontier

该步骤读取 Step 3 产生的 `frontier.json`，不重新跑模型，用不同 serving-path 假设估算刹车系统开销是否抵消节省的 loop depth。

This step reads `frontier.json` from Step 3 and does not rerun the model. It
estimates whether the halting controller overhead offsets saved loop depth
under different serving-path assumptions.

```bash
python experiments/run_overhead_frontier.py \
  --config configs/experiments/overhead_frontier.json
```

输出 / Outputs:

```text
runs/overhead_frontier/overhead_frontier.json
```

可读表格 / Human-readable table:

```bash
python scripts/plot_overhead_frontier.py \
  --overhead-frontier runs/overhead_frontier/overhead_frontier.json
```

默认比较三种刹车系统实现假设：

Default controller profiles:

```text
hidden_only_fast_path        # hidden-state features only, GPU-resident
hybrid_exit_only_lm_head     # hidden-state features plus one exit-time LM head
logits_every_depth_cpu_sync  # full-vocab logits and CPU sync at every depth
```

---

## 6. 真实模型实验入口 / Real Model Entrypoints

### 6.1 Ouro-1.4B

```bash
python experiments/collect_traces.py \
  --config configs/experiments/collect_traces_ouro_1_4b.json

python experiments/run_signal_prediction.py \
  --trace runs/ouro_1_4b_traces/teacher_forced.jsonl \
  --output-dir runs/ouro_1_4b_signal

python experiments/run_frontier.py \
  --trace runs/ouro_1_4b_traces/teacher_forced.jsonl \
  --output-dir runs/ouro_1_4b_frontier
```

### 6.2 Ouro-2.6B

```bash
python experiments/collect_traces.py \
  --config configs/experiments/collect_traces_ouro_2_6b.json

python experiments/run_frontier.py \
  --trace runs/ouro_2_6b_traces/teacher_forced.jsonl \
  --output-dir runs/ouro_2_6b_frontier
```

### 6.3 LoopFormer

```bash
python experiments/collect_traces.py \
  --config configs/experiments/collect_traces_loopformer.json

python experiments/run_frontier.py \
  --trace runs/loopformer_traces/teacher_forced.jsonl \
  --output-dir runs/loopformer_frontier
```

### 6.4 Base-Loop-EE / TMLT-EE

```bash
python experiments/collect_traces.py \
  --config configs/experiments/collect_traces_base_loop_ee.json

python experiments/collect_traces.py \
  --config configs/experiments/collect_traces_tmlt_ee.json
```

### 6.5 Controlled LoopTiny-style benchmark

```bash
python experiments/collect_traces.py \
  --config configs/experiments/collect_traces_looptiny.json

python experiments/run_residual_convergence.py \
  --trace runs/looptiny_traces/teacher_forced.jsonl \
  --output-dir runs/looptiny_residual_convergence
```

---

## 7. 实验输出解释 / Output Interpretation

- `teacher_forced.jsonl`
  每行是一个 token position 的全 depth trace。
  Each line stores all depth states for one token position.

- `feature_rows.jsonl`
  每行是一个 token-depth pair 的特征与 label。
  Each line stores features and labels for one token-depth pair.

- `metrics.json`
  风险预测器的 train/test 指标。
  Train/test metrics for risk prediction.

- `frontier.json`
  不同方法和目标风险下的平均 depth 与风险。
  Average depth and risk under different methods and target risks.

- `overhead_frontier.json`
  在不同刹车系统开销假设下的有效延迟、净节省、break-even 和 speedup。
  Effective latency, net saving, break-even status, and speedup under different
  controller overhead assumptions.

- `calibration.json`
  target risk、empirical risk、coverage 和 group thresholds。
  Target risk, empirical risk, coverage, and group thresholds.

- `ablation.json`
  confidence、DAR、token-stage 特征消融结果。
  Feature ablation results.

---

## 8. 复现注意事项 / Reproducibility Notes

- 论文实验前请在 `configs/models/*.json` 中固定 `revision`。
  Pin `revision` in `configs/models/*.json` before paper experiments.

- `runs/` 默认被 `.gitignore` 忽略，避免误传大量 trace。
  `runs/` is ignored by default to avoid committing large traces.

- Cross-domain / cross-model 结果只报告 empirical transfer，不声明形式化校准保证。
  Cross-domain / cross-model results are empirical transfer results, not formal
  calibration guarantees.

- 真实模型的 depth-control 字段依赖 remote code，第一次接入时需要用小样本 smoke test 验证。
  Real model depth-control fields depend on remote code; validate with a small
  smoke test before large runs.

- 开销感知实验中的 `loop_step_ms` 和 profile 参数是 serving-path 假设；论文实验应在目标 GPU 上实测后替换默认值。
  `loop_step_ms` and profile parameters in the overhead experiment are
  serving-path assumptions; replace defaults with measurements on the target GPU
  for paper experiments.

---

## 9. 推荐阅读 / Recommended Docs

- [详细实验方案 / Detailed Experiment Plan](docs/experiment_plan.md)
- [实验协议 / Experiment Protocol](docs/experiment_protocol.md)
- [论文计划 / Paper Plan](docs/paper_plan.md)
- [风险控制 / Risk Mitigation](docs/risk_mitigation.md)
- [文件索引 / File Index](docs/file_index.md)
