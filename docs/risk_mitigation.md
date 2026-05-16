# 风险控制 / Risk Mitigation

本文方向有明确潜力，但也存在若干容易被评审质疑的风险。下面列出主要风险和应对方式。

This research direction has clear potential, but it also has several risks that
reviewers may challenge. The main risks and mitigations are listed below.

## 1. Depth Attention Probe 可能被认为是后验解释

**风险 / Risk:**  
Depth attention probe 可能被认为是 post-hoc feature engineering，而不是模型内部真实机制。

The depth attention probe may be viewed as post-hoc feature engineering rather
than a faithful explanation of the base model internals.

**应对 / Mitigation:**  
明确将其定位为 **halting feature extractor**，而不是对 Ouro 内部机制的解释。

Position it explicitly as a **halting feature extractor**, not as an explanation
of Ouro's internal mechanism.

必须做消融：

Required ablation:

```text
confidence only
confidence + residual novelty
confidence + depth attention stability
SafeLoop-DAR full
```

## 2. Full-depth 输出不一定正确

**风险 / Risk:**  
`y_full` 只是满循环输出，不一定等于真实正确答案。

`y_full` is only the full-depth output and may not be the ground-truth answer.

**应对 / Mitigation:**  
同时报告三类指标：

Report three metrics separately:

```text
teacher-consistency risk
task-degradation risk
ground-truth task quality
```

## 3. Loop Depth 与 Layer Depth 不完全等价

**风险 / Risk:**  
Ouro 的 recurrent depth 和 LayerSkip / 普通 Transformer 的 layer depth 语义不同。

Ouro's recurrent depth and LayerSkip / ordinary Transformer layer depth have
different semantics.

**应对 / Mitigation:**  
跨模型实验只验证 depth-state signal 的预测价值，不声称两类 depth 机制等价。

Cross-model experiments only test whether depth-state signals are predictive; do
not claim mechanistic equivalence between recurrent depth and layer depth.

## 4. 校准假设在分布漂移下可能失效

**风险 / Risk:**  
Calibration 通常依赖 calibration set 和 test set 近似同分布。跨任务、跨模型时该假设可能不成立。

Calibration usually assumes that calibration and test data are approximately
exchangeable. This may fail across tasks or models.

**应对 / Mitigation:**  
只在 in-domain setting 中声明 calibration validity；cross-domain 结果只作为 empirical transfer 报告。

Claim calibration validity only in in-domain settings; report cross-domain
results as empirical transfer.

## 5. 平均循环步数下降不一定等于真实延迟下降

**风险 / Risk:**  
Probe 计算、trace hook、batch 碎片化和 kernel launch overhead 可能抵消 wall-clock speedup。

Probe computation, trace hooks, batch fragmentation, and kernel launch overhead
may offset wall-clock speedups.

**应对 / Mitigation:**  
主论文卖点放在 adaptive computation efficiency，而不是绝对 serving speedup。

Frame the main contribution as adaptive computation efficiency rather than
guaranteed serving speedup.

必须分开报告：

Report separately:

```text
average depth
estimated compute saving
measured latency
probe overhead
controller overhead
effective speedup
break-even rate
```

本仓库将该风险落实为 E10 `Overhead-Aware Risk-Compute Frontier`：先用可配置 profile 估算
hidden-only、hybrid、logits-every-depth 三种路径，再在正式论文实验中替换为目标 GPU 的实测开销。

The repository turns this risk into E10, `Overhead-Aware Risk-Compute Frontier`:
first estimate hidden-only, hybrid, and logits-every-depth paths with
configurable profiles, then replace the defaults with measured costs on the
target GPU for paper experiments.

## 6. Benchmark 过多导致主线分散

**风险 / Risk:**  
如果同时铺开太多 benchmark，论文可能显得主张过宽、每个实验不够深入。

Too many benchmarks can make the paper feel unfocused and shallow.

**应对 / Mitigation:**  
主实验聚焦：

Focus main experiments on:

```text
Ouro-1.4B / Ouro-2.6B
LoopFormer / Base-Loop-EE
LoopTiny controlled study
Math / Code / Tool-Agent representative tasks
```

其他模型如 MELT、PLT、Attractor Models 可作为 optional appendix。

Other models such as MELT, PLT, and Attractor Models can be optional appendix
studies.
