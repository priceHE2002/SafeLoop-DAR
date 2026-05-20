# 论文计划 / Paper Plan

## 标题 / Title

**SafeLoop: Risk-Calibrated Test-Time Scaling for Looped Language Models**

中文题目：

**SafeLoop：面向循环语言模型的风险校准 Test-Time Scaling**

## 核心主张 / Main Claim

循环语言模型的 test-time compute 不应该只由固定 recurrent depth 或输出置信度决定，而应该结合：

- 深度状态是否收敛；
- 新一轮循环是否仍然提供残差新信息；
- 当前 token 和生成阶段的错误风险是否可控。

Test-time compute in looped LMs should not be determined only by fixed
recurrent depth or output confidence. It should combine:

- whether depth states have converged;
- whether the next recurrent step still adds residual novelty;
- whether token/stage-specific error risk is calibrated and acceptable.

## 主要证据 / Primary Evidence

1. **残差新颖性与深度注意力稳定性优于单纯 confidence 特征。**  
   Residual novelty and depth attention stability predict safe halting better
   than entropy / margin-only confidence baselines.

2. **风险校准的 test-time scaling 可以在控制 in-domain 任务退化风险的同时降低平均循环深度。**
   Risk-calibrated test-time scaling controls in-domain task degradation risk
   while reducing average loop depth.

3. **刹车系统开销必须低于节省的 loop depth，hidden-only fast path 是推荐实现路径。**
   Controller overhead must be lower than saved loop depth; the hidden-only fast
   path is the recommended implementation path.

4. **Token / 阶段感知校准可以降低高代价错误。**
   Token/stage-aware calibration reduces high-cost errors for math, code,
   JSON/tool calls, and retrieved entities.

5. **深度状态信号可以有限迁移到 Ouro 以外的循环或深度自适应模型。**
   Depth-state signals transfer beyond Ouro to other looped or depth-adaptive
   models in a limited and explicitly qualified sense.

## 论文边界 / Scope

本文不声称：

- 完全解释 Ouro 内部真实机制；
- 将 loop depth 与 ordinary Transformer layer depth 视为完全等价；
- 在跨域设置下提供严格校准保证；
- 在没有专门 serving runtime 的情况下必然带来 wall-clock 加速。

This paper does not claim:

- to fully explain the internal mechanism of Ouro;
- that recurrent loop depth and ordinary Transformer layer depth are identical;
- formal calibration guarantees under cross-domain distribution shift;
- guaranteed wall-clock speedup without specialized serving support.

本文真正要证明的是：

> 深度状态稳定性与残差新颖性是预测剩余风险的有效信号；结合风险校准后，可以在可控任务退化风险下降低循环语言模型的平均计算深度。

The central claim is:

> Depth-state stability and residual novelty are useful predictors of residual
> risk; with risk calibration, they can reduce average recurrent computation
> under controlled task-degradation risk.
