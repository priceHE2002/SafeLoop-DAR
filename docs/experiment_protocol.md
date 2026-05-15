# 实验协议 / Experiment Protocol

本项目将实验分成两种 trace 模式：`teacher-forced` 和 `free-generation`。

This project separates experiments into two trace modes: `teacher-forced` and
`free-generation`.

## Teacher-Forced Trace / 教师强制 Trace

Teacher-forced trace 固定前缀，在同一个 token 位置比较不同 depth 的候选输出。

Teacher-forced traces keep prefixes fixed and compare candidate outputs from
different depths at the same token position.

用途 / Used for:

- 构造 safe-exit 标签 / safe-exit label construction
- 训练风险预测器 / risk predictor training
- 特征消融实验 / feature ablation
- 风险校准 / calibration
- 分析 depth signal 是否能预测继续计算收益 / analyzing whether depth signals predict marginal gain

这样做可以避免自回归错误级联污染 token-level 分析。

This avoids contaminating token-level analysis with autoregressive error
cascades.

## Free-Generation Trace / 自由生成 Trace

Free-generation trace 让模型真实生成完整答案。

Free-generation traces let the model produce a complete answer.

用途 / Used for:

- 端到端准确率 / end-to-end accuracy
- JSON / 代码格式合法性 / format validity
- 工具调用成功率 / tool-call success
- 输出长度与延迟 / output length and latency
- 真实错误级联分析 / real autoregressive error-cascade analysis

## 风险标签 / Risk Labels

本项目至少支持三类风险标签：

The project supports at least three risk labels:

- **Teacher-consistency risk / 教师一致性风险**  
  提前退出 token 是否不同于 full-depth token。  
  Whether the early-exit token differs from the full-depth token.

- **Task-degradation risk / 任务退化风险**  
  提前退出后最终任务结果是否比 full-depth 更差。  
  Whether early exit degrades the final task output compared with full depth.

- **High-risk token risk / 高风险 token 风险**  
  数字、代码标识符、JSON 参数、工具参数、实体等 token 是否出错。  
  Whether high-risk tokens such as numbers, code identifiers, JSON values, tool
  arguments, and entities are wrong.

## 校准声明 / Calibration Claims

风险保证只在 in-domain calibration / test split 上声明。

Risk guarantees are claimed only for in-domain calibration/test splits.

跨任务、跨模型实验只作为经验泛化结果报告，不声称形式化保证。

Transfer experiments across tasks or models are reported as empirical
generalization, not as formal guarantees.

## 推荐实验顺序 / Recommended Experiment Order

1. 使用 mock adapter 收集 teacher-forced trace。  
   Collect teacher-forced traces with the mock adapter.

2. 训练 safe-exit risk predictor。  
   Train the safe-exit risk predictor.

3. 跑 risk-compute frontier。  
   Build the risk-compute frontier.

4. 跑 calibration validity。  
   Evaluate calibration validity.

5. 接入 Ouro / LoopFormer 等真实模型。  
   Connect real models such as Ouro and LoopFormer.

