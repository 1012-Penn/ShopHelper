# Claim Grounding R2 开发记录

## 观察到的问题

已有 `rag-quality-v3.1` 对大量引用保留 `uncertain`。字符规则能把完全
相同的句子判为 supported，也能用数字/实体交集发现一部分冲突，但无法
可靠处理同义支持、条件缺失、单位等价和跨版本语义。直接扩大全答案
语义裁判又会增加调用面与误拒风险。

## 定位

典型失败不是“模型回答错”，而是证据判断粒度太粗：

- “售价 299 元”和“标价二百九十九元”没有表面重叠；
- “半年”和“六个月”、“1.5 小时”和“90 分钟”是同一事实；
- v3.1 的数字集合规则会把等价的 24 个月/两年判成 unsupported；
- 只给通用规则、未给商品状态的证据不足以确定适用折扣；
- 其他订单的证据不能证明当前订单，陈旧物流快照也不能证明“今日送达”。

## 方案选择

将任务收敛为单 claim 三态判断，并加两道保护：

1. 结构化硬冲突层先执行，语义模型不能覆盖；
2. 默认只在 v3.1 为 `uncertain` 时调用一次 backend。

官方模型卡比较后没有下载本地权重：BGE-M3 zero-shot 主权重 1.14 GB，
超过 600 MB 上限；428 MB 多语 MiniLMv2 NLI 可作后续候选。本轮实际
使用 DeepSeek Flash 结构化 JSON，因此只声称“选择性 API claim 判定”。

## 结果

dev：

- v3.1：3/12 分类正确，11/12 弃权；
- selective：10/12 正确，2/12 弃权，supported 5/6 precision；
- 8 次真实请求，$0.002699376。

首次 held-out：

- v3.1：14/32 正确，21/32 弃权；
- selective：22/32 正确，6/32 弃权；
- supported precision 2/2，但 recall 只有 2/8；
- 16 次真实请求，$0.004682460。

首次 held-out 发现 `as_of_date` 与硬编码政策边界实现错误。原结果逐题
保留；修复后第二轮得到 25/32，但明确标记为事后 bugfix，不作为独立验证。

## 为什么没有默认启用

虽然 `uncertain` 显著减少，最终 unsupported precision 只有 17/23，
条件缺失样本存在误判；更关键的是 v3.1 的 false unsupported 会让
单位等价 claim 无法进入语义层，supported recall 仅 3/8。该机制可以
用于离线复核队列，但不具备“默认拒答或改答案”的充分证据。

## 工程证据

- 独立 SQLite 预算账本使用 `BEGIN IMMEDIATE`，并发预留受硬上限约束；
  pending 成本计入已占预算，provider usage 后才释放差额。
- 每轮最多 2 次重试且仅重试 429；失败请求费用也保守入账。
- exact response cache 在预算预留前命中；缓存重放延迟与在线 P95 分开。
- 26 次 provider 请求全部 settled，总费用 $0.00813324，无超限或残留 pending。
- API `not_entailment` 永不映射 `unsupported`；解析失败/超时/上下文外引用
  保持 `uncertain`。

## 下一步

只在一个新的独立 held-out 上验证两项改动：

1. 先做确定性的单位/数量等价归一化，避免 v3.1 false unsupported 短路；
2. 为绝对量词 claim 增加人工双标，判断“全量断言与条件规则冲突”应归
   `unsupported` 还是 `uncertain`。

未完成第 1 项前，不把 selective grounding 接入默认答案路径。
