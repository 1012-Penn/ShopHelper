# Adaptive Routing R2 Dev Notes

日期：2026-10-27

## 问题

旧多 Agent held-out 为 17/20 对单 Agent 15/20，复杂桶 6/8 对 3/8，
但固定多 Agent 每成功任务成本约高 6.4 倍。旧结果说明“复杂时可能值得升级”，
不足以支持所有请求升级，也没有自动路由实现。

## 定位与设计

1. 只允许 user 问题、已有可见历史、可见运行预算进入 `RoutingInput`；
   额外字段在 Pydantic 边界直接拒绝。
2. 规则识别显式订单实体、跨订单比较、政策+订单依赖、改口、缺槽和写确认风险。
3. 默认单 Agent；只有显式跨订单比较，或多实体政策依赖且发生改口才升级。
4. 写风险只作为特征记录，不能扩权，也不能单独触发多 Agent。
5. Dispatcher 在选择后只调用一个 executor。单路异常直接保留，禁止失败后切换路径。
6. 可选结构化 judge 只允许处理预声明的不确定带；默认与冻结运行均为 rules-only。

## 实现中发现的错误

### `astream` 代理缺失

第一次 dev 真实运行中，固定单 Agent 在模型调用前报
`_RetryingGuardedModel has no attribute astream`。原因是 429 包装器只代理了
`ainvoke`。修复增加流式入口，且只允许首 chunk 前重试；已经输出部分文本后不重放。
失败轨迹与账本保留在 `runs/dev-smoke`。

### Request envelope 过紧

第二次 dev 中两条长历史请求为 6188–6246 bytes，超过 6000 上限。
修复将 envelope 调到 7000 bytes，并重新划分四段 hard caps；路由策略与题集不变。
失败轨迹保留在 `runs/dev-smoke-v2`。

### 累计账本行污染单次指标

重跑 dev 时，`calls_for_case` 会返回同一场景此前的调用，导致行级成本被放大。
修复为每次运行生成唯一 attempt ID，并只汇总该 ID 下的 ledger rows；回归测试覆盖。

### 评分器误判

首次零调用重评发现三类问题：

- “请问哪笔订单”包含“状态”时被宽泛 answer 规则先截获；
- Markdown 粗体会打断负向语境例外；
- “不退货/不建工单/不再处理”没有映射到“不会”的语义。

第二次重评又发现“先给政策结论，再问是否提供订单号”被误判为纯澄清。
最终规则先识别具体订单/政策事实，只有没有事实结论时才判 clarify。
所有版本均保留，最终为 scorer v3。

## 结果

- 固定单 Agent：27/34，$0.00941760。
- 固定多 Agent：26/34，$0.06267488。
- Adaptive：30/34，$0.02669863。
- Adaptive vs single：3W/0L/31T。
- Adaptive vs multi：5W/1L/28T。
- 8/34 场景升级多 Agent；误升级 0，漏升级 1。
- 规则路由 190,000 次 CPU 基准：P50 23.967 us / P95 51.291 us。

自适应在本样本中优于两个固定端点，但它不是无条件默认：相比单 Agent 总费用约
2.84 倍、每成功任务费用约 2.55 倍。

## 复用提示

- 生产接线必须注入现有会话、工具和确认边界，不能直接复制实验 fixture。
- 单 Agent 的 SSE/run 恢复与多 Agent 进程内 graph 并非同语义；router API 当前只返回归一化 JSON。
- 写预览必须检查真实 preview 对象和 `confirm_key`，不能因为回答文本写了“预览”就判成功。
- 继续调规则应使用新的 dev 集，不可用已经看过的 24 题 held-out。
