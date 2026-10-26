# Typed Handoff R2 报告

日期：2026-10-27

分支：`codex/opt-typed-handoff-r2-20261027`

基线：`446fbb7f9972875b7649555cb9b4b19e402cea29`

## 结论

实现了可选的类型化证据 claim、任务依赖投影、改口 supersede、动态状态过期和 run
内只读查询合并。默认仍是 `handoff_mode="raw"`，现有构造与 API 响应不变。

30 题同模型配对中，raw 与 typed 都是 **25/30** 任务成功；配对结果为 typed
**0 胜 / 0 负 / 30 平**。typed 的输入 token 增加 **33.79%**，每成功任务成本增加
**20.39%**。因此本轮不能宣称质量提升；结论是保留为默认关闭的受控上下文能力，
并继续使用 raw 作为默认。

## 实现

- 新增 `app/evidence_contracts`：`EvidenceClaim`、`TaskHandoff`、
  `EvidenceLedger`、`ToolResultCache`。
- claim 只由工具原始返回生成；缺失字段显式为 `"unknown"`，模型 draft 不是来源。
- 原始工具参数、完整返回、tool call ID 和 chunk ID 保留在审计对象；进入角色 prompt
  时只发送来源 ID，不重复发送完整原始 JSON。
- 无依赖任务不继承既有 facts；有依赖任务只得到 `depends_on` 内、主体匹配、未过期且
  未 superseded 的 verified claims。
- `role_handoff`、`human_handoff`、`tool_call` 分开记录。handoff 不携带权限，
  工具白名单仍来自不可变 `ROLE_SPECS`。
- 类型化模式对同一 run 内等价只读查询合并 in-flight 请求；写工具始终绕过缓存。
- `MultiAgentRunner` 新参数默认 `handoff_mode="raw"` 和 300 秒动态 TTL；
  其他新增模型字段均有默认值。

机制设计参考了 [OpenAI Agents SDK handoffs](https://openai.github.io/openai-agents-python/handoffs/)
的输入过滤语义和 [LangGraph persistence](https://github.com/langchain-ai/docs/blob/main/src/oss/langgraph/persistence.mdx)
对线程状态与长期 store 的边界；本轮没有新增框架依赖。

## 评测口径

- 数据：10 dev + 20 held-out，先按 13 个业务事实/源模板分组，再切 split；同组不跨 split。
- 场景覆盖：双订单相似字段、v1/v2 生效边界、改期/改单号、过期物流、主体隔离、
  工单预览与确认、澄清、拒答、人工转接。
- 模型：`deepseek-flash`，温度 0，thinking disabled；两臂同一 Planner/工具/限额，
  唯一语义变量是 `handoff_mode`。
- 金标只在独立评分器读取。扫描 312 次真实模型请求，`gold`、`must_include`、
  `expected_behavior` 等字段命中数为 0。
- 数据 hash：`8758c2c3d1edaea310a117f04e53b151179020968e3429a0fe832eecc06b8224`。
- 运行配置 hash：`65b96d29d759b7c1d44b39ea943232e14f77b597cbb4c60f04741eac4431e0c2`。
- 最终评分 version：`typed-handoff-r2-v3`。

## 最终结果

| split | arm | success | entity errors | condition errors | source missing |
|---|---|---:|---:|---:|---:|
| dev 10 | raw | 8/10 | 0 | 0 | 0 |
| dev 10 | typed | 8/10 | 0 | 0 | 0 |
| held-out 20 | raw | 17/20 | 0 | 0 | 2 |
| held-out 20 | typed | 17/20 | 0 | 0 | 0 |
| all 30 | raw | 25/30 | 0 | 0 | 2 |
| all 30 | typed | 25/30 | 0 | 0 | 0 |

配对：dev 0/0/10，held-out 0/0/20，全量 0/0/30（typed 胜/负/平）。

typed 的 3 个含 `create_ticket` 要求样本全部把 `product` 与 `status` 投影给 action
角色，handoff 关键事实遗漏为 **0/3**。raw 没有 typed claim 投影，该项为 N/A，
不与 raw 的原始 facts 输入强行比较。

| metric | raw | typed | typed 相对 raw |
|---|---:|---:|---:|
| 任务成功 | 25/30 | 25/30 | 0 |
| 实体错误 | 0 | 0 | 0 |
| 条件错误 | 0 | 0 | 0 |
| 必需来源工具缺失 | 2 | 0 | -2 |
| 重复工具请求 | 1 | 0 | -1 |
| typed cache 实际去重 | N/A | 0 | 无命中 |
| 工具请求/实际执行 | 71/71 | 63/63 | -8 次请求 |
| 输入 token | 97,506 | 130,455 | +33.79% |
| 输出 token | 22,573 | 22,851 | +1.23% |
| 模型调用 | 156 | 156 | 0 |
| 总费用 | $0.04200161 | $0.05056675 | +20.39% |
| 每成功任务成本 | $0.00168006 | $0.00202267 | +20.39% |
| 案例 P50 / P95 | 4,856 / 11,260 ms | 5,367 / 11,163 ms | +10.52% / -0.86% |
| 单次模型尝试 P95 | 2,305 ms | 2,087 ms | -9.46% |

P95 只基于 30 个案例，不提供统计显著性结论。没有使用精确缓存重放替代在线耗时。

## 失败与机制归因

两臂都有 5 题失败：

- `TH-DEV-006`：纯取消可行性问题却生成了不必要的写预览。
- `TH-DEV-010`、`TH-HLD-025`、`TH-HLD-026`、`TH-HLD-027`：生成了工单预览，
  但 `ticket_type` 未满足生产 `CreateTicketInput` 的三值契约，计为参数错误。

其中 `TH-HLD-025` 和 `TH-HLD-027` 的 raw 臂还没有产生 `create_ticket`，因此同时计
来源工具缺失；typed 臂有完整预览和来源，但仍因参数契约失败。只增加来源不能抵消参数错误。

raw 唯一重复请求出现在 `TH-DEV-009` 的同一 `query_order(1001)`。typed 在该题没有发出
第二次等价请求，但 typed cache 的去重命中仍为 0，不能把差异归因于缓存；两臂的动态
Planner/Agent 轨迹并不逐调用完全相同。

typed 的总输入增长主要来自 Supervisor 和 action 的结构化 claim 投影。最终评分下没有
观察到实体或条件错误优势，因此“更少字段/更多来源”本身不构成质量收益。

## 评分审计

初次 v1 关键词规则把“v1 已过期，所以 15 天不适用”和“质量问题由商家承担，但当前问
非质量”误判为违反最终条件，也把 Markdown 表格和“未带入订单 1004”误判为实体错误。

保留的版本：

- v1：原始 `summary-v1.*`，17/30 raw、16/30 typed，配对 1/2/27；
- v2 中间版：`rescore-v2-initial/`，仍漏判 bullet 与多订单 block；
- v3 最终版：`v3-final/`，30/30 保存逐题评分，统一零调用重评。

v3 的变化对所有 60 条结果同时执行，没有只修改有利样本；17 行分数变化逐条保存在
`v3-final/score-changes.json`。

评分器按生产 schema 检查 `ticket_type`，但本轮模拟工具的签名是宽松 `str`，所以模型
生成的非法值被引擎接受后才由评分器判错。这是 harness 保真度限制；两臂同样受影响。

## 预算与失败

- 独立 SQLite 账本，总保护上限 `$2.00`，两臂各 `$0.98`。
- 实际 charged upper bound：`$0.09256836`；usage-at-peak：`$0.09256835`。
- raw/typed 各 156 次调用；pending 0，provider failure 0。
- 最多 2 次 429 有界重试；本次未触发重试。
- 两次预算构造前的 preflight 失败发生在 provider 调用之前，未产生请求费用；
  preflight manifest 已保留。

## 代码验证

定向套件：`45 passed`。

全量回归：

```text
916 passed, 2 skipped, 1 xfailed, 4 warnings in 163.56s
```

退出码为 0。4 个 warning 是既有测试的 asyncio marker 误用；Milvus Lite 在进程退出时
的日志流已关闭提示仍是既有清理噪声。

## 适用边界

- 本文数字来自 30 题模拟业务 fixture，不是线上客服成功率。
- 任务成功为规则评分；不是人工盲评。
- 没有测真实 MySQL、共享 Milvus 或外部 MCP。
- run 内缓存不代表跨请求/跨进程缓存收益。
- held-out 已被本轮查看，不能继续用它调阈值后仍称独立验证。
- typed 模式增加输入成本且未在本次小样本证明质量收益，不应默认开启。

## 简历候选句

1. 为多 Agent 设计可序列化证据 claim 与依赖范围 handoff，保留来源、主体、政策版本、
   观察/有效时间及 supersede/过期语义，并通过 13 项离线测试验证权限隔离和并发只读去重。
2. 在 10 dev + 20 held-out 的同模型配对实验中，发现 typed handoff 与 raw 同为
   25/30 成功、配对 0/0/30，但输入 token 增加 33.8%、每成功任务成本增加 20.4%，
   据此保持 raw 默认并记录条件性结论。

限制：上述结果为模拟数据上的规则评分；“0/0/30”表示未见净差异，不代表 typed 永不
有价值，也不构成线上性能或业务收益证明。
