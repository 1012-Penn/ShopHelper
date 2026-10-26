# 第三轮优化最终验收

日期：2026-10-27

共同基线：`59e233f5145298846f8870b74e351ac3db98f5e1`

最终集成分支：`codex/night-final-r3-integration-20261027`

代码、测试与逐题证据 commit：`163ad99f85ef63c2680b31c268aab87568656dae`

本轮只追加这一轮，不提出第四轮。

## 结论

| 方向 | 源提交 | 验收结论 | 默认状态 |
|---|---|---|---|
| 订单事实身份与证据作用域 | 代码 `40934f1`，文档 `0d541c1` | 命名空间修复有效；held-out 主体匹配 `5/20 → 18/20`、任务成功 `5/20 → 17/20`；scope 只改善 R@K，不改善任务成功 | legacy 裸数字契约保留；scope 仅在工具收到明确主体/版本/日期时启用 |
| Schema 动作完成契约 | `1b754d8` | 工程契约与安全门通过；真实 dev 配对任务成功 `5/8 = 5/8`，额外预览 `1→0`、创建调用尝试 `8→2`；held-out 被评测基础设施污染 | 关闭，`MULTI_AGENT_ACTION_CONTRACTS_ENABLED=false` |
| 证据绑定答案计划 | 代码 `05266eb`，文档 `c7291dd` | 条件保留强，但 held-out 严格成功为负；原评 `13/20` 对自由文本 `16/20`，事后审计 `16/20` 对 `18/20` | 关闭，`ANSWER_PLAN_ENABLED=false` |

三项指标不能相加，也不是同一次实验。三个源分支和最终 smoke 各自使用独立账本。

## 事实身份与作用域

### 核心修复

- 裸数字 `1001` 固定解析为 `legacy_demo:1001`。
- 完整 `ORD-1001` 固定解析为 `corpus:ORD-1001`。
- 两者不再按共同后缀合并，也不会用一个来源覆盖另一个来源。
- 结构化订单返回 `subject/source/source_uri/version/as_of/missing_fields`；缺失商品不会被相似商品补齐。
- 检索 scope 只排除可证明冲突的主体、政策版本或日期；泛化政策保留，scope 缺失标为 unknown。
- scope 过滤为空时不会回退到已判定冲突的证据。

### held-out

| Arm | 任务成功 | 主体正确 | 条件正确 | 实体串用 | 误拒 | R@K | P95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 5/20 | 5/20 | 9/20 | 7/20 | 11/20 | 8/9 | 15,126 ms |
| identity | 17/20 | 18/20 | 20/20 | 0/20 | 0/20 | 8/9 | 9,893 ms |
| identity + scope | 17/20 | 18/20 | 19/20 | 0/20 | 0/20 | 9/9 | 5,140 ms |

identity 对 baseline 为 `14胜/2负/4平`；scope 对 identity 为 `0胜/0负/20平`。
scope 的价值是更短上下文、更高检索覆盖和更低 P95，不是回答成功率提升。

### 保留失败

- `FSR3-HLD-011` 缺少签收时间。回答给出正确条件，但 action 标签与 gold 的 clarify 不一致，仍计失败。
- `FSR3-HLD-017/018` 在 held-out 中把假设日期的年份 `2026` 误识别为裸订单号；held-out 后已加回归修复，但未重跑，原失败保留。
- `FSR3-HLD-017` 的正确条件句触发简单禁止短语评分，未事后改有利样本。

## Schema 动作契约

### 验收机制

- 从真实 registry JSON Schema 读取允许参数、必填项和枚举。
- `conversation_id` 由当前可信 run 注入；模型不能提交 `confirm_key`，也不能自行 `confirmed=True`。
- 文本“已预览/已创建”不算成功；只有结构化 preview artifact 才能进入
  `awaiting_confirmation`。
- 参数/动作缺口最多修复一次；普通查询和取消可行性问题不自动建预览。
- 客户端确认仍走既有 `/api/tickets/confirm` 和 ToolEngine 写拦截。
- 已登记预览被取消后，旧 confirm key 不能再建单；确认字段必须与预览一致。
- 关闭功能时，`/api/multi-agent/chat` 与 `/api/tickets/cancel` 保持旧响应形状。

### 真实 dev 配对

| Arm | 任务成功 | 配对 | 正确预览 | 正确取消 | 额外预览 | 创建尝试 | 未确认写 | 费用 | P95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 5/8 | 基准 | 7/8 | 7/8 | 1 | 8 | 0 | $0.0131205 | 7,167 ms |
| action contract | 5/8 | `2胜/2负/4平` | 6/8 | 6/8 | 0 | 2 | 0 | $0.0115809 | 6,819 ms |

这不是整体任务成功率提升。它只支持“错误额外动作更少、创建尝试更少”的工程结论。

首轮 held-out 两臂各有 `17/20` 题受请求包/调用上限污染，因此不可用于质量判断；
原始轨迹和账本保留，held-out 未重跑。

## 证据绑定答案计划

### held-out

| Arm | 原评严格成功 | 审计严格成功 | 条件 | 误拒 | 调用 | 费用 | P95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| free text | 16/20 | 18/20 | 19/20 | 0 | 20 | $0.00530490 | 2,274 ms |
| answer plan | 13/20 | 16/20 | **20/20** | 4 | 27 | $0.04426290 | 22,391 ms |

原评为冻结后唯一一次 held-out；审计版修正三个确定性评分缺陷并重评全部样本，
属于事后非独立复算。两版结论都为负，因此不把审计版 `16/20` 当提升。

失败集中在把自由文本能力、否定和多事实 claim 锚定到原文 quote 时过于严格。
uncertain 从未算正确，也没有事后抬高引用阈值。

## 真实应用组合 smoke

固定任务和停止规则在调用前写入
`tests/eval/optimization_r3_smoke/manifest.json`。新账本上限 `$2`、80 calls、
每 call 最多两次重试、600 秒墙钟上限。

### `attempt-01`，保留基础设施失败

4 个 `/api/chat` 任务完成，20 次真实 provider 调用，`$0.01120026`，pending 0。
写出汇总时脚本从错误层级读取 `route_nodes`，在多 Agent 任务前停止。失败目录和账本原样保留。

### `attempt-02`，v2 固定任务

- 5 个 `/api/chat` 任务和 2 个 multi-Agent 任务全部返回；
- 61 次真实 provider 调用，`$0.042805572`，pending 0；
- 技术传输失败 0；
- 3/5 chat 实际经过 `answer_plan` 节点；
- `DEV-ORDER-1002-TOOL` 调度 `query_order("1002")` 并成功；
- `ORD-1001` 调度 `query_order("ORD-1001")`，金额 248、状态“未支付”；回包也显式表明
  商品字段缺失。简单 scorer 因“无法确认商品名称”把它归为行为拒答，因此该题自动分为失败，
  但工具主体和事实本身正确；
- 取消可行性题没有生成预览、没有工单写入，返回结构化 `kind=none`；
- “创建售后工单预览”题在 multi-Agent 聚合阶段返回 failed，未产生 preview artifact，
  也没有写入。因此本次真实模型组合路径没有完成 preview→confirm 证明。

preview/confirm 的 HTTP、同键幂等、取消后旧键阻断、字段篡改阻断由真实应用工厂、SQLite
ToolEngine 和 fake registry 集成测试覆盖；这不等同于上述真实模型 smoke 已成功完成该动作。

## 默认开关与回滚

```text
STRUCTURED_OUTPUT_MODE=auto
ADAPTIVE_ROUTING_ENABLED=false
MULTI_AGENT_HANDOFF_MODE=raw
MULTI_AGENT_ACTION_CONTRACTS_ENABLED=false
ANSWER_PLAN_ENABLED=false
ANSWER_VERIFICATION_ENABLED=false
CLAIM_GROUNDING_ENABLED=false
RAG_GAP_TRIGGER_ENABLED=false
```

`ANSWER_PLAN_ENABLED=true` 时必须提供非空 `ANSWER_PLAN_LEDGER_PATH`，否则应用工厂失败。
动作契约关闭时，multi-Agent 和取消端点的公开响应保持旧形状。

## 验证与费用

- 组合定向回归：`180 passed`。
- 最终全量：`1110 passed, 2 skipped, 1 xfailed`，退出码 0。
- 既有 reliability fault acceptance：通过；32/32 隔离 SQLite run，0 模型调用。
- `compileall`、`git diff --check`：通过。
- 本轮改动范围 Ruff：除冻结的 fact harness 外全部通过。
- 冻结的 `tests/eval/fact_scope_r3/harness.py` 仍有 5 个 lint finding；为保持冻结 manifest
  的源码 hash 和实验可追溯性，未修改。
- 当前全仓 Ruff 仍有 `485` 项，不得表述为全仓 lint 绿。第二轮报告中的“Ruff 通过”声明已撤回。
- 文本证据未发现 API key 或 Bearer token。

费用拆分：

- fact scope：`$0.1852206`；
- action contract：`$0.1295586`；
- answer plan：`$0.100874172`；
- integration smoke 两次：`$0.054005832`；
- 合计 `$0.469659204`。

这些是不同题集、账本和运行时开销，不是一次运行成本，也不是单次请求单价。

## 未验证与风险

- order identity 的新 held-out 已查看，规则修复后没有新的独立 held-out。
- action held-out 被基础设施污染；最终边界修复后的真实两臂未重跑。
- 真实组合 smoke 的 preview 题失败，action preview→confirm 只有 source dev 与集成测试证据。
- answer plan 在固定证据上为负结果；组合 smoke 的自动严格分也只有 `1/5`，且 P95 约 51 秒。
- 多 Agent 路径仍没有与 `/api/chat` 等价的持久 run、SSE 重放和完整 usage 恢复声明。
- 所有业务数据均为模拟数据，不能写成真实客服 ROI、生产准确率或线上 SLA。
