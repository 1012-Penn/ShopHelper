# 第二轮优化集成验收

日期：2026-10-27

共同基线：`446fbb7f9972875b7649555cb9b4b19e402cea29`

集成分支：`codex/optimization-r2-integration-20261027`

代码集成 commit：`d615c52`

## 总览

| 方向 | 源 commit | 验收结论 | 默认状态 |
|---|---|---|---|
| 成本感知路由 | `40482fa847edfbf8e10b8a7aa51e24aaef88d8dd` | 通过；效果有净收益，但样本小且仅在模拟题 | 关闭，显式 `ADAPTIVE_ROUTING_ENABLED=true` 才挂载 |
| 选择性 claim grounding | `6f680733dcee7e875ef8a0735d3aa58e22a8c49f` | 工程可运行；效果是条件收益/负结果，不适合默认拒答 | 关闭 |
| typed evidence/handoff | `6334e43636aaface0c70cabf811219b213be96bb` | 接线与来源约束通过；没有净质量收益且增加上下文成本 | `MULTI_AGENT_HANDOFF_MODE=raw` |
| 无标签端到端评测 | `3d6fd9fdca6905db316812f611ff6b8cd1472475` | 诊断通过；发现并修复真实结构化输出兼容故障 | 独立 harness，不开生产实验策略 |

四个源实验都基于 `446fbb7` 各自单变量运行。不能把它们的分母、成功数或百分比相加。

## 独立复核

- 四套题集均为 dev/held-out 分组且源模板组不交叉：
  - 路由 10 + 24；
  - claim 12 + 32；
  - typed 10 + 20；
  - E2E 8 + 20。
- 路由输入 schema 只含用户问题、允许历史和可见预算；额外 gold 字段直接 422。
- 路由三臂从保存轨迹重算为 `30/34`、`27/34`、`26/34`，与报告一致；8 个场景选择多 Agent。
- claim 从逐 claim 结果重算 dev 与 held-out 混淆矩阵，未把 `uncertain` 计入正确。
- typed 从 60 条保存轨迹重算成功数、输入 token、费用和配对结果。
- E2E 的 450 条历史诊断与保存聚合一致：dev 三模式完全同答 43/60，至少两模式同答 56/60。
- 未发现 API key 或凭证进入本轮提交的结果目录。

## 各方向结果

### 路由

| Arm | 成功 | P50/P95 | 调用 | 费用 | 每成功费用 |
|---|---:|---:|---:|---:|---:|
| fixed single | 27/34 | 1,912 / 4,407 ms | 72 | $0.009418 | $0.000349 |
| fixed multi | 26/34 | 6,851 / 16,061 ms | 203 | $0.062675 | $0.002411 |
| adaptive | **30/34** | 2,054 / 10,654 ms | 111 | $0.026699 | $0.000890 |

held-out 为 adaptive 22/24、single 20/24、multi 18/24。自适应相对 single 为 `3胜/0负/31平`，
相对 multi 为 `5胜/1负/28平`；误升级 0，漏升级 1（`AR2-DEV-009`）。

收益主要来自多实体桶 `7/7` 对 single `4/7`。simple 为 `14/14` 对 `14/14`，
说明规则没有把简单题升级。代价是自适应总费用约为 single 的 2.84 倍。

保留一个技术失败：`AR2-HLD-015/fixed_single` 的 7,208-byte request envelope 超过冻结上限，
留在分母，未选择性补跑。

集成：

- 新增 `AppAdaptiveRoutingService`，使用现有 session store、tool engine、model 和单 Agent run 执行器。
- `/api/adaptive-routing/chat` 只在显式开关下挂载。
- 单 Agent 分支复用持久 run、usage 和事件；该 HTTP 端点返回 JSON，不提供 SSE。
- 多 Agent 分支仍是进程内 graph，没有持久 run、SSE 重放或持久 usage。
- dispatcher 只执行选定分支，不因失败切换另一臂。

### Claim grounding

| 数据/版本 | v3.1 rules | selective |
|---|---:|---:|
| dev 分类正确 | 3/12 | 10/12 |
| 首次 held-out 分类正确 | 14/32 | 22/32 |
| 修复后 held-out，非独立 | — | 25/32 |

修复版 supported precision `3/3`、recall `3/8`；unsupported precision `17/23`、
recall `17/17`；uncertain 6/32。26 次 provider 请求总计 `$0.00813324`，
在线请求 P95 `2,493.19 ms`，缓存重放 P95 约 `1.03 ms`。

首次 held-out 结果与修复后结果分开保存。修复针对 `as_of_date` 语义和硬编码 v1/v2 边界，
属于冻结后 bugfix，不是独立复现。

机制能拦截更多明确矛盾，但 `supported recall 3/8` 与两个 uncertain 被升为 unsupported
说明还不足以默认改答或拒答。集成将它接在现有 `AnswerVerifier` 语义接口后，需要同时显式设置：

```text
ANSWER_VERIFICATION_ENABLED=true
CLAIM_GROUNDING_ENABLED=true
CLAIM_GROUNDING_LEDGER_PATH=<per-run sqlite path>
```

预算路径为空时应用工厂启动失败，避免把无界调用悄悄接入默认路径。

### Typed evidence/handoff

| 指标 | raw | typed |
|---|---:|---:|
| 任务成功 | 25/30 | 25/30 |
| dev / held-out | 8/10 · 17/20 | 8/10 · 17/20 |
| 配对 typed 对 raw | — | 0胜 / 0负 / 30平 |
| 输入 token | 97,506 | 130,455（+33.79%） |
| 每成功任务费用 | $0.001680 | $0.002023（+20.39%） |
| 案例 P95 | 11,260 ms | 11,163 ms |

来源缺失从 raw 2 降至 typed 0，但 typed 实际 cache dedup 为 0，不能把调用次数差异全部归因于缓存。
加入来源约束没有带来任务成功净增益，因此保持 raw 默认。

集成复核另外修复了三个源实验未覆盖的问题：

- 重复查询得到同一 `observed_sequence` 时，后一次真实观测现在会 supersede 旧 claim；
- ToolEngine 的错误字符串不再在 TTL 内被当作成功读结果复用；
- Supervisor 的 typed prompt 也移除 `source.raw_result`，完整原文仅留在审计对象。

这些修复没有新增真实模型效果证据，所以不重写上面的 25/30 结果。

### 无标签 E2E 与生产修复

历史诊断：

- dev 基线 12/60 正确、17 模型/行为错误、26 引用 uncertain、5 unsupported；
- frozen-v1 基线 17/90 正确、28 模型/行为错误、33 uncertain、12 unsupported；
- 旧回答 prompt 确实包含 `expected_behavior`，三模式共享缓存输出；
- 历史 chunk ID 可映射，未发现证据 ID 断链；旧 trace 没有 finish reason，截断只能记为未观测。

源实验默认生产路径出现 `Thinking mode does not support this tool_choice`，28/28 技术失败；
JSON 兼容臂技术失败 0/28，但严格成功只有 4/28，其中知识引用没有任何确定 supported。

集成修复：

- `app/llm.py` 增加集中的 `make_structured_extractor`；
- DeepSeek thinking 模型在 `auto` 模式使用 JSON Schema 文本 + Pydantic 校验，普通聊天流不变；
- resolver、expander、intent、evidence judge、rewrite、gap analyzer、多 Agent Planner、
  after-sale extractor 和 flywheel 结构化调用均改走该入口；
- 真实单次结构化 smoke 成功，146 input / 221 output token，成本 `$0.000309`。

集成后的真实应用 dev smoke（全新精确缓存、独立 `$2` 账本）：

| 指标 | 结果 |
|---|---:|
| 严格成功 | 2/8 |
| 技术失败 | 0/8 |
| 事实层 | 6/6 |
| 条件层 | 2/2 |
| 实体层 | 3/3 |
| 引用 | 3 uncertain / 2 unsupported / 2 tool-only / 1 not-required |
| P50 / P95 | 8.793 / 22.448 s |
| 生成 / 裁判费用 | $0.011128 / $0.006607 |
| 总费用 | $0.017735 |

这不是新 held-out 效果，只证明结构化协议故障已在真实应用路径消失。严格成功仍低，
引用不支持/待复核仍不能算正确。

同时修复两个低成本真实问题：

- `query_product` 不再随机生成价格和库存，改为确定性商品目录；未知商品显式返回未找到；
- 纯退款/售后政策问题在缺订单号时先走检索，不再一律索取订单号；具体“这一单”仍保留订单选择器。

未统一 `app.orders` 与《订单与支付》中的订单事实。两者仍是不同业务源，需下一轮以新数据版本修复并重跑。

## 默认开关与回滚

```text
STRUCTURED_OUTPUT_MODE=auto
ADAPTIVE_ROUTING_ENABLED=false
MULTI_AGENT_HANDOFF_MODE=raw
ANSWER_VERIFICATION_ENABLED=false
CLAIM_GROUNDING_ENABLED=false
RAG_GAP_TRIGGER_ENABLED=false
```

关闭 adaptive 即不挂载 `/api/adaptive-routing/chat`。typed 与 claim 都只通过显式配置启用。
没有新增数据库迁移；claim 预算走独立 SQLite ledger。

## 验证与费用

- 定向集成回归：`219 passed`。
- 集成全量：`1011 passed, 2 skipped, 1 xfailed`，退出码 0。
- 既有可靠性 fault acceptance：通过；32/32 隔离 SQLite run，0 模型调用。
- Ruff：原“本轮修改的代码/测试均通过”声明已撤回。后续复跑无法复现全仓通过；
  第三轮验收报告记录了当前实际结果，不以测试通过替代 lint 证据。
- `git diff --check`：通过。
- 源会话费用合计 `$0.344887`；本轮集成新增 API 费用 `$0.018044`（含结构化 smoke），
  全程使用独立账本，未复用其他会话 pending。
- 最终状态：未 push。main 状态在交付时记录于最终回复。

## 未验证与风险

- adaptive 只有 34 个模拟任务，held-out 在最终报告前已被查看；不能据此继续调规则并称独立验证。
- adaptive 单路复用持久 run，但 JSON 端点本身不是 SSE；multia 路仍无持久恢复/usage。
- claim grounding 的 gold 仍待领域人工复核，且当前 supported recall 太低，不应默认拒答。
- typed 修复后没有重新跑真实两臂，不能把工程修复当成效果提升。
- 当前 E2E dev smoke 不是 held-out；严格引用支持仍接近零。
- `app.orders` 与知识库冲突、固定退款流程的其他边界、工单预览对象缺失仍需下一轮处理。
