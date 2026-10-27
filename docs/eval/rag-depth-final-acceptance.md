# RAG 深度补强最终集成验收

日期：2026-10-27。集成分支：`codex/rag-depth-final-integration-20261027`。基线：本地 `main=60f29ce54cd2f169a9619569d9bd453d8887d416`。本报告只描述本地集成与验证，不表示已推送。

## 验收结论

四个来源分支已完成代码与证据集成，并通过真实应用组合冒烟证明工厂接线可用：

1. 证据面选择保留 Cross-Encoder 排序，但为订单、商品、条件、政策版本等显式证据面保留互补材料；
2. 精确静态缓存包装嵌入与重排输入，语料版本缺失时 fail closed；
3. 受护 Query 改写默认开启，锚点/否定/纠错被破坏时回退原句；
4. 170 题答案质量实验保持为实验证据，不改变生产答案生成策略。

默认配置仍保持保守：证据面与缓存均显式 opt-in，受护改写默认开启且可用 `QUERY_GUARD_ENABLED=false` 回滚。集成验收通过，允许合入本地 `main`；引用冒烟中的 1 个 unsupported 与 2 个 uncertain 不阻塞集成，因为本轮没有把该 3 题冒烟宣称成答案质量提升。

## 集成矩阵

| 方向 | 来源提交 | 集成状态 | 默认策略 |
|---|---|---|---|
| 查询条件感知证据面 | `4ab7b32661448cfb93b5750115e31706b1225b1e` | 已集成，含地区运价与保修质量判定修复 | `RETRIEVAL_EVIDENCE_SLATE_ENABLED=false` |
| 精确嵌入/重排缓存 | `9ca99fcf54a2c9df498c1b0a35a37b8b42ed34be` | 已接入 `make_retrieval_chain` | `RETRIEVAL_CACHE_ENABLED=false`；开启必须显式提供 `RETRIEVAL_CACHE_CORPUS_VERSION` |
| 受护 Query 改写 | `cfee119f66918c06fdb42a98cfef244ed85ecdf3` | 已接入 resolve/rewrite 与评测入口 | `QUERY_GUARD_ENABLED=true`，可一键回滚 |
| 170 题答案质量 | `cacefa9221845749ab7cc1097309a96985156c9e` | 代码、逐题结果与报告已集成 | 不改变生产生成 prompt / 模型 / 温度 |

集成后关键检索配置为：chunk `500/80`、dense/BM25 各 50 候选、RRF 融合、一次 Cross-Encoder rerank 请求返回最多 50 个候选、最终上下文仍为 10 条。证据面选择只改变这 50 个候选进入最终 10 条的组合，不增加 rerank 调用。

## 证据覆盖

### 首轮冻结结果

| 范围 | 基线 Top-5 完整覆盖 | 证据面 | 配对结果 |
|---|---:|---:|---:|
| dev 43 | 26/43 | 42/43 | +16/0 |
| frozen regression 65 | 49/65 | 65/65 | +16/0 |
| 独立 holdout 20 | 13/20 | 18/20 | +6/1 |
| 合并 128 | 88/128 | 125/128 | +38/1 |

独立 20 题是唯一更强的新颖性证据；dev/frozen 包含开发与已看题回归，不能当独立泛化估计。

### 集成后零调用重放

地区运价与保修质量判定两处集成修复后，对已保存候选重放，未新增 provider 调用：

| 范围 | 基线 | 集成后证据面 | 配对结果 |
|---|---:|---:|---:|
| dev 43 | 26/43 | 42/43 | 不变 |
| frozen regression 65 | 49/65 | 65/65 | 不变 |
| 已保存 holdout 响应重放 20 | 13/20 | 20/20 | +7/0 |
| 合并 128 | 88/128 | 127/128 | +39/0 |

`20/20` 是对已经看过的独立 holdout 响应做事后修复重放，不是新的独立实验。对外表述仍应使用首轮独立结果 `13/20 -> 18/20`；`127/128` 是回顾性集成回归口径。

## 最终答案质量

170 题同题生成对照使用相同模型、prompt、温度和 Top-10 上下文，唯一差异是证据集合/顺序：

| 范围 | 基线 | 证据面 | 配对 |
|---|---:|---:|---:|
| 126 道证据答案题 | 94/126 | 102/126 | +8 净增 |
| 170 道总题（不确定留在分母） | 137/170 | 145/170 | 9 胜 / 1 负 / 160 平 |
| dev 42 | 27/42 | 30/42 | 3/0/39 |
| frozen 64 | 49/64 | 53/64 | 5/1/58 |
| 独立 20 | 18/20 | 19/20 | 1/0/19 |

澄清题两臂均为 16/17，另 1 题目标歧义保留 uncertain；应拒答题两臂均为 27/27。唯一真实答案退化是 `FRZ-ENTITY-02`：明示证据支持心率监测，但 slate 回答把另一张表的字段缺失误判成冲突。

17 道“baseline 漏、slate 齐”的题中16道可评答案题中，最终转移为：8 个 correct→correct、2 个 incorrect→correct、5 个 incorrect→incorrect、1 个 incorrect→uncertain。只有 2 题是覆盖补齐的直接答案收益；另外 6 道净增益来自两臂原本 Top-10 均完整的上下文组成/排序变化。不能把旧 Top-5 覆盖提升 28.9pp 写成答案正确率提升。

## 受护 Query 改写

冻结 20 题三臂同题比较：

| 方案 | Top-5 完整覆盖 | 意图保持 | MRR@10 |
|---|---:|---:|---:|
| 原话 | 17/20 | 15/20 | 0.6667 |
| 既有改写 | 20/20 | 17/20 | 0.8042 |
| 受护改写 | 20/20 | 20/20 | 0.7767 |

受护方案相对原话为覆盖 +3/0/17、意图 +5/0/15；相对既有改写为覆盖 0/0/20、意图 +3/0/17。MRR 低于既有改写，是保留约束和回退带来的取舍。冻结重放逐题一致，新增 0 次模型调用。

## 精确静态缓存

真实 130 chunk 私有索引上的重复请求对照：

| 指标 | 冷缓存 | 热缓存 |
|---|---:|---:|
| P50 检索延迟 | 739.8 ms | 35.8 ms |
| P95 检索延迟 | 2083.8 ms | 49.3 ms |
| 嵌入+重排 provider HTTP | 16 | 0 |

不重复 warm 请求 P50/P95 为 27.8/45.7 ms。语料价格从 299 改为 329 后，新语料版本首个请求 miss、下一个请求 hit，证明版本失效有效。最终有效运行 110 次 HTTP、0 失败、0 重试。缓存只覆盖静态嵌入/重排输入，不缓存订单状态、生成答案或证据判断；不能因当前价格表列出免费而宣称货币节省。

## 真实应用组合冒烟

冒烟入口是 `scripts.eval_e2e_quality_r2`，通过 `create_app`、真实 `/api/chat`、LangGraph、工具注册中心、检索服务与数据库私有副本运行。固定 3 道 dev 题：`DEV-DIRECT-E300`、`DEV-SHIP-REMOTE`、`DEV-POLICY-V2`。独立台账上限 $2、provider 调用上限 40、墙钟上限 600 秒。

运行时同时开启：

- `RETRIEVAL_EVIDENCE_SLATE_ENABLED=true`
- `RETRIEVAL_RERANK_POOL_K=50`
- `RETRIEVAL_CACHE_ENABLED=true`
- `RETRIEVAL_CACHE_CORPUS_VERSION=rag-depth-smoke-20261027-v1`
- `QUERY_REWRITE_ENABLED=true`
- `QUERY_GUARD_ENABLED=true`

结果：

| 检查项 | 结果 |
|---|---:|
| 收到非空答案 | 3/3 |
| resolve guard=pass | 3/3 |
| rerank pool=50 | 3/3 |
| 最终证据 10 条 | 3/3 |
| 事实检查 | 3/3 |
| 条件检查 | 2/2 |
| 实体检查 | 1/1 |
| 技术失败 | 0 |
| 引用 uncertain / unsupported | 2 / 1 |
| provider 记录 | 22，全部 succeeded |
| 实际估算费用 | $0.00826266 |

逐题证据面分别为 `product_entry`、`remote_shipping`、`v2_window+v1_window`，说明三类规则在真实应用工厂中都生效。引用失败说明生成层角标选择仍不稳定：例如政策题把多个相邻编号合并引用，评分器判 unsupported；运费题的“13 元起”依赖两个来源的组合，评分器保留 uncertain。这是质量边界，不是检索或接线失败。

## 成本

| 项目 | 有效/结算口径 |
|---|---:|
| 证据面实验 | $0.000000 |
| 缓存实验 | $0.003698 |
| 受护改写实验 | $0.116715 |
| 170 题答案质量有效链路 | $0.499860 |
| 来源分支小计 | $0.620273 |
| 集成后零调用重放 | $0.000000 |
| 真实应用组合冒烟 | $0.008263 |
| 最终集成有效小计 | $0.628535 |

答案质量分支另有包含失败尝试、截断重跑和未结算 reservation 的保守 SQLite 台账 `$0.866871 / $2.00`，不应与有效链路相加后称供应商发票。所有金额均基于冻结价格表估算。

评测 CLI 的 provider 调用次数上限按单次运行对象计数，包括该运行尚未结算的预留；多个进程共用一个账本时，每个进程各自执行该次数上限，不能把它当作跨进程共享的总次数上限。美元预算由账本加锁共享约束。正式本轮冒烟为单进程运行。

## 最终验证

接手验收时修复通用评测入口误强制实验功能、错误记录固定预算、历史调用计入本轮上限，并增加线程并发预留回归。

- Scoped Ruff（本分支 changed Python files）：`All checks passed!`
- 定向回归（证据面、缓存、受护改写、答案质量、真实应用 CLI 与相关检索/图/工具契约）：修复后 `241 passed`
- 全量测试：修复后 `1213 passed, 2 skipped, 1 xfailed, 6 warnings`，耗时 169.01s，退出码 0
- `git diff --check`：通过
- 冒烟与集成重放证据凭据扫描：未发现 API key
- 全量测试退出后 Milvus Lite 对已关闭 stream 输出清理日志，未改变 pytest 退出码或汇总结果

Lint 口径仅覆盖本分支 changed Python files；全仓库历史 lint 问题不在本轮修复范围内。

## 产物

- 集成重放汇总：`tests/eval/evidence_slate/runs/integration-v2-aggregate.json`
- 真实应用冒烟契约：`tests/eval/e2e_quality_r2/results/rag-depth-smoke-v1/run-contract.json`
- 冒烟逐题结果：`tests/eval/e2e_quality_r2/results/rag-depth-smoke-v1/attempt-01/dev/cases.jsonl`
- 冒烟预算：`tests/eval/e2e_quality_r2/results/rag-depth-smoke-v1/session-budget.json`
- 来源报告：`retrieval-evidence-slate-report.md`、`retrieval-static-cache-report.md`、`query-rewrite-history-ablation-report.md`、`answer-quality-depth-report.md`
