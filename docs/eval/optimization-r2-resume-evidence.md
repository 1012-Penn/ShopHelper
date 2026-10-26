# 第二轮优化简历证据

日期：2026-10-27

集成分支：`codex/optimization-r2-integration-20261027`

以下数字均来自模拟电商任务。所有分母、held-out 和成本代价必须一起保留。

## 推荐写进简历的 2–3 项

### 1. 可解释成本路由

可写：

> 为 34 个模拟客服任务实现可解释的单/多 Agent 路由；同模型三臂对照中自适应完成 30/34，
> 固定单 Agent 27/34、固定多 Agent 26/34，并在 24 道 held-out 上取得 22/24 对 20/24。

带成本：

> 路由只升级 8/34 个场景，误升级 0；相对单 Agent 多 3 个成功，但总费用约 2.84 倍。
> 规则路由 CPU P95 约 0.051 ms。

证据：

- scorer v3，10 dev + 24 held-out；
- adaptive 对 single `3/0/31`，对 multi `5/1/28`；
- 多实体桶 7/7 对 single 4/7；
- 费用 `$0.026699` 对 `$0.009418` / `$0.062675`。

限制：

- 规划型规则只利用用户可见文本，题目规模和领域仍小；
- `AR2-HLD-015/fixed_single` 有一次 envelope 技术失败，仍在该臂分母；
- 写预览/取消在所有臂仍失败，不能把“文本里写了预览”当成写安全成功；
- held-out 已查看，后续不能继续调阈值后称独立验证。

### 2. DeepSeek thinking 的端到端兼容诊断

可写：

> 定位 DeepSeek thinking 模式对 `tool_choice` 的实现级拒绝，在真实 `create_app` +
> `POST /api/chat` 路径上复现 28/28 技术降级；集中 JSON-Schema 提取层后技术失败降为 0/28。

集成后真实 dev smoke：

> 8 题 dev 真实运行技术失败 0/8，事实层 6/6、条件 2/2、实体 3/3；
> 严格成功仍为 2/8，引用 uncertain/unsupported 未算正确。

证据：

- 历史默认 28/28 技术失败；兼容臂 0/28；
- 当前应用 dev 结果在
  `tests/eval/e2e_quality_r2/results/current-app-dev-20261027/`；
- 单次结构化真实 smoke 成本 `$0.000309`；
- 当前 dev 8 题运行成本 `$0.017735`。

限制：

- 0/8 技术失败是接线结果，不是 RAG 质量提升；
- 2/8 严格成功来自无知识引用题，知识类确定性 supported 仍为零；
- 新 dev smoke 是 8 题，不是新的 held-out。

### 3. 负结果驱动的机制取舍

可写：

> 在同一 30 题模拟集上对照 raw 与 typed evidence handoff：两臂均为 25/30、
> 配对 0胜0负30平，但 typed 输入 token 增加 33.8%、每成功成本增加 20.4%，
> 因而保持 raw 默认。

或 claim grounding：

> 在 32 条 held-out claim 上将 v3.1 规则 14/32 提升到选择性 grounding 22/32，
> 并把 17 条确定矛盾中的 16 条识别出来；supported recall 仅 2/8，因此只用于离线复核。

限制：

- typed 是规则评分的模拟任务，0/0/30 表示本样本未见净差异；
- claim 首轮 held-out 是一次真实运行，修复后 25/32 是冻结后 bugfix，不是独立复现；
- 两条都未证明线上或业务收益。

## 不推荐的表述

- “自适应路由普遍优于单 Agent”：只有 34 个模拟题。
- “多 Agent 比单 Agent 快”：固定 multi P95 为 16.1s，远慢于 single 4.4s。
- “typed handoff 减少 8 次工具调用”：只有 raw 一次重复请求，typed cache 实际去重为 0，
  动态 Planner 轨迹不同。
- “claim grounding 默认提升回答质量”：它增加了拒答风险且 supported recall 低。
- “严格成功率 4/28”或“2/8”解释为 RAG 质量：当前样本主要由引用覆盖与固定流程失败构成。
- 把源分支费用相加后说成单次运行成本。四个源实验各自有独立 `$2` 保护账本。

## 数字口径

| 方向 | 数据 | 模型 | 评分/版本 | 主要比较 |
|---|---|---|---|---|
| 路由 | 10 dev + 24 held-out | deepseek-flash | adaptive routing scorer v3 | 30/34 vs 27/34 vs 26/34 |
| claim | 12 dev + 32 held-out | deepseek-flash | claim-grounding-r2-v1 | 14/32 → 22/32；bugfix 25/32 非独立 |
| typed | 10 dev + 20 held-out | deepseek-flash | typed-handoff-r2-v3 | 25/30 = 25/30 |
| E2E | 8 dev（集成后 smoke） | deepseek-flash | e2e-quality-r2-score-v2 | 历史 28/28 → 兼容 0/28；集成 dev 0/8 技术失败、strict 2/8 |
| E2E 历史 | 7 RAG/retrieval groups，450 dev/frozen 轨迹 | 旧回答模型 | rag-quality-v3.1 | 零调用诊断 |

所有模拟数据都不能写成真实客服 ROI、生产 SLA 或线上准确率。
