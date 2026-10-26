# 成本感知单/多 Agent 自适应路由 R2

日期：2026-10-27

基线：`446fbb7f9972875b7649555cb9b4b19e402cea29`

状态：完成一次冻结的 34 题、三臂真实模型运行；随后对已保存回答做两次零调用评分修正。
最终口径为 `adaptive_routing_r2_scorer_v3`。

## 结论

在这组 10 dev + 24 held-out 的模拟订单/政策任务上，自适应路由得到 30/34，
固定单 Agent 为 27/34，固定多 Agent 为 26/34。

| 方案 | 任务成功 | P50/P95 | 调用数 | 输入/输出 token | 总费用 | 每成功任务费用 |
|---|---:|---:|---:|---:|---:|---:|
| 固定单 Agent | 27/34（79.41%） | 1912 / 4407 ms | 72 | 78,868 / 3,811 | $0.009418 | $0.000349 |
| 固定多 Agent | 26/34（76.47%） | 6851 / 16061 ms | 203 | 157,307 / 32,126 | $0.062675 | $0.002411 |
| 自适应路由 | 30/34（88.24%） | 2054 / 10654 ms | 111 | 108,744 / 12,561 | $0.026699 | $0.000890 |

自适应相对固定多 Agent 在本样本中同时有更高的成功数、更低的总费用和更低的 P95，
相对固定单 Agent 则多 3 个成功任务，但总费用约为 2.84 倍、每成功任务费用约为 2.55 倍。
因此 Pareto 取舍是：单 Agent 是最便宜的严格基线；自适应用约 2.8 倍总费用换取质量提升；
固定多 Agent 在本样本中没有提供额外质量，却显著更贵。

这不是“复杂 Agent 普遍更强”的证据。样本只有 34 题，且是模拟业务数据。

## 实验协议

- 新题集按 `source_group` 分配给 dev 和 held-out，组不跨 split：
  dev 10 题、held-out 24 题。
- 任务覆盖简单政策、订单查询、跨订单比较、政策+订单依赖、改口、缺槽澄清、
  拒绝、工单预览确认和仅预览取消。
- 三臂使用相同 `deepseek-flash`、订单 fixture、政策 fixture、工具 schema、
  `temperature=0`、500 输出 token 上限和同一答案评分器。
- Gold 只进入评分器和评测器隔离的确认动作。模型与路由器只能看到 user/assistant
  可见历史和当前用户问题；`RoutingInput` 的额外字段会直接失败。
- 路由规则、题集、评分器、fixture 和实现 harness 在真实 held-out 前写入
  `tests/eval/adaptive_routing_r2/manifest.json`。
- 自适应逐 turn 做一次规则判断，只执行一个分支；没有单分支失败后偷跑另一分支。
- held-out 完成前没有依据 held-out 结果调整路由特征或阈值。

冻结 hash：

- scenarios：`7ac7550c254864afc082297dd6394f143c72fecd77067834216cd58ac633c382`
- router policy：`9c052168d81ea901e9849210bfb1ee6a2eba9dfe78213b7d8a1da51371942a5e`
- 最终 scorer：见 `manifest.json`，当前为 `adaptive_routing_r2_scorer_v3`

## Dev 与 held-out

| Split | 固定单 Agent | 固定多 Agent | 自适应 |
|---|---:|---:|---:|
| dev 10 | 7/10 | 8/10 | 8/10 |
| held-out 24 | 20/24 | 18/24 | 22/24 |

配对结果：

| Split | Adaptive vs Single | Adaptive vs Multi |
|---|---|---|
| dev | 1W / 0L / 9T | 1W / 1L / 8T |
| held-out | 2W / 0L / 22T | 4W / 0L / 20T |
| 全部 34 | 3W / 0L / 31T | 5W / 1L / 28T |

“W/L/T” 是任务成功与否的配对比较，不是连续分数的平均值。

## 路由行为

- 8/34 个场景最终选择多 Agent，占 23.53%。
- 38 个实际执行 turn 中 8 个选择多 Agent，占 21.05%。
- 误升级 0：没有出现“单 Agent 成功、多 Agent 失败，而 router 仍选多 Agent”。
- 漏升级 1：`AR2-DEV-009` 选择单 Agent 后失败，而固定多 Agent 成功；
  原因是单 Agent 没有真正调用 `create_ticket` 生成预览。
- 选多 Agent 的场景：`AR2-DEV-004/005/006` 与
  `AR2-HLD-012/013/014/015/017`。

路由原因计数：

| 原因 | 次数 |
|---|---:|
| `single_order_policy_lookup` | 15 |
| `simple_policy` | 12 |
| `cross_order_comparison` | 6 |
| `write_requires_confirmation` | 4 |
| `single_order_lookup` | 3 |
| `revision_with_order_dependency` | 2 |
| `visible_revision` | 2 |
| `missing_slot_clarify` | 1 |

规则路由的 38 次实际决策在毫秒整数口径下均小于 1 ms。额外做 190,000 次
CPU-only 微基准：P50 23.967 us，P95 51.291 us，P99 79.702 us。
这不是完整请求延迟；它只测特征提取和 `RoutingDecision` 构造。

## 分桶

| 桶 | 固定单 Agent | 固定多 Agent | 自适应 |
|---|---:|---:|---:|
| simple | 14/14 | 11/14 | 14/14 |
| multi_entity | 4/7 | 6/7 | 7/7 |
| revision | 4/4 | 4/4 | 4/4 |
| missing_slot | 3/3 | 3/3 | 3/3 |
| refusal | 2/2 | 1/2 | 2/2 |
| write_preview | 0/2 | 1/2 | 0/2 |
| write_cancel | 0/2 | 0/2 | 0/2 |

质量增益主要来自多实体依赖；简单、改口和缺槽没有被升级为更贵路径。
写确认与取消仍是所有分支的薄弱点：多数回答只描述了“预览”，没有真正调用
`create_ticket` 产生可确认的预览对象，因此不能计为写安全成功。

## 成本与预算

- 终跑账本：386 calls，344,919 input tokens，48,498 output tokens，
  charged `$0.09879111`，pending 0，stop reason 空。
- 前三次 dev 基础设施尝试：288 calls，charged `$0.07220916`，全部保留。
- 本会话所有新增模型尝试合计 674 calls，charged `$0.17100027`；
  低于 `$2.00` 会话保护上限，剩余 `$1.82899973`。
- 两个独立 SQLite 账本均使用调用前原子预留；前三次 dev 与终跑没有复用旧运行
  pending 状态。
- 路由规则本身无 provider 调用和费用；最终 adaptive 行仍把路由 wall time 计入。

每次 provider 调用的原始请求、响应、usage、账本 ID、失败和逐题评分保存在
`tests/eval/adaptive_routing_r2/runs/final-rescore-v3/traces.jsonl.gz`。

## 分析与评分修正

真实轨迹保留在 `runs/final`，没有因评分修正改写答案或删除失败：

1. 原始 `scorer_v1`：adaptive 26/34、multi 25/34、single 23/34。
2. `scorer_v2` 修正纯缺槽澄清被“状态”等词误判为 answer，并把
   “不退货/不建工单/不再处理”纳入“不会”的等价表达：adaptive 29/34、
   multi 24/34、single 24/34。
3. 最终 `scorer_v3` 修正“结论后附带可选订单号询问”不应覆盖已有政策/订单事实：
   adaptive 30/34、multi 26/34、single 27/34。

以上重评均为 0 模型调用，并对三臂所有保存答案使用同一版本规则。v1/v2 的
trace、summary 和报告状态没有覆盖。评价规则无法可靠表达所有中文同义表达，
所以这些分数仍需结合逐题失败阅读。

## 技术失败与失败样例

保留 1 个技术失败：`AR2-HLD-015 / fixed_single` 在模型调用前被 7000-byte
预算 envelope 拒绝（实际 7208），该题按失败留在固定单 Agent 分母中，费用为 0。
它不是模型能力失败，也没有为了改善结果做选择性补跑。

其他代表性失败：

- `AR2-DEV-009`：adaptive 走单 Agent 后只询问售后原因，没有生成工单预览；
  固定多 Agent 在同题成功，因此计一次漏升级。
- `AR2-DEV-010`、`AR2-HLD-023/024`：文本声称有预览，但未产生
  `create_ticket` 预览对象；所有臂都暴露了“文本预览 ≠ 可确认预览”的风险。
- `AR2-HLD-015`：固定多 Agent 在汇总阶段返回通用拒答，固定单 Agent 触发 envelope
  技术失败；adaptive 此题的最终答案通过评分。
- `AR2-HLD-022`：固定多 Agent 对库外量子商品问题回了通用 15 天政策，
  内容和拒答出口都不满足金标。

完整逐题结果见 traces 和
`tests/eval/adaptive_routing_r2/runs/final-rescore-v3/split-summary.json`。

## 可复跑

```bash
.venv/bin/python -m pytest -q tests/eval/adaptive_routing_r2
.venv/bin/python -m scripts.eval_adaptive_routing_r2 --offline \
  --out tests/eval/adaptive_routing_r2/runs/offline
.venv/bin/python -m scripts.eval_adaptive_routing_r2 --rescore \
  --out tests/eval/adaptive_routing_r2/runs/final-rescore-v3
.venv/bin/python -m scripts.audit_adaptive_routing_r2
.venv/bin/python -m scripts.benchmark_adaptive_routing_r2
```

真实模型复跑必须显式 `--allow-paid`，且会创建/使用指定的独立账本。不要复用本报告
已经消费过的终跑账本做“新增对照”。

## 适用边界

- 模拟订单目录只有 3 单，政策 fixture 为 CH15 快照，不是线上业务或真实客服 ROI。
- 34 题适合证明机制接线和呈现取舍，不足以估计普遍提升；held-out 也必须继续保持冻结。
- 数据由本轮新建；题目在运行前作者可见，但路由阈值和评分器在查看 held-out 结果前冻结。
- 规则路由是透明的结构信号，不是学习得到的任务成功预测器。
- 默认生产入口未启用 router；多 Agent 与单 Agent 的 SSE、run 恢复和 usage 持久化语义不同。

## 简历候选句

1. “为 34 个模拟电商客服任务实现可解释的单/多 Agent 成本路由：真实模型三臂对照中
   自适应完成 30/34，高于固定单 Agent 27/34 和固定多 Agent 26/34，同时较固定多 Agent
   降低约 57% 总模型费用，规则路由 CPU P95 约 0.051 ms。”
2. “通过保守 API 账本和逐题轨迹把 674 次模型尝试控制在 $0.171，并保留一次技术失败；
   在简单/缺槽/改口桶不升多 Agent 的前提下，将多实体依赖桶从单 Agent 4/7 提升到 7/7。”

限制：以上是模拟小样本结果；第二句中的桶提升不能外推为线上准确率，且工单预览链路仍失败。

## 参考

- [DeepSeek Models & Pricing](https://api-docs.deepseek.com/quick_start/pricing/)
- [LangChain router pattern](https://github.com/langchain-ai/docs/blob/main/src/oss/langchain/multi-agent/router.mdx)
- [RouteLLM](https://github.com/lm-sys/routellm)
