# 历史补全与受保护 Query 改写消融

日期：2026-10-27。分支：`codex/query-rewrite-ablation`。检索证据基点：
`4ab7b32661448cfb93b5750115e31706b1225b1e`。

## 结论

在同一 8 文档、130 chunk、242 fact 的 CH11 快照上，对原话、既有 resolve+rewrite、
受保护 resolve+rewrite 做同题检索对照。三者固定 `hybrid_rerank`、50 个候选、
Top-10 上下文和同一 reranker，只改变送入检索链的查询。

冻结集 20 题中：

| 方案 | Top-5 完整覆盖 | Top-10 完整覆盖 | MRR@10 | 意图保持 | 错误实体 |
|---|---:|---:|---:|---:|---:|
| 原话 | 17/20 | 18/20 | 0.6667 | 15/20 | 0 |
| 既有改写 | **20/20** | **20/20** | **0.8042** | 17/20 | 0 |
| 受保护改写 | **20/20** | **20/20** | 0.7767 | **20/20** | 0 |

相对原话，受保护方案 Top-5 完整覆盖 `+3/-0/17 平`，意图保持 `+5/-0/15 平`。
相对既有改写，受保护方案 Top-5 为 `0/-0/20 平`，意图保持 `+3/-0/17 平`。
因此旧改写已经找得全，但会吞掉否定、改口或实体纠错；本方案的主要新增价值是把
“可检索”和“问法约束仍正确”同时保住，而不是继续堆高已经饱和的覆盖。

## 适用边界

- 原话最适合已显式包含型号、订单号、日期和条件的查询。冻结集中 5/5 显式对照与
  4/4“显式实体 + 口语属性”在原话组已进入 Top-5；此时额外 LLM 调用没有覆盖收益。
- 历史含真实省略或代词时必须补全。冻结集历史缺口原话 `3/5`，两种改写均 `5/5`。
- 否定、改口以及“不是 A，是 B”需要保护器。旧改写在冻结集否定/改口题仅
  `17/20` 总体意图保持，其中 3 次回归；受保护方案为 `20/20`。
- 本结果只评价检索证据覆盖与查询约束，不评价最终答案正确、引用支持或幻觉率。
  当前 8 文档模拟客服语料之外，不能直接外推开放域知识库。
- 保护器是显式约束审计，不是通用语义证明。它保护订单/型号、ISO 日期、
  政策版本、数字单位与否定极性；无法证明所有同义表达完全等价。

## 逐缺口结果

冻结集按覆盖缺口出题，没有按 200 题硬凑。

| 缺口 | 题数 | 原话 Top-5 / 意图 | 既有改写 Top-5 / 意图 | 受保护 Top-5 / 意图 |
|---|---:|---:|---:|---:|
| 历史补全 | 5 | 3/5 · 0/5 | 5/5 · 5/5 | 5/5 · 5/5 |
| 词面缺口 | 4 | 4/4 · 4/4 | 4/4 · 4/4 | 4/4 · 4/4 |
| 否定/改口 | 5 | 4/5 · 5/5 | 5/5 · 3/5 | 5/5 · 5/5 |
| 显式完整对照 | 5 | 5/5 · 5/5 | 5/5 · 5/5 | 5/5 · 5/5 |
| 实体纠错 | 1 | 1/1 · 1/1 | 1/1 · 0/1 | 1/1 · 1/1 |

“词面缺口”样本保留用户明确给出的产品型号，只测试属性口语到标准表达的归一；
没有要求模型从“键盘”猜出某个具体型号。同义换词不重复计数成独立样本。

## 三个面试案例

1. **原话失败。** `FRZ-QR-HIST-04` 的当前问法是“它呢？”，历史最后确认对象是
   `ORD-1006`。原话在 Top-10 没有任何金标证据；既有改写补出
   `查询订单 ORD-1006 的状态`，状态证据排名 1。省略问法直接检索时，既不包含对象，
   也不包含要查的业务属性，无法仅靠向量相似度可靠恢复。

2. **改写成功。** `FRZ-QR-HIST-03` 的当前问法是“那它能刷公交吗？”，历史对象是
   `SH-W20`。原话 Top-10 未命中；受保护方案构造
   `SH-W20 是否支持刷公交？`，NFC 证据排名 1。补全实体和新属性后，查询同时覆盖了
   “谁”和“问什么”。

3. **改写副作用。** `FRZ-QR-NEG-04` 原文是“已激活的虚拟商品不是也能退吗？”，
   既有改写曾得到“已激活的虚拟商品可以退吗？”，把否定问法翻转；受保护方案在
   归一化前识别极性变化并回退原文。另一例 `FRZ-QR-ENTITY-01` 中，既有改写删掉了
   `ORD-1005` 的纠错上下文，受保护方案保留“不是 ORD-1005，查 ORD-1007”。

这就是保留原话基线的理由：改写能解决省略和词面缺口，但也可能在不缺证据的显式题上
产生无收益成本，并在否定或纠错题上改变问题。仅看召回会掩盖第二种错误。

## 运行、失败与费用

冻结首跑器在工作区干净的 `85f5ad983b22d6205175ce2c08b1f00152840e82` 上执行：

- 索引快照 SHA-256：`b6b1fa237d03d2c0f8ddcfe8e003f13c4cb584f0764ac04c2d57ae834bbdca73`
- 冻结题集 canonical SHA-256：`640083f1147aef2a58912cede7abff77413f3d0adb4ecea35beb82b420013f1b`
- 60/60 逐题轨迹完成；0 请求失败、0 结构解析失败、0 重试。
- 80 次付费查询模型调用（每组 resolve/rewrite 各 20 次），另 110 次检索侧
  embedding/rerank 尝试；总计 190 次外部 API 尝试。
- 实际估算费用 **USD 0.056311**。既有改写 `$0.0206205`，受保护方案
  `$0.0356901`，原话 `$0`。答案生成 0 次，LLM 裁判 0 次。
- 本轮私有台账累计到冻结首跑为 `$0.116715`（包含 dev v1-v3），USD 2.00 独立账本
  从未超限。执行前最坏规划估计为 `$0.20544`，不是已发生费用。
- 首跑端到端 P50/P95：原话 `1.064s / 1.312s`，既有改写
  `5.432s / 12.475s`，受保护 `6.554s / 30.713s`。这次受保护 P95 包含真实
  rerank cache miss，不能把缓存重放延迟当在线速度。

缓存重放新增 **0** 次模型调用、**$0**；`frozen-v1-replay/verification.json` 对
三项查询构造结果、全部 Top-10 排名和稳定聚合做逐题比较，`ok=true`、0 差异。

## 复现

```bash
python3 -m scripts.query_rewrite_ablation.runner \
  --split frozen-v1 \
  --cases tests/eval/query_rewrite_ablation/frozen-v1.seeds.jsonl \
  --frozen-lock tests/eval/query_rewrite_ablation/frozen-v1.lock.json \
  --out tests/eval/query_rewrite_ablation/runs/frozen-v1-replay \
  --credential-env-file /path/to/.env \
  --allow-paid

python3 -m scripts.query_rewrite_ablation.verify \
  --baseline tests/eval/query_rewrite_ablation/runs/frozen-v1/responses.jsonl \
  --replay tests/eval/query_rewrite_ablation/runs/frozen-v1-replay/responses.jsonl \
  --out tests/eval/query_rewrite_ablation/runs/frozen-v1-replay/verification.json
```

凭据只从进程环境或 `--credential-env-file` 读取，不进入结果、日志、报告或 git。
运行输入副本位于 gitignored `data/`；旧题集、旧回答和旧报告没有被覆盖。

## 产物

- 实现：`app/query_guard.py`、`app/rewrite.py`、`app/graph/entry.py`
- 运行器：`scripts/query_rewrite_ablation/runner.py`
- Dev/冻结题：`tests/eval/query_rewrite_ablation/`
- 逐题响应、聚合、契约：[`runs/frozen-v1/`](../../tests/eval/query_rewrite_ablation/runs/frozen-v1/)
- 零调用重放与比较：[`runs/frozen-v1-replay/`](../../tests/eval/query_rewrite_ablation/runs/frozen-v1-replay/)

## 简历候选

> 在 ShopHelper 对 20 道冻结客服问法构建原话/旧改写/约束保护改写三臂检索消融
> （同索引、50 候选、hybrid_rerank），将 Top-5 完整证据覆盖从 17/20 提升至 20/20、
> 意图保持从 15/20 提升至 20/20；冻结重放 0 模型调用且逐题查询与排名完全一致。

面试口径：20 道是冻结验证分母，覆盖 8 份模拟客服文档；完整覆盖要求全部金标证据
组进入 Top-5，不等于答案正确率。MRR 在冻结集上旧改写略高于受保护方案，不能用单一
指标宣称全面领先；净收益是覆盖与约束同时保持。
