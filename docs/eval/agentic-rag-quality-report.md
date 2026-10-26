# 受控 RAG 与最终答案复核

日期：2026-10-27。分支：`codex/interview-rag-quality-20261027`。基线：
`main@392e058d195bf731deff9ef6b4c2113da2a36a8b`。

## 结论

本轮没有证明最终答案复核或新的检索触发能提升严格答对率，因此两者都保持默认关闭。
可交付的确定成果是：

- 修正了旧评分尺子的标点后引用、时长单位、日期窗口和背景事实误判，并保留三态引用判定；
- 新增独立候选答案复核，能拦截“问题没给订单号却引入 `ORD-1006`”这类串单；
- 在固定 10 题 dev 小集上完成真实模型复核，3 次有限修订后严格答对仍为 0/9，
  且事实层答对从 1/9 降到 0/9，因此不能把复核吹成质量提升；
- 结构缺口触发只命中 `DEV-MULTI-01`，最终因 token 预算停止，并新增一次误拒；
  检索 R@5 仍为 2/10，条件覆盖从 0.45 降到 0.40，因此默认旧路径保持不变。

旧问题、金标、回答和 CH16 聚合均未改写。新评分写在
`tests/eval/rag_quality/runs/`，旧数据仍位于 `tests/eval/rag_ablation/runs/`。

可复现身份：Milvus 快照 tree SHA-256
`b6b1fa237d03d2c0f8ddcfe8e003f13c4cb584f0764ac04c2d57ae834bbdca73`；
overlay SHA-256 `86918bf704daf756c510a8fed6611fa90c08150b0bdcc9e8da1cd37a220b3f4a`；
固定 dev 小集 SHA-256 `4b4909aacc31f3ddc89fc94215fcb038441ffee971801adbbcba3c4b0e7b0ba3`。

## 评分版本 `rag-quality-v3`

旧 CH16 报告中的严格通过数受到以下问题影响：句号后的 `[n]` 被拆到无正文片段；
`半年`/`6 个月`、`一年`/`12 个月`未归一；日期规则只做字面匹配；部分背景 fact
被当成答案必答事实；字符 4-gram 被当作语义支持。

v3 的具体处理：

- 角标紧邻句号之后时归属前一个陈述句；
- 金额、时长和中文数字做确定性归一，并比较规范化日期与窗口内天数；
- 引用返回 `supported / unsupported / uncertain`。只有数字或实体的确定性冲突判
  `unsupported`；仅词面不同或无可靠规则时判 `uncertain`，不再由 4-gram 放行；
- `tests/eval/overlays/rag-quality-v3.json` 逐题记录背景 fact、条件或 fact 等价表达
  及修订依据，不修改原 gold；
- 聚合同时报严格三态结果和独立的事实层结果。`uncertain` 不塞进正确或错误：
  - 严格结果要求事实、条件、行为和引用支持同时确定；
  - 事实层结果只排确定性事实/条件/行为错误，引用待复核单列。

## 已有响应的统一离线重评

零模型调用、零费用重评了 dev/frozen-v1 的所有七组历史响应。“事实正确”在此指已确定的
gold fact、条件和行为均通过，不含引用三态；它不是把 uncertain 计入正确的总准确率。

| split | group | R@3 | R@5 | 事实正确 | 条件均值 | 引用 S/U/Q | 误拒 | P95 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| dev | RAG baseline | 21/43 | 26/43 | 29/41 | 0.729 | 1/11/31 | 2 | 45.913 |
| dev | decompose_once | 21/43 | 25/43 | 30/41 | 0.688 | 1/12/31 | 1 | 20,071.848* |
| dev | controlled_loop | 21/43 | 25/43 | 30/41 | 0.688 | 1/12/31 | 1 | 256.942* |
| dev | vector | 21/43 | 27/43 | 28/41 | 0.729 | 1/12/38 | 0 | 33.675 |
| dev | BM25 | 20/43 | 25/43 | 29/41 | 0.688 | 2/13/40 | 0 | 15.246 |
| dev | RRF | 20/43 | 29/43 | 26/41 | 0.646 | 2/13/36 | 0 | 54.109 |
| dev | RRF+rerank | 21/43 | 26/43 | 29/41 | 0.729 | 1/11/31 | 2 | 50.460 |
| frozen-v1 | RAG baseline | 41/65 | 50/65 | 43/64 | 0.708 | 0/21/44 | 3 | 80.783 |
| frozen-v1 | decompose_once | 41/65 | 49/65 | 44/64 | 0.681 | 0/22/44 | 2 | 25,037.331* |
| frozen-v1 | controlled_loop | 41/65 | 49/65 | 44/64 | 0.681 | 0/22/44 | 2 | 217.958* |
| frozen-v1 | vector | 37/65 | 46/65 | 47/64 | 0.722 | 3/20/54 | 0 | 7,738.978 |
| frozen-v1 | BM25 | 32/65 | 39/65 | 45/64 | 0.694 | 2/14/64 | 0 | 6,637.368 |
| frozen-v1 | RRF | 41/65 | 47/65 | 46/64 | 0.681 | 0/19/57 | 0 | 8,234.907 |
| frozen-v1 | RRF+rerank | 41/65 | 50/65 | 43/64 | 0.708 | 0/21/44 | 3 | 21,057.628 |

`S/U/Q` 分别表示确定支持、确定冲突、待人工/语义复核。失败的引用不能被当作支持；
大量 `Q` 也意味着不能把这张表读成“引用已经修复”。

严格三态结果仍保留在机器产物中。RAG 三组 dev 为确定正确 1/18、确定错误 17、
待复核 23；frozen-v1 为确定正确 0/33、确定错误 33、待复核 32。两组分母没有把
23 或 32 个引用待复核样本伪装成正确或错误。

所有 `*` 延迟含 CH16 既有长调用/超时记录，且大多来自缓存重放或本地路径；它们不是本轮
新增服务的在线 P95。

## 首轮停止与结构缺口触发

原 controlled_loop 的 dev/frozen 首轮放行分别为 43/60、67/90。frozen 剩余 23 题中
只有 `FRZ-MULTI-01`、`FRZ-MULTI-02`、`FRZ-CONSTRAINT-13` 三题有事实金标；其余主要是
澄清或应拒答，因此“补检索覆盖面小”不能单独归咎于循环实现。

原证据闸在 RAG 三组的通过数为：dev baseline/decompose_once/controlled_loop
`43/44/44`（共 60），frozen-v1 `67/68/68`（共 90）。这些是材料闸结论，不是答案正确率；
RAG 对照仍存在各模式默认闸门时机不同的问题，不能把答案差异单独归因给检索机制。
原 controlled_loop 的停止分布为 dev `43 first_pass / 5 no_new / 4 token / 4 clarify /
3 refuse / 1 sufficient`，frozen-v1 为 `67 / 10 / 4 / 4 / 4 / 1`。

`RetrievalService.retrieve` 已把 `EvidenceJudge` 结论合入 `low_confidence`；首轮并没有绕过
已执行的语义闸。新触发只作为默认关闭的独立实验变量：首轮判定通过后，若查询结构显示
订单商品与目录资格等跨证据阶段缺口，才进入一次有界补检索，且复用首轮判断结论，不重复
调用 judge。

固定 dev 小集中仅 `DEV-MULTI-01` 命中：

触发原因：`会员折扣缺少当前商品目录资格:SH-C60,SH-E300`。

| 指标 | 原 controlled_loop | 新触发 | 变化 |
|---|---:|---:|---:|
| R@3 / R@5（10 题） | 2/10 / 2/10 | 2/10 / 2/10 | 0 / 0 |
| 事实正确 | 1/9 | 1/9 | 0 |
| gold fact 均值 | 0.75 | 0.65 | -0.10 |
| 条件均值 | 0.45 | 0.40 | -0.05 |
| 误拒 | 1 | 2 | +1 |
| 首轮停止 | 9 | 8 | 触发 1 题 |

该题在分析器给出两个补检索后，未得到可放行证据，最终 `token_budget` 停止。实际调用为
7 次外部尝试、3 次付费模型尝试，成本 `$0.0013977`；随后精确缓存重放为 0 次外部调用。
结论：在现有 3,000 token 循环预算下无收益，`rag_gap_trigger_enabled` 保持默认 `false`。

## 最终答案独立复核

新增 `app/answer_verifier.py`，不把 `EvidenceJudge.answered` 当最终答案放行依据。
它先检查：

- 引用编号存在；
- 当前订单/商品实体没有串用；问题未提供订单号时，答案不得另选订单冒充当前订单；
- 当前订单的签收日期没有被其他订单日期替换；
- 用户给出的签收日期与订单记录冲突时必须显式纠正；
- v1/v2 与签收日期、7/15 天窗口一致；
- 引用证据能确认候选答案中的关键数值。

随后调用独立结构化语义裁判。仅确定性或语义 `fail` 触发一次答案修订；修订后仍不通过
则安全澄清/拒答。开关默认关闭；开启时最终 token 缓冲，工具状态和 citations 仍实时输出。

固定 10 题 dev 对照对 baseline、decompose_once、controlled_loop 三组已有回答分别执行：

| 指标 | 复核前 | 复核后 |
|---|---:|---:|
| 严格正确（确定分母） | 0/9 | 0/9 |
| 事实正确（确定分母） | 1/9 | 0/9 |
| 条件均值 | 0.45–0.55 | 0.45 |
| 引用 S/U/Q | 0/5/4 | 0/3–4/5–6 |
| 误拒 | 1 | 2 |
| 复核状态 | — | 8 pass / 2 refuse |
| 修订（每组） | — | 3 次逻辑修订；三组共 9 次逐题变化，首次在线生成共 3 次、其余为精确缓存重放 |

有代表性的逐题证据：

- `DEV-CONSTRAINT-01`：修订稿凭空引入 `ORD-1006`。确定性复核识别为“问题未提供的订单”，
  再次验证后安全拒答，证明串单防线实际生效；但它同时增加了误拒，不能算质量收益。
- `DEV-CONSTRAINT-05`：语义裁判指出原答案错误声称缺少用户已给出的满减门槛和折后金额；
  修订仍未形成可靠答案并安全拒答，严格结果没有改善。
- `DEV-MULTI-04` / `DEV-CONFLICT-05`：回答能指出订单实际签收日与用户陈述不同；确定性
  引用检查仍不能判断整句语义支持，因此保留为待复核，不硬判答对。
- `DEV-MULTI-01`：语义复核放行，但答案自己承认缺少商品在售/原价资格，条件得分仅 0.5；
  说明“语义复核 pass”不能替代任务评分。

第一版复核过宽，10 题均被 pass，已作为负结果原样保存在
`tests/eval/rag_quality/verified/run-1-dev/`。收紧实体和拒答规则后的真实运行保存在
`tests/eval/rag_quality/verified/run-2-dev/`；最终缓存重评保存在
`tests/eval/rag_quality/verified/dev/`。三份目录不互相覆盖。

## 成本、延迟与调用审计

价格使用既有已验证快照 `2026-10-26`：`deepseek-flash` 输入 `$0.30/M`、缓存命中
`$0.006/M`、输出 `$1.20/M`；BGE-M3/reranker 标准端点为 0。执行前估算 38 次调用、
`$0.04601`；运行时硬上限 `$2.00`，独立共享台账和 append-only attempt journal。

| 阶段 | 外部尝试 | 付费尝试 | 失败 | 成本 USD |
|---|---:|---:|---:|---:|
| verifier-v1（保留为负结果） | 10 | 10 | 0 | 0.0213039 |
| verifier-v2（实体收紧后） | 17 | 17 | 0 | 0.05223386 |
| controlled gap trigger | 7 | 3 | 0 | 0.0013977 |
| 最终缓存重评 | 0 | 0 | 0 | 0 |
| 合计 | 34 | 30 | 0 | 0.074935464 |

在线 attempt latency：答案复核 P50 `5.436s`、P95 `61.992s`（n=17，含 14 次裁判和
3 次修订）；触发实验 P50 `0.830s`、P95 `18.729s`（n=7）。这些是单请求耗时，
不是端到端 SLA。缓存重放不计入在线 P95。剩余保护额度为 `$1.925064536`。

## 已知边界

- 语义引用支持仍无法仅靠确定性规则可靠判断；RAG 历史重评有大量 `uncertain`，
  需要独立语义裁判或人工复核后才能得到可信的引用支持率。
- 评测回答 prompt 仍携带 `expected_behavior`，这是继承 CH11/CH16 的历史设计；
  本报告不把该口径下的结果外推为无泄漏的线上准确率。
- 新触发仅按 dev 结构错例设计；frozen 只做回归，不据此继续调阈值。
- 真实复核只覆盖固定 10 题 dev 小集；三组回答大多来自既有缓存，组间共享的是历史答案，
  不能把复核成本当作完整在线生成成本。
- 该工程仍是模拟数据上的面试项目，不是生产客服质量或 ROI 证明。

## 可直接用于简历的诚实候选句

- “构建受控 RAG 的三态答案评测与独立最终答案复核，覆盖引用归属、单位/日期归一、
  订单实体绑定和最多一次修订；逐题审计 10 题 dev 对照，发现复核在缺乏更强语义判分时
  会引入误拒，因此保持功能默认关闭。”
- “对 60 题 dev 与 90 题 frozen-v1 的历史 RAG 回答做零调用统一重评，区分确定事实正确、
  引用冲突和语义待复核，定位跨订单日期串用与政策条件错误，并为 34 次真实模型尝试保留
  `$0.0749` 逐次成本账本。”

## 可复现入口

```bash
/root/workplace/ShopHelper/.venv/bin/python -m scripts.eval_rag_quality --offline-rescore
/root/workplace/ShopHelper/.venv/bin/python -m scripts.eval_rag_quality \
  --verify-dev-subset --allow-paid --max-cost-usd 2.0 \
  --credential-env-file /root/workplace/ShopHelper/.env
/root/workplace/ShopHelper/.venv/bin/python -m scripts.eval_rag_quality \
  --run-controlled-trigger --allow-paid --max-cost-usd 2.0 \
  --credential-env-file /root/workplace/ShopHelper/.env
```

机器产物：

- `tests/eval/rag_quality/aggregate.json`：七组、两 split 的统一重评；
- `tests/eval/rag_quality/runs/<split>/<experiment>/<group>/rescore.jsonl`：逐题重评；
- `tests/eval/rag_quality/verified/dev/cost-reconciliation.json`：真实运行/缓存重放边界；
- `tests/eval/rag_quality/verified/run-1-dev/`：第一版过宽复核的负结果；
- `tests/eval/rag_quality/verified/run-2-dev/`：收紧后的真实复核与 attempts；
- `tests/eval/rag_quality/gap-trigger/run-1-dev/`：触发实验真实轨迹；
- `tests/eval/overlays/rag-quality-v3.json`：逐项评分修订依据。
