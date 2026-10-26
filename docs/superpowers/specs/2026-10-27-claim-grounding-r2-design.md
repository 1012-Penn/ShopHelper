# 选择性 Claim Grounding R2 设计

日期：2026-10-27
基线：`446fbb7f9972875b7649555cb9b4b19e402cea29`

## 目标

只在 `rag-quality-v3.1` 的引用判定为 `uncertain` 时，对单条 claim
和候选证据做一次轻量三态判断，减少无法解释的弃权；不能以降低 precision、
漏掉确定性矛盾或增加错误拒答换取更少的 `uncertain`。

## 接口

`app/claim_grounding` 提供纯函数和小型策略对象，不接入默认聊天入口。

- `ClaimInput`：claim、主体/型号/日期/版本/单位、候选证据
  `evidence_id + text`。不包含 gold、expected behavior、评分规则或允许动作。
- `GroundingResult`：`supported | unsupported | uncertain`、理由、
  实际使用的证据 ID、backend/model 版本和 provider usage。
- `ClaimGrounder`：先运行确定性冲突层；存在确定性矛盾时直接
  `unsupported`，语义 backend 无覆盖权。只有 v3.1 规则为 `uncertain`
  且 `only_uncertain=true`（默认）时才调用 backend。
- backend 异常、超时、无候选证据、JSON/字段解析失败、引用越界或不一致，
  一律保留 `uncertain`，不生成确定结论。
- 二分类 NLI 的 `not_entailment` 不得映射为 `unsupported`；只有明确
  entailment 可映射 `supported`，其他结果映射 `uncertain`。若未来接入三分类
  NLI，显式 contradiction 也需要高置信阈值。

`ClaimGroundingAnswerSemantic` 实现现有 `AnswerVerifier` 的
`semantic.invoke(prompt) -> AnswerVerdict` 形状。它解析 verifier 的固定 prompt
区块，逐句执行 v3.1 规则和选择性 grounding，再把 claim 结果聚合为
`pass/fail/uncertain`。不修改 `app/answer_verifier.py`。

## Backend 选型

官方资料与约束：

1. `MoritzLaurer/bge-m3-zeroshot-v2.0`：多语、CPU/GPU 可运行、MIT，
   但主权重约 1.14 GB，超过本轮 600 MB 下载上限；其二分类标签为
   entailment/not_entailment，不能把 not_entailment 直接解释为矛盾。
2. `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli`：428 MB 权重、0.1B、
   MIT、XNLI 中文准确率 0.721，CPU 可部署；三分类能区分 neutral 和
   contradiction。适合作为后续本地路径，但本轮只选一个 backend 做真实对照。
3. 现有 DeepSeek Flash 单 claim 结构化判定：支持三态，可要求引用证据 ID
   和矛盾类型；无需下载权重，按 provider usage 结算。本轮选择此项并实际运行。

因此报告只能称“选择性 API claim 判定”，不能称已部署本地 NLI。

## 数据与冻结

- dev 12 条：从既有 v3.1 `uncertain` 样本中抽取 claim，保留原 case/trace/
  chunk 来源。
- held-out 32 条：新建对抗 claim，覆盖同义支持、条件缺失、否定、单位、
  日期边界、跨订单串用和证据过期；gold 来自明确业务事实的独立标注，
  语义不明样本保留 `uncertain`。
- 以 `group_id` 绑定业务事实与源模板；同一 group 不跨 dev/held-out。
  冻结后，先固定数据、评分和配置 SHA-256，再允许读取 held-out 结果。
- gold 不进入 backend prompt，仅由独立评分器读取。

## 对照与指标

两组：

- `v3_1_rules`：现有字符/数字/实体规则，不作语义调用。
- `selective_grounding`：相同规则，仅对 `uncertain` 调用选定 backend。

输出逐 claim 的输入、证据、规则状态、backend 原始响应、最终状态、
usage、延迟与来源。聚合：

- 三态 precision/recall、macro-F1、supported precision 及样本分母；
- `uncertain` 覆盖率与弃权率；
- 确定性矛盾漏检；
- 实体、条件、单位、日期/版本错误；
- semantic 调用比例及 token/费用；
- 在线调用 P50/P95；精确缓存重放单列，不能当在线 P95。

成功口径：

- 确定性冲突漏检必须为 0；
- 最终不能把 gold `unsupported` 升为 `supported`；
- 若 `uncertain` 降低但 supported precision 或 unsupported recall 下降，
  视为没有质量收益。

## 预算与失败

- 本轮新增 provider 调用保护上限为 `$2.00`，独立 SQLite 账本，不复用旧 pending。
- 每次调用前用保守 token/价格估算原子预留；并发调用允许多个 pending。
- 结束后按 provider usage 结算；工具/裁判/失败尝试也记入独立 attempt 记录。
- 最多 2 次有界重试，仅对 429；退避有上限。解析失败和超时保留 uncertain。
- 优先查询本实验精确响应缓存；缓存命中不产生 provider 费用。

## 兼容与回滚

- 只新增 `app/claim_grounding/`、本目录评测、脚本和报告；不修改共享入口、
  `app/answer_verifier.py`、`scripts/eval_checks.py`、已有结果或数据库。
- 所有构造参数有兼容默认值；默认关闭 selective backend，不影响现有行为。
- 接线只需在创建 `AnswerVerifier` 时传入 adapter；回滚则继续使用原 semantic
  或 `None`。
- 下游答案修订默认不启用，本轮只记录 claim 层结果。
