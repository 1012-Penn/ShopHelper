# 检索证据面重排开发记录

## 观察

旧 frozen-v1 的 15 个 Top-5 miss 全部存在于 50 条 RRF 候选内，9 个也在旧 Top-10。多跳题需要“订单/商品事实 + 对应政策”，多约束题需要两个条件片段，政策题需要版本分支；单条 rerank 分数不能表达这些组合约束。

## 定位

逐题比较候选池、Top-10 和 Top-5 后确认，问题不是 dense/BM25 没召回，而是精排后用前 10/取前 5 的截断。来源限额和字符 MMR 已在协调会话做过零 API 试验，dev 没有超过 26/43，因此本轮未重复盲扫。

## 改动

`app/evidence_slate.py` 用查询中的显式实体、动作、条件和政策版本生成可解释槽位；从最多 50 个已精排候选中选证据，最后仍输出 10 条。`RetrievalService` 默认行为不变，只有 `RETRIEVAL_EVIDENCE_SLATE_ENABLED=true` 才启用。规则不读取金标。

开发过程按 TDD 增加纯选择测试、服务接线测试和对照汇总测试。

## 同题结果

- dev：`26/43 -> 42/43`，16 增 0 失；组召回@5 `0.7558 -> 0.9884`。
- frozen-v1 旧记录：`50/65 -> 65/65`，15 增 0 失；这是已看题集回归。
- 独立 holdout：`13/20 -> 18/20`，6 增 1 失；组召回@5 `0.8417 -> 0.9583`。
- 独立 holdout 唯一退化 `SLATE-H20`：通用运费片段替代了地区运价表；`SLATE-H19` 还漏掉第三个“损坏判定”证据面。

## 成本与边界

holdout 端到端检索 P50/P95 `878/1022 ms`，纯证据面选择约 `0.23/0.32 ms`。本轮 231 次 API 请求全部按 2026-10-26 价格快照计为 `$0`，0 失败、0 重试，预算护栏为 `$2`。没有执行答案生成或语义裁判。

规则适用于当前稳定的商品/订单/售后文档标题与 ID 语义。证据面识别不是通用语义理解；新语料需重新验证。默认关闭，建议监控后再开。

## 冻结与复现

方法先提交为 `881fa06`，独立题集随后提交为 `aeab239` 并只运行一次。复现：

```bash
python3 -m scripts.eval_evidence_slate \
  --split retrieval-slate-holdout-v1 \
  --cases tests/eval/splits/retrieval-slate-holdout-v1.jsonl \
  --out /tmp/evidence-slate-holdout \
  --credential-env-file /path/to/read-only/.env \
  --allow-paid --max-cost-usd 2
```

详细数字和面试口径见 `docs/eval/retrieval-evidence-slate-report.md`。
