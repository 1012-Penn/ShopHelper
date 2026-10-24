# 评测数据变更记录

## 2026-10-25 · regression-v1 标签审计（v2-overlay）

- 旧文件 `ch04_eval_set.jsonl`、`ch09_junk_set.jsonl` 一个字节未改；本文件对应的 overlay 是 `tests/eval/overlays/regression-v1.labels.json`。
- 旧报告与旧标签全部保留，不覆盖历史成绩；新口径只用于 ch11 起的同 contract 对照。

| 题号 | 旧标签 | 新标签 | 理由 |
|---|---|---|---|
| C01 | answer | answer_with_conditions | 7 天无理由需商品完好；非质量退货运费责任未说明 |
| C02 | answer | answer_with_conditions | 1–3 天只适用微信/支付宝；银行卡最长 7 天 |
| C04 | answer | answer_with_conditions | “这个东西”缺 SKU/会员信息；95 折是否适用未知，不能承诺 |
| C05 | answer | verify_then_answer | 知识库没有“开胶属于质量问题”的判定规则，只能给核验流程 |
| E01 | answer | answer_with_conditions | 凑单包邮需合并支付、同地址、均未发货 |
| E03 | answer | answer_with_conditions | 定制类仅在有质量问题时可退，不是七天后仍可无理由退 |
| E04 | answer | verify_then_answer | 降噪损坏是否属质量问题需检测，不能直接下结论 |
| E06 | answer | answer_with_conditions | 95 折按商品原价、会员身份和在售状态适用 |
| E08 | answer | answer_with_conditions | 满 99 包邮后偏远地区仍需补 5 元 |
| F04 | answer | answer_with_conditions | 大件取件需预约，费用与时间窗以客服确认为准 |

- 变更只加字段（`expected_behavior`、`evidence_state`、条件、禁用断言、槽位、核验标记），不改原题、原证据、原 `answerable`。
- 若要重新校准证据闸，只能使用 dev/regression，不得用 frozen-v1 调阈值。
