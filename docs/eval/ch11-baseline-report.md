# ch11 评测与语料基线报告（M1）

- 运行日期：2026-10-25
- 语料：8 份文档 / 130 chunk / 242 条结构化事实 / 99 个证据单元
- 数据集：dev 60、frozen-v1 90、regression-v1 60；旧 47+13 文件未改
- frozen-v1 SHA-256：`f518fe1bd1d7c3c3cd95896a55a06f8160c3f5c82529c1639c8f6e6437fdabf7`
- Milvus build_id：`ebfd49478b37ef387c8b124319a450ed73a86ae2a4b9f4fc28d6a52b2ccaa0ba`
- corpus SHA-256：`52d1048465d0b842ab4165c851a47211d2e35ad6fd02ea68010a3fa5b0be7fa6`
- facts SHA-256：`07ffaa92aad72336443a31b3f7dd26ca381eef8b23a0f86e8f3f0efa44fcbb24`
- 回答/证据判断/裁判模型：`deepseek-flash`，回答与裁判温度 0.0
- 价格表：已核对，`deepseek-flash` 按峰值价保守估算；bge-m3/reranker 标准版免费但按 $0.01/M 占位
- 证据校准：旧 `config/evidence_calibration.json` 绑定旧语料，本轮 contract 标 `stale`，未覆盖

## 1. 边界

1. 本报告是**评测回答策略**（`scripts/eval_prompts.py` 的专用 prompt + 确定性检查），不是线上 system prompt/完整 LangGraph 意图与工具链。
2. 工具题只记录 `tool_fixtures`，L4 标 `not_applicable`，留给 ch12 跑完整工具链。
3. 语料是合成数据，不能外推到真实业务分布。
4. Milvus 是停写后复制的 Lite 快照；不是生产集群。contract 记录快照 SHA 与五项一致性结果。
5. 历史 3 文档/59 chunk 数字只作参照，见 `ch11-historical-baseline.md`；语料/index/题型已变，不与本表拼“提升”。

## 2. L1 检索（frozen-v1，n=65 有证据题）

| 指标 | 值 | 分母 |
|---|---:|---|
| R@3 | 0.6154 | 40/65 |
| R@5 | 0.7538 | 49/65 |
| R@10 | 0.9077 | 59/65 |
| MRR@10 | 0.5116 | 65 |
| group_recall | 0.9538 | 65 |
| single@1 | 0.3538 | 65 |
| evidence_precision | 0.1385 | 65 |
| hard_negative_hit_rate | 1.0000 | 负例题（诊断项，不等于误答） |

分桶 R@3：single_hop 1.000、entity_mismatch 0.889、multi_constraint 0.643、policy_conflict 0.250、multi_hop 0.188。单跳已饱和，**多跳与版本冲突是主要瓶颈**。

regression-v1（n=47 有证据题）：R@3 0.8298、R@5 0.9362、R@10 0.9362、MRR 0.6025、group_recall 0.9574。regression 的改写模式是 identity（旧组合集没有独立改写缓存），contract 中已如实记录。

## 3. L2 证据

| 指标 | frozen-v1 |
|---|---:|
| gate_false_accept | 0 |
| gate_false_block | 3 |
| conflict_detection_rate | 1.0000 |
| partial_handling_rate | 0.9286 |

- 误放行 0；误拦 3 例（FRZ-MULTI-11、FRZ-MULTI-14、FRZ-CONSTRAINT-13），分母只算“金标 sufficient、非澄清题”。
- 证据判断器固定温度 0、结果进缓存；同一快照/数据集/缓存复跑时 L2 只读缓存。
- 旧证据校准未覆盖新语料；L2 主要依赖 LLM 证据蕴含判断器，contract 标 `calibration_status=stale`。

## 4. L3 回答

| 指标 | frozen-v1 | regression-v1 |
|---|---:|---:|
| strict accuracy | 0.1774（11/62） | 不可算（旧题无 gold_facts） |
| partial accuracy | 0.8468 | — |
| condition_coverage | 0.7361（n=36 有必需条件题） | 0.4333 |
| forbidden_claim_rate | 0.0000（0/57） | 0.0000 |
| clarify_slot_recall | 0.7500 | — |
| clarify_precision | 0.5500 | — |
| unsupported_claim_rate（claim 级微平均） | 0.5267（79/150） | 0.4851 |

- 严格正确率低的主因是**引用支持**：模型常一条答案同时标 `[1][2]`，但并非每个引用都真正支持该句；`citation_unsupported` 45 次。
- 条件覆盖只统计有 `required_conditions` 的题；`answer_missing_condition` 11 次，说明有条件结论仍常被说成无条件。
- 禁用断言为 0；实体错配题的“不是 104 键”是纠正，检查器按完整短语 + 否定前缀匹配，不把否定误判成断言。

## 5. L4 行为

| 指标 | frozen-v1 |
|---|---:|
| under_refusal_rate | 0.0000（0/16） |
| over_refusal_rate | 0.1000（7/70） |
| correct_refusal_rate | 1.0000 |
| conditional_compliance | 0.7143 |
| verification_compliance | 0.7500 |
| L4 分母（剔除 4 道工具题） | 86 |

- 应拒未拒为 0；误拒集中在缺槽位的 clarify 题（闸先拒答，未追问）与个别多跳题。
- 有条件题合规率 0.7143；需核验题合规率 0.75，仍有 1 例把“需检测”说成确定结论。

## 6. L5 过程

frozen-v1（full 首跑）：157 次真实 LLM 调用（67 次回答 + 90 次证据判断）、tokens 103834/27055、成本 $0.1019、P50/P95 4193/15584 ms。
frozen-v1（同 contract 复跑）：0 次新增调用，67 题命中回答缓存、90 题命中证据判断缓存；回答等价成本 $0.0636，真实新增 $0.0000。
regression-v1（full 首跑）：106 次调用、成本 $0.0631；复跑 0 次新增，46 题命中回答缓存、60 题命中证据判断缓存。
累计台账（含改写、嵌入、重排、回答、证据判断）：$0.4644 / $10.00。
所有付费客户端（嵌入、重排、证据判断、裁判、回答）都经过 `allow_paid + pricing.verified + 台账` 守卫；缓存命中不重复计费，报告中同时给“等价成本”和“真实新增”。

## 7. 阻断项

| 项 | frozen-v1 | regression-v1 |
|---|---:|---:|
| 硬负例误放行 | 0/10 | 0/0 |
| OOD 误放行 | 0/6 | — |
| 禁用断言 | 0/57 | 0/10 |
| 无证据断言（claim 级） | 79/150 | 32/63 |
| 应拒未拒 | 0/16 | 0/13 |

硬负例 `FRZ-HARD-10` 原问“表带可以单独买吗”其实可由兼容配件表回答，已改为“表带单独买多少钱”并重新冻结；修复后误放行为 0。无证据断言仍高，是下一阶段最需要修的回答层问题。

## 8. 结论与下一章输入

1. 共同评测契约已建立：同一 frozen-v1、同一 contract、同一指标口径；ch12/ch13 不得另造题。
2. 检索层已经不是单跳瓶颈；多跳、政策版本冲突、条件组合是主要失败源。
3. 回答层最大的问题是引用支持与条件合规，不是检索召回；ch12 的补检索不能掩盖这两项。
4. 旧 3 文档 R@10=1.000 不能作为新语料的对照结论；新语料 R@10=0.9077 是新的基线。
5. 本轮不改生产 `config/evidence_calibration.json`；需要时只用 dev/regression 重校准，并重新冻结 contract。
