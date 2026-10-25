# ch11 验收记录

- 日期：2026-10-25
- 命令：`python -m scripts.acceptance_ch11 --snapshot data/eval-snapshots/ebfd49478b37ef387c8b124319a450ed73a86ae2a4b9f4fc28d6a52b2ccaa0ba.db`
- 结果：退出码 0

```text
[ok] corpus: docs=8 chunks=130 facts=242
[ok] frozen-v1: n=90 buckets={'single_hop': 18, 'multi_hop': 16, 'multi_constraint': 14, 'entity_mismatch': 10, 'policy_conflict': 8, 'hard_negative': 10, 'clarify': 8, 'ood': 6}
[ok] regression-v1: n=60 C04/C05 labels present; old files unchanged
[ok] leakage: blocking=0 warning=1
[ok] budget: $0.4644/$10.00
[ok] offline-harness: eval_ch11 --mode offline frozen-v1
[ok] snapshot: ok=true, mysql=130, milvus=130, pending=0, build_id=ebfd49478b37ef387c8b124319a450ed73a86ae2a4b9f4fc28d6a52b2ccaa0ba, build_manifest_match=true
[ok] tests: 154 passed
```

## 说明

- 快照校验使用五项：build_id、pending=0、id 集合、逐 chunk 文本/类别、快照文件 SHA。
- `excluded_runtime_ids=[28]` 是旧审核块 `review_queue:4`，不属于 8 文档语料；快照一致性只比较 130 个文档 chunk。
- `leakage` 的 1 条 warning 是 regression-v1/A04 与售后手册的 8-gram 相似度 0.71，属于旧题与知识文档的正常词面接近，不是 blocking。
- 累计费用 $0.4644 含改写、嵌入、重排、回答、证据判断；上限 $10。
- 全量测试中的 FinanceBench 依赖失败不属于 ch11；本章验收只跑 ch11 相关 16 个测试文件。
