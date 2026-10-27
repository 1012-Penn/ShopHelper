# Query 改写评测集

| Split | 用途 | 题数 | 运行结果 |
|---|---|---:|---|
| `dev.seeds.jsonl` | 开发与参数比较 | 20 | 历史结果仅用于回归 |
| `frozen-v1.seeds.jsonl` | 首轮冻结验证 | 20 | 已运行并保留 |
| `frozen-v2.seeds.jsonl` | 扩展未调参 holdout | 60 | 结果单独存放 |

`frozen-v2` 覆盖 10 道历史指代/省略、8 道话题切换、8 道实体/订单/数值约束、
8 道歧义澄清、8 道否定/改口、10 道无需改写对照和 8 道多约束联查。每道题都有
独立的 `semantic_family`；没有把同义模板重复计数。

可回答题的金标事实、证据单元与逐字 quote 由 `corpus/` 真源派生，并通过
`scripts.eval_dataset.validate_cases` 校验。歧义题标记为 `intent_only`，只评查询
是否保留歧义和实体边界，不进入检索覆盖分母。

冻结规则见 [`docs/eval/query-rewrite-expanded-rubric.md`](../../../docs/eval/query-rewrite-expanded-rubric.md)。
运行时必须同时提供 `frozen-v2.lock.json`；哈希或题数漂移时执行器会拒绝付费调用。

复现命令：

```bash
python3 -m scripts.query_rewrite_ablation.runner \
  --split frozen-v2 \
  --cases tests/eval/query_rewrite_ablation/frozen-v2.seeds.jsonl \
  --frozen-lock tests/eval/query_rewrite_ablation/frozen-v2.lock.json \
  --out tests/eval/query_rewrite_ablation/runs/frozen-v2 \
  --credential-env-file /path/to/.env \
  --allow-paid
```
