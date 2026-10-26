# Claim Grounding R2 复现

## 文件和冻结

- `data/dev.jsonl`：12 条 dev claim；
- `data/heldout.jsonl`：32 条独立对抗 claim；
- `frozen-config.json`：冻结数据、prompt、价格表和停止规则；
- `results/`：逐 claim 结果、原始 prompt/response、账本与聚合；
- `cache/exact-responses.sqlite`：本地 exact-cache，不提交凭据。

冻结后不再根据 held-out 修改 prompt、数据或停止规则。

## 命令

```bash
PY=/root/workplace/ShopHelper/.venv/bin/python

$PY -m scripts.eval_claim_grounding_r2 freeze
$PY -m scripts.eval_claim_grounding_r2 \
  run --mode rules --split dev --run-id rules-dev-1
$PY -m scripts.eval_claim_grounding_r2 \
  run --mode selective --split dev --run-id selective-dev-1 \
  --allow-paid --credential-env /root/workplace/ShopHelper/.env \
  --max-cost-usd 2 --measure-cache-replay
$PY -m scripts.eval_claim_grounding_r2 \
  run --mode selective --split heldout --run-id selective-heldout-1 \
  --allow-paid --credential-env /root/workplace/ShopHelper/.env \
  --max-cost-usd 2 --measure-cache-replay
$PY -m scripts.eval_claim_grounding_r2 rescore --run-id selective-heldout-1
```

`--allow-paid` 是付费调用的显式闸门。脚本不会把 API key 写入结果。
预算账本上限硬性限制在不超过 $2.00；已有 run-id 默认拒绝覆盖。

## 产物口径

- `results.jsonl`：逐 claim 的 v3.1、硬规则、最终三态、使用证据和来源；
- `attempts.jsonl`：每次真实/prompt cache 尝试及原始模型响应；
- `budget-records.json`：预留、usage 与结算；
- `aggregate.json`：该 run 作为最终 arm 的聚合；
- `derived-comparison.json`：同一批 claim 上 v3.1、hard guard 和 selective
  三个零调用重算视图；
- `cache-replay.json`：精确缓存重放延迟，不是线上 P95。
