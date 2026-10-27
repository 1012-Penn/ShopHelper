# Evidence-slate evaluation artifacts

- `runs/dev/`: rule-development split, replayed from the same reranker output.
- `runs/frozen-regression/`: previously seen frozen-v1, regression only.
- `runs/holdout/`: frozen independent 20-case set, run once before any rule change.
- `runs/latency/`: ten paired top-10/top-50 reranker latency probes in two request orders.
- `attempt-reconciliation.json`: aggregate API attempts and cost guardrail.

Reproduce the independent run:

```bash
python3 -m scripts.eval_evidence_slate \
  --split retrieval-slate-holdout-v1 \
  --cases tests/eval/splits/retrieval-slate-holdout-v1.jsonl \
  --out /tmp/evidence-slate-holdout \
  --credential-env-file /path/to/read-only/.env \
  --allow-paid --max-cost-usd 2
```

Full method, metrics, failure cases, and enable recommendation:
[`docs/eval/retrieval-evidence-slate-report.md`](../../../docs/eval/retrieval-evidence-slate-report.md).
