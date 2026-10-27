# Answer-quality depth artifacts

- `evidence/`: fixed Top-5 and Top-10 evidence arms for all 170 questions.
- `runs/dev-final3/`, `runs/frozen-v1-final3/`, `runs/retrieval-slate-holdout-v1-final3/`:
  final per-question answers, evidence, anonymous judge output, deterministic score,
  usage, latency, and generation-repair history.
- `aggregate.json`: split, denominator, bucket, evidence-state, coverage-attribution,
  judge, cost, and paired summaries.
- `manual-review.jsonl`: 17 audited uncertain, regression, behavior, and correction cases.
- `scoring-overlay-v1.json`: versioned corrections and their case-level basis.
- `experiment-ledger.sqlite3`: independent fail-closed budget ledger for all model calls.

The original generated answers are preserved in `generation_history` when a
`finish_reason=length` response had to be retried. Final artifacts have no empty or
length-truncated answers and no technical failures.

Reproduce the report from committed responses:

```bash
python3 -m scripts.eval_answer_quality_depth rescore \
  --run-dir tests/eval/answer_quality_depth/runs/dev-final3
python3 -m scripts.eval_answer_quality_depth rescore \
  --run-dir tests/eval/answer_quality_depth/runs/frozen-v1-final3
python3 -m scripts.eval_answer_quality_depth rescore \
  --run-dir tests/eval/answer_quality_depth/runs/retrieval-slate-holdout-v1-final3
python3 -m scripts.eval_answer_quality_depth report
```

Run the verification:

```bash
python3 -m pytest -q tests/eval/answer_quality_depth
```

The experiment is simulated data; it does not validate real order reads or writes.
Scoring v7 is a post-response audit. The original generation-time freeze is
`frozen-config.json`; `analysis-config-v7.json` records the final scoring audit.
