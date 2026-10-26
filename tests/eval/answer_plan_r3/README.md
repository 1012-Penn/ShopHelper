# Answer Plan R3 Fixed-Evidence Evaluation

This directory contains a frozen 8-dev / 20-held-out diagnostic for comparing:

- `free_text`: one final-answer call over the same candidate evidence;
- `answer_plan`: a structured plan, deterministic binding validation, at most one
  repair call, and a rendered final answer.

The dataset is split by source group and evidence template. Model inputs include
only the user question and candidate evidence. Gold facts, expected behavior and
scoring fields are used only by the independent scorer.

## Reproduce

```bash
/root/workplace/ShopHelper/.venv/bin/python -m scripts.eval_answer_plan_r3 \
  --split dev --allow-paid --max-cost-usd 2.0
```

Held-out is guarded by `runs/heldout-once.json` and must not be rerun.
The independent ledger is `runs/answer-plan-r3-paid.sqlite3`.

## Rescore Saved Trajectories

```bash
/root/workplace/ShopHelper/.venv/bin/python -m scripts.rescore_answer_plan_r3 \
  --results tests/eval/answer_plan_r3/runs/heldout-20261027T024707Z/results.jsonl \
  --output /tmp/answer-plan-r3-heldout-v2/results.jsonl \
  --scoring-version v2
```

`score-v1` is the frozen primary result. `score-v2` is a post-run audit that
fixes three deterministic scoring defects; it is not independent validation.
