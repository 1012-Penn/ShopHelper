# Typed Handoff R2 Fixtures

The 30 scenarios are split by `group` before arm execution: all rows in one source
template group belong to exactly one split. `dev` has 10 rows and `held-out` has
20. Gold is visible only to `scoring.py`; harness projections contain the current
user message and prior assistant/user turns only.

The catalog and policy files are isolated simulated fixtures. They are not copied
from or written to shared MySQL/Milvus.

Run the frozen live experiment once:

```bash
/root/workplace/ShopHelper/.venv/bin/python \
  scripts/typed_handoff_r2_eval.py --allow-paid
```

The harness refuses to overwrite existing results or the isolated ledger. Re-score
the preserved responses without model calls:

```bash
/root/workplace/ShopHelper/.venv/bin/python \
  scripts/typed_handoff_r2_rescore.py
```

The paid run used scoring v1 as initially frozen. The final semantic scorer is v3;
every response was re-scored uniformly with zero model calls. Do not replace v1
artifacts or remove difficult cases.
