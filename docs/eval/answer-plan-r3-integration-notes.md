# Answer Plan R3 Integration Notes

## Scope

This branch intentionally does not edit `app/main.py`, `app/config.py`,
`app/graph/builder.py`, `app/graph/agent.py` or shared evaluation checks.
The optional node and factory are complete in `app/answer_plan/`.

Unified acceptance can either:

1. decline production wiring and keep the module/report as negative evidence;
2. apply the minimal patch in `answer-plan-r3-integration.patch` behind a
   default-off flag.

Given held-out did not show net quality gain, default should remain off.

## Required Settings

```text
ANSWER_PLAN_ENABLED=false
ANSWER_PLAN_LEDGER_PATH=
ANSWER_PLAN_MAX_COST_USD=2.0
ANSWER_PLAN_PRICING_PATH=config/pricing.yaml
```

`ANSWER_PLAN_ENABLED=true` with an empty ledger path fails during app creation.
This avoids silently creating an unbounded model path.

## Factory

`build_optional_answer_plan_runtime(settings, base_model=..., pricing=...)`
returns `None` unless enabled. When enabled it:

- builds an independent SQLite ledger and refuses pending reservations;
- wraps the shared model with pre-call estimates, atomic reservations, usage
  settlement and at most two transport retries;
- reuses `make_structured_extractor`, so DeepSeek thinking keeps the central
  JSON-schema compatibility path;
- returns a node factory that emits progress before the final answer.

## Minimal Graph Placement

For knowledge/FAQ paths:

```text
retrieve -> gate -> answer_plan -> log
agentic_retrieve -> answer_plan -> log
```

Keep order/tool-only and complaint routes on the existing agent:

```text
intent -> agent -> log
```

The new node consumes already retrieved evidence; it is not a replacement for
tool-using order queries. It must not be mounted after the legacy agent has
already generated and streamed a final answer.

## Compatibility

- All new settings have default-off behavior.
- Existing agent, AnswerVerifier, claim grounding, retrieval and tool paths are
  unchanged.
- New claim validation remains conservative; `uncertain` is not promoted.
- Rollback is `ANSWER_PLAN_ENABLED=false`; no DB migration is needed.
- Budget ledger is independent from claim-grounding and E2E ledgers.

## Conflicts

- Do not wire both `AnswerVerifier` and answer plan on the same final text
  without an explicit design review; it duplicates semantic checks and costs.
- Do not pass raw `state["evidence"]` without source metadata. The adapter
  preserves source IDs/titles when present, but missing metadata reduces
  attribution quality.
- Do not treat this fixed-evidence experiment as real retrieval quality evidence.

## Not Verified

The patch has not been applied to shared files. Full production graph,
SSE/citation persistence, tool-only tasks and real retrieval snapshots require
separate integration acceptance.
