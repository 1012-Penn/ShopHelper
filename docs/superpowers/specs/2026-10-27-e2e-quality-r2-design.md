# End-to-End RAG Quality Baseline R2 Design

## Goal

Establish a trustworthy, no-behavior-hint baseline for the real ShopHelper chat path, and explain
why the historical RAG numbers are not direct estimates of production quality.

This work uses a new 8-case development set and 20-case held-out set. It evaluates the production
FastAPI application, `POST /api/chat`, compiled graph, retrieval service, evidence gate, and tools.
It is not a fixed-evidence answer-prompt benchmark.

## Evidence Boundaries

- Historical traces are read-only. The existing dev/frozen-v1 split, scoring overlay, reports, and
  denominators are never rewritten.
- The old held-out set is design evidence, not a new independent validation set.
- New case sources are grouped by business fact/source template before split assignment. Dev and
  held-out do not share a source template.
- Case data, scorer, model/config contract, snapshot identity, and cache namespace are hashed before
  the paid run. No tuning follows observation of held-out results.
- Model input may contain the user's turns, production prompts, retrieved business documents, order
  tool results, and tool schemas. It must not contain `expected_behavior`, gold fact IDs, grading
  rules, required chunk IDs, or a final answer.
- Gold facts and scoring metadata are available only to the scorer after graph execution.
- The order catalog is deterministic business fixture data. Knowledge metadata comes from a private
  copy of the verified CH16 SQLite database; vectors come from a verified copy of the CH16 Milvus
  snapshot. No shared MySQL or Milvus is migrated or written.

## Historical Diagnosis

A zero-call analyzer reports, per historical case:

- source trace and original answer;
- expected and observed behavior;
- mapped evidence `[n] -> chunk_id -> source text`;
- fact, condition, and citation layer outcomes;
- whether the answer was reused by another historical RAG mode;
- one technical outcome class: model error, wiring/capture error, scoring/review uncertainty,
  not applicable, or technical failure.

The classifier preserves every historical case and never turns `uncertain` into either certainty
or hallucination. Exact multi-mode answer reuse is counted, not inferred from shared filenames.

## New Evaluation

### Runtime

- Build the app through the production `create_app()` factory.
- Run each user turn through `POST /api/chat` with an isolated SQLite database and snapshot copy.
- Instrument without changing production behavior:
  - wrap model/embedding/rerank clients for exact caching, budget reservation, usage and raw I/O;
  - wrap the compiled graph to retain the final state;
  - wrap the existing tool engine to retain tool arguments and results.
- Keep the default production switches unchanged: hybrid rerank RAG, answer verification off, gap
  trigger off.
- Use `deepseek-flash` as the business model from the project configuration. Coding-model names are
  not written into business configuration.

### Cost and failures

- The whole run has an independent persistent $2.00 protection cap.
- Every provider call reserves atomically before dispatch; pending reservations support concurrency.
- Successful calls settle against provider usage where available, otherwise conservative estimates.
- Embedding, rerank, evidence judging, generation, verification, and failed attempts are recorded.
- Network retries are bounded to two retries with bounded 429 backoff.
- Existing exact embedding/rerank caches are copied and reused. Real answer-model runs are not
  replaced by historical cache replay.

### Scoring

The deterministic scorer reports each layer separately:

- expected vs observed behavior;
- required facts with entity, number, and unit;
- required conditions and forbidden claims;
- citation existence and tri-state semantic support;
- required evidence/source coverage, entity consistency, and latest user correction;
- route, tool selection/arguments/completion, and technical failure;
- task success only when all required layers pass and citation support is positive.

`uncertain` is reported as `review_required`; it is neither certain success nor hallucination.
Answers requiring semantic judgment are written to a manual review queue. Any optional model label
is named a model label, not human adjudication.

### Deliverables

- `scripts/eval_e2e_quality_r2.py` runner;
- private harness/scoring modules;
- frozen dev/held-out cases and integrity manifest;
- focused input-boundary/cache/budget/scoring tests;
- per-attempt JSONL with prompts, responses, tool I/O and source mappings;
- aggregate results with token, cost and end-to-end P95;
- report, dev notes, and minimal integration notes.

## Compatibility and rollback

No production entry, graph, shared scorer, schema, or historical artifact is edited. The runner is
standalone and default-off. Integration is unnecessary unless diagnosis identifies a production
bug; in that case integration notes specify a minimal reproduced patch.
