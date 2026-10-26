# Adaptive Routing R2 Implementation Plan

Date: 2026-10-27

Baseline: `446fbb7f9972875b7649555cb9b4b19e402cea29`

Branch: `codex/opt-adaptive-routing-r2-20261027`

## Phase 1: Routing Contract

1. Add failing contract tests for strict visible input, deterministic features, simple
   versus complex decisions, write-risk permission invariance, and policy hashing.
2. Implement the immutable routing models and feature extractor in
   `app/adaptive_routing/`.
3. Add a failing test for optional structured judgment; implement bounded uncertain-band
   classification and cost recording.
4. Add a failing test for exact-once dispatch and no failure fallback; implement response
   normalization and route/execution timing.

## Phase 2: Optional API Boundary

1. Add failing tests for an independently mounted FastAPI factory using an injected
   dispatcher service.
2. Implement the optional router/factory without importing or changing `app.main`.
3. Document the existing single/multi session, SSE, persistence, and preview differences
   and the smallest production adapter patch.

## Phase 3: Frozen Evaluation

1. Author 10 dev and 24 held-out cases in source-template groups, plus manifest and
   scoring rubric.
2. Hash every scenario, rubric, route policy, model setting, and stop rule before the
   held-out run.
3. Add offline contract tests proving gold never crosses the executor/router input and
   all three arms receive the same user turns.
4. Implement the standalone evaluation script with single/multi/adaptive arms, paired
   metrics, route diagnostics, and unknown-usage conservatism.
5. Run offline harness validation before any paid request.

## Phase 4: Real-Model Comparison

1. Use an isolated SQLite ledger with a USD 2.00 atomic pre-call cap and concurrent
   pending reservations; never reuse old ledger state.
2. Run the 10 dev cases first only to detect infrastructure failures, without changing
   frozen routing rules.
3. Run the frozen 34-case three-arm comparison once. Preserve raw prompts, responses,
   provider usage, failures, and all cases in the denominator.
4. Perform at most two bounded transient retries; no held-out tuning or selective reruns.

## Phase 5: Verification and Handoff

1. Run targeted routing, evaluator, and existing multi-Agent compatibility tests.
2. Run the full repository suite once after targeted checks.
3. Write the report, dev notes, and integration notes with actual evidence, limitations,
   rollback steps, and two evidence-backed resume candidates.
4. Commit only this branch and report the final commit and unperformed integration steps.
