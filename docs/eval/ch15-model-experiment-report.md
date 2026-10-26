# CH15 Real-Model Paired Experiment

**Status:** `complete`
**Frozen config:** `sha256:5acb5125defd5678da54c5f4a874765be128ee0dad0d1096b9402301478d9410`
**Scoring version:** `semantic_task_v2`
**Scope:** deterministic simulated scenarios only; no production/customer data.
**Semantic task success** uses the model-generated answer, required safe evidence and database assertions.
The original strict contract score remains visible; it required the full action sequence and literal free-text arguments.
Refusal alignment uses each scenario's frozen gold; it is not assumed successful by default.

Runs: 60 arm-runs / 30 frozen pairs.

## Completion metrics

| Arm | Semantic task success | Original strict contract | Unconfirmed writes | Reasonable handoff | Refusal gold alignment |
|---|---:|---:|---:|---:|---:|
| Fixed refund flow | 20/30 (66.7%) | 11/30 (36.7%) | 0/0 (N/A) | 1/1 (100.0%) | 24/30 (80.0%) |
| Constrained Agent | 28/30 (93.3%) | 1/30 (3.3%) | 0/0 (N/A) | 1/1 (100.0%) | 29/30 (96.7%) |

Each metric includes its numerator and denominator. Handoff may be N/A when no frozen case requires handoff. The paid paired set exposes only read tools, so its write-attempt denominator is zero; all SQLite ticket counts are separately checked as zero.

## Tool and parameter metrics

| Arm | Exact action sequence | Required tool coverage | Required identifiers and schema | Literal call agreement |
|---|---:|---:|---:|---:|
| Fixed refund flow | 17/30 (56.7%) | 30/30 (100.0%) | 34/34 (100.0%) | 2/34 (5.9%) |
| Constrained Agent | 24/30 (80.0%) | 30/30 (100.0%) | 68/68 (100.0%) | 34/72 (47.2%) |

Required tool coverage uses the final turn's needs and lets valid evidence from earlier turns survive a revision. Required identifiers and schema check order IDs exactly and require non-empty tool arguments. Literal call agreement is retained only as a diagnostic.

## Paired semantic comparison

- fixed: 20/30 (66.7%)
- autonomous: 28/30 (93.3%)
- Autonomous minus fixed task-success delta: +26.7%.

## Cost and execution

- Charged upper bound: $0.044894 / $17.40.
- Usage-based cost at pinned peak rates: $0.044894.
- Pending reserved amount: $0.000000.
- Stop reason: none.
- Cost uses provider token usage at pinned peak rates; invoice charges can be lower during off-peak hours. Calls without cache breakdown or usage are charged at the conservative upper bound.
- Every model call is serial, max_retries=0, and bounded before dispatch. Provider errors stop subsequent calls; missing usage is charged at the full per-call envelope.
- Calls and cost by arm:
  - fixed: 175/378 calls; usage cost at peak rates $0.024311; charged upper bound $0.024311; cost-upper-bound calls 0; cap $9.20.
  - autonomous: 166/336 calls; usage cost at peak rates $0.020583; charged upper bound $0.020583; cost-upper-bound calls 0; cap $8.20.
- `ch15-model-case-report.jsonl` stores each case's full prompt/response trace, graph/tool trace, DB assertion results and score. It must remain simulated-only.

## Scoring correction

- The original task-success boolean mixed user-task completion with exact tool order and literal free-text FAQ wording. It failed equivalent values such as `1 到 3 个工作日` versus `1到3个工作日`.
- This report now separates semantic task success from the original strict contract and exact call diagnostics.
- Natural-language clarification now recognizes requests such as `麻烦提供一下订单号` and post-lookup confirmation such as `核对订单号`. Explicit refusals still follow each case's gold.
- `CH15-RFD-010` is explicitly corrected because the user excluded order 1001 while the original gold required querying it. `CH15-REV-009` changes from `answer` to `clarify` because the receipt date needed to answer is unavailable.
- Task success uses the final user request, lets earlier valid read evidence survive a revision, and treats harmless extra reads separately from exact action compliance. Wrong or missing order identifiers, forbidden actions, refusal mismatches, required handoff, prohibited claims and database assertions remain failures.
- Existing traces were re-scored offline. No model or external business call was made, and each trace records both `legacy_score` and the semantic score.

## Remaining failures

- `CH15-REV-004` / `fixed`: includes_correct, outcome_correct.
- `CH15-REV-006` / `fixed`: handoff_correct, outcome_correct, refusal_correct.
- `CH15-REV-008` / `autonomous`: handoff_correct, includes_correct, outcome_correct, refusal_correct.
- `CH15-REV-008` / `fixed`: handoff_correct, includes_correct, outcome_correct, refusal_correct.
- `CH15-REV-009` / `fixed`: handoff_correct, outcome_correct, refusal_correct.
- `CH15-REV-011` / `autonomous`: includes_correct.
- `CH15-REV-011` / `fixed`: includes_correct.
- `CH15-REV-012` / `fixed`: includes_correct.
- `CH15-RFD-005` / `fixed`: includes_correct.
- `CH15-RFD-007` / `fixed`: handoff_correct, outcome_correct, refusal_correct.
- `CH15-RFD-013` / `fixed`: handoff_correct, outcome_correct, refusal_correct.
- `CH15-RFD-014` / `fixed`: handoff_correct, includes_correct, outcome_correct, refusal_correct.

## Case results

- `CH15-REV-001` / `autonomous`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-REV-001:autonomous:attempt-1`, calls=9, latency=9566 ms, failures=none, diagnostics=none
- `CH15-REV-001` / `fixed`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-REV-001:fixed:attempt-1`, calls=10, latency=10799 ms, failures=none, diagnostics=none
- `CH15-REV-002` / `autonomous`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-REV-002:autonomous:attempt-1`, calls=8, latency=8827 ms, failures=none, diagnostics=none
- `CH15-REV-002` / `fixed`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-REV-002:fixed:attempt-1`, calls=9, latency=9898 ms, failures=none, diagnostics=none
- `CH15-REV-003` / `autonomous`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-REV-003:autonomous:attempt-1`, calls=8, latency=7784 ms, failures=none, diagnostics=handoff_correct
- `CH15-REV-003` / `fixed`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-REV-003:fixed:attempt-1`, calls=9, latency=9759 ms, failures=none, diagnostics=handoff_correct
- `CH15-REV-004` / `autonomous`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-REV-004:autonomous:attempt-1`, calls=8, latency=7427 ms, failures=none, diagnostics=none
- `CH15-REV-004` / `fixed`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-REV-004:fixed:attempt-1`, calls=6, latency=6215 ms, failures=outcome_correct, includes_correct, diagnostics=none
- `CH15-REV-005` / `autonomous`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-REV-005:autonomous:attempt-1`, calls=8, latency=8565 ms, failures=none, diagnostics=none
- `CH15-REV-005` / `fixed`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-REV-005:fixed:attempt-1`, calls=8, latency=9891 ms, failures=none, diagnostics=none
- `CH15-REV-006` / `autonomous`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-REV-006:autonomous:attempt-1`, calls=8, latency=7712 ms, failures=none, diagnostics=none
- `CH15-REV-006` / `fixed`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-REV-006:fixed:attempt-1`, calls=8, latency=8397 ms, failures=refusal_correct, handoff_correct, outcome_correct, diagnostics=none
- `CH15-REV-007` / `autonomous`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-REV-007:autonomous:attempt-1`, calls=8, latency=7274 ms, failures=none, diagnostics=none
- `CH15-REV-007` / `fixed`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-REV-007:fixed:attempt-1`, calls=8, latency=7521 ms, failures=none, diagnostics=none
- `CH15-REV-008` / `autonomous`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-REV-008:autonomous:attempt-1`, calls=8, latency=7646 ms, failures=refusal_correct, handoff_correct, outcome_correct, includes_correct, diagnostics=none
- `CH15-REV-008` / `fixed`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-REV-008:fixed:attempt-1`, calls=8, latency=7969 ms, failures=refusal_correct, handoff_correct, outcome_correct, includes_correct, diagnostics=none
- `CH15-REV-009` / `autonomous`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-REV-009:autonomous:attempt-1`, calls=7, latency=6739 ms, failures=none, diagnostics=none
- `CH15-REV-009` / `fixed`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-REV-009:fixed:attempt-1`, calls=7, latency=7548 ms, failures=refusal_correct, handoff_correct, outcome_correct, diagnostics=none
- `CH15-REV-010` / `autonomous`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-REV-010:autonomous:attempt-1`, calls=8, latency=7466 ms, failures=none, diagnostics=none
- `CH15-REV-010` / `fixed`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-REV-010:fixed:attempt-1`, calls=8, latency=7633 ms, failures=none, diagnostics=none
- `CH15-REV-011` / `autonomous`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-REV-011:autonomous:attempt-1`, calls=7, latency=6755 ms, failures=includes_correct, diagnostics=none
- `CH15-REV-011` / `fixed`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-REV-011:fixed:attempt-1`, calls=9, latency=10072 ms, failures=includes_correct, diagnostics=none
- `CH15-REV-012` / `autonomous`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-REV-012:autonomous:attempt-1`, calls=8, latency=7946 ms, failures=none, diagnostics=none
- `CH15-REV-012` / `fixed`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-REV-012:fixed:attempt-1`, calls=9, latency=9104 ms, failures=includes_correct, diagnostics=none
- `CH15-RFD-001` / `autonomous`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-001:autonomous:attempt-1`, calls=4, latency=3970 ms, failures=none, diagnostics=none
- `CH15-RFD-001` / `fixed`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-001:fixed:attempt-2`, calls=7, latency=3972 ms, failures=none, diagnostics=none
- `CH15-RFD-002` / `autonomous`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-002:autonomous:attempt-1`, calls=4, latency=4159 ms, failures=none, diagnostics=none
- `CH15-RFD-002` / `fixed`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-002:fixed:attempt-1`, calls=4, latency=3304 ms, failures=none, diagnostics=none
- `CH15-RFD-003` / `autonomous`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-003:autonomous:attempt-1`, calls=4, latency=4265 ms, failures=none, diagnostics=none
- `CH15-RFD-003` / `fixed`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-003:fixed:attempt-1`, calls=4, latency=4427 ms, failures=none, diagnostics=none
- `CH15-RFD-004` / `autonomous`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-004:autonomous:attempt-1`, calls=4, latency=4293 ms, failures=none, diagnostics=none
- `CH15-RFD-004` / `fixed`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-004:fixed:attempt-1`, calls=4, latency=4670 ms, failures=none, diagnostics=none
- `CH15-RFD-005` / `autonomous`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-005:autonomous:attempt-1`, calls=4, latency=4445 ms, failures=none, diagnostics=none
- `CH15-RFD-005` / `fixed`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-005:fixed:attempt-1`, calls=4, latency=4517 ms, failures=includes_correct, diagnostics=none
- `CH15-RFD-006` / `autonomous`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-006:autonomous:attempt-1`, calls=4, latency=4611 ms, failures=none, diagnostics=none
- `CH15-RFD-006` / `fixed`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-006:fixed:attempt-1`, calls=4, latency=4715 ms, failures=none, diagnostics=none
- `CH15-RFD-007` / `autonomous`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-007:autonomous:attempt-1`, calls=4, latency=4648 ms, failures=none, diagnostics=none
- `CH15-RFD-007` / `fixed`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-007:fixed:attempt-1`, calls=4, latency=4467 ms, failures=refusal_correct, handoff_correct, outcome_correct, diagnostics=none
- `CH15-RFD-008` / `autonomous`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-008:autonomous:attempt-1`, calls=3, latency=3500 ms, failures=none, diagnostics=none
- `CH15-RFD-008` / `fixed`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-008:fixed:attempt-1`, calls=2, latency=2243 ms, failures=none, diagnostics=none
- `CH15-RFD-009` / `autonomous`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-009:autonomous:attempt-1`, calls=4, latency=4588 ms, failures=none, diagnostics=none
- `CH15-RFD-009` / `fixed`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-009:fixed:attempt-1`, calls=4, latency=4253 ms, failures=none, diagnostics=none
- `CH15-RFD-010` / `autonomous`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-010:autonomous:attempt-1`, calls=4, latency=4987 ms, failures=none, diagnostics=none
- `CH15-RFD-010` / `fixed`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-010:fixed:attempt-1`, calls=4, latency=5013 ms, failures=none, diagnostics=none
- `CH15-RFD-011` / `autonomous`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-011:autonomous:attempt-1`, calls=4, latency=4163 ms, failures=none, diagnostics=none
- `CH15-RFD-011` / `fixed`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-011:fixed:attempt-1`, calls=4, latency=4459 ms, failures=none, diagnostics=none
- `CH15-RFD-012` / `autonomous`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-012:autonomous:attempt-1`, calls=4, latency=3969 ms, failures=none, diagnostics=none
- `CH15-RFD-012` / `fixed`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-012:fixed:attempt-1`, calls=4, latency=4425 ms, failures=none, diagnostics=none
- `CH15-RFD-013` / `autonomous`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-013:autonomous:attempt-1`, calls=4, latency=4184 ms, failures=none, diagnostics=none
- `CH15-RFD-013` / `fixed`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-013:fixed:attempt-1`, calls=5, latency=5118 ms, failures=refusal_correct, handoff_correct, outcome_correct, diagnostics=none
- `CH15-RFD-014` / `autonomous`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-014:autonomous:attempt-1`, calls=4, latency=4581 ms, failures=none, diagnostics=none
- `CH15-RFD-014` / `fixed`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-014:fixed:attempt-1`, calls=4, latency=4602 ms, failures=refusal_correct, handoff_correct, outcome_correct, includes_correct, diagnostics=none
- `CH15-RFD-015` / `autonomous`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-015:autonomous:attempt-1`, calls=4, latency=4216 ms, failures=none, diagnostics=none
- `CH15-RFD-015` / `fixed`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-015:fixed:attempt-1`, calls=4, latency=4061 ms, failures=none, diagnostics=none
- `CH15-RFD-016` / `autonomous`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-016:autonomous:attempt-1`, calls=4, latency=4417 ms, failures=none, diagnostics=none
- `CH15-RFD-016` / `fixed`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-016:fixed:attempt-1`, calls=4, latency=4525 ms, failures=none, diagnostics=none
- `CH15-RFD-017` / `autonomous`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-017:autonomous:attempt-1`, calls=4, latency=4386 ms, failures=none, diagnostics=none
- `CH15-RFD-017` / `fixed`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-017:fixed:attempt-1`, calls=5, latency=5971 ms, failures=none, diagnostics=none
- `CH15-RFD-018` / `autonomous`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-018:autonomous:attempt-1`, calls=4, latency=4343 ms, failures=none, diagnostics=none
- `CH15-RFD-018` / `fixed`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-018:fixed:attempt-1`, calls=5, latency=6322 ms, failures=none, diagnostics=none

All rates describe this frozen set of simulated scenarios only. They are not real support outcomes or customer-service ROI.
