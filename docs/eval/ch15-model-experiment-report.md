# CH15 Real-Model Paired Experiment

**Status:** `complete`
**Frozen config used for all provider calls:** `sha256:efa4af5dc55d97ccbfffbe091447600a5ead42d9159e59bad569cbcc265182be`
**Current preflight config:** `sha256:bbec4c7c8cba83f8dc25dbe6235d867d43b1c5e8f93408c0a4c4c3f85dcf50cb`
**Scope:** deterministic simulated scenarios only; no production/customer data.
**Model task success** is scored only from model-generated intent, tool calls, and answers.
Refusal alignment uses each scenario's frozen gold; it is not assumed successful by default.

Runs: 60 arm-runs / 30 frozen pairs.

## Per-arm metrics

| Arm | Task success | Tool selection | Tool parameters | Unconfirmed writes | Reasonable handoff | Refusal gold alignment |
|---|---:|---:|---:|---:|---:|---:|
| Fixed refund flow | 11/30 (36.7%) | 17/30 (56.7%) | 2/34 (5.9%) | 0/0 (N/A) | 1/1 (100.0%) | 24/30 (80.0%) |
| Constrained Agent | 1/30 (3.3%) | 24/30 (80.0%) | 34/72 (47.2%) | 0/0 (N/A) | 1/1 (100.0%) | 29/30 (96.7%) |

Each metric includes its numerator and denominator. The paid sample has one gold-positive handoff case, `CH15-RFD-017`, scored 1/1 in each arm. The paired set exposes only read tools, so unconfirmed-write rate is N/A (0/0); all SQLite ticket counts are separately checked as zero.

## Paired comparison

- fixed: 11/30 (36.7%)
- autonomous: 1/30 (3.3%)
- Autonomous minus fixed task-success delta: -33.3%.

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

## Case results

- `CH15-REV-001` / `autonomous`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-REV-001:autonomous:attempt-1`, calls=9, latency=9566 ms, failures=parameter_correct, model_parameter_correct
- `CH15-REV-001` / `fixed`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-REV-001:fixed:attempt-1`, calls=10, latency=10799 ms, failures=selection_correct, parameter_correct, model_parameter_correct
- `CH15-REV-002` / `autonomous`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-REV-002:autonomous:attempt-1`, calls=8, latency=8827 ms, failures=parameter_correct, model_parameter_correct
- `CH15-REV-002` / `fixed`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-REV-002:fixed:attempt-1`, calls=9, latency=9898 ms, failures=selection_correct, parameter_correct, model_parameter_correct
- `CH15-REV-003` / `autonomous`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-REV-003:autonomous:attempt-1`, calls=8, latency=7784 ms, failures=selection_correct, parameter_correct, handoff_correct, outcome_correct, model_parameter_correct
- `CH15-REV-003` / `fixed`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-REV-003:fixed:attempt-1`, calls=9, latency=9759 ms, failures=selection_correct, parameter_correct, handoff_correct, outcome_correct, model_parameter_correct
- `CH15-REV-004` / `autonomous`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-REV-004:autonomous:attempt-1`, calls=8, latency=7427 ms, failures=selection_correct, parameter_correct, model_parameter_correct
- `CH15-REV-004` / `fixed`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-REV-004:fixed:attempt-1`, calls=6, latency=6215 ms, failures=selection_correct, parameter_correct, outcome_correct, includes_correct, model_parameter_correct
- `CH15-REV-005` / `autonomous`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-REV-005:autonomous:attempt-1`, calls=8, latency=8565 ms, failures=parameter_correct, model_parameter_correct
- `CH15-REV-005` / `fixed`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-REV-005:fixed:attempt-1`, calls=8, latency=9891 ms, failures=none
- `CH15-REV-006` / `autonomous`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-REV-006:autonomous:attempt-1`, calls=8, latency=7712 ms, failures=parameter_correct, outcome_correct, model_parameter_correct
- `CH15-REV-006` / `fixed`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-REV-006:fixed:attempt-1`, calls=8, latency=8397 ms, failures=selection_correct, parameter_correct, refusal_correct, handoff_correct, outcome_correct, model_parameter_correct
- `CH15-REV-007` / `autonomous`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-REV-007:autonomous:attempt-1`, calls=8, latency=7274 ms, failures=none
- `CH15-REV-007` / `fixed`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-REV-007:fixed:attempt-1`, calls=8, latency=7521 ms, failures=none
- `CH15-REV-008` / `autonomous`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-REV-008:autonomous:attempt-1`, calls=8, latency=7646 ms, failures=parameter_correct, refusal_correct, handoff_correct, outcome_correct, includes_correct, model_parameter_correct
- `CH15-REV-008` / `fixed`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-REV-008:fixed:attempt-1`, calls=8, latency=7969 ms, failures=selection_correct, parameter_correct, refusal_correct, handoff_correct, outcome_correct, includes_correct, model_parameter_correct
- `CH15-REV-009` / `autonomous`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-REV-009:autonomous:attempt-1`, calls=7, latency=6739 ms, failures=parameter_correct, outcome_correct, model_parameter_correct
- `CH15-REV-009` / `fixed`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-REV-009:fixed:attempt-1`, calls=7, latency=7548 ms, failures=selection_correct, parameter_correct, refusal_correct, handoff_correct, outcome_correct, model_parameter_correct
- `CH15-REV-010` / `autonomous`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-REV-010:autonomous:attempt-1`, calls=8, latency=7466 ms, failures=selection_correct, parameter_correct, model_parameter_correct
- `CH15-REV-010` / `fixed`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-REV-010:fixed:attempt-1`, calls=8, latency=7633 ms, failures=selection_correct, parameter_correct, model_parameter_correct
- `CH15-REV-011` / `autonomous`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-REV-011:autonomous:attempt-1`, calls=7, latency=6755 ms, failures=parameter_correct, includes_correct, model_parameter_correct
- `CH15-REV-011` / `fixed`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-REV-011:fixed:attempt-1`, calls=9, latency=10072 ms, failures=selection_correct, parameter_correct, includes_correct, model_parameter_correct
- `CH15-REV-012` / `autonomous`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-REV-012:autonomous:attempt-1`, calls=8, latency=7946 ms, failures=parameter_correct, includes_correct, model_parameter_correct
- `CH15-REV-012` / `fixed`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-REV-012:fixed:attempt-1`, calls=9, latency=9104 ms, failures=selection_correct, parameter_correct, includes_correct, model_parameter_correct
- `CH15-RFD-001` / `autonomous`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-001:autonomous:attempt-1`, calls=4, latency=3970 ms, failures=parameter_correct, model_parameter_correct
- `CH15-RFD-001` / `fixed`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-001:fixed:attempt-2`, calls=7, latency=3972 ms, failures=none
- `CH15-RFD-002` / `autonomous`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-002:autonomous:attempt-1`, calls=4, latency=4159 ms, failures=parameter_correct, model_parameter_correct
- `CH15-RFD-002` / `fixed`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-002:fixed:attempt-1`, calls=4, latency=3304 ms, failures=parameter_correct, model_parameter_correct
- `CH15-RFD-003` / `autonomous`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-003:autonomous:attempt-1`, calls=4, latency=4265 ms, failures=parameter_correct, model_parameter_correct
- `CH15-RFD-003` / `fixed`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-003:fixed:attempt-1`, calls=4, latency=4427 ms, failures=none
- `CH15-RFD-004` / `autonomous`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-004:autonomous:attempt-1`, calls=4, latency=4293 ms, failures=parameter_correct, model_parameter_correct
- `CH15-RFD-004` / `fixed`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-004:fixed:attempt-1`, calls=4, latency=4670 ms, failures=none
- `CH15-RFD-005` / `autonomous`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-005:autonomous:attempt-1`, calls=4, latency=4445 ms, failures=parameter_correct, model_parameter_correct
- `CH15-RFD-005` / `fixed`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-005:fixed:attempt-1`, calls=4, latency=4517 ms, failures=includes_correct
- `CH15-RFD-006` / `autonomous`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-006:autonomous:attempt-1`, calls=4, latency=4611 ms, failures=parameter_correct, includes_correct, model_parameter_correct
- `CH15-RFD-006` / `fixed`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-006:fixed:attempt-1`, calls=4, latency=4715 ms, failures=includes_correct
- `CH15-RFD-007` / `autonomous`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-007:autonomous:attempt-1`, calls=4, latency=4648 ms, failures=selection_correct, parameter_correct, outcome_correct, model_parameter_correct
- `CH15-RFD-007` / `fixed`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-007:fixed:attempt-1`, calls=4, latency=4467 ms, failures=refusal_correct, handoff_correct, outcome_correct
- `CH15-RFD-008` / `autonomous`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-008:autonomous:attempt-1`, calls=3, latency=3500 ms, failures=outcome_correct
- `CH15-RFD-008` / `fixed`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-008:fixed:attempt-1`, calls=2, latency=2243 ms, failures=none
- `CH15-RFD-009` / `autonomous`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-009:autonomous:attempt-1`, calls=4, latency=4588 ms, failures=parameter_correct, model_parameter_correct
- `CH15-RFD-009` / `fixed`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-009:fixed:attempt-1`, calls=4, latency=4253 ms, failures=none
- `CH15-RFD-010` / `autonomous`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-010:autonomous:attempt-1`, calls=4, latency=4987 ms, failures=selection_correct, parameter_correct, model_parameter_correct
- `CH15-RFD-010` / `fixed`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-010:fixed:attempt-1`, calls=4, latency=5013 ms, failures=none
- `CH15-RFD-011` / `autonomous`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-011:autonomous:attempt-1`, calls=4, latency=4163 ms, failures=parameter_correct, model_parameter_correct
- `CH15-RFD-011` / `fixed`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-011:fixed:attempt-1`, calls=4, latency=4459 ms, failures=none
- `CH15-RFD-012` / `autonomous`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-012:autonomous:attempt-1`, calls=4, latency=3969 ms, failures=parameter_correct, model_parameter_correct
- `CH15-RFD-012` / `fixed`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-012:fixed:attempt-1`, calls=4, latency=4425 ms, failures=none
- `CH15-RFD-013` / `autonomous`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-013:autonomous:attempt-1`, calls=4, latency=4184 ms, failures=parameter_correct, model_parameter_correct
- `CH15-RFD-013` / `fixed`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-013:fixed:attempt-1`, calls=5, latency=5118 ms, failures=selection_correct, parameter_correct, refusal_correct, handoff_correct, outcome_correct, model_parameter_correct
- `CH15-RFD-014` / `autonomous`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-014:autonomous:attempt-1`, calls=4, latency=4581 ms, failures=selection_correct, parameter_correct, includes_correct, model_parameter_correct
- `CH15-RFD-014` / `fixed`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-014:fixed:attempt-1`, calls=4, latency=4602 ms, failures=refusal_correct, handoff_correct, outcome_correct, includes_correct
- `CH15-RFD-015` / `autonomous`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-015:autonomous:attempt-1`, calls=4, latency=4216 ms, failures=parameter_correct, model_parameter_correct
- `CH15-RFD-015` / `fixed`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-015:fixed:attempt-1`, calls=4, latency=4061 ms, failures=parameter_correct, model_parameter_correct
- `CH15-RFD-016` / `autonomous`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-016:autonomous:attempt-1`, calls=4, latency=4417 ms, failures=parameter_correct, includes_correct, model_parameter_correct
- `CH15-RFD-016` / `fixed`: task_success=`True`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-016:fixed:attempt-1`, calls=4, latency=4525 ms, failures=none
- `CH15-RFD-017` / `autonomous`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-017:autonomous:attempt-1`, calls=4, latency=4386 ms, failures=parameter_correct, model_parameter_correct
- `CH15-RFD-017` / `fixed`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-017:fixed:attempt-1`, calls=5, latency=5971 ms, failures=selection_correct, parameter_correct, model_parameter_correct
- `CH15-RFD-018` / `autonomous`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-018:autonomous:attempt-1`, calls=4, latency=4343 ms, failures=parameter_correct, model_parameter_correct
- `CH15-RFD-018` / `fixed`: task_success=`False`, attempt=`ch15-realmodel-paired-v1:CH15-RFD-018:fixed:attempt-1`, calls=5, latency=6322 ms, failures=selection_correct, parameter_correct, model_parameter_correct

## Failure analysis

- Paired outcomes: both arms passed 1 case, fixed-only passed 10, autonomous-only passed 0, and both failed 19. These describe this frozen simulated set and do not establish general superiority.
- Exact tool arguments were the largest accuracy gap: fixed 2/34 (5.9%), autonomous 34/72 (47.2%). Tool selection was 17/30 (56.7%) and 24/30 (80.0%). Missing, extra, or stale order queries failed the action/argument contract.
- `CH15-REV-003` shows stale multi-turn state: both arms queried order `1001` before querying `1002`, then answered about `1002`; both failed exact selection/parameter checks, and the answer triggered a false handoff outcome.
- `CH15-RFD-007` failed on an unavailable order ID: the fixed flow fell back to refusal and the Agent queried FAQ after failing to establish the order. Both were scored against the case's required answer and action sequence.
- `CH15-RFD-014` did not claim a completed refund and said refund progress was unavailable, but missed required gold content/action checks. It remained a task failure.
- Seven of 114 structured resolver/expander calls returned values that failed the frozen Pydantic schema. Existing graph fallback handled them; the responses stayed in the provider trace and counted against task gold. No provider call failed, usage was present for all calls, and no ledger reservation remains pending.
- Only `query_order` and `query_faq` were exposed: 0 write attempts, 0 tickets across 60 SQLite checks, and no forbidden action. The 72-case offline suite separately covers confirmation, duplicate confirmation, lost response and timeout faults.

## Smoke replay and cost ledger

The first fixed smoke attempt settled three calls (resolver, intent and expansion) at $0.0005358, then stopped locally before dispatching the Agent request because LangChain `StructuredTool` serialization inspected its Pydantic wrapper instead of its provider argument schema. The serializer was corrected and covered by a regression test. The smoke replay used attempt ID `ch15-realmodel-paired-v1:CH15-RFD-001:fixed:attempt-2`; both attempts remain in the per-case call trace. The same SQLite ledger preserves those three calls and records the config transition from `sha256:54fd2f87399f0b1672715ca14b62af7819d979be5268c5f7bfdbdb5379e3078a` to the execution config shown above. After the run, the explanatory design document was moved from `source_sha256` to `historical_source_sha256`; executable sources, prompts, scenarios, gold and scoring code were unchanged. That documentation-only transition is recorded in the ledger. The current preflight hash is `sha256:bbec4c7c8cba83f8dc25dbe6235d867d43b1c5e8f93408c0a4c4c3f85dcf50cb`. A third ledger history row records the Markdown table-spacing correction; it preserves all 341 settled calls and $0.044893914 and does not alter executable behavior, prompts, scenarios, gold or scoring. All prior calls are included in the reported fixed-arm count and cost.

All rates describe this frozen set of simulated scenarios only. They are not real support outcomes or customer-service ROI.
