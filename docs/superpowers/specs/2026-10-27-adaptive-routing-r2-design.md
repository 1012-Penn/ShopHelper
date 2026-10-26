# Cost-Aware Adaptive Routing R2

Date: 2026-10-27

Status: implementation specification; actual metrics are added only from a completed frozen run.

## Objective

Decide once, before execution, whether a customer-service turn should use the existing
single-Agent path or the existing multi-Agent graph. The router must improve the
quality/cost trade-off only when evidence supports doing so; it is acceptable for the
router to be cheaper but not better than either fixed endpoint.

## Trust Boundary

`RoutingInput` has exactly three visible fields:

- current user question;
- conversation history visible to the agent;
- remaining run time and cost budget.

Gold labels, expected behavior, hidden buckets, expected tools, scoring rules, and
scenario identifiers are not accepted. Inputs with additional fields fail validation.
The router can observe only the same business evidence available before either graph
starts.

## Routing Contract

The initial policy is deterministic and explainable:

| Signal | Detection from visible text |
|---|---|
| order/entity count | distinct explicit 3-6 digit order IDs |
| cross-order comparison | at least two entities plus comparison/separate-treatment wording |
| policy-order dependency | an order context plus cancellation, return, refund, shipping, or evidence wording |
| correction/revision | correction, changed question, or replacement wording |
| missing slot | demonstrative order reference without a resolvable visible order ID |
| write-confirmation risk | ticket creation, submission, approval, or confirmation wording |

Default route:

- `multi` when there is an explicit cross-order comparison, or at least two entities
  with policy/action dependency and correction/revision;
- `single` for simple policy lookup, one-order lookup, missing-slot clarification,
  refusal, and write-only/confirmation turns;
- write risk is recorded but never expands permissions or independently upgrades to
  multi-Agent.

An optional one-call structured judge is supported only for a predeclared uncertain
band. It receives the same visibility projection, has a bounded output, records
confidence and cost, and cannot raise permissions. The default and frozen evaluation
policy use rules only; no hidden classifier call is assumed.

The policy version, feature definitions, score threshold, and optional-judge threshold
are frozen before held-out execution.

## Dispatcher Contract

The dispatcher accepts two injectable executors and invokes exactly one after routing.
It never executes both on failure, retries by switching arms, or selects a better
answer. Both executors receive the same immutable `RoutingInput`; any attempt to
change permissions is outside the dispatcher boundary.

Responses are normalized from the common existing shapes (`final_reply`/`reply`, graph
state dicts, Pydantic models, or plain strings) while retaining the raw executor
result. The result records:

- decision target, reasons, and features;
- classifier usage, confidence, tokens, and cost;
- routing wall time;
- execution wall time and total wall time;
- selected executor call count;
- normalized response and any single execution error.

Routing time is included in total latency. Any optional classifier cost is included
in total cost. The application integration remains optional and disabled by default.

## Existing Execution Semantics

The router reuses, but does not imitate, existing execution infrastructure:

- session continuity, persistence, SSE, and resume belong to the existing single-Agent
  route;
- multi-Agent has its own process-local graph and does not have equivalent persistent
  run/SSE recovery semantics;
- both current paths use the existing tool engine and ticket preview/confirmation
  boundary, but their response fields and write-preview lifecycle differ;
- no router implementation may claim identical streaming or recovery behavior.

The integration handoff supplies a minimal adapter patch; production defaults remain
unchanged.

## Evaluation Design

The new dataset contains 10 dev and 24 held-out cases. Cases are partitioned by source
business fact and question template before execution, not randomly after seeing
outcomes. Coverage includes simple policy/order lookup, cross-order dependency,
correction, missing slots, refusal, write confirmation, and preview cancellation.

Before any held-out execution, the dataset, scoring rubric, router features, thresholds,
model settings, pricing, and stop rules are hashed into a manifest. Development may
use only the 10 dev cases. Held-out is executed once after freeze. Infrastructure
failures may be retried without changing model prompts or routing policy; a repaired
version preserves all earlier failed artifacts and is reported separately.

Three paired arms use the same `deepseek-flash` model, order fixture, policy fixture,
tools, answer prompt limits, and output cap:

1. fixed single-Agent;
2. fixed multi-Agent;
3. adaptive dispatch.

The adaptive arm records its selected path and route features for every case. The
route-only cost is zero for the frozen rules policy; routing wall time is still
measured. Provider usage comes from the persistent guarded ledger. Gold is visible
only to the deterministic scorer after execution.

## Acceptance Metrics

- task success with numerator/denominator and paired wins/losses/ties;
- relative to each fixed endpoint: adaptive wins, losses, ties, false upgrades, and
  missed upgrades;
- success split by simple, multi-entity dependency, correction, missing slot, and
  write/preview category;
- cold-path P50/P95 including routing overhead;
- model calls, input/output tokens, total cost, and cost per successful task;
- multi-Agent share and route reason/feature counts;
- every failed case and technical failure preserved.

No minimum positive lift is assumed. The report states the observed Pareto trade-off,
including negative or inconclusive results.

## Stop Rules

- One complete frozen three-arm run.
- A maximum of two bounded retries for transient 429/5xx provider failures, with
  backoff and every failed cost counted.
- No retuning on held-out results.
- The session-level protected API cap is USD 2.00 with atomic pre-call reservations.
- If the cap would be exceeded, stop and report completed cases and the reason.

## Out of Scope

- Training or downloading a routing model.
- Exposing chain-of-thought or model-internal logits.
- Changing default `/api/chat`.
- Claiming online latency from cached replay.
- Combining uncommitted gains from the evidence-grounding or structured-handoff
  branches into this branch's comparison.
