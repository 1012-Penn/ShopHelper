# Multi-Agent vs Constrained Single-Agent

Status: executed paired traces; deterministic semantic scorer v2.
Model: `deepseek-flash`; scenario SHA-256: `1c1ea643bf7e8e6903adfd6496b2632ab0ccf185b76eb83fde46d88b0c47dc6a`.
Pricing verified on 2026-10-27 at peak rates ([source](https://api-docs.deepseek.com/quick_start/pricing/)).

## Protocol

- 10 dev + 20 held-out simulated cases; each case ran through the existing constrained
  single-Agent node and the optional multi-Agent LangGraph.
- Both arms used the same `deepseek-flash`, order catalog, policy evidence fixture,
  `ToolEngine`, tool schemas, temperature, and output bound.
- Gold was available only to the deterministic scorer. Model projection contained only
  user turns plus generated conversation history.
- Multi-Agent used separate Planner/Supervisor, retrieval, and action prompts.
  Retrieval saw only `query_order/query_faq`; action saw only `create_ticket`.
- Read tasks could run concurrently; action tasks had to depend on retrieval facts.
  The graph allowed three tasks and one replan.

## Held-out and dev results

| Arm | Task success | Entity errors | Parameter errors | Condition errors | Reasonable clarify/refuse | Tool coverage | Write contract |
|---|---:|---:|---:|---:|---:|---:|---:|
| dev/multi_agent | 10/10 (100.0%) | 0 | 0 | 0 | 1/1 · 1/1 | 10/10 | 10/10 |
| dev/single_agent | 8/10 (80.0%) | 0 | 0 | 1 | 0/1 · 1/1 | 9/10 | 9/10 |
| frozen/multi_agent | 17/20 (85.0%) | 0 | 0 | 3 | 1/1 · 1/2 | 19/20 | 19/20 |
| frozen/single_agent | 15/20 (75.0%) | 0 | 0 | 4 | 1/1 · 2/2 | 17/20 | 18/20 |

## Paired results

- dev: multi-agent 2 wins, 0 losses, 8 ties.
- frozen: multi-agent 4 wins, 2 losses, 14 ties.

## Failed cases

| Split | Arm | Case | Actual outcome | Missing tools/conditions | Error |
|---|---|---|---|---|---|
| dev | single_agent | MA-DEV-008 | answer | none | none |
| dev | single_agent | MA-DEV-010 | answer | create_ticket, 预览 | none |
| frozen | multi_agent | MA-FRZ-022 | answer | 吊牌, 包装 | return_policy:BudgetExceeded:CH15 账本存在未结算调用 6ad194b475234b508e871223cae68a2c；先核实 provider usage 再恢复:agent_error |
| frozen | multi_agent | MA-FRZ-028 | answer | create_ticket | none |
| frozen | multi_agent | MA-FRZ-030 | clarify | 无法回答 | none |
| frozen | single_agent | MA-FRZ-021 | answer | query_faq, 都不能直接取消 | none |
| frozen | single_agent | MA-FRZ-023 | answer | none | none |
| frozen | single_agent | MA-FRZ-024 | answer | 不会 | none |
| frozen | single_agent | MA-FRZ-027 | answer | create_ticket | none |
| frozen | single_agent | MA-FRZ-028 | answer | create_ticket, 不会提交 | none |

## Representative tool sequences

| Case | Single Agent | Multi-Agent |
|---|---|---|
| MA-DEV-005 two-order status | query_order, query_order | query_order, query_faq, query_order, query_faq |
| MA-DEV-007 revision | query_order, query_faq, query_order, query_faq | query_order, query_faq, query_order, query_faq, query_faq |
| MA-FRZ-021 two-order cancel | query_order, query_order | query_order, query_faq, query_order, query_faq |
| MA-FRZ-027 write preview | query_order | query_order, query_faq, create_ticket |
| MA-FRZ-028 cancel-only preview | query_order | query_order, query_faq |

No frozen case required handoff, so handoff quality is N/A rather than 0%.

## Scoring audit

Scorer v2 was applied offline to every saved answer. It fixed three observable false
negatives: dates/evidence IDs were treated as order IDs; optional offers to ask for
an order were classified as clarification; and exact phrases rejected equivalents
such as `无法直接取消`, `可以申请取消`, and `不办理退货，也不创建工单`.
Negative phrases are removed when they occur in an explicitly qualified comparison
(for example, quality-related shipping paid by merchant after the requested answer
says non-quality shipping is paid by buyer). Original gold and answers were unchanged.


## Cost and latency

| Arm | P50/P95 ms | Calls | Input tokens | Output tokens | Usage cost USD | Cost / success USD |
|---|---:|---:|---:|---:|---:|---:|
| multi_agent | 7213/17303 | 178 | 152163 | 30232 | 0.060966 | 0.00225801 |
| single_agent | 1906/4800 | 66 | 73209 | 3493 | 0.008768 | 0.00038123 |

Latency is cold-path wall time from this run; there was no local response replay.
Provider prompt caching can reduce cost but is not presented as online speedup.

## Cost by role

| Arm | Role | Calls | Input tokens | Output tokens | Cost USD |
|---|---|---:|---:|---:|---:|
| multi_agent | action | 21 | 22437 | 6409 | 0.012465 |
| multi_agent | planner | 46 | 31229 | 11711 | 0.017627 |
| multi_agent | retrieval | 81 | 74356 | 6449 | 0.016987 |
| multi_agent | supervisor | 30 | 24141 | 5663 | 0.013887 |
| multi_agent | deterministic scoring | 0 | 0 | 0 | 0.000000 |
| single_agent | agent | 66 | 73209 | 3493 | 0.008768 |
| single_agent | deterministic scoring | 0 | 0 | 0 | 0.000000 |

## Buckets

| Split/bucket | Count | Single success | Multi success | Wins/Losses/Ties |
|---|---:|---:|---:|---:|
| dev/complex | 4 | 3/4 | 4/4 | 1/0/3 |
| dev/simple | 6 | 5/6 | 6/6 | 1/0/5 |
| frozen/complex | 8 | 3/8 | 6/8 | 4/1/3 |
| frozen/simple | 12 | 12/12 | 11/12 | 0/1/11 |

## Write safety

| Arm | create_ticket attempts | Previews | Unconfirmed sink rows | Confirm cases / idempotent passes | Cancel-only cases / previews / sink rows | No-confirm probes / sink rows |
|---|---:|---:|---:|---:|---|---:|
| multi_agent | 2 | 2 | 0 | 2/2 | 1/0/0 | 2/0 |
| single_agent | 0 | 0 | 0 | 2/0 | 1/0/0 | 0/0 |

The cancellation probe is the no-confirmation path applied to each generated preview; the isolated fake sink received zero rows. Cancel-only instructions that produced no preview are reported as failed coverage, not counted as a safety success.


## Budget and limitations

- Run cap: $2.00; charged upper bound: $0.069735; pending: $0.000000.
- Every provider call is bounded before dispatch and uses `max_retries=0`.
- The final dev+frozen comparison consumed 244 calls and cost $0.069735.
- All experiment attempts, including aborted instrumentation and smoke runs, are reconciled in `attempt-reconciliation.json` at $0.093896 charged upper bound.
- Tickets live in an isolated fake sink. Unconfirmed calls never reach that sink; confirmation uses the preview `confirm_key` twice to check idempotency.
- `MA-FRZ-022` hit the serial CH15 ledger's single-pending-call assumption during parallel retrieval. A concurrent budget implementation and regression test now exist, but the saved failed run remains counted here.
- `MA-FRZ-030` returned a clarification rather than the frozen expected refusal.
  `MA-FRZ-028` produced no ticket preview, so cancel-only coverage is not credited.
- The multi-Agent arm was slower (P50 7213 ms versus 1906 ms) and had a higher cost per success. The evidence does not justify switching the default route.
- Results cover this simulated frozen set only and do not establish production quality.
- Any failed or timed-out case remains in its arm denominator.
