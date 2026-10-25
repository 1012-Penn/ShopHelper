# CH15 Paid Model Experiment Proposal — Not Executed

**Status:** Not executed. Offline validation only. The user has clarified that the experiment does not need to fit CH11's shared $10 budget; the CH15 experiment uses its own $17.40 cap. This budget clarification does not start a paid provider run.

## Proposed comparison

Run only the 30 paired simulated cases: 18 refund-guidance scenarios and 12 multi-turn revision scenarios. Both arms use the same frozen scenario text, deterministic order/policy fixtures, tool registry, ToolEngine schema and confirmation gate. Run sequentially, with one model configuration pinned for both arms.

| Group | Cases | User turns | Maximum generations | Generations per turn | Hard token-cost envelope | Proposed stop cap |
|---|---:|---:|---:|---:|---:|---:|
| Existing fixed refund flow | 30 | 42 | 378 | 9 | $8.34 | $9.20 |
| Constrained Agent route | 30 | 42 | 336 | 8 | $7.42 | $8.20 |
| **Total** | **60 arm-runs** | **84 arm-turns** | **714** | — | **$15.76** | **$17.40** |

The fixed-flow ceiling allows two preprocessing generations (resolve and intent), one refund query expansion, and up to six Agent generations for each of the 42 turns: `(2 + 1 + 6) × 42 = 378`. The constrained Agent ceiling allows resolve, intent, and up to six Agent generations: `(2 + 6) × 42 = 336`. Retries are set to zero. These are maximum call counts, not expected usage.

## Price basis and calculation

Price snapshot checked 2026-10-26 against the [official DeepSeek Models & Pricing page](https://api-docs.deepseek.com/quick_start/pricing/). The proposal pins `deepseek-flash` (DeepSeek-V4.1-Flash-1017) and uses peak prices: $0.30 per million input tokens for cache misses and $1.20 per million output tokens. The budget assumes every input token misses cache.

Per-generation worst case: `65,536 × $0.30 / 1,000,000 + 2,000 × $1.20 / 1,000,000 = $0.0220608`.

| Group | Token-envelope calculation | Cost before reserve | 10% reserve | Rounded hard cap |
|---|---|---:|---:|---:|
| Fixed flow | 378 × $0.0220608 | $8.3389824 | $0.8338982 | $9.20 |
| Constrained Agent | 336 × $0.0220608 | $7.4124288 | $0.7412429 | $8.20 |
| **Total** | 714 generations | **$15.7514112** | **$1.5751411** | **$17.40** |

The provider price may change. Recheck the official page and the configured context/output limits on the run date. The $17.40 total is a separate CH15 envelope; the CH11 $10 ledger and frozen evaluation set remain untouched. Stop if either arm's cap or generation limit is reached.

## Execution and safety controls

1. Load the verified price table and construct `PaidRunBudget` with `allow_paid=True` only for an explicitly arranged run. The guard defaults to rejecting calls. `guard_graph_models` must wrap the Agent, resolver, intent and expansion models with the same arm budget before passing them to `build_graph`; the intent escalator is disabled.
2. Before starting, confirm the selected model and current official price. `validate_paid_settings` requires `llm_max_retries=0`, `intent_escalation_enabled=false`, at most six Agent steps and at most 2,000 output tokens. The guard reserves the full worst-case per-call cost before dispatch, counts timed-out/failed attempts against the limit, and rejects request envelopes above 65,536 bytes/token-envelope. No retry or hidden generation is allowed.
3. Keep deterministic policy and order data. Do not load production credentials, connect to MCP, query customer systems, or import live order records.
4. Do not call `/api/tickets/confirm` in the paid comparison. A model-initiated `create_ticket` call remains behind the existing confirmation preview and cannot persist; there is no refund execution tool. Confirmation and response-loss tests remain offline-only.
5. Run serially and preserve per-case prompts, model-generated intent/tool arguments/replies, tool results and traces for review. The budget guard reserves worst-case cost rather than reconciling actual provider usage; capture provider usage as an audit record, but it cannot raise the cap.
6. The paired result will describe performance on these simulated scenarios only. It will not be reported as real customer-service ROI or production success rate. It may report Agent task success only after the real model independently generates intent, tool arguments and answers and the same predeclared gold is applied.

The offline CLI remains deterministic and cannot run paid calls. The budget guard is implemented as a reusable pre-call wrapper, but no paid experiment executor has been wired to the graph yet; the future runner must wrap every model call before it can be considered ready. This proposal and guard do not mean the paid experiment has run. No paid model call has been made.
