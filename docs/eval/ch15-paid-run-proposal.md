# CH15 Paid Model Experiment Proposal — Pending Arrangement

**Status:** Not executed. Offline validation only; waiting for the user's unified arrangement before any paid provider call.

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

The provider price may change. Recheck the official page and the configured context/output limits on the run date; stop if either arm's cap or generation limit is reached. No budget above these caps is implied.

## Execution and safety controls

1. Before starting, confirm the selected model, current price, exact two arm caps, and no-retry setting. Explicitly set `llm_max_retries=0` and `intent_escalation_enabled=false` so the call ceiling remains valid; use the same model for resolve, intent, expansion and Agent generations.
2. Keep deterministic policy and order data. Do not load production credentials, connect to MCP, query customer systems, or import live order records.
3. Do not call `/api/tickets/confirm` in the paid comparison. A model-initiated `create_ticket` call remains behind the existing confirmation preview and cannot persist; there is no refund execution tool. Confirmation and response-loss tests remain offline-only.
4. Run serially, record provider token usage for each generation, and stop before the next call if an arm's hard cap or generation count is reached. Preserve per-case prompts, tool requests/results and traces for review.
5. The paired result will describe performance on these simulated scenarios only. It will not be reported as real customer-service ROI or production success rate.

This is a costed plan, not authorization to run it. No paid model call has been made.
