# CH15 Real-Model Paired Experiment Run Plan

**Status:** Complete. All 30 paired simulated scenarios / 60 arm-runs completed with actual model-generated intents, tool calls and answers. Peak-priced usage cost was **$0.044894** against the separate $17.40 CH15 cap.

Task success was 11/30 (36.7%) for the fixed flow and 1/30 (3.3%) for the constrained Agent, a descriptive paired difference of -33.3 percentage points. Full numerators, denominators, per-case traces and failures are in `docs/eval/ch15-model-experiment-report.md` and `docs/eval/ch15-model-case-report.jsonl`.

## Proposed comparison

Run only the 30 paired simulated cases: 18 refund-guidance scenarios and 12 multi-turn revision scenarios, covering 42 user turns per arm. Both arms use the same frozen scenario text, deterministic order/policy fixtures, production ToolEngine, and read-only `query_order` / `query_faq` capability set. The fixed arm uses the existing fixed refund route; the constrained Agent arm routes refund/aftersales intents to the production ReAct node. Both arms independently call the real resolver and intent model. Run serially with one pinned model configuration.

| Group | Cases | User turns | Maximum generations | Generations per turn | Hard token-cost envelope | Proposed stop cap |
|---|---:|---:|---:|---:|---:|---:|
| Existing fixed refund flow | 30 | 42 | 378 | 9 | $8.34 | $9.20 |
| Constrained Agent route | 30 | 42 | 336 | 8 | $7.42 | $8.20 |
| **Total** | **60 arm-runs** | **84 arm-turns** | **714** | — | **$15.76** | **$17.40** |

The fixed-flow ceiling allows two preprocessing generations (resolve and intent), one refund query expansion, and up to six Agent generations for each of the 42 turns: `(2 + 1 + 6) × 42 = 378`. The constrained Agent ceiling allows resolve, intent, and up to six Agent generations: `(2 + 6) × 42 = 336`. A fixed-arm `CH15-RFD-001` graph smoke case is the first task and counts inside the 378-call / $9.20 fixed cap. Retries are set to zero. These are maximum call counts, not expected usage.

## Price basis and calculation

Price snapshot checked 2026-10-26 against the [official DeepSeek Models & Pricing page](https://api-docs.deepseek.com/quick_start/pricing/). The proposal pins `deepseek-flash` (DeepSeek-V4.1-Flash-1017) and uses peak prices: $0.006 per million cache-hit input, $0.30 per million cache-miss input and $1.20 per million output tokens. The budget assumes every input token misses cache. Provider usage includes cache hit/miss counts; result cost will use those counts at peak rates, so it may overstate the invoice when the run occurs off-peak.

Per-generation worst case: `65,536 × $0.30 / 1,000,000 + 2,000 × $1.20 / 1,000,000 = $0.0220608`.

| Group | Token-envelope calculation | Cost before reserve | 10% reserve | Rounded hard cap |
|---|---|---:|---:|---:|
| Fixed flow | 378 × $0.0220608 | $8.3389824 | $0.8338982 | $9.20 |
| Constrained Agent | 336 × $0.0220608 | $7.4124288 | $0.7412429 | $8.20 |
| **Total** | 714 generations | **$15.7514112** | **$1.5751411** | **$17.40** |

The provider price may change. Recheck the official page and the configured context/output limits on the run date. The $17.40 total is a separate CH15 envelope; the CH11 $10 ledger and frozen evaluation set remain untouched. Stop if either arm's cap or generation limit is reached.

## Execution and safety controls

1. The frozen file `docs/eval/ch15-model-experiment-config.json` pins the 30 scenario IDs, 30 execution/rubric source hashes, model, price basis, route modes, model-visible tools and limits. Structured-output schemas are included in each pre-call envelope and audit record. The evaluator passes only `turn.message` plus actual generated conversation history to the graph; prewritten actors, expected intents/actions, and gold are used only by post-run scoring.
2. Use `deepseek-flash`, temperature 0, reasoning disabled, max output 2,000, provider timeout 45 seconds, `max_retries=0`, intent escalation disabled and Agent step limit 6. The model sees only `query_order` and `query_faq`; both are read-only. Each arm gets an isolated in-memory SQLite database. No MCP, production order service, or ticket confirmation endpoint is used.
3. The persistent SQLite ledger at `data/ch15_paid_run.sqlite3` is separate from CH11. It commits the worst-case reservation before every provider call, records request/schema, raw response, tokens, latency and failure status, and settles against provider usage. Missing usage or uncertain failures are charged at the full per-call envelope. Any provider error stops future calls; startup rejects a stopped or unresolved ledger before generating new traces. The command runs serially, so concurrent runs are not supported.
4. The required zero-cost preflight is `python3 scripts/ch15_model_experiment.py --preflight`. After `OPENAI_API_KEY` is configured locally, run the paid smoke with `python3 scripts/ch15_model_experiment.py --allow-paid --smoke`; if it completes, run the paired set with `python3 scripts/ch15_model_experiment.py --allow-paid`. The smoke case is counted once in the fixed arm, and completed case traces are reused when the paired run resumes.
5. The CLI requires `--allow-paid`; the database enforces both arm caps and the $17.40 total cap before every generation. The call ceilings include the smoke case. Any timeout/provider error counts against both spend and call ceilings. No model/tool retry is configured.
6. The paid 30-case set contains one gold-positive handoff case (`CH15-RFD-017`) and exposes no write tool. Report reasonable handoff as 1/1 per arm, unconfirmed write attempt rate as N/A (0 write attempts), and separately report the zero-ticket SQLite assertion. Refusal behavior is scored against each paid case's gold.
7. Report task success, tool selection, parameter accuracy, unconfirmed write safety, reasonable handoff, and refusal/gold alignment with numerator and denominator. Preserve the exact model prompts/responses, generated intents/tool calls, tool results, graph frames, database assertions, and per-case score in `docs/eval/ch15-model-case-report.jsonl`.
8. Results describe this frozen simulated set only. They are not real customer-service outcomes or ROI. Do not report task success until the model has independently generated intent, tool arguments and answers and every row has been scored against the frozen gold.

Completed report: `docs/eval/ch15-model-experiment-report.md`; per-case trace: `docs/eval/ch15-model-case-report.jsonl`. The run recorded 175 fixed-flow calls ($0.024311) and 166 constrained-Agent calls ($0.020583), with no pending reservations or provider errors. The first smoke attempt's three settled calls remain in the same ledger and are counted in the fixed total. The offline 72-scenario contract report remains a separate scripted-model result.
