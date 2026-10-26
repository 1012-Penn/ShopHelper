# CH15 Real-Model Paired Experiment Report

**Status: Not run — local provider credential unavailable.** The smoke case and paired model experiment have not made a provider request. Observed cost is **$0.00** and no real-model success rate is available.

## Frozen run plan

The verified preflight pins config `sha256:54fd2f87399f0b1672715ca14b62af7819d979be5268c5f7bfdbdb5379e3078a` and 30 simulated cases / 42 user turns per arm. The full manifest has 72 simulated scenarios; the paid comparison uses 18 refund-guidance and 12 multi-turn revision cases.

| Arm | Simulated cases | User turns | Maximum generations | Hard cap | Observed generations | Observed cost |
|---|---:|---:|---:|---:|---:|---:|
| Fixed refund flow | 30 | 42 | 378 | $9.20 | 0 | $0.00 |
| Constrained Agent | 30 | 42 | 336 | $8.20 | 0 | $0.00 |
| **Total** | **60 arm-runs** | **84 arm-turns** | **714** | **$17.40** | **0** | **$0.00** |

The smoke is `CH15-RFD-001` on the fixed graph and counts inside that arm's generation and cost cap. Provider prices are pinned at peak rates: $0.006/M cache-hit input, $0.30/M cache-miss input and $1.20/M output. The pre-call upper envelope treats all input as cache misses and is $0.0220608 per generation at 65,536 input and 2,000 output tokens. Reported usage cost will use the provider's cache-hit/miss breakdown when available; peak pricing can overstate the invoice during off-peak hours.

## Safety and scoring

The runner calls the real LangGraph resolver, intent classifier, fixed refund path or Agent route, Agent node, read-only ToolEngine tools and isolated in-memory SQLite fixtures. Model-visible tools are only `query_order` and `query_faq`; no model write tool or confirmation endpoint is exposed. Policy and order facts are deterministic simulated fixtures. Gold, expected intents/actions and scripted actors are projected out of model inputs.

The durable CH15 SQLite ledger reserves maximum call cost before dispatch, records prompts, structured-output schemas, raw responses, usage and latency, and charges provider usage when returned. Missing usage and uncertain failures are charged at the full reservation. A provider error stops further calls; a restart with an unresolved call fails closed. This ledger is separate from CH11.

When run, every metric will include a numerator and denominator. Task completion requires all frozen answer, action, parameter, refusal/handoff and database checks to pass. Parameter accuracy counts missing and extra tool calls in its denominator. The paid set contains no gold-positive handoff case and exposes no write operation, so those rates will be N/A with their denominators shown as zero; the zero-ticket database assertion will still be reported. Refusal alignment will be scored against each case's gold.

## Preflight and blocker

`python3 scripts/ch15_model_experiment.py --preflight` passed with zero provider calls and verified all 30 frozen source/fixture hashes. The `--allow-paid --smoke` launch stopped before creating a ledger or calling the provider because `OPENAI_API_KEY` is not configured in this worktree or environment. The user has said budget is not a concern; the remaining requirement is a credential available locally through `.env` or the environment.

After the credential is configured, run the smoke first and then the paired sample:

```bash
python3 scripts/ch15_model_experiment.py --allow-paid --smoke
python3 scripts/ch15_model_experiment.py --allow-paid
```

The smoke case's completed trace is reused by the full run. Until those commands complete, the offline 72-scenario report remains the only measured result. It reports scripted scenario contract pass rates, not real-model task success, and none of these results represent real customer-service ROI.
