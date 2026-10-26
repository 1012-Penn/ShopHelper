# CH15 Offline Contract Evaluation Summary

Run date: 2026-10-26.

Scenarios: 72 fixed simulated cases; executed arm runs: 102.

The actor is a deterministic scripted model substitute. This run validates graph wiring, tool contracts and scoring; it does not measure a paid LLM or real customer-service outcomes. Agent task success rate was not measured.

## Metrics by arm — scripted scenario contract pass

| Arm | Scripted scenario contract pass | Tool selection | Parameter accuracy | Unconfirmed write rate | Reasonable handoff | Refusal/gold alignment |
|---|---:|---:|---:|---:|---:|---:|
| fixed | 29/30 (96.7%) | 30/30 (100.0%) | 3/3 (100.0%) | N/A (0/0) | 1/1 (100.0%) | 29/30 (96.7%) |
| autonomous | 62/72 (86.1%) | 69/72 (95.8%) | 81/85 (95.3%) | 0/7 (0.0%) | 9/9 (100.0%) | 71/72 (98.6%) |

The scripted scenario contract pass column is not an Agent task success rate. The actor receives prewritten intents, tool actions and responses; no model independently generated those decisions in this run.

`Parameter accuracy` is N/A where the fixed subgraph performs no ToolEngine call with model-supplied arguments. `Unconfirmed write rate` uses persisted writes without a matching confirmation over all audited write attempts. Refusal alignment compares each result to its case gold.

## Paired scripted contract pass: actor variants on the default intent router

| Arm | Contract-passing scenarios / paired scenarios | Turns |
|---|---:|---:|
| fixed | 29/30 (96.7%) | 42 |
| autonomous | 28/30 (93.3%) | 42 |

Paired scripted contract pass difference (autonomous minus fixed): -3.3 percentage points. Both offline arms used the production default route and arm-specific prewritten intents/actors; this is not a controlled route ablation or evidence that one approach has a higher Agent task success rate. The real-model runner adds an explicit fixed-vs-autonomous route mode.
