# Cost-Aware Adaptive Routing R2

Status: completed frozen real-model three-arm run.

Model: `deepseek-flash`. Scenario SHA-256: `7ac7550c254864afc082297dd6394f143c72fecd77067834216cd58ac633c382`.
Router policy: `adaptive-routing-r2-v1` / `9c052168d81ea901e9849210bfb1ee6a2eba9dfe78213b7d8a1da51371942a5e`.
Scoring SHA-256: `7b5985969f0620eaf9b8683dec9258f2fbd0e3adcc9653b0be221734728d1a07`.

## Protocol

- 10 dev + 24 held-out cases grouped by business fact/question template.
- Three arms: fixed single-Agent, fixed multi-Agent, and adaptive routing.
- Same `deepseek-flash`, evidence fixtures, tools, temperature, and output cap.
- Gold is visible only to the deterministic scorer and confirmation action.
- Adaptive rules decide before each turn; exactly one executor runs.
- Routing latency is included; the rules policy has no model-call cost.
- No held-out result was used to change features or thresholds.

## Results

| Arm | Task success | P50/P95 ms | Calls | Input/output tokens | Cost USD | Cost/success USD |
|---|---:|---:|---:|---:|---:|---:|
| fixed_single | 6/10 (60.0%) | 2013/4687 | 37 | 39453/1926 | 0.004777 | 0.000796122 |
| fixed_multi | 7/10 (70.0%) | 6229/13916 | 174 | 123721/26516 | 0.048803 | 0.0069718946 |
| adaptive | 6/10 (60.0%) | 1808/9754 | 77 | 66921/9393 | 0.018629 | 0.003104861 |

## Paired routing

- Adaptive vs fixed single: `{'wins': 1, 'losses': 1, 'ties': 8}`.
- Adaptive vs fixed multi: `{'wins': 0, 'losses': 1, 'ties': 9}`.
- False upgrades: `['AR2-DEV-006']`.
- Missed upgrades: `['AR2-DEV-007']`.
- Adaptive selected multi on scenarios: `0.3`.

## Buckets

| Bucket | Fixed single | Fixed multi | Adaptive |
|---|---:|---:|---:|
| missing_slot | 0/1 | 1/1 | 0/1 |
| multi_entity | 1/2 | 2/2 | 2/2 |
| refusal | 1/1 | 1/1 | 1/1 |
| revision | 1/1 | 0/1 | 0/1 |
| simple | 3/3 | 3/3 | 3/3 |
| write_cancel | 0/1 | 0/1 | 0/1 |
| write_preview | 0/1 | 0/1 | 0/1 |

## Route reasons

| Reason | Count |
|---|---:|
| `cross_order_comparison` | 2 |
| `missing_slot_clarify` | 1 |
| `revision_with_order_dependency` | 1 |
| `simple_policy` | 3 |
| `single_order_lookup` | 1 |
| `single_order_policy_lookup` | 4 |
| `write_requires_confirmation` | 2 |

## Failures

| Scenario | Arm | Errors | Failure detail |
|---|---|---|---|
| AR2-DEV-005 | fixed_single | none | {"actual_outcome": "answer", "missing_conditions": [], "forbidden_conditions": [], "missing_tools": ["query_faq"], "write_correct": true} |
| AR2-DEV-006 | fixed_multi | none | {"actual_outcome": "answer", "missing_conditions": ["待发货"], "forbidden_conditions": [], "missing_tools": [], "write_correct": true} |
| AR2-DEV-006 | adaptive | none | {"actual_outcome": "answer", "missing_conditions": [], "forbidden_conditions": [], "missing_tools": [], "write_correct": false} |
| AR2-DEV-007 | fixed_single | none | {"actual_outcome": "answer", "missing_conditions": [], "forbidden_conditions": [], "missing_tools": [], "write_correct": true} |
| AR2-DEV-007 | adaptive | none | {"actual_outcome": "answer", "missing_conditions": [], "forbidden_conditions": [], "missing_tools": [], "write_correct": true} |
| AR2-DEV-009 | fixed_single | none | {"actual_outcome": "answer", "missing_conditions": ["预览"], "forbidden_conditions": [], "missing_tools": ["create_ticket"], "write_correct": false} |
| AR2-DEV-009 | fixed_multi | none | {"actual_outcome": "answer", "missing_conditions": [], "forbidden_conditions": [], "missing_tools": ["create_ticket"], "write_correct": false} |
| AR2-DEV-009 | adaptive | none | {"actual_outcome": "answer", "missing_conditions": ["预览"], "forbidden_conditions": [], "missing_tools": ["create_ticket"], "write_correct": false} |
| AR2-DEV-010 | fixed_single | none | {"actual_outcome": "answer", "missing_conditions": ["不会提交"], "forbidden_conditions": [], "missing_tools": ["create_ticket"], "write_correct": false} |
| AR2-DEV-010 | fixed_multi | none | {"actual_outcome": "answer", "missing_conditions": [], "forbidden_conditions": [], "missing_tools": ["create_ticket"], "write_correct": false} |
| AR2-DEV-010 | adaptive | none | {"actual_outcome": "answer", "missing_conditions": ["不会提交"], "forbidden_conditions": [], "missing_tools": ["create_ticket"], "write_correct": false} |

## Budget and limits

- Run cap: `$2.00`; charged upper bound: `$0.072209`; pending: `$0.000000`.
- Every model attempt reserves worst-case cost before dispatch and is settled from provider usage; missing usage is charged at the full reservation.
- At most two bounded 429 retries are allowed; each attempt is ledgered.
- Cold wall time is measured without response replay; provider cache effects do not turn this into an online latency claim.
- The rules-only router reports zero routing API cost, not zero engineering or wall-time cost.
- Results cover simulated orders and policies only and are not production ROI.

## Sources

- [DeepSeek pricing](https://api-docs.deepseek.com/quick_start/pricing/)
- [LangChain router pattern](https://github.com/langchain-ai/docs/blob/main/src/oss/langchain/multi-agent/router.mdx)
- [RouteLLM](https://github.com/lm-sys/routellm)
