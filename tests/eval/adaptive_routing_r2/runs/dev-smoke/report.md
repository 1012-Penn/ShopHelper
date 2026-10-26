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
| fixed_single | 0/10 (0.0%) | 16/27 | 0 | 0/0 | 0.000000 | N/A |
| fixed_multi | 7/10 (70.0%) | 7518/16172 | 58 | 40440/9011 | 0.016510 | 0.0023585897 |
| adaptive | 2/10 (20.0%) | 39/8486 | 14 | 10907/2281 | 0.004391 | 0.002195562 |

## Paired routing

- Adaptive vs fixed single: `{'wins': 2, 'losses': 0, 'ties': 8}`.
- Adaptive vs fixed multi: `{'wins': 0, 'losses': 5, 'ties': 5}`.
- False upgrades: `[]`.
- Missed upgrades: `['AR2-DEV-001', 'AR2-DEV-002', 'AR2-DEV-003', 'AR2-DEV-007', 'AR2-DEV-008']`.
- Adaptive selected multi on scenarios: `0.2`.

## Buckets

| Bucket | Fixed single | Fixed multi | Adaptive |
|---|---:|---:|---:|
| missing_slot | 0/1 | 1/1 | 0/1 |
| multi_entity | 0/2 | 2/2 | 2/2 |
| refusal | 0/1 | 1/1 | 0/1 |
| revision | 0/1 | 0/1 | 0/1 |
| simple | 0/3 | 3/3 | 0/3 |
| write_cancel | 0/1 | 0/1 | 0/1 |
| write_preview | 0/1 | 0/1 | 0/1 |

## Route reasons

| Reason | Count |
|---|---:|
| `cross_order_comparison` | 2 |
| `missing_slot_clarify` | 1 |
| `simple_policy` | 3 |
| `single_order_lookup` | 1 |
| `single_order_policy_lookup` | 4 |
| `write_requires_confirmation` | 2 |

## Failures

| Scenario | Arm | Errors | Failure detail |
|---|---|---|---|
| AR2-DEV-001 | fixed_single | harness:AttributeError:'_RetryingGuardedModel' object has no attribute 'astream' | {"actual_outcome": "answer", "missing_conditions": ["15天", "吊牌"], "forbidden_conditions": [], "missing_tools": ["query_faq"], "write_correct": true} |
| AR2-DEV-001 | adaptive | routing_executor:AttributeError:'_RetryingGuardedModel' object has no attribute 'astream' | {"actual_outcome": "answer", "missing_conditions": ["15天", "吊牌"], "forbidden_conditions": [], "missing_tools": ["query_faq"], "write_correct": true} |
| AR2-DEV-002 | fixed_single | harness:AttributeError:'_RetryingGuardedModel' object has no attribute 'astream' | {"actual_outcome": "answer", "missing_conditions": ["1001", "已签收"], "forbidden_conditions": [], "missing_tools": ["query_order"], "write_correct": true} |
| AR2-DEV-002 | adaptive | routing_executor:AttributeError:'_RetryingGuardedModel' object has no attribute 'astream' | {"actual_outcome": "answer", "missing_conditions": ["1001", "已签收"], "forbidden_conditions": [], "missing_tools": ["query_order"], "write_correct": true} |
| AR2-DEV-003 | fixed_single | harness:AttributeError:'_RetryingGuardedModel' object has no attribute 'astream' | {"actual_outcome": "answer", "missing_conditions": ["1002", "不能直接取消"], "forbidden_conditions": [], "missing_tools": ["query_order", "query_faq"], "write_correct": true} |
| AR2-DEV-003 | adaptive | routing_executor:AttributeError:'_RetryingGuardedModel' object has no attribute 'astream' | {"actual_outcome": "answer", "missing_conditions": ["1002", "不能直接取消"], "forbidden_conditions": [], "missing_tools": ["query_order", "query_faq"], "write_correct": true} |
| AR2-DEV-004 | fixed_single | harness:AttributeError:'_RetryingGuardedModel' object has no attribute 'astream' | {"actual_outcome": "answer", "missing_conditions": ["1001", "1002", "已签收", "已发货"], "forbidden_conditions": [], "missing_tools": ["query_order"], "write_correct": true} |
| AR2-DEV-005 | fixed_single | harness:AttributeError:'_RetryingGuardedModel' object has no attribute 'astream' | {"actual_outcome": "answer", "missing_conditions": ["1002", "1003", "1003"], "forbidden_conditions": [], "missing_tools": ["query_order", "query_faq"], "write_correct": true} |
| AR2-DEV-006 | fixed_single | harness:AttributeError:'_RetryingGuardedModel' object has no attribute 'astream' | {"actual_outcome": "answer", "missing_conditions": ["1003", "待发货"], "forbidden_conditions": [], "missing_tools": ["query_order", "query_faq"], "write_correct": true} |
| AR2-DEV-006 | fixed_multi | none | {"actual_outcome": "answer", "missing_conditions": ["待发货"], "forbidden_conditions": [], "missing_tools": [], "write_correct": true} |
| AR2-DEV-006 | adaptive | routing_executor:AttributeError:'_RetryingGuardedModel' object has no attribute 'astream' | {"actual_outcome": "answer", "missing_conditions": ["1003", "待发货"], "forbidden_conditions": [], "missing_tools": ["query_order", "query_faq"], "write_correct": true} |
| AR2-DEV-007 | fixed_single | harness:AttributeError:'_RetryingGuardedModel' object has no attribute 'astream' | {"actual_outcome": "answer", "missing_conditions": ["订单号"], "forbidden_conditions": [], "missing_tools": [], "write_correct": true} |
| AR2-DEV-007 | adaptive | routing_executor:AttributeError:'_RetryingGuardedModel' object has no attribute 'astream' | {"actual_outcome": "answer", "missing_conditions": ["订单号"], "forbidden_conditions": [], "missing_tools": [], "write_correct": true} |
| AR2-DEV-008 | fixed_single | harness:AttributeError:'_RetryingGuardedModel' object has no attribute 'astream' | {"actual_outcome": "answer", "missing_conditions": ["无法回答"], "forbidden_conditions": [], "missing_tools": [], "write_correct": true} |
| AR2-DEV-008 | adaptive | routing_executor:AttributeError:'_RetryingGuardedModel' object has no attribute 'astream' | {"actual_outcome": "answer", "missing_conditions": ["无法回答"], "forbidden_conditions": [], "missing_tools": [], "write_correct": true} |
| AR2-DEV-009 | fixed_single | harness:AttributeError:'_RetryingGuardedModel' object has no attribute 'astream' | {"actual_outcome": "answer", "missing_conditions": ["预览", "确认"], "forbidden_conditions": [], "missing_tools": ["query_order", "create_ticket"], "write_correct": false} |
| AR2-DEV-009 | fixed_multi | none | {"actual_outcome": "answer", "missing_conditions": [], "forbidden_conditions": [], "missing_tools": ["create_ticket"], "write_correct": false} |
| AR2-DEV-009 | adaptive | routing_executor:AttributeError:'_RetryingGuardedModel' object has no attribute 'astream' | {"actual_outcome": "answer", "missing_conditions": ["预览", "确认"], "forbidden_conditions": [], "missing_tools": ["query_order", "create_ticket"], "write_correct": false} |
| AR2-DEV-010 | fixed_single | harness:AttributeError:'_RetryingGuardedModel' object has no attribute 'astream' | {"actual_outcome": "answer", "missing_conditions": ["预览", "不会提交"], "forbidden_conditions": [], "missing_tools": ["query_order", "create_ticket"], "write_correct": false} |
| AR2-DEV-010 | fixed_multi | none | {"actual_outcome": "answer", "missing_conditions": [], "forbidden_conditions": [], "missing_tools": ["create_ticket"], "write_correct": false} |
| AR2-DEV-010 | adaptive | routing_executor:AttributeError:'_RetryingGuardedModel' object has no attribute 'astream' | {"actual_outcome": "answer", "missing_conditions": ["预览", "不会提交"], "forbidden_conditions": [], "missing_tools": ["query_order", "create_ticket"], "write_correct": false} |

## Budget and limits

- Run cap: `$2.00`; charged upper bound: `$0.020901`; pending: `$0.000000`.
- Every model attempt reserves worst-case cost before dispatch and is settled from provider usage; missing usage is charged at the full reservation.
- At most two bounded 429 retries are allowed; each attempt is ledgered.
- Cold wall time is measured without response replay; provider cache effects do not turn this into an online latency claim.
- The rules-only router reports zero routing API cost, not zero engineering or wall-time cost.
- Results cover simulated orders and policies only and are not production ROI.

## Sources

- [DeepSeek pricing](https://api-docs.deepseek.com/quick_start/pricing/)
- [LangChain router pattern](https://github.com/langchain-ai/docs/blob/main/src/oss/langchain/multi-agent/router.mdx)
- [RouteLLM](https://github.com/lm-sys/routellm)
