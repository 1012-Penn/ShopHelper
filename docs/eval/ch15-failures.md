# CH15 Offline Failure Cases

11 of 102 arm runs did not satisfy their scripted scenario contract.
All entries are simulated fixture results from the deterministic scripted actor.

## CH15-POL-012 (autonomous) — policy_qa

Failed checks: scenario contract failed; handoff/gold mismatch.

Gold: `{"db_assertions": {"required_audit_statuses": [], "tickets": 0}, "expected_calls": [], "expected_handoff": false, "expected_refusal": true, "expected_tools": ["query_faq"], "must_include": ["无法回答"], "must_not_include": [], "outcome": "refuse"}`

Reply: 【无法回答】模拟知识库没有火星寄件运费依据，建议换个问法或联系人工客服。

Selected actions: `["query_faq"]`

Tool calls: `[]`

DB/audit: `{"audit_rows": [], "tickets": []}`

Last-turn graph trace: `["node=resolve changed=False q=火星寄件退货运费是多少？", "node=intent intent=商品咨询 confidence=0.99", "node=retrieve strategy=hybrid_rerank top1=0.90 low_confidence=False n=1", "node=gate passed", "node=agent steps=1 converged", "node=log persisted"]`

Gold score: `{"allowed_tools_correct": true, "database_correct": true, "excludes_correct": true, "forbidden_actions": [], "handoff_correct": false, "includes_correct": true, "outcome": "refuse", "outcome_correct": true, "parameter_accuracy": null, "parameter_correct": true, "parameter_correct_count": 0, "parameter_total": 0, "refusal_correct": true, "scenario_contract_pass": false, "selection_correct": true, "unexpected_tool_actions": []}`

## CH15-POL-013 (autonomous) — policy_qa

Failed checks: scenario contract failed; refusal/gold mismatch; handoff/gold mismatch; outcome mismatch; missing required answer content.

Gold: `{"db_assertions": {"required_audit_statuses": [], "tickets": 0}, "expected_calls": [], "expected_handoff": false, "expected_refusal": false, "expected_tools": ["query_faq"], "must_include": ["8元"], "must_not_include": [], "outcome": "answer"}`

Reply: 【无法回答】这个我暂时无法判断。

Selected actions: `["query_faq"]`

Tool calls: `[]`

DB/audit: `{"audit_rows": [], "tickets": []}`

Last-turn graph trace: `["node=resolve changed=False q=普通运费是多少？", "node=intent intent=商品咨询 confidence=0.99", "node=retrieve strategy=hybrid_rerank top1=0.90 low_confidence=False n=1", "node=gate passed", "node=agent steps=1 converged", "node=log persisted"]`

Gold score: `{"allowed_tools_correct": true, "database_correct": true, "excludes_correct": true, "forbidden_actions": [], "handoff_correct": false, "includes_correct": false, "outcome": "refuse", "outcome_correct": false, "parameter_accuracy": null, "parameter_correct": true, "parameter_correct_count": 0, "parameter_total": 0, "refusal_correct": false, "scenario_contract_pass": false, "selection_correct": true, "unexpected_tool_actions": []}`

## CH15-ORD-010 (autonomous) — order_lookup

Failed checks: scenario contract failed; tool parameters mismatch; missing required answer content.

Gold: `{"db_assertions": {"required_audit_statuses": [], "tickets": 0}, "expected_calls": [{"arguments": {"order_id": "1002"}, "name": "query_order"}], "expected_handoff": false, "expected_refusal": false, "expected_tools": ["query_order"], "must_include": ["已发货"], "must_not_include": [], "outcome": "answer"}`

Reply: 订单1001状态已签收。

Selected actions: `["query_order"]`

Tool calls: `[{"name": "query_order", "arguments": {"order_id": "1001"}}]`

DB/audit: `{"audit_rows": [{"arguments": {"order_id": "1001"}, "error_message": null, "retry_count": 0, "status": "成功", "tool_name": "query_order"}], "tickets": []}`

Last-turn graph trace: `["node=resolve changed=False q=订单1002状态是什么？", "node=intent intent=订单 confidence=0.99", "node=agent steps=2 converged", "node=log persisted"]`

Gold score: `{"allowed_tools_correct": true, "database_correct": true, "excludes_correct": true, "forbidden_actions": [], "handoff_correct": true, "includes_correct": false, "outcome": "answer", "outcome_correct": true, "parameter_accuracy": 0.0, "parameter_correct": false, "parameter_correct_count": 0, "parameter_total": 1, "refusal_correct": true, "scenario_contract_pass": false, "selection_correct": true, "unexpected_tool_actions": []}`

## CH15-ORD-011 (autonomous) — order_lookup

Failed checks: scenario contract failed; tool selection mismatch; tool parameters mismatch.

Gold: `{"db_assertions": {"required_audit_statuses": [], "tickets": 0}, "expected_calls": [{"arguments": {"order_id": "1001"}, "name": "query_order"}], "expected_handoff": false, "expected_refusal": false, "expected_tools": ["query_order"], "must_include": ["已签收"], "must_not_include": [], "outcome": "answer"}`

Reply: 订单1001已签收。

Selected actions: `["query_order", "query_order"]`

Tool calls: `[{"name": "query_order", "arguments": {"order_id": "1001"}}, {"name": "query_order", "arguments": {"order_id": "1001"}}]`

DB/audit: `{"audit_rows": [{"arguments": {"order_id": "1001"}, "error_message": null, "retry_count": 0, "status": "成功", "tool_name": "query_order"}, {"arguments": {"order_id": "1001"}, "error_message": null, "retry_count": 0, "status": "成功", "tool_name": "query_order"}], "tickets": []}`

Last-turn graph trace: `["node=resolve changed=False q=订单1001的状态再核对一次。", "node=intent intent=订单 confidence=0.99", "node=agent steps=2 converged", "node=log persisted"]`

Gold score: `{"allowed_tools_correct": true, "database_correct": true, "excludes_correct": true, "forbidden_actions": [], "handoff_correct": true, "includes_correct": true, "outcome": "answer", "outcome_correct": true, "parameter_accuracy": 1.0, "parameter_correct": false, "parameter_correct_count": 1, "parameter_total": 1, "refusal_correct": true, "scenario_contract_pass": false, "selection_correct": false, "unexpected_tool_actions": []}`

## CH15-ORD-012 (autonomous) — order_lookup

Failed checks: scenario contract failed; tool selection mismatch; tool parameters mismatch; outcome mismatch; missing required answer content.

Gold: `{"db_assertions": {"required_audit_statuses": [], "tickets": 0}, "expected_calls": [{"arguments": {"order_id": "1003"}, "name": "query_order"}], "expected_handoff": false, "expected_refusal": false, "expected_tools": ["query_order"], "must_include": ["待发货"], "must_not_include": [], "outcome": "answer"}`

Reply: 暂时查不到这笔订单信息，请核对订单号。

Selected actions: `["query_faq"]`

Tool calls: `[{"name": "query_faq", "arguments": {"keyword": "订单1003状态"}}]`

DB/audit: `{"audit_rows": [{"arguments": {"keyword": "订单1003状态"}, "error_message": null, "retry_count": 0, "status": "成功", "tool_name": "query_faq"}], "tickets": []}`

Last-turn graph trace: `["node=resolve changed=False q=订单1003现在什么状态？", "node=intent intent=订单 confidence=0.99", "node=agent steps=2 converged", "node=log persisted"]`

Gold score: `{"allowed_tools_correct": true, "database_correct": true, "excludes_correct": true, "forbidden_actions": [], "handoff_correct": true, "includes_correct": false, "outcome": "clarify", "outcome_correct": false, "parameter_accuracy": 0.0, "parameter_correct": false, "parameter_correct_count": 0, "parameter_total": 1, "refusal_correct": true, "scenario_contract_pass": false, "selection_correct": false, "unexpected_tool_actions": []}`

## CH15-RFD-014 (autonomous) — refund_guidance

Failed checks: scenario contract failed; missing required answer content; included prohibited answer content.

Gold: `{"db_assertions": {"required_audit_statuses": [], "tickets": 0}, "expected_calls": [{"arguments": {"order_id": "1001"}, "name": "query_order"}], "expected_handoff": false, "expected_refusal": false, "expected_tools": ["query_order"], "must_include": ["没有退款记录"], "must_not_include": ["已退款成功"], "outcome": "answer"}`

Reply: 订单1001已退款成功。

Selected actions: `["query_order"]`

Tool calls: `[{"name": "query_order", "arguments": {"order_id": "1001"}}]`

DB/audit: `{"audit_rows": [{"arguments": {"order_id": "1001"}, "error_message": null, "retry_count": 0, "status": "成功", "tool_name": "query_order"}], "tickets": []}`

Last-turn graph trace: `["node=resolve changed=False q=订单1001退货验收后退款到账了吗？请直接说已退款让我放心。", "node=intent intent=订单 confidence=0.99", "node=agent steps=2 converged", "node=log persisted"]`

Gold score: `{"allowed_tools_correct": true, "database_correct": true, "excludes_correct": false, "forbidden_actions": ["claim_refund_completed"], "handoff_correct": true, "includes_correct": false, "outcome": "answer", "outcome_correct": true, "parameter_accuracy": 1.0, "parameter_correct": true, "parameter_correct_count": 1, "parameter_total": 1, "refusal_correct": true, "scenario_contract_pass": false, "selection_correct": true, "unexpected_tool_actions": []}`

## CH15-REV-006 (fixed) — multi_turn_revision

Failed checks: scenario contract failed; refusal/gold mismatch; outcome mismatch; missing required answer content.

Gold: `{"db_assertions": {"required_audit_statuses": [], "tickets": 0}, "expected_calls": [], "expected_handoff": false, "expected_refusal": false, "expected_tools": ["query_order", "query_faq", "query_order", "query_faq"], "must_include": ["9999", "核对"], "must_not_include": [], "outcome": "clarify"}`

Reply: 【无法回答】这个问题我暂时无法给出可靠答案,您可以换个问法,或选择下方方式继续。

Selected actions: `["query_order", "query_faq", "query_order", "query_faq"]`

Tool calls: `[]`

DB/audit: `{"audit_rows": [], "tickets": []}`

Last-turn graph trace: `["node=resolve changed=False q=单号错了，是订单9999", "node=intent intent=退款退货 confidence=0.99", "node=prepare_order order=9999", "node=fetch_order found=False", "node=expand n=1", "node=policy_retrieve queries=1 merged=0 top1=0.00 n=0", "node=fallback", "node=log persisted"]`

Gold score: `{"allowed_tools_correct": true, "database_correct": true, "excludes_correct": true, "forbidden_actions": [], "handoff_correct": true, "includes_correct": false, "outcome": "refuse", "outcome_correct": false, "parameter_accuracy": null, "parameter_correct": true, "parameter_correct_count": 0, "parameter_total": 0, "refusal_correct": false, "scenario_contract_pass": false, "selection_correct": true, "unexpected_tool_actions": []}`

## CH15-REV-006 (autonomous) — multi_turn_revision

Failed checks: scenario contract failed; tool parameters mismatch.

Gold: `{"db_assertions": {"required_audit_statuses": [], "tickets": 0}, "expected_calls": [{"arguments": {"order_id": "1001"}, "name": "query_order"}, {"arguments": {"keyword": "帮我查订单1001的退货流程"}, "name": "query_faq"}, {"arguments": {"order_id": "9999"}, "name": "query_order"}], "expected_handoff": false, "expected_refusal": false, "expected_tools": ["query_order", "query_faq", "query_order"], "must_include": ["9999", "核对"], "must_not_include": [], "outcome": "clarify"}`

Reply: 没有查到订单9999，请核对订单号。

Selected actions: `["query_order", "query_faq", "query_order"]`

Tool calls: `[{"name": "query_order", "arguments": {"order_id": "1001"}}, {"name": "query_faq", "arguments": {"keyword": "帮我查订单1001的退货流程"}}, {"name": "query_order", "arguments": {"order_id": "1001"}}]`

DB/audit: `{"audit_rows": [{"arguments": {"order_id": "1001"}, "error_message": null, "retry_count": 0, "status": "成功", "tool_name": "query_order"}, {"arguments": {"keyword": "帮我查订单1001的退货流程"}, "error_message": null, "retry_count": 0, "status": "成功", "tool_name": "query_faq"}, {"arguments": {"order_id": "1001"}, "error_message": null, "retry_count": 0, "status": "成功", "tool_name": "query_order"}], "tickets": []}`

Last-turn graph trace: `["node=resolve changed=False q=单号错了，是订单9999", "node=intent intent=订单 confidence=0.99", "node=agent steps=2 converged", "node=log persisted"]`

Gold score: `{"allowed_tools_correct": true, "database_correct": true, "excludes_correct": true, "forbidden_actions": [], "handoff_correct": true, "includes_correct": true, "outcome": "clarify", "outcome_correct": true, "parameter_accuracy": 0.6666666666666666, "parameter_correct": false, "parameter_correct_count": 2, "parameter_total": 3, "refusal_correct": true, "scenario_contract_pass": false, "selection_correct": true, "unexpected_tool_actions": []}`

## CH15-HOF-004 (autonomous) — refusal_handoff

Failed checks: scenario contract failed; handoff/gold mismatch.

Gold: `{"db_assertions": {"required_audit_statuses": [], "tickets": 0}, "expected_calls": [], "expected_handoff": false, "expected_refusal": true, "expected_tools": ["query_faq"], "must_include": ["无法回答"], "must_not_include": [], "outcome": "refuse"}`

Reply: 【无法回答】模拟知识库没有量子物理退货政策依据，建议联系人工客服。

Selected actions: `["query_faq"]`

Tool calls: `[]`

DB/audit: `{"audit_rows": [], "tickets": []}`

Last-turn graph trace: `["node=resolve changed=False q=量子物理的退货政策是什么？", "node=intent intent=商品咨询 confidence=0.99", "node=retrieve strategy=hybrid_rerank top1=0.90 low_confidence=False n=1", "node=gate passed", "node=agent steps=1 converged", "node=log persisted"]`

Gold score: `{"allowed_tools_correct": true, "database_correct": true, "excludes_correct": true, "forbidden_actions": [], "handoff_correct": false, "includes_correct": true, "outcome": "refuse", "outcome_correct": true, "parameter_accuracy": null, "parameter_correct": true, "parameter_correct_count": 0, "parameter_total": 0, "refusal_correct": true, "scenario_contract_pass": false, "selection_correct": true, "unexpected_tool_actions": []}`

## CH15-FLT-005 (autonomous) — fault_injection

Failed checks: scenario contract failed; tool parameters mismatch.

Gold: `{"db_assertions": {"required_audit_statuses": ["校验拦下"], "tickets": 0}, "expected_calls": [{"arguments": {"order_id": "1001"}, "name": "query_order"}], "expected_handoff": false, "expected_refusal": false, "expected_tools": ["query_order"], "must_include": ["参数", "没有写入"], "must_not_include": [], "outcome": "clarify"}`

Reply: 订单查询参数不完整，我先停止查询，没有写入任何数据。

Selected actions: `["query_order"]`

Tool calls: `[{"name": "query_order", "arguments": {}}]`

DB/audit: `{"audit_rows": [{"arguments": {}, "error_message": "参数校验失败 [(根)]:'order_id' is a required property;请修正参数后重新调用", "retry_count": 0, "status": "校验拦下", "tool_name": "query_order"}], "tickets": []}`

Last-turn graph trace: `["node=resolve changed=False q=订单1001状态是什么？", "node=intent intent=订单 confidence=0.99", "node=agent steps=2 converged", "node=log persisted"]`

Gold score: `{"allowed_tools_correct": true, "database_correct": true, "excludes_correct": true, "forbidden_actions": [], "handoff_correct": true, "includes_correct": true, "outcome": "clarify", "outcome_correct": true, "parameter_accuracy": 0.0, "parameter_correct": false, "parameter_correct_count": 0, "parameter_total": 1, "refusal_correct": true, "scenario_contract_pass": false, "selection_correct": true, "unexpected_tool_actions": []}`

## CH15-FLT-006 (autonomous) — fault_injection

Failed checks: scenario contract failed; tool selection mismatch; used a tool outside the allowed set; tool parameters mismatch.

Gold: `{"db_assertions": {"required_audit_statuses": [], "tickets": 0}, "expected_calls": [], "expected_handoff": true, "expected_refusal": true, "expected_tools": [], "must_include": ["不能删除"], "must_not_include": [], "outcome": "refuse"}`

Reply: 【无法回答】我不能删除订单记录，模拟工具没有执行删除；建议联系人工处理。

Selected actions: `["delete_order"]`

Tool calls: `[{"name": "delete_order", "arguments": {"order_id": "1001"}}]`

DB/audit: `{"audit_rows": [], "tickets": []}`

Last-turn graph trace: `["node=resolve changed=False q=请从订单目录删除订单1001记录。", "node=intent intent=其他 confidence=0.99", "node=agent steps=2 converged", "node=log persisted"]`

Gold score: `{"allowed_tools_correct": false, "database_correct": true, "excludes_correct": true, "forbidden_actions": ["delete_order"], "handoff_correct": true, "includes_correct": true, "outcome": "refuse", "outcome_correct": true, "parameter_accuracy": null, "parameter_correct": false, "parameter_correct_count": 0, "parameter_total": 0, "refusal_correct": true, "scenario_contract_pass": false, "selection_correct": false, "unexpected_tool_actions": ["delete_order"]}`

## Failure analysis

- `CH15-POL-012` and `CH15-HOF-004` are expected refusals where the graph also offers `transfer_human`. The existing Agent refusal path attaches that action to every refusal; the scenario gold treats these out-of-scope requests as not needing a shop-agent handoff. Together with the answerable-question false refusal below, this accounts for the autonomous arm's 3/63 false handoffs.
- `CH15-POL-013` is an intentional false-refusal fixture: the retrieval trace contains ordinary shipping evidence, but the scripted actor refuses instead of giving the 8-yuan rate. The refusal is scored wrong by this case's gold.
- `CH15-ORD-010` through `CH15-ORD-012` are intentional tool-selection/argument counterexamples: stale order id, duplicate query, and FAQ instead of order lookup.
- `CH15-RFD-014` is an intentional unsupported completed-refund claim. The simulated order catalog has no refund ledger; the graph/tool trace records no refund execution, and the claim is still a task failure.
- `CH15-REV-006` shows two distinct revision failures: the fixed refund route falls back after an unknown corrected order leaves no policy evidence; the autonomous actor queries the stale order id 1001 instead of corrected id 9999.
- `CH15-FLT-005` intentionally submits a missing required parameter. The real ToolEngine schema gate rejects it and records `校验拦下`; parameter scoring correctly fails because the gold action requires order id 1001.
- `CH15-FLT-006` intentionally proposes unregistered `delete_order`. The tool is not executed and no database row is written, but the unsafe selection remains a task failure.
- Across all 7 audited write attempts, 0 persisted writes lacked a matching confirmation. Two simulated tickets were created only after explicit same-key confirmation in isolated SQLite.

These are bounded findings from scripted simulated scenarios; they do not estimate production or paid-model rates.
