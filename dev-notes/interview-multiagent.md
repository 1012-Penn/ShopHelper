# Interview Multi-Agent: implementation and evidence

## Scope

Added an opt-in LangGraph path at `app/multi_agent/`. The default single-Agent graph,
configuration, and application entrypoint are unchanged.

The Supervisor/Planner emits a structured plan with at most three tasks. Retrieval and
action Agents have independent message histories, system prompts, and hard tool
allowlists. Retrieval tasks without dependencies run concurrently; an action task must
declare a completed retrieval dependency. Shared state records run/task/agent IDs,
serializable facts, evidence IDs, tool calls, errors, replan count, and stop reasons.

`ToolEngine` remains the only execution boundary. Model-originated writes are intercepted
as ticket previews. Explicit confirmation must use `execute_confirmed`; the preview
contains the existing confirmation fields plus `confirm_key` for idempotency.

## Files

- `app/multi_agent/models.py`: serializable plan/outcome/state contracts.
- `app/multi_agent/roles.py`: Planner, retrieval, action, and Supervisor prompts/whitelists.
- `app/multi_agent/runtime.py`: task execution, dependency scheduling, guards, LangGraph.
- `app/multi_agent/api.py`: optional router and standalone app factory.
- `scripts/eval_multi_agent.py`: paired runner, protected ledger, rescoring, cost audit.
- `tests/eval/multi_agent/scenarios.jsonl`: 10 dev + 20 frozen simulated cases.
- `tests/test_multi_agent_*.py`: graph, isolation, write, timeout, budget, scoring tests.

## Observed issue -> diagnosis -> change

1. A first real batch exposed a graph-level token exception escaping final synthesis.
   A red test now proves planning and finalization failures return `status=failed` plus
   structured errors instead of aborting the batch.
2. The reused CH15 paid ledger assumed only one in-flight call. Two independent retrieval
   tasks hit this limitation in `MA-FRZ-022`. A concurrent persistent ledger now reserves
   every pending call atomically; the original failed answer remains in the frozen result.
3. Initial scoring treated dates and evidence IDs as order entities, considered an optional
   offer to ask for an order as clarification, and missed safe equivalent wording.
   Scorer v2 is data-driven and was applied to all saved answers with zero model calls.

## Frozen evaluation

Model: `deepseek-flash`, temperature 0, thinking disabled, zero provider retries.
Model calls and scorer used the same tools/evidence fixture for both arms. Gold never
entered a model prompt. See `docs/eval/multi-agent-comparison-report.md`.

- Held-out: multi-Agent 17/20 (85.0%), single Agent 15/20 (75.0%).
- Paired held-out: 4 wins, 2 losses, 14 ties.
- Combined 30 cases: multi-Agent 27/30, single Agent 23/30.
- Multi-Agent P50/P95: 7213/17303 ms; single Agent: 1906/4800 ms.
- Final call cost: $0.069735. All smoke/debug attempts: $0.093896 charged upper bound.
- Multi-Agent was about 5.9x the single-Agent cost per successful task. Do not switch the
  default route on this evidence.

Write coverage is reported separately: two multi-Agent models generated previews and both
confirmed once with the same key into the fake sink. No unconfirmed row reached the sink.
The cancel-only model case produced no preview, so it is a failed coverage case rather
than a 0/0 safety success.

## Commands

```bash
/root/workplace/ShopHelper/.venv/bin/python -m pytest -q \
  tests/test_multi_agent_runtime.py \
  tests/test_multi_agent_contracts.py \
  tests/test_multi_agent_eval.py

/root/workplace/ShopHelper/.venv/bin/python scripts/eval_multi_agent.py \
  --offline --split dev --out /tmp/multi-agent-offline

/root/workplace/ShopHelper/.venv/bin/python scripts/eval_multi_agent.py \
  --rescore-all

/root/workplace/ShopHelper/.venv/bin/python scripts/eval_multi_agent.py \
  --cost-audit
```

The paid path requires explicit `--allow-paid`, a separate ledger, and at most $2.
Saved full prompt/response/tool traces are gzip-compressed under
`tests/eval/multi_agent/runs/`.

Final verification on this branch:

```text
19 passed  # multi-agent targeted tests
833 passed, 2 skipped, 1 xfailed, 4 warnings  # full pytest regression
```

The warnings are pre-existing async-mark warnings in CH07/CH08 tests. Milvus Lite emits
non-fatal logging errors while releasing temporary servers after pytest exits.

## Honest resume candidates

- “为客服 Agent 实现 Supervisor/Planner + 检索 Agent + 行动 Agent 的 LangGraph
  协作闭环，以结构化依赖调度并行只读任务，并在 30 个模拟场景上与现有单 Agent
  同题配对；多 Agent 27/30、单 Agent 23/30。”
- “为多 Agent 增加工具白名单、子任务超时/全局 token 和时间护栏、写入预览与幂等确认，
  并用逐调用账本和 10 dev/20 held-out 轨迹复核收益与成本；观察到多 Agent 单任务成本
  约为单 Agent 的 5.9 倍，因此保留单 Agent 为默认路由。”

These statements describe simulated scenarios only and do not claim customer ROI or
production readiness.
