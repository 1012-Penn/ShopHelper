# Multi-Agent Minimal Collaboration Loop

## Goal

Build an optional LangGraph path that is genuinely multi-agent and directly comparable with
the existing constrained single Agent. The default product graph remains unchanged.

## Runtime contract

- `Supervisor/Planner` produces a structured plan with at most three tasks.
- `RetrievalAgent` can only call `query_order` and `query_faq`.
- `ActionAgent` can only call `create_ticket`; it receives prior agent facts through shared state.
- Every task has a stable run/task/agent ID, explicit dependencies, and its own message history.
- Retrieval tasks without dependencies may run in parallel. An action task must depend on a
  retrieval task, so policy judgment cannot race ahead of order evidence.
- Shared state contains JSON-serializable business facts, evidence IDs, errors, stop reasons,
  handoffs, and tool/write-preview events. Raw gold labels are never an input.
- The graph permits one replan, at most three planned tasks per plan, and enforces global step,
  token, wall-clock, and child timeout limits. A failed child is recorded and cannot be silently
  dropped from evaluation.
- Write tools still pass through `ToolEngine`. Model-originated calls are intercepted as previews;
  only `execute_confirmed` may reach the isolated fake write sink.

## Execution

The LangGraph nodes are `supervisor_plan -> execute_tasks -> supervisor_finalize`, with an optional
one-shot `supervisor_replan` branch after task failures. The task executor resolves dependency
waves and uses `asyncio.gather` for independent retrieval tasks. Each agent loops over its own
message list and only its whitelisted tools.

The Supervisor calls the planner and writes the final answer from shared facts/drafts. The planner,
retrieval Agent, action Agent, and final synthesis have separate system prompts and call labels.

## Evaluation

- Freeze 10 development and 20 held-out simulated scenarios before the paid comparison.
- Cover simple controls, two-order comparisons, policy/order dependencies, user revisions,
  clarification, refusals, and ticket preview/confirm/cancel.
- Run the existing single Agent node and the multi-Agent graph on the same user-visible turns,
  model, tool registry, evidence fixtures, and injected price accounting.
- Keep gold outside all model inputs. Score task outcome, missing/wrong identifiers or conditions,
  reasonable clarification/refusal, tool and handoff order, latency, calls, tokens, and cost.
- Analyze paired results by simple/complex buckets. Timeouts and failures remain in the denominator.
- Use a separate SQLite budget/attempt ledger with a hard $2 run cap, one pre-call reservation,
  zero provider retries, and post-call usage reconciliation.
- Cache replay, if used, is reported separately from cold calls. The paid run must include actual
  model confirmation and cancellation coverage for the fake write sink.

## Delivery

Add only `app/multi_agent/*`, evaluation assets/scripts, tests, and documentation. Provide a router
factory plus a runnable CLI. Put any changes to existing application files in
`dev-notes/integration-notes.md` as small integration patches. Rollback is removal of the optional
router/factory; the existing graph and default route stay untouched.
