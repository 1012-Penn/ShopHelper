# CH15 Agent Task Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fixed, auditable 72-case simulated task set and an offline end-to-end evaluator that runs the real ShopHelper LangGraph, tool engine and confirmation contract.

**Architecture:** Keep production modules unchanged. Store scenario golds and deterministic policy facts under `tests/eval/tasks`; build a test-only SQLite/ASGI runtime around the production graph and tools, with a scripted local actor and injectable fault tools. Emit paired fixed-refund versus constrained-autonomous results, per-case traces, aggregate denominators and a cost proposal without invoking a paid model.

**Tech Stack:** Python 3.10+, pytest/pytest-asyncio, FastAPI/ASGITransport, LangGraph, SQLAlchemy SQLite, existing ShopHelper tool registry and engine.

**Spec:** `docs/superpowers/specs/2026-10-25-ch15-agent-task-eval-design.md`, implementing `docs/superpowers/specs/2026-10-25-ch15-ch17-interview-evidence-roadmap.md` §1–3.

## Global Constraints

- Base commit is `dde19f3420b6da00c8e28dd9f503e1de4b6cfde9` from `origin/main`; branch is `codex/ch15-agent-task-eval`.
- Keep all changes in the requested isolated worktree; leave the original checkout and all other worktrees untouched.
- Do not modify CH11 frozen cases, CH11 historical reports, or the production default refund flow.
- Use deterministic simulated business facts; never call a real model, MCP server, customer system, or refund system during offline validation.
- Keep all test writes in per-run SQLite memory databases and require explicit confirmation for the real `create_ticket` write contract.
- Label every result as simulated. Do not claim a real customer outcome, model improvement, or business ROI.
- Do not push or merge.

---

### Task 1: Freeze the scenario contract, fixtures, and rubric

**Files:**
- Create: `tests/eval/tasks/scenarios.jsonl`
- Create: `tests/eval/tasks/fixtures/policies.json`
- Create: `tests/eval/tasks/test_ch15_contract.py`
- Create: `docs/eval/ch15-design-and-rubric.md`
- Create: `docs/superpowers/specs/2026-10-25-ch15-agent-task-eval-design.md`
- Create: `docs/superpowers/plans/2026-10-25-ch15-agent-task-eval.md`

**Interfaces:** Each JSONL row has `id`, `category`, `comparison_scope`, `turns`, `gold`, `allowed_tools`, `forbidden_actions`, and deterministic actor scripts. Each arm's `gold` has database assertions; each turn records user text, exact fixed/autonomous intent fixture, expected actions and actor script. Contract tests verify 72 unique IDs, required fields, valid categories/outcomes, aligned per-turn and aggregate gold, allowed tool subsets, and database assertions.

- [x] Add contract tests first and run them to confirm the missing scenario file fails for the expected reason.
- [x] Author exactly 72 static scenarios: 14 policy, 12 order, 18 refund guidance, 12 multi-turn revision, 10 refusal/handoff, 6 fault/confirmation.
- [x] Add deterministic policy fixture statements for return eligibility, shipping, refund timing, return review and return shipment timing; never import CH11 frozen rows.
- [x] Encode exact expected tool sequence and arguments per scenario; include prohibited write claims/actions and SQLite postconditions.
- [x] Run the contract tests and confirm every scenario has a complete gold before reading generated results.
- [x] Review the design/rubric against the roadmap; define each rate with a numerator, denominator and N/A rule.
- [x] Commit this coherent dataset/design slice.

### Task 2: Build an offline real-graph runner and scorer

**Files:**
- Create: `tests/eval/tasks/harness.py`
- Create: `tests/eval/tasks/test_ch15_harness.py`
- Create: `scripts/eval_agent_tasks.py`

**Interfaces:** `load_scenarios(path) -> list[dict]` reads the static manifest. `score_case(scenario, arm, trace) -> dict` returns task, tool-selection, parameter, refusal, handoff and safety values. `run_suite(scenarios) -> (run_records, summary)` builds a new in-memory runtime for each arm/scenario and invokes the production `build_graph` directly with `initial_state`; it collects emitted frames, graph state traces, ToolEngine calls/results/audits and SQLite rows. Confirmation tests use the production ticket ASGI route. The CLI writes a self-contained per-arm JSONL report and summary Markdown.

- [x] Write scorer and runtime contract tests first, including a golden that treats a refusal as successful only for a refusal gold.
- [x] Run the tests and verify they fail because the harness functions are absent.
- [x] Implement the in-memory SQLite runtime with real `build_graph`, registry/`ToolEngine`, ticket route, deterministic resolver/expander/retriever and scripted actor.
- [x] Route paired refund cases through the production fixed subgraph when the intent remains refund/aftersales; use the test-only intent fixture for revised order/policy/withdrawal turns and the autonomous path.
- [x] Record per-turn graph traces/frames, tool calls/results, audit rows, final replies, handoff/refusal flags and database state.
- [x] Implement exact action-sequence and parameter scoring; summaries include numerators, denominators, and null for zero denominators.
- [x] Run the CH15 tests and full local simulated CLI suite; no provider client or MCP request is made.
- [x] Commit this runner/test slice.

### Task 3: Exercise confirmation and fault injection end to end

**Files:**
- Modify: `tests/eval/tasks/harness.py`
- Modify: `tests/eval/tasks/scenarios.jsonl`
- Modify: `tests/eval/tasks/test_ch15_harness.py`

**Interfaces:** The runtime registers a test-only slow read tool for fault cases, sends user-approved ticket confirmation through `/api/tickets/confirm`, and applies scenario fault instructions only after the actual graph or tool endpoint has run. `db_assertions` are evaluated against the isolated SQLite ORM state after each case.

- [x] Add failing tests proving same-key repeated confirmation creates one row, an intentionally discarded post-commit response followed by same-key replay creates one row, and timeout produces a timeout audit without a ticket.
- [x] Run those tests to confirm the current harness lacks the fault paths.
- [x] Implement six fault cases: duplicate confirmation, response lost after committed write, tool timeout, attempted unconfirmed write, invalid arguments, and unknown tool.
- [x] Keep direct model write attempts behind the existing Engine permission intercept; only the test's simulated user confirmation endpoint may persist a ticket.
- [x] Assert final row counts and audit statuses; include refusal/handoff outcomes in their pre-frozen scenario gold.
- [x] Run the CH15 harness tests and fault cases offline; commit this fault-test slice.

### Task 4: Produce reports, interview notes, and verify the branch

**Files:**
- Create: `docs/eval/ch15-case-report.jsonl`
- Create: `docs/eval/ch15-offline-report.md`
- Create: `docs/eval/ch15-failures.md`
- Create: `docs/eval/ch15-paid-run-proposal.md`
- Create: `dev-notes/ch15.md`
- Modify: `docs/eval/ch15-design-and-rubric.md`

- [x] Run the complete offline suite and write the per-case trace report from the frozen scenarios.
- [x] Derive every aggregate count and rate from the JSONL run report; show numerator and denominator for each requested metric and for both paired arms.
- [x] List concrete failed cases with expected behavior, observed behavior, trace evidence, likely cause, and bounded impact.
- [x] Document paid-model volume by arm and a cost envelope using the current official price snapshot; mark the proposal pending human arrangement and do not invoke the API.
- [x] Write `dev-notes/ch15.md` with design decisions, simulator limitations, task metrics, contrast result and resume-ready candidate wording with “simulated scenarios” explicit.
- [x] Run the CH15 test files, related tool/graph/idempotency regressions, and the full repository pytest suite.
- [x] Inspect the diff and git status to verify only this worktree changed and CH11/production defaults were untouched; create a final coherent documentation/report commit.
- [x] Leave the branch/worktree in place; do not push or merge.

