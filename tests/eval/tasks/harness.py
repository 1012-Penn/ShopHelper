"""CH15 deterministic task/tool evaluation over the production LangGraph and ToolEngine.

The scripted actor is an offline substitute for the paid chat model. Orders, policy
facts, SQLite persistence and confirmation routes are local simulation fixtures.
"""
from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from langchain.tools import tool
from langchain_core.messages import AIMessage, AIMessageChunk
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import Settings
from app.context import build_layered, derive_budgets
from app.graph.builder import build_graph, initial_state
from app.models import Base, Ticket, ToolAuditLog
from app.routers import tickets as ticket_router
from app.store import ConversationStore
from app.tools.audit import ToolAuditStore
from app.tools.base import ToolContext, ToolRecord, ToolRegistryV2
from app.tools.builtin import register_builtin
from app.tools.engine import ToolEngine
from app.guard import is_refusal


TASK_DIR = Path(__file__).resolve().parent
SCENARIO_PATH = TASK_DIR / "scenarios.jsonl"
POLICY_PATH = TASK_DIR / "fixtures" / "policies.json"


def load_scenarios(path: Path | None = None) -> list[dict]:
    source = path or SCENARIO_PATH
    return [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()]


class FixtureKB:
    def __init__(self, facts: list[dict]):
        self._rows = {
            int(fact["id"]): SimpleNamespace(
                id=int(fact["id"]), questions=fact["question"], answer=fact["answer"],
                category=fact["category"], section_path=fact["section_path"],
            )
            for fact in facts
        }

    def get_chunks(self, chunk_ids: list[int]) -> list[SimpleNamespace]:
        return [self._rows[cid] for cid in chunk_ids if cid in self._rows]


class FixtureRetrieval:
    """Deterministic keyword fixture with ranked, fixed scores and no network access."""

    def __init__(self, facts: list[dict]):
        self.facts = facts

    @staticmethod
    def _score(query: str, fact: dict) -> int:
        q = query.casefold()
        score = sum(3 for term in fact.get("match_any", []) if term.casefold() in q)
        for group in fact.get("match_all", []):
            if group and all(term.casefold() in q for term in group):
                score += 2
        return score

    def retrieve(self, query: str, **_kwargs):
        ranked = [(fact, self._score(query, fact)) for fact in self.facts]
        ranked = [(fact, score) for fact, score in ranked if score]
        ranked.sort(key=lambda pair: (-pair[1], int(pair[0]["id"])))
        items = [SimpleNamespace(chunk_id=int(fact["id"]), score=0.9 - i * 0.01)
                 for i, (fact, _score) in enumerate(ranked[:10])]
        return FixtureRetrievalResult(items, low_confidence=not items,
                                      reason="模拟政策 fixture 未命中" if not items else "")


class FixtureRetrievalResult:
    def __init__(self, items, *, low_confidence: bool, reason: str):
        self.items = items
        self.low_confidence = low_confidence
        self.reason = reason
        self.filter_fallback = False
        self.degraded = False

    def snapshot(self, top_k: int | None = None):
        items = self.items if top_k is None else self.items[:top_k]
        return [{"rank": i + 1, "chunk_id": item.chunk_id, "score": item.score,
                 "rerank_score": item.score} for i, item in enumerate(items)]


class ScriptedStructuredModel:
    def __init__(self, kind: str):
        self.kind = kind
        self.message = ""
        self.intent = "其他"

    async def ainvoke(self, _input, _config=None, **_kwargs):
        if self.kind == "resolve":
            return SimpleNamespace(resolved=self.message, changed=False)
        if self.kind == "expand":
            return SimpleNamespace(queries=[self.message])
        return AIMessage(content=json.dumps({"intent": self.intent, "confidence": 0.99},
                                            ensure_ascii=False))


class ScriptedActor:
    """Replay a fixed proposal/reply through the actual graph's streaming model API."""

    def __init__(self, scenario_id: str, arm: str, call_sink: list[dict]):
        self.scenario_id = scenario_id
        self.arm = arm
        self.call_sink = call_sink
        self.actor = {"tool_calls": [], "reply": ""}
        self.turn = 0
        self.generation = 0

    def set_turn(self, turn_number: int, actor: dict):
        self.turn = turn_number
        self.actor = actor
        self.generation = 0

    def bind_tools(self, _tools):
        return self

    async def astream(self, _messages, _config=None, **_kwargs):
        if self.generation == 0 and self.actor.get("tool_calls"):
            chunks = []
            for index, call in enumerate(self.actor["tool_calls"]):
                args = dict(call.get("arguments") or {})
                proposed = {"name": call["name"], "arguments": args}
                self.call_sink.append(proposed)
                chunks.append({"name": call["name"], "args": json.dumps(args, ensure_ascii=False),
                               "id": f"{self.scenario_id}-{self.arm}-{self.turn}-{index}",
                               "index": index})
            self.generation += 1
            yield AIMessageChunk(content="", tool_call_chunks=chunks)
            return
        if self.generation == 0:
            self.generation += 1
        else:
            self.generation += 1
        yield AIMessageChunk(content=self.actor.get("reply", ""))


class OfflinePool:
    def __init__(self):
        self.insertions = []

    def insert(self, source, conversation_id, question, reason="", retrieved_chunks=None):
        self.insertions.append({"source": source, "conversation_id": conversation_id,
                                "question": question, "reason": reason,
                                "retrieved_chunks": retrieved_chunks})
        return len(self.insertions)


def _db_factory():
    engine = create_engine("sqlite+pysqlite:///:memory:",
                           connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return engine, sessionmaker(bind=engine, expire_on_commit=False)


def _read_db(factory, session_id: int):
    with factory() as session:
        tickets = session.scalars(select(Ticket).where(
            Ticket.conversation_id == session_id).order_by(Ticket.ticket_no)).all()
        audits = session.scalars(select(ToolAuditLog).where(
            ToolAuditLog.conversation_id == session_id).order_by(ToolAuditLog.id)).all()
        return ([{"ticket_no": row.ticket_no, "conversation_id": row.conversation_id,
                 "confirm_key": row.confirm_key, "description": row.description,
                 "ticket_type": row.ticket_type} for row in tickets],
                [{"tool_name": row.tool_name, "status": row.status,
                  "arguments": row.arguments, "retry_count": row.retry_count,
                  "error_message": row.error_message} for row in audits])


async def _confirm_preview(app: FastAPI, preview: dict, *, repeat: bool,
                           drop_first_response: bool = False):
    requests = []
    responses = []
    payload = {
        "conversation_id": preview["conversation_id"],
        "tool_name": "create_ticket",
        "description": preview["description"],
        "ticket_type": preview.get("ticket_type", "售后"),
        "confirm_key": preview["confirm_key"],
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://ch15.test") as client:
        for request_index in range(2 if repeat else 1):
            requests.append(dict(payload))
            response = await client.post("/api/tickets/confirm", json=payload)
            if drop_first_response and request_index == 0:
                # Simulate the committed response being lost between endpoint and caller.
                responses.append({"client_visible": False,
                                  "injected_fault": "drop_after_commit_before_response_delivery"})
            else:
                responses.append({"client_visible": True,
                                  "status_code": response.status_code,
                                  "body": response.json() if response.content else None})
    return requests, responses


def _actual_outcome(reply: str, frames: list[dict], suggested_actions: list[str]) -> str:
    if any(frame.get("type") == "order_selector" for frame in frames):
        return "clarify"
    if is_refusal(reply):
        return "refuse"
    if re.search(r"请.{0,8}(?:提供|补充|核对订单号|核对单号|确认单号)|请先核实|(?:暂时|无法)判断|参数.{0,8}不完整|需要.{0,8}(?:订单号|单号)", reply):
        return "clarify"
    if "transfer_human" in suggested_actions:
        return "handoff"
    return "answer"


def _claim_is_present(text: str, phrase: str) -> bool:
    """Treat explicit denials as non-claims while retaining an exact phrase rubric."""
    for match in re.finditer(re.escape(phrase), text):
        prefix = text[max(0, match.start() - 12):match.start()]
        if not re.search(
                r"(?:不能|不可以|不应|无法|不承诺|未承诺|没有显示|没有|并非|并未|"
                r"并无|尚未|暂未|未显示|不存在)\s*$", prefix):
            return True
    return False


def score_case(row: dict, arm: str, trace: dict) -> dict:
    gold = row["gold"][arm]
    reply = trace.get("reply", "") or ""
    expected_tools = list(gold["expected_tools"])
    observed_tools = list(trace.get("tool_actions", []))
    selection_correct = observed_tools == expected_tools
    allowed_tools = set(row["allowed_tools"])
    unexpected_tool_actions = [name for name in observed_tools if name not in allowed_tools]
    allowed_tools_correct = not unexpected_tool_actions
    expected_calls = list(gold["expected_calls"])
    observed_calls = list(trace.get("tool_calls", []))
    parameter_total = len(expected_calls)
    parameter_correct_count = sum(
        expected == observed
        for expected, observed in zip(expected_calls, observed_calls)
    )
    parameter_correct = (parameter_correct_count == parameter_total
                         and len(observed_calls) == parameter_total)
    parameter_accuracy = (parameter_correct_count / parameter_total
                          if parameter_total else None)

    refused = is_refusal(reply)
    refusal_correct = refused is bool(gold["expected_refusal"])
    actual_handoff = "transfer_human" in trace.get("suggested_actions", [])
    handoff_correct = actual_handoff is bool(gold["expected_handoff"])
    turn_frames = trace.get("turn_frames") or [trace.get("frames", [])]
    last_turn_frames = turn_frames[-1]
    outcome = _actual_outcome(reply, last_turn_frames,
                              trace.get("suggested_actions", []))
    outcome_correct = outcome == gold["outcome"]

    displayed = []
    for frame in last_turn_frames:
        if frame.get("type") == "actions":
            displayed.extend(item.get("label", "") for item in frame.get("items", []))
        elif frame.get("type") == "order_selector":
            displayed.append("选择订单")
        elif frame.get("type") == "ticket_preview":
            displayed.append("工单预览卡片")
    combined_reply = "\n".join([reply, *displayed])
    includes_correct = all(term in combined_reply for term in gold["must_include"])
    excludes_correct = all(not _claim_is_present(combined_reply, term)
                           for term in gold["must_not_include"])

    assertions = gold["db_assertions"]
    tickets = trace.get("tickets", [])
    tickets_correct = len(tickets) == assertions["tickets"]
    statuses = Counter(trace.get("audit_statuses", []))
    required_statuses = Counter(assertions.get("required_audit_statuses", []))
    audit_correct = all(statuses[status] >= count for status, count in required_statuses.items())
    database_correct = tickets_correct and audit_correct
    forbidden = list(trace.get("forbidden_actions", []))
    scenario_contract_pass = all((selection_correct, allowed_tools_correct,
                                  parameter_correct, refusal_correct,
                                  handoff_correct, outcome_correct, includes_correct,
                                  excludes_correct, database_correct, not forbidden))

    return {
        "scenario_contract_pass": scenario_contract_pass,
        "selection_correct": selection_correct,
        "allowed_tools_correct": allowed_tools_correct,
        "unexpected_tool_actions": unexpected_tool_actions,
        "parameter_correct": parameter_correct,
        "parameter_correct_count": parameter_correct_count,
        "parameter_total": parameter_total,
        "parameter_accuracy": parameter_accuracy,
        "refusal_correct": refusal_correct,
        "handoff_correct": handoff_correct,
        "outcome": outcome,
        "outcome_correct": outcome_correct,
        "includes_correct": includes_correct,
        "excludes_correct": excludes_correct,
        "database_correct": database_correct,
        "forbidden_actions": forbidden,
    }


async def run_case(row: dict, arm: str, *, refund_flow_mode: str = "fixed") -> dict:
    """Run one scripted task arm through build_graph, ToolEngine, SQLite and confirm API."""
    facts = json.loads(POLICY_PATH.read_text(encoding="utf-8"))["facts"]
    kb = FixtureKB(facts)
    retrieval = FixtureRetrieval(facts)
    db_engine, factory = _db_factory()
    store = ConversationStore(factory)
    session_id = await store.resolve(None)
    audit = ToolAuditStore(factory)
    pool = OfflinePool()
    registry = ToolRegistryV2()
    register_builtin(registry, ToolContext(session_factory=factory, service=retrieval,
                                           kb=kb, top_k=10))

    fault = row.get("fault") or {}
    timeout = float(fault.get("timeout_seconds", 2.0))
    settings = Settings(_env_file=None, openai_api_key="offline-scripted-model",
                        evidence_entailment_enabled=False, query_rewrite_enabled=False,
                        tool_timeout_seconds=timeout, tool_retry_attempts=1,
                        max_agent_steps=6)
    if fault.get("type") == "tool_timeout":
        @tool
        async def ch15_slow_read(delay_ms: int) -> str:
            """CH15 isolated asynchronous read fixture that deterministically exceeds timeout."""
            await asyncio.sleep(delay_ms / 1000)
            return json.dumps({"ok": True, "simulated": True})

        registry.register(ToolRecord(ch15_slow_read, source="builtin", access="read",
                                     label="CH15 慢查询 fixture"))

    engine = ToolEngine(registry, audit, timeout_seconds=settings.tool_timeout_seconds,
                        retry_attempts=settings.tool_retry_attempts)
    actual_calls: list[dict] = []
    model = ScriptedActor(row["id"], arm, actual_calls)
    resolver = ScriptedStructuredModel("resolve")
    expander = ScriptedStructuredModel("expand")
    intent_model = ScriptedStructuredModel("intent")
    graph = build_graph(model, engine, settings, store, pool, retrieval, kb,
                        resolver=resolver, intent_model=intent_model, expander=expander,
                        refund_flow_mode=refund_flow_mode)
    app = FastAPI()
    app.include_router(ticket_router.router)
    app.state.store = store
    app.state.engine = engine
    app.state.registry = registry

    engine_events = []
    original_execute = engine.execute

    async def traced_execute(name, args_json, **kwargs):
        result = await original_execute(name, args_json, **kwargs)
        try:
            args = json.loads(args_json or "{}")
        except (TypeError, json.JSONDecodeError):
            args = args_json
        engine_events.append({"name": name, "arguments": args,
                              "confirmed": bool(kwargs.get("confirmed")),
                              "result": result})
        return result

    engine.execute = traced_execute

    frames: list[dict] = []
    turn_frames: list[list[dict]] = []
    replies: list[str] = []
    graph_traces = []
    final_state = {}
    fault_trace: dict = {"type": (row.get("fault") or {}).get("type"),
                         "confirmation_requests": [], "confirmation_responses": []}
    deterministic_id = 0

    class DeterministicUUID:
        def __init__(self, value):
            self.hex = value

    def fixed_uuid():
        nonlocal deterministic_id
        deterministic_id += 1
        return DeterministicUUID(f"ch15-confirm-{deterministic_id:04d}")

    def fixed_ticket_no():
        nonlocal deterministic_id
        deterministic_id += 1
        return f"TCH15{deterministic_id:08d}"

    with ExitStack() as stack:
        from unittest.mock import patch
        stack.enter_context(patch("uuid.uuid4", fixed_uuid))
        stack.enter_context(patch("app.tickets.new_ticket_no", fixed_ticket_no))
        for turn_index, turn in enumerate(row["turns"]):
            message = turn["message"]
            resolver.message = message
            expander.message = message
            intent_model.intent = turn["intents"][arm]
            model.set_turn(turn_index + 1, turn["actors"][arm])
            current_rows = await store.get_rows(session_id)
            budgets = derive_budgets(settings)
            layered = build_layered(current_rows, [], "",
                                   settings.ctx_layer2_assistant_head_chars,
                                   budgets.sliding)
            state = initial_state(session_id, message,
                                  history_rows=[] if turn_index == 0 else None,
                                  layered=layered)
            queue = asyncio.Queue()
            config = {"configurable": {"thread_id": str(session_id), "sink": queue}}
            final_state = await graph.ainvoke(state, config)
            current_frames = []
            while not queue.empty():
                frame = queue.get_nowait()
                frames.append(frame)
                current_frames.append(frame)
            turn_frames.append(current_frames)
            replies.append(final_state.get("final_reply", "") or "")
            graph_traces.append(final_state.get("trace", []))

        fault_type = (row.get("fault") or {}).get("type")
        previews = [frame for frame in frames if frame.get("type") == "ticket_preview"]
        if fault_type in {"confirm_twice_same_key", "drop_first_confirm_response_then_retry"}:
            if not previews:
                fault_trace["error"] = "Agent 没有产出确认预览卡片"
            else:
                requests, responses = await _confirm_preview(
                    app, previews[-1], repeat=True,
                    drop_first_response=(fault_type == "drop_first_confirm_response_then_retry"))
                fault_trace["confirmation_requests"] = requests
                fault_trace["confirmation_responses"] = responses
                fault_trace["response_dropped_after_commit"] = (
                    fault_type == "drop_first_confirm_response_then_retry")

    tickets, audit_rows = _read_db(factory, session_id)
    suggested_actions = list(final_state.get("suggested_actions", []))
    if not suggested_actions:
        actions_frames = [frame for frame in frames if frame.get("type") == "actions"]
        suggested_actions = [item.get("action") for frame in actions_frames
                             for item in frame.get("items", [])]
    tool_actions = [frame["name"] for frame in frames
                    if frame.get("type") == "tool_status" and frame.get("name") != "expand"]
    # Model calls always produce tool_status frames; preserve the attempted action even if
    # the engine rejects it (including an unregistered dangerous name).
    observed_call_args = actual_calls
    confirmed_keys = {request["confirm_key"] for request in fault_trace["confirmation_requests"]}
    unconfirmed_tickets = [ticket for ticket in tickets
                           if not ticket["confirm_key"] or ticket["confirm_key"] not in confirmed_keys]
    forbidden_actions = []
    if any(call["name"] == "delete_order" for call in actual_calls):
        forbidden_actions.append("delete_order")
    if unconfirmed_tickets:
        forbidden_actions.append("unconfirmed_ticket_persist")
    refund_claim_phrases = ("已退款", "已经退款", "成功退款", "退款成功", "退款完成")
    if any(_claim_is_present("\n".join(replies), phrase)
           for phrase in refund_claim_phrases):
        forbidden_actions.append("claim_refund_completed")
    if any(call["name"] in {"refund", "refund.execute"} for call in actual_calls):
        forbidden_actions.append("refund.execute")

    trace = {
        "reply": replies[-1] if replies else "",
        "replies": replies,
        "frames": frames,
        "turn_frames": turn_frames,
        "tool_actions": tool_actions,
        "tool_calls": observed_call_args,
        "engine_events": engine_events,
        "graph_trace": graph_traces,
        "suggested_actions": suggested_actions,
        "tickets": tickets,
        "ticket_count": len(tickets),
        "audit_rows": audit_rows,
        "audit_statuses": [row["status"] for row in audit_rows],
        "write_attempt_count": sum(row["tool_name"] == "create_ticket" for row in audit_rows),
        "unconfirmed_persisted_count": len(unconfirmed_tickets),
        "forbidden_actions": forbidden_actions,
        "fault_trace": fault_trace,
        "pool_insertions": pool.insertions,
        "intent_by_turn": [turn["intents"][arm] for turn in row["turns"]],
        "resolved_messages": [turn["message"] for turn in row["turns"]],
    }
    trace["score"] = score_case(row, arm, trace)
    db_engine.dispose()
    return trace


def _rate(numerator: int, denominator: int) -> dict:
    return {"numerator": numerator, "denominator": denominator,
            "rate": round(numerator / denominator, 4) if denominator else None}


def summarize_runs(scenarios: list[dict], results: list[dict]) -> dict:
    scenario_by_id = {row["id"]: row for row in scenarios}
    summary = {"arms": {}, "paired_comparison": {}, "case_count": len(scenarios),
               "run_count": len(results)}
    for arm in ("fixed", "autonomous"):
        selected = [result for result in results if result["arm"] == arm]
        if not selected:
            continue
        complete = sum(bool(item["score"]["scenario_contract_pass"]) for item in selected)
        selected_correct = sum(bool(item["score"]["selection_correct"]) for item in selected)
        param_ok = sum(item["score"]["parameter_correct_count"] for item in selected)
        param_total = sum(item["score"]["parameter_total"] for item in selected)
        refusal_correct = sum(bool(item["score"]["refusal_correct"]) for item in selected)
        refusal_gold = sum(bool(scenario_by_id[item["scenario_id"]]["gold"][arm]
                                ["expected_refusal"]) for item in selected)
        handoff_correct = sum(bool(item["score"]["handoff_correct"])
                              for item in selected
                              if scenario_by_id[item["scenario_id"]]["gold"][arm]
                              ["expected_handoff"])
        handoff_expected = sum(bool(scenario_by_id[item["scenario_id"]]["gold"][arm]
                                    ["expected_handoff"]) for item in selected)
        write_attempts = sum(item["write_attempt_count"] for item in selected)
        unconfirmed = sum(item["unconfirmed_persisted_count"] for item in selected)
        false_handoffs = sum(
            "transfer_human" in item["suggested_actions"]
            for item in selected
            if not scenario_by_id[item["scenario_id"]]["gold"][arm]["expected_handoff"]
        )
        false_handoff_total = sum(
            not scenario_by_id[item["scenario_id"]]["gold"][arm]["expected_handoff"]
            for item in selected
        )
        summary["arms"][arm] = {
            "runs": len(selected),
            "scripted_scenario_contract_pass": _rate(complete, len(selected)),
            "tool_selection_accuracy": _rate(selected_correct, len(selected)),
            "parameter_accuracy": _rate(param_ok, param_total),
            "parameter_na_runs": sum(item["score"]["parameter_total"] == 0 for item in selected),
            "unconfirmed_write_rate": _rate(unconfirmed, write_attempts),
            "reasonable_handoff_rate": _rate(handoff_correct, handoff_expected),
            "false_handoff_rate": _rate(false_handoffs, false_handoff_total),
            "refusal_gold_alignment": _rate(refusal_correct, len(selected)),
            "gold_refusal_cases": refusal_gold,
        }

    paired = [row for row in scenarios if row["comparison_scope"] == "paired_refund"]
    paired_by_id = {row["id"]: row for row in paired}
    for arm in ("fixed", "autonomous"):
        arm_results = [item for item in results
                       if item["arm"] == arm and item["scenario_id"] in paired_by_id]
        completed = sum(bool(item["score"]["scenario_contract_pass"])
                        for item in arm_results)
        summary["paired_comparison"][arm] = {
            "scripted_scenario_contract_pass": _rate(completed, len(paired)),
            "scenario_count": len(paired),
            "turn_count": sum(len(row["turns"]) for row in paired),
        }
    fixed_rate = summary["paired_comparison"].get("fixed", {}).get(
        "scripted_scenario_contract_pass", {})
    autonomous_rate = summary["paired_comparison"].get("autonomous", {}).get(
        "scripted_scenario_contract_pass", {})
    if autonomous_rate.get("denominator", 0) and fixed_rate.get("denominator", 0):
        delta = (autonomous_rate["numerator"] / autonomous_rate["denominator"]
                 - fixed_rate["numerator"] / fixed_rate["denominator"])
        summary["paired_comparison"]["paired_contract_pass_delta"] = round(delta, 4)
    else:
        summary["paired_comparison"]["paired_contract_pass_delta"] = None
    return summary


async def run_suite(scenarios: list[dict] | None = None):
    rows = scenarios or load_scenarios()
    results = []
    for row in rows:
        arms = ("fixed", "autonomous") if row["comparison_scope"] == "paired_refund" else ("autonomous",)
        for arm in arms:
            trace = await run_case(row, arm)
            results.append({"scenario_id": row["id"], "category": row["category"],
                            "comparison_scope": row["comparison_scope"], "arm": arm,
                            "scenario_record": {
                                "id": row["id"], "category": row["category"],
                                "comparison_scope": row["comparison_scope"],
                                "allowed_tools": row["allowed_tools"],
                                "forbidden_actions": row["forbidden_actions"],
                                "fault": row.get("fault"),
                                "gold": row["gold"][arm],
                                "turns": [{"message": turn["message"],
                                           "intent": turn["intents"][arm],
                                           "expected": turn["expected"][arm],
                                           "actor": turn["actors"][arm]}
                                          for turn in row["turns"]]},
                            **trace})
    return results, summarize_runs(rows, results)


def render_summary_markdown(summary: dict) -> str:
    lines = ["# CH15 Offline Contract Evaluation Summary", "",
             f"Run date: {summary.get('run_date', 'not recorded')}.", "",
             f"Scenarios: {summary['case_count']} fixed simulated cases; "
             f"executed arm runs: {summary['run_count']}.", "",
             "The actor is a deterministic scripted model substitute. This run validates "
             "graph wiring, tool contracts and scoring; it does not measure a paid LLM or "
             "real customer-service outcomes. Agent task success rate was not measured.", "",
             "## Metrics by arm — scripted scenario contract pass", "",
             "| Arm | Scripted scenario contract pass | Tool selection | Parameter accuracy | Unconfirmed write rate | "
             "Reasonable handoff | Refusal/gold alignment |", "|---|---:|---:|---:|---:|---:|---:|"]
    for arm, metrics in summary["arms"].items():
        def fmt(key):
            item = metrics[key]
            return "N/A (0/0)" if item["rate"] is None else (
                f"{item['numerator']}/{item['denominator']} ({item['rate']:.1%})")
        lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            arm, fmt("scripted_scenario_contract_pass"), fmt("tool_selection_accuracy"),
            fmt("parameter_accuracy"), fmt("unconfirmed_write_rate"),
            fmt("reasonable_handoff_rate"), fmt("refusal_gold_alignment")))
    lines += ["", "The scripted scenario contract pass column is not an Agent task success rate. "
              "The actor receives prewritten intents, tool actions and responses; no model "
              "independently generated those decisions in this run.", "",
              "`Parameter accuracy` is N/A where the fixed subgraph performs no "
              "ToolEngine call with model-supplied arguments. `Unconfirmed write rate` "
              "uses persisted writes without a matching confirmation over all audited "
              "write attempts. Refusal alignment compares each result to its case gold.", "",
              "## Paired scripted contract pass: actor variants on the default intent router", "",
              "| Arm | Contract-passing scenarios / paired scenarios | Turns |", "|---|---:|---:|"]
    for arm in ("fixed", "autonomous"):
        metric = summary["paired_comparison"][arm]
        rate = metric["scripted_scenario_contract_pass"]
        lines.append(f"| {arm} | {rate['numerator']}/{rate['denominator']} "
                     f"({rate['rate']:.1%}) | {metric['turn_count']} |")
    delta = summary["paired_comparison"]["paired_contract_pass_delta"]
    delta_text = "N/A (0/0)" if delta is None else f"{delta * 100:+.1f} percentage points"
    lines += ["", "Paired scripted contract pass difference (autonomous minus fixed): "
              f"{delta_text}. Both offline arms used the production default route and "
              "arm-specific prewritten intents/actors; this is not a controlled route "
              "ablation or evidence that one approach has a higher Agent task success rate. "
              "The real-model runner adds an explicit fixed-vs-autonomous route mode.", ""]
    return "\n".join(lines)
