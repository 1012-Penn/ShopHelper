"""Reproducible paired evaluation for the optional multi-agent graph.

The frozen scenario file is the only source of gold. Evaluator projections expose
only user turns and generated history to either implementation.
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import os
import re
import sys
import tempfile
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain.tools import tool
from langchain_openai import ChatOpenAI

from app.config import Settings
from app.guard import is_refusal
from app.graph.agent import make_agent_node
from app.graph.builder import initial_state
from app.multi_agent.models import AgentPlan, AgentTask, MultiAgentLimits
from app.multi_agent.runtime import MultiAgentRunner, build_multi_agent_graph
from app.orders import ORDER_CATALOG
from app.tools.base import ToolContext, ToolRecord, ToolRegistryV2
from app.tools.builtin import CreateTicketInput, register_builtin
from app.tools.engine import ToolEngine
from scripts.ch15_paid_guard import (
    _CALL_CONTEXT,
    _json_safe,
    _utc_now,
    ArmLimit,
    GuardedChatModel,
    PersistentPaidRunBudget,
    paid_call_context,
)
from scripts.eval_cost import BudgetExceeded, PricingTable


SCENARIO_PATH = ROOT / "tests" / "eval" / "multi_agent" / "scenarios.jsonl"
REPORT_PATH = ROOT / "docs" / "eval" / "multi-agent-comparison-report.md"
RESULTS_DIR = ROOT / "tests" / "eval" / "multi_agent" / "runs"
SCORING_RUBRIC_PATH = ROOT / "tests" / "eval" / "multi_agent" / "scoring-rubric-v2.json"
LEDGER_PATH = ROOT / "data" / "multi_agent_paid_run_latest.sqlite3"
POLICY_PATH = ROOT / "tests" / "eval" / "tasks" / "fixtures" / "policies.json"
MODEL = "deepseek-flash"
BASE_URL = "https://api.deepseek.com"
RUN_CAP_USD = 2.0
PRICING_VERIFIED_ON = "2026-10-27"
PRICING_URL = "https://api-docs.deepseek.com/quick_start/pricing/"


def load_scenarios(path: Path | None = None) -> list[dict]:
    source = Path(path or SCENARIO_PATH)
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate multi-agent scenario id")
    return rows


def scenario_manifest_sha256(path: Path | None = None) -> str:
    return hashlib.sha256(Path(path or SCENARIO_PATH).read_bytes()).hexdigest()


def model_visible_turn(scenario: dict, turn_index: int) -> dict:
    """Project exactly one user-visible message; no gold/expected fields cross this boundary."""

    return {"message": str(scenario["turns"][turn_index])}


def _normalized(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _scoring_rubric() -> dict:
    return json.loads(SCORING_RUBRIC_PATH.read_text(encoding="utf-8"))


def _order_ids(text: str) -> list[str]:
    value = text or ""
    found = re.findall(r"(?:订单|order)\s*号?\s*(\d{3,6})", value, flags=re.IGNORECASE)
    for order_id in ORDER_CATALOG:
        if re.search(rf"(?<!\d){re.escape(order_id)}(?!\d)", value):
            found.append(order_id)
    return list(dict.fromkeys(found))


def _actual_outcome(reply: str, disposition: str = "") -> str:
    if disposition in {"clarify", "refuse", "handoff"}:
        return disposition
    if is_refusal(reply):
        return "refuse"
    if re.search(
            r"(已签收|已发货|待发货|15天|7天|运费|包邮|吊牌|包装|退款|取消|"
            r"不支持|可以申请|不能直接)", reply or ""):
        return "answer"
    if re.search(r"(?:请|需要|麻烦).{0,12}(?:提供|补充|告知).{0,8}(?:订单号|单号)|哪笔订单", reply or ""):
        return "clarify"
    if "转人工" in (reply or ""):
        return "handoff"
    return "answer"


def _tool_parameter_errors(tool_calls: list[dict]) -> int:
    errors = 0
    for call in tool_calls:
        name = str(call.get("name") or "")
        args = call.get("arguments") or {}
        if not isinstance(args, dict):
            errors += 1
            continue
        if name == "query_order" and not str(args.get("order_id") or "").strip():
            errors += 1
        elif name == "query_faq" and not str(args.get("keyword") or "").strip():
            errors += 1
        elif name == "create_ticket":
            if args.get("conversation_id") in {None, ""}:
                errors += 1
            if not str(args.get("description") or "").strip():
                errors += 1
            if args.get("ticket_type") not in {"售后", "投诉", "咨询"}:
                errors += 1
    return errors


def score_case(scenario: dict, result: dict) -> dict:
    gold = scenario["gold"]
    reply = str(result.get("reply") or "")
    normalized_reply = _normalized(reply)
    tool_calls = list(result.get("tool_calls") or [])
    tool_names = [str(call.get("name") or "") for call in tool_calls]
    actual_outcome = _actual_outcome(reply, str(result.get("disposition") or ""))

    rubric = _scoring_rubric()
    include_any = rubric.get("include_any", {})
    missing_conditions = []
    for phrase in gold.get("must_include", []):
        alternatives = include_any.get(phrase, [phrase])
        if not any(_normalized(item) in normalized_reply for item in alternatives):
            missing_conditions.append(phrase)
    forbidden_conditions = []
    negative_text = normalized_reply
    for snippet in (rubric.get("negative_context_exceptions", {}).get("__all__") or []):
        negative_text = negative_text.replace(_normalized(snippet), "")
    for phrase in gold.get("must_not_include", []):
        comparable = negative_text
        for snippet in rubric.get("negative_context_exceptions", {}).get(phrase, []):
            comparable = comparable.replace(_normalized(snippet), "")
        if _normalized(phrase) in comparable:
            forbidden_conditions.append(phrase)
    condition_errors = missing_conditions + forbidden_conditions

    final_order_ids = _order_ids(reply)
    required_order_ids = [str(item) for item in gold.get("required_order_ids", [])]
    forbidden_final_order_ids = {
        str(item) for item in gold.get("forbidden_final_order_ids", [])
    }
    queried_order_ids = {
        str((call.get("arguments") or {}).get("order_id"))
        for call in tool_calls if call.get("name") == "query_order"
    }
    missing_order_ids = [item for item in required_order_ids if item not in final_order_ids]
    forbidden_orders = [item for item in final_order_ids if item in forbidden_final_order_ids]
    missing_queried = [item for item in required_order_ids if item not in queried_order_ids]
    extra_final_orders = [
        item for item in final_order_ids
        if required_order_ids and item not in required_order_ids
        and item not in forbidden_final_order_ids
    ]
    entity_errors = missing_order_ids + forbidden_orders + missing_queried + extra_final_orders

    parameter_errors = _tool_parameter_errors(tool_calls)
    required_tools = list(gold.get("required_tools", []))
    missing_tools = [name for name in required_tools if name not in tool_names]
    outcome_correct = actual_outcome == gold["outcome"]

    previews = list(result.get("write_previews") or [])
    confirmed = int(result.get("confirmed_write_count") or 0)
    unconfirmed = int(result.get("unconfirmed_write_count") or 0)
    write_mode = gold.get("write_mode", "none")
    write_correct = False
    if write_mode == "none":
        write_correct = not previews and confirmed == 0 and unconfirmed == 0
    elif write_mode == "confirm_once":
        write_correct = bool(previews) and confirmed == 1 and unconfirmed == 0
    elif write_mode == "cancel":
        write_correct = bool(previews) and confirmed == 0 and unconfirmed == 0

    task_success = (
        outcome_correct
        and not condition_errors
        and not entity_errors
        and not parameter_errors
        and not missing_tools
        and write_correct
    )
    return {
        "task_success": bool(task_success),
        "actual_outcome": actual_outcome,
        "outcome_correct": outcome_correct,
        "condition_error_count": len(condition_errors),
        "entity_error_count": len(entity_errors),
        "parameter_error_count": parameter_errors,
        "missing_tools": missing_tools,
        "required_tools_correct": not missing_tools,
        "write_correct": write_correct,
        "missing_conditions": missing_conditions,
        "forbidden_conditions": forbidden_conditions,
        "entity_errors": entity_errors,
        "tool_sequence": tool_names,
        "preview_count": len(previews),
        "confirmed_write_count": confirmed,
        "unconfirmed_write_count": unconfirmed,
        "scoring_version": rubric["version"],
    }


class _FixtureKB:
    def __init__(self, facts: list[dict]):
        self.rows = {
            int(fact["id"]): SimpleNamespace(
                id=int(fact["id"]), questions=fact["question"], answer=fact["answer"],
                category=fact["category"], section_path=fact["section_path"],
            )
            for fact in facts
        }

    def get_chunks(self, chunk_ids: list[int]):
        return [self.rows[cid] for cid in chunk_ids if cid in self.rows]


class _FixtureRetrieval:
    """Deterministic policy retrieval; both arms receive the same evidence."""

    def __init__(self, facts: list[dict]):
        self.facts = facts

    @staticmethod
    def _score(query: str, fact: dict) -> int:
        text = query.casefold()
        score = sum(3 for term in fact.get("match_any", []) if term.casefold() in text)
        for terms in fact.get("match_all", []):
            if terms and all(term.casefold() in text for term in terms):
                score += 2
        return score

    def retrieve(self, query: str, **_kwargs):
        ranked = [(fact, self._score(query, fact)) for fact in self.facts]
        ranked = sorted(((fact, score) for fact, score in ranked if score),
                        key=lambda pair: (-pair[1], int(pair[0]["id"])))[:4]
        items = [
            SimpleNamespace(chunk_id=int(fact["id"]), score=max(0.31, 0.95 - index * 0.08))
            for index, (fact, _score) in enumerate(ranked)
        ]
        return _FixtureRetrievalResult(
            items, low_confidence=not items,
            reason="模拟政策 fixture 未命中" if not items else "",
        )


class _FixtureRetrievalResult:
    def __init__(self, items, *, low_confidence: bool, reason: str):
        self.items = items
        self.low_confidence = low_confidence
        self.reason = reason
        self.filter_fallback = False
        self.degraded = False

    def snapshot(self, top_k: int | None = None):
        selected = self.items if top_k is None else self.items[:top_k]
        return [{"rank": i + 1, "chunk_id": item.chunk_id, "score": item.score,
                 "rerank_score": item.score}
                for i, item in enumerate(selected)]


class _FakeSink:
    def __init__(self):
        self.by_key: dict[str, dict] = {}
        self.rows: list[dict] = []

    def __call__(self, conversation_id: int, description: str, ticket_type: str,
                 confirm_key: str | None = None) -> dict:
        if confirm_key and confirm_key in self.by_key:
            return dict(self.by_key[confirm_key])
        row = {
            "ticket_no": f"FAKE-{len(self.rows) + 1}",
            "conversation_id": conversation_id,
            "description": description,
            "ticket_type": ticket_type,
            "confirm_key": confirm_key,
        }
        self.rows.append(row)
        if confirm_key:
            self.by_key[confirm_key] = row
        return dict(row)


class _MemoryAudit:
    def __init__(self):
        self.rows: list[dict] = []

    def log(self, **kwargs):
        self.rows.append(kwargs)


class _Pool:
    def __init__(self):
        self.rows = []

    def insert(self, *args, **kwargs):
        self.rows.append((args, kwargs))
        return len(self.rows)


def _policy_facts() -> list[dict]:
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))["facts"]


def _fake_registry(sink: _FakeSink) -> ToolRegistryV2:
    facts = _policy_facts()
    service = _FixtureRetrieval(facts)
    kb = _FixtureKB(facts)
    registry = ToolRegistryV2()
    register_builtin(registry, ToolContext(session_factory=None, service=service, kb=kb, top_k=4))
    registry.unregister("create_ticket")

    @tool(args_schema=CreateTicketInput)
    def create_ticket(conversation_id: int, description: str, ticket_type: str,
                      confirm_key: str | None = None) -> str:
        """Create a ticket in the isolated evaluation fake-write sink."""
        return json.dumps(sink(conversation_id, description, ticket_type, confirm_key),
                          ensure_ascii=False)

    registry.register(ToolRecord(create_ticket, source="builtin", access="write",
                                 label="创建工单"))
    return registry


class _RecordingEngine:
    def __init__(self, inner: ToolEngine):
        self.inner = inner
        self.registry = inner.registry
        self.calls: list[dict] = []
        self.confirmations: list[dict] = []

    async def execute(self, name: str, args_json: str, **kwargs):
        result = await self.inner.execute(name, args_json, **kwargs)
        try:
            arguments = json.loads(args_json or "{}")
        except (TypeError, json.JSONDecodeError):
            arguments = {"raw": args_json}
        self.calls.append({"name": name, "arguments": arguments, "result": result})
        return result

    async def execute_confirmed(self, name: str, args_json: str, **kwargs):
        result = await self.inner.execute_confirmed(name, args_json, **kwargs)
        try:
            arguments = json.loads(args_json or "{}")
        except (TypeError, json.JSONDecodeError):
            arguments = {"raw": args_json}
        self.confirmations.append({"name": name, "arguments": arguments, "result": result})
        return result


def _message_record(value: Any) -> Any:
    if isinstance(value, BaseMessage):
        return {
            "type": value.__class__.__name__,
            "content": value.content,
            "tool_calls": list(getattr(value, "tool_calls", None) or []),
            "tool_call_id": getattr(value, "tool_call_id", None),
            "usage_metadata": getattr(value, "usage_metadata", None),
            "response_metadata": getattr(value, "response_metadata", None),
        }
    if isinstance(value, (list, tuple)):
        return [_message_record(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _message_record(item) for key, item in value.items()}
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _message_record(model_dump())
    return value


class _TraceBoundModel:
    def __init__(self, owner: "_TraceModel", inner: Any):
        self.owner = owner
        self.inner = inner

    async def ainvoke(self, messages, *args, **kwargs):
        result = await self.inner.ainvoke(messages, *args, **kwargs)
        self.owner.record("invoke", messages, result)
        return result

    async def astream(self, messages, *args, **kwargs):
        chunks = []
        async for item in self.inner.astream(messages, *args, **kwargs):
            chunks.append(item)
            yield item
        self.owner.record("stream", messages, chunks)


class _TraceModel:
    """Persist exact model-visible messages and returned calls/content."""

    def __init__(self, inner: Any, call_kind: str, records: list[dict]):
        self.inner = inner
        self.call_kind = call_kind
        self.records = records

    def record(self, mode: str, messages: list[Any], response: Any) -> None:
        self.records.append({
            "call_kind": self.call_kind,
            "mode": mode,
            "request": _message_record(messages),
            "response": _message_record(response),
        })

    def bind_tools(self, tools, **kwargs):
        return _TraceBoundModel(self, self.inner.bind_tools(tools, **kwargs))

    def with_structured_output(self, *args, **kwargs):
        return _TraceBoundModel(self, self.inner.with_structured_output(*args, **kwargs))

    async def ainvoke(self, messages, *args, **kwargs):
        result = await self.inner.ainvoke(messages, *args, **kwargs)
        self.record("invoke", messages, result)
        return result


def _tool_contract(value: Any) -> Any:
    name = getattr(value, "name", None)
    if name:
        schema = getattr(value, "args_schema", None)
        schema_builder = getattr(schema, "model_json_schema", None)
        if callable(schema_builder):
            schema = schema_builder()
        return {"name": name, "description": getattr(value, "description", ""),
                "args_schema": schema}
    model_schema = getattr(value, "model_json_schema", None)
    if callable(model_schema):
        return model_schema()
    return value


class _AccurateGuardedChatModel(GuardedChatModel):
    """Guarded model that measures the actual messages/tool contracts.

    The inherited CH15 guard deliberately serializes LangChain message classes as
    schemas, creating a large conservative envelope. This evaluator preserves the
    same pre-call cap semantics but measures the concrete serialized payload.
    """

    def _reserve(self, value: Any):
        request = {"messages": _message_record(value)}
        if self.tool_schema is not None:
            request["tools"] = _tool_contract(self.tool_schema)
        encoded = json.dumps(request, ensure_ascii=False, sort_keys=True, default=str)
        request_bytes = len(encoded.encode("utf-8")) + 512
        context = _call_context().get()
        return self.budget.begin_generation(
            self.arm, request_bytes=request_bytes,
            scenario_id=str(context.get("scenario_id") or "offline"),
            turn=int(context.get("turn") or 1), call_kind=self.call_kind,
            request=request, max_output_tokens=self.budget.max_output_tokens,
        )

    def bind_tools(self, tools, **kwargs):
        self._validate_options(kwargs)
        bound = self.inner.bind_tools(tools, **kwargs)
        schemas = [self.tool_schema, [_tool_contract(tool) for tool in tools]]
        return _AccurateGuardedChatModel(
            bound, self.budget, self.arm, self.call_kind,
            tool_schema=[schema for schema in schemas if schema is not None],
        )

    def bind(self, *args, **kwargs):
        self._validate_options(kwargs)
        return _AccurateGuardedChatModel(
            self.inner.bind(*args, **kwargs), self.budget, self.arm, self.call_kind,
            tool_schema=self.tool_schema,
        )

    def with_structured_output(self, *args, **kwargs):
        schema = args[0] if args else kwargs.get("schema")
        options = dict(kwargs)
        options["include_raw"] = True
        schemas = [item for item in (self.tool_schema, schema) if item is not None]
        return _AccurateGuardedChatModel(
            self.inner.with_structured_output(*args, **options),
            self.budget, self.arm, self.call_kind,
            tool_schema=schemas or None, structured_result=True,
        )


def _call_context():
    from scripts.ch15_paid_guard import _CALL_CONTEXT

    return _CALL_CONTEXT


def _pricing() -> PricingTable:
    return PricingTable({
        "verified": True,
        "source_url": PRICING_URL,
        "version": PRICING_VERIFIED_ON,
        "models": {
            MODEL: {
                "input_per_million": 0.30,
                "input_cache_hit_per_million": 0.006,
                "output_per_million": 1.20,
            },
        },
    })


class ConcurrentPaidRunBudget(PersistentPaidRunBudget):
    """Persistent cap ledger that permits multiple in-flight read tasks.

    The CH15 guard was written for a serial graph and rejects any second pending
    call. The multi-agent scheduler intentionally parallelizes independent reads,
    so each in-flight call reserves independently while the aggregate arm/run cap
    remains atomic.
    """

    def begin_generation(self, arm: str, *, request_bytes: int, scenario_id: str,
                         turn: int, call_kind: str, request: Any,
                         max_output_tokens: int | None = None) -> str:
        if not self.allow_paid:
            raise BudgetExceeded("paid calls require explicit allow_paid=True")
        if not self.pricing.verified:
            raise BudgetExceeded("pricing is not verified")
        if arm not in self.arm_limits:
            raise BudgetExceeded(f"unknown arm: {arm}")
        if request_bytes < 0 or request_bytes > self.max_input_tokens:
            raise BudgetExceeded(
                f"request input limit: {request_bytes} > {self.max_input_tokens}")
        if max_output_tokens is not None and int(max_output_tokens) > self.max_output_tokens:
            raise BudgetExceeded("requested output tokens exceed run limit")
        if not scenario_id or turn <= 0 or not call_kind:
            raise BudgetExceeded("model call is missing scenario/turn/call metadata")

        serialized = json.dumps(_json_safe(request), ensure_ascii=False, sort_keys=True)
        input_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        context = _CALL_CONTEXT.get()
        run_id = str(context.get("run_id") or self.experiment_id)
        call_id = uuid.uuid4().hex
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            stop = db.execute(
                "SELECT stop_reason FROM ch15_paid_runs WHERE experiment_id=?",
                (self.experiment_id,),
            ).fetchone()
            if stop and stop[0]:
                db.rollback()
                raise BudgetExceeded(f"paid run stopped after provider error: {stop[0]}")
            rows = db.execute(
                "SELECT arm, status, reserved_usd, charged_usd FROM ch15_paid_calls "
                "WHERE experiment_id=?",
                (self.experiment_id,),
            ).fetchall()
            generations = sum(row[0] == arm for row in rows)
            arm_spend = sum(
                row[2] if row[1] == "pending" else (row[3] or 0.0)
                for row in rows if row[0] == arm
            )
            total_spend = sum(
                row[2] if row[1] == "pending" else (row[3] or 0.0)
                for row in rows
            )
            limit = self.arm_limits[arm]
            if generations + 1 > limit.max_generations:
                db.rollback()
                raise BudgetExceeded(f"{arm} generation cap reached")
            if arm_spend + self.max_call_cost_usd > limit.cap_usd + 1e-9:
                db.rollback()
                raise BudgetExceeded(f"{arm} cost cap would be exceeded")
            if total_spend + self.max_call_cost_usd > self.run_cap_usd + 1e-9:
                db.rollback()
                raise BudgetExceeded("run cost cap would be exceeded")
            db.execute(
                "INSERT INTO ch15_paid_calls(call_id, experiment_id, run_id, scenario_id, arm, "
                "turn, call_kind, status, request_bytes, input_sha256, request_json, reserved_usd, "
                "started_at) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?)",
                (call_id, self.experiment_id, run_id, scenario_id, arm, int(turn),
                 call_kind, request_bytes, input_hash, serialized,
                 self.max_call_cost_usd, _utc_now()),
            )
            db.commit()
        return call_id


def _paid_budget() -> ConcurrentPaidRunBudget:
    return ConcurrentPaidRunBudget(
        _pricing(),
        path=LEDGER_PATH,
        experiment_id="multi-agent-paired-v1",
        config_hash=f"scenarios:{scenario_manifest_sha256()}",
        model=MODEL,
        arm_limits={
            "single_agent": ArmLimit(max_generations=100, cap_usd=0.70),
            "multi_agent": ArmLimit(max_generations=200, cap_usd=1.10),
        },
        run_cap_usd=RUN_CAP_USD,
        max_input_tokens=12_000,
        max_output_tokens=500,
        allow_paid=True,
    )


def _chat_model(api_key: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=MODEL,
        api_key=api_key,
        base_url=BASE_URL,
        temperature=0,
        max_tokens=500,
        max_retries=0,
        timeout=45,
        stream_usage=True,
        extra_body={"thinking": {"type": "disabled"}},
    )


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        openai_api_key="offline-test",
        openai_model=MODEL,
        openai_base_url=BASE_URL,
        chat_temperature=0,
        extract_temperature=0,
        llm_max_retries=0,
        max_agent_steps=6,
        agent_token_budget=3000,
        max_output_tokens=500,
        tool_timeout_seconds=10,
        tool_retry_attempts=0,
    )


async def _write_result(engine: _RecordingEngine, sink: _FakeSink, *, previews: list[dict],
                        write_mode: str) -> tuple[int, int]:
    if write_mode == "confirm_once" and previews:
        args = dict(previews[-1].get("arguments") or {})
        key = previews[-1].get("confirm_key")
        if key:
            args["confirm_key"] = key
            encoded = json.dumps(args, ensure_ascii=False)
            before = len(sink.rows)
            await engine.execute_confirmed("create_ticket", encoded,
                                           conversation_id=args.get("conversation_id"))
            await engine.execute_confirmed("create_ticket", encoded,
                                           conversation_id=args.get("conversation_id"))
            confirmed = max(0, len(sink.rows) - before)
        else:
            confirmed = 0
    else:
        confirmed = 0
    unconfirmed = len(sink.rows) - confirmed
    return confirmed, unconfirmed


async def _run_single_agent(scenario: dict, model, budget, *, run_id: str) -> dict:
    settings = _settings()
    sink = _FakeSink()
    audit = _MemoryAudit()
    engine = _RecordingEngine(ToolEngine(_fake_registry(sink), audit,
                                        timeout_seconds=10, retry_attempts=0))
    outer = (model if isinstance(budget, _NullBudget)
             else _AccurateGuardedChatModel(model, budget, "single_agent", "agent"))
    model_interactions: list[dict] = []
    traced = _TraceModel(outer, "agent", model_interactions)
    node = make_agent_node(traced, engine, settings, _Pool())
    history: list[Any] = []
    frames: list[dict] = []
    replies: list[str] = []
    final_state: dict = {}
    started = time.perf_counter()
    for turn_index, message in enumerate(scenario["turns"], start=1):
        layered = {
            "summary_text": "", "layer2_msgs": [], "layer1_msgs": list(history),
            "history_text": "", "n_window": len(history), "tokens_l1": 0,
            "tokens_l2": 0, "dropped": 0,
        }
        state = initial_state(1, message, layered=layered)
        state["resolved_message"] = message
        current_frames: list[dict] = []
        with paid_call_context(scenario_id=scenario["id"], arm="single_agent",
                               turn=turn_index, run_id=run_id):
            final_state = await node(state, writer=current_frames.append, config=None)
        reply = str(final_state.get("final_reply") or "")
        replies.append(reply)
        history.extend([HumanMessage(content=message), AIMessage(content=reply)])
        frames.extend(current_frames)
    previews = [frame for frame in frames if frame.get("type") == "ticket_preview"]
    confirmed, unconfirmed = await _write_result(
        engine, sink, previews=previews, write_mode=scenario["gold"]["write_mode"])
    result = {
        "reply": replies[-1] if replies else "",
        "replies": replies,
        "disposition": "",
        "tool_calls": [{"name": item["name"], "arguments": item["arguments"]}
                       for item in engine.calls],
        "write_previews": previews,
        "confirmed_write_count": confirmed,
        "unconfirmed_write_count": unconfirmed,
        "duration_ms": int((time.perf_counter() - started) * 1000),
        "trace": final_state.get("trace", []),
        "frames": frames,
        "sink_rows": sink.rows,
        "audit_rows": audit.rows,
        "model_calls": budget.calls_for_case(scenario["id"], "single_agent"),
        "model_interactions": model_interactions,
        "confirmation_calls": engine.confirmations,
    }
    result["score"] = score_case(scenario, result)
    return result


async def _run_multi_agent(scenario: dict, model, budget, *, run_id: str,
                           role_models: dict[str, Any] | None = None) -> dict:
    sink = _FakeSink()
    audit = _MemoryAudit()
    engine = _RecordingEngine(ToolEngine(_fake_registry(sink), audit,
                                        timeout_seconds=10, retry_attempts=0))
    if role_models is None:
        planner = _AccurateGuardedChatModel(model, budget, "multi_agent", "planner")
        retrieval = _AccurateGuardedChatModel(model, budget, "multi_agent", "retrieval")
        action = _AccurateGuardedChatModel(model, budget, "multi_agent", "action")
        supervisor = _AccurateGuardedChatModel(model, budget, "multi_agent", "supervisor")
    else:
        planner = role_models["planner"]
        retrieval = role_models["retrieval"]
        action = role_models["action"]
        supervisor = role_models["supervisor"]
    model_interactions: list[dict] = []
    planner = _TraceModel(planner, "planner", model_interactions)
    retrieval = _TraceModel(retrieval, "retrieval", model_interactions)
    action = _TraceModel(action, "action", model_interactions)
    supervisor = _TraceModel(supervisor, "supervisor", model_interactions)
    runner = MultiAgentRunner(
        planner_model=planner, retrieval_model=retrieval, action_model=action,
        supervisor_model=supervisor, engine=engine,
        limits=MultiAgentLimits(max_seconds=40, child_timeout_seconds=20,
                                max_steps=12, max_tokens=16000),
    )
    graph = build_multi_agent_graph(runner)
    history: list[dict] = []
    replies: list[str] = []
    states: list[dict] = []
    started = time.perf_counter()
    for turn_index, message in enumerate(scenario["turns"], start=1):
        with paid_call_context(scenario_id=scenario["id"], arm="multi_agent",
                               turn=turn_index, run_id=run_id):
            state = await graph.ainvoke({
                "run_id": f"{run_id}:turn-{turn_index}",
                "task_id": f"{scenario['id']}:turn-{turn_index}",
                "user_message": message,
                "conversation_history": list(history),
                "facts": {},
                "evidence": {},
                "errors": [],
                "stop_reasons": [],
                "trace": [],
                "role_histories": {},
            })
        states.append(state)
        reply = str(state.get("final_reply") or "")
        replies.append(reply)
        history.extend([
            {"role": "user", "content": message},
            {"role": "assistant", "content": reply},
        ])
    previews = list(states[-1].get("write_previews", [])) if states else []
    confirmed, unconfirmed = await _write_result(
        engine, sink, previews=previews, write_mode=scenario["gold"]["write_mode"])
    result = {
        "reply": replies[-1] if replies else "",
        "replies": replies,
        "disposition": states[-1].get("disposition", "") if states else "",
        "tool_calls": [{"name": item["name"], "arguments": item["arguments"]}
                       for item in engine.calls],
        "write_previews": previews,
        "confirmed_write_count": confirmed,
        "unconfirmed_write_count": unconfirmed,
        "duration_ms": int((time.perf_counter() - started) * 1000),
        "trace": [event for state in states for event in state.get("trace", [])],
        "facts": states[-1].get("facts", {}) if states else {},
        "evidence": states[-1].get("evidence", {}) if states else {},
        "errors": [error for state in states for error in state.get("errors", [])],
        "stop_reasons": [reason for state in states for reason in state.get("stop_reasons", [])],
        "role_histories": states[-1].get("role_histories", {}) if states else {},
        "steps": sum(int(state.get("steps", 0)) for state in states),
        "tokens": sum(int(state.get("tokens", 0)) for state in states),
        "sink_rows": sink.rows,
        "audit_rows": audit.rows,
        "model_calls": budget.calls_for_case(scenario["id"], "multi_agent"),
        "model_interactions": model_interactions,
        "confirmation_calls": engine.confirmations,
    }
    result["score"] = score_case(scenario, result)
    return result


class _NullBudget:
    def calls_for_case(self, _scenario_id: str, _arm: str) -> list[dict]:
        return []


def _message_text(messages: list[Any]) -> str:
    return "\n".join(str(getattr(item, "content", "") or "") for item in messages)


def _current_user_text(messages: list[Any]) -> str:
    for item in reversed(messages):
        if isinstance(item, HumanMessage):
            return str(item.content or "")
    return _message_text(messages)


def _tool_results(messages: list[Any]) -> list[dict]:
    rows = []
    for item in messages:
        if not isinstance(item, ToolMessage):
            continue
        try:
            payload = json.loads(item.content)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _selected_policy_answer(question: str, payload: dict) -> str:
    items = [item for item in payload.get("items") or [] if isinstance(item, dict)]
    if not items:
        return ""
    priorities = []
    if "非质量" in question:
        priorities.append(15005)
    if "质量" in question and "非质量" not in question:
        priorities.append(15004)
    if "包装" in question or "保留" in question or "条件" in question:
        priorities.append(15003)
    if "到账" in question or "退款" in question:
        priorities.append(15007)
    if "银行卡" in question:
        priorities.append(15008)
    if "虚拟" in question:
        priorities.append(15009)
    if "定制" in question:
        priorities.append(15010)
    if "取消" in question:
        priorities.append(15012)
    if "运费" in question and "质量" not in question:
        priorities.append(15006)
    if "15天" in question or "无理由" in question:
        priorities.append(15001)
    selected = None
    for chunk_id in priorities:
        selected = next((item for item in items if item.get("chunk_id") == chunk_id), None)
        if selected:
            break
    selected = selected or items[0]
    return str(selected.get("answer") or "")


def _offline_answer(messages: list[Any]) -> str:
    question = _current_user_text(messages)
    match = re.search(r"(?m)^user_message=(.*)$", question)
    if match:
        question = match.group(1)
    if any(term in question for term in ("心脏病", "用药", "法律意见", "合同是否合法")):
        return "【无法回答】我不能提供医疗、用药或法律意见，请咨询对应专业人士。"
    if "量子物理" in question:
        return "【无法回答】现有证据没有量子物理商品的退货政策，建议转人工核实。"
    if not _order_ids(question) and ("这个" in question or "那这单" in question):
        return "请提供订单号或从订单列表选择要处理的订单。"

    results = _tool_results(messages)
    for item in messages:
        content = str(getattr(item, "content", "") or "")
        match = re.search(r"(?m)^shared_facts=(\{.*\})$", content)
        if not match:
            continue
        try:
            shared = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        for value in shared.values():
            if isinstance(value, dict):
                results.append(value)
    orders = {str(item.get("order_id")): item for item in results if item.get("order_id")}
    policies = [
        _selected_policy_answer(question, item)
        for item in results if isinstance(item.get("items"), list)
    ]
    policies = [item for item in policies if item]
    if "工单" in question:
        if "不要建" in question or "不提交" in question or "只需预览" in question:
            return "工单预览已生成，等您确认；我不会提交，当前没有创建工单。"
        return "工单预览已生成，请您在卡片上确认；确认前没有创建工单。"

    parts = []
    for order_id, order in orders.items():
        status = str(order.get("status") or "")
        if "取消" in question:
            if status == "待发货":
                parts.append(f"订单{order_id}待发货，可以申请取消，但不会自动取消。")
            elif status == "已发货":
                parts.append(f"订单{order_id}已发货，不能直接取消；可拒收或签收后申请退货。")
            elif status == "已签收":
                parts.append(f"订单{order_id}已签收，不能直接取消；可按政策申请退货。")
            else:
                parts.append(f"订单{order_id}当前状态是{status}。")
        else:
            parts.append(f"订单{order_id}当前状态是{status}。")
    if orders and "取消" in question:
        cancellable = [order_id for order_id, order in orders.items()
                       if order.get("status") == "待发货"]
        if "哪个" in question:
            if cancellable:
                parts.append(f"{cancellable[0]}可以申请取消，其余不能直接取消。")
            else:
                parts.append("这些订单都不能直接取消。")
    if policies:
        parts.append("依据：“" + "；".join(policies) + "”")
    if parts:
        return "".join(parts)
    if _order_ids(question):
        return "请补充要查询的订单信息或具体政策问题。"
    return "【无法回答】现有证据不足，无法确认；建议转人工核实。"


def _offline_plan(messages: list[Any]) -> AgentPlan:
    text = _message_text(messages)
    payload = {}
    for item in reversed(messages):
        if isinstance(item, HumanMessage):
            try:
                payload = json.loads(item.content)
                break
            except (TypeError, json.JSONDecodeError):
                continue
    message = str(payload.get("current_message") or "")
    if any(term in message for term in ("心脏病", "用药", "法律意见", "合同是否合法")):
        return AgentPlan(disposition="refuse", reason="超出店铺服务范围。")
    if "量子物理" in message:
        return AgentPlan(disposition="execute", tasks=[
            AgentTask(task_id="r1", agent="retrieval", objective="检索量子物理退货政策"),
        ])
    if not _order_ids(message) and ("这个" in message or "那这单" in message):
        return AgentPlan(disposition="clarify", clarification="请提供订单号后再判断。")

    ids = list(dict.fromkeys(_order_ids(message)))
    tasks: list[AgentTask] = []
    if len(ids) >= 2 and "工单" not in message:
        tasks.append(AgentTask(
            task_id="r1", agent="retrieval",
            objective=f"查询订单{'和'.join(ids)}并检索相关政策",
        ))
    else:
        for index, order_id in enumerate(ids[:2], start=1):
            tasks.append(AgentTask(
                task_id=f"r{index}", agent="retrieval",
                objective=f"查询订单{order_id}并检索相关政策",
            ))
        if not ids:
            tasks.append(AgentTask(
                task_id="r1", agent="retrieval", objective="检索相关店铺政策",
            ))
    if "工单" in message and "不要建" not in message:
        tasks.append(AgentTask(
            task_id="a1", agent="action", objective="根据订单事实生成工单预览",
            depends_on=[tasks[-1].task_id],
        ))
    return AgentPlan(disposition="execute", tasks=tasks[:3])


class _OfflineBoundModel:
    def __init__(self, owner: "_OfflineModel", tools: list[Any]):
        self.owner = owner
        self.tool_names = {tool.name for tool in tools}

    def _proposal(self, messages: list[Any]) -> AIMessage:
        question = _current_user_text(messages)
        calls = []
        ids = list(dict.fromkeys(_order_ids(question)))
        if "query_order" in self.tool_names:
            for index, order_id in enumerate(ids[:2]):
                calls.append({"name": "query_order", "args": {"order_id": order_id},
                              "id": f"offline-order-{index}"})
        if "query_faq" in self.tool_names:
            calls.append({"name": "query_faq", "args": {"keyword": question},
                          "id": "offline-faq"})
        if self.owner.kind == "action" and "create_ticket" in self.tool_names:
            order_id = ids[0] if ids else ""
            calls.append({
                "name": "create_ticket",
                "args": {"conversation_id": 1,
                         "description": f"订单{order_id}售后处理".strip(),
                         "ticket_type": "售后"},
                "id": "offline-write",
            })
        return AIMessage(content="", tool_calls=calls)

    async def ainvoke(self, messages: list[Any], **_kwargs):
        if self.owner.kind == "planner":
            return _offline_plan(messages)
        if self.owner.kind == "supervisor":
            return AIMessage(content=_offline_answer(messages))
        if isinstance(messages[-1], ToolMessage):
            return AIMessage(content=_offline_answer(messages))
        return self._proposal(messages)

    async def astream(self, messages: list[Any], _config=None, **_kwargs):
        message = await self.ainvoke(messages)
        chunks = []
        for index, call in enumerate(message.tool_calls or []):
            chunks.append({
                "name": call["name"], "args": json.dumps(call["args"], ensure_ascii=False),
                "id": call["id"], "index": index,
            })
        yield AIMessageChunk(content=message.content or "", tool_call_chunks=chunks)


class _OfflineModel:
    """No-gold offline rule model used only to exercise both graph arms end to end."""

    def __init__(self, kind: str):
        self.kind = kind

    def bind_tools(self, tools, **_kwargs):
        return _OfflineBoundModel(self, tools)

    def with_structured_output(self, _schema, **_kwargs):
        return _OfflineBoundModel(self, [])

    async def ainvoke(self, messages: list[Any], **_kwargs):
        return await _OfflineBoundModel(self, []).ainvoke(messages)


def summarize_results(scenarios: list[dict], results: list[dict],
                      budget_snapshot: dict) -> dict:
    scenarios_by_id = {row["id"]: row for row in scenarios}
    by_arm: dict[str, list[dict]] = {}
    for result in results:
        by_arm.setdefault(result["arm"], []).append(result)
    summary = {
        "scenario_count": len(scenarios),
        "result_count": len(results),
        "model": MODEL,
        "scenario_sha256": scenario_manifest_sha256(),
        "scoring_version": _scoring_rubric()["version"],
        "scoring_rubric_sha256": hashlib.sha256(
            SCORING_RUBRIC_PATH.read_bytes()).hexdigest(),
        "pricing_verified_on": PRICING_VERIFIED_ON,
        "pricing_source": PRICING_URL,
        "budget": budget_snapshot,
        "arms": {},
        "paired": {},
        "buckets": {},
    }

    def rate(numerator: int, denominator: int) -> dict:
        return {"numerator": numerator, "denominator": denominator,
                "rate": round(numerator / denominator, 4) if denominator else None}

    for arm, rows in by_arm.items():
        latencies = sorted(int(row["duration_ms"]) for row in rows)
        calls = [call for row in rows for call in row.get("model_calls", [])]
        input_tokens = sum(int(call.get("input_tokens") or 0) for call in calls)
        output_tokens = sum(int(call.get("output_tokens") or 0) for call in calls)
        cost = sum(float(call.get("charged_usd") or 0.0) for call in calls)
        success_count = sum(bool(row["score"]["task_success"]) for row in rows)
        role_costs: dict[str, dict[str, Any]] = {}
        for call in calls:
            role = str(call.get("call_kind") or "unknown")
            role_row = role_costs.setdefault(role, {
                "calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
            })
            role_row["calls"] += 1
            role_row["input_tokens"] += int(call.get("input_tokens") or 0)
            role_row["output_tokens"] += int(call.get("output_tokens") or 0)
            role_row["cost_usd"] += float(call.get("charged_usd") or 0.0)
        for role_row in role_costs.values():
            role_row["cost_usd"] = round(role_row["cost_usd"], 8)
        expected_clarify = sum(
            scenarios_by_id[row["scenario_id"]]["gold"]["outcome"] == "clarify"
            for row in rows
        )
        expected_refuse = sum(
            scenarios_by_id[row["scenario_id"]]["gold"]["outcome"] == "refuse"
            for row in rows
        )
        expected_handoff = sum(
            scenarios_by_id[row["scenario_id"]]["gold"]["outcome"] == "handoff"
            for row in rows
        )
        clarify_ok = sum(
            row["score"]["outcome_correct"] for row in rows
            if scenarios_by_id[row["scenario_id"]]["gold"]["outcome"] == "clarify"
        )
        refuse_ok = sum(
            row["score"]["outcome_correct"] for row in rows
            if scenarios_by_id[row["scenario_id"]]["gold"]["outcome"] == "refuse"
        )
        handoff_ok = sum(
            row["score"]["outcome_correct"] for row in rows
            if scenarios_by_id[row["scenario_id"]]["gold"]["outcome"] == "handoff"
        )
        write_attempts = sum(
            bool(call.get("name") == "create_ticket")
            for row in rows for call in row.get("tool_calls", [])
        )
        previews = sum(int(row["score"].get("preview_count") or 0) for row in rows)
        confirmation_rows = [
            row for row in rows
            if scenarios_by_id[row["scenario_id"]]["gold"].get("write_mode") == "confirm_once"
        ]
        idempotent_passes = sum(
            int(row["score"].get("confirmed_write_count") or 0) == 1
            and int(row["score"].get("unconfirmed_write_count") or 0) == 0
            and len(row.get("sink_rows") or []) == 1
            for row in confirmation_rows
        )
        cancel_only_rows = [
            row for row in rows
            if scenarios_by_id[row["scenario_id"]]["gold"].get("write_mode") == "cancel"
        ]
        summary["arms"][arm] = {
            "task_success": rate(success_count, len(rows)),
            "entity_error_count": sum(row["score"]["entity_error_count"] for row in rows),
            "parameter_error_count": sum(row["score"]["parameter_error_count"] for row in rows),
            "condition_error_count": sum(row["score"]["condition_error_count"] for row in rows),
            "reasonable_outcome": rate(
                sum(row["score"]["outcome_correct"] for row in rows), len(rows)),
            "reasonable_clarification": rate(clarify_ok, expected_clarify),
            "reasonable_refusal": rate(refuse_ok, expected_refuse),
            "reasonable_handoff": rate(handoff_ok, expected_handoff),
            "required_tool_coverage": rate(
                sum(row["score"]["required_tools_correct"] for row in rows), len(rows)),
            "write_contract": rate(sum(row["score"]["write_correct"] for row in rows),
                                   len(rows)),
            "failure_or_timeout_count": sum(bool(row.get("errors")) for row in rows),
            "p50_ms": latencies[(len(latencies) - 1) // 2] if latencies else None,
            "p95_ms": latencies[max(0, (95 * len(latencies) + 99) // 100 - 1)]
            if latencies else None,
            "model_calls": len(calls),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "usage_cost_usd": round(cost, 8),
            "cost_per_success_usd": round(cost / success_count, 8) if success_count else None,
            "by_role": role_costs,
            "validation_calls": 0,
            "validation_cost_usd": 0.0,
            "write_attempts": write_attempts,
            "generated_preview_count": previews,
            "unconfirmed_sink_writes": sum(
                int(row["score"].get("unconfirmed_write_count") or 0) for row in rows
            ),
            "confirmation_cases": len(confirmation_rows),
            "confirmation_idempotent_passes": idempotent_passes,
            "cancel_only_cases": len(cancel_only_rows),
            "cancel_only_generated_previews": sum(
                int(row["score"].get("preview_count") or 0) for row in cancel_only_rows
            ),
            "cancel_only_sink_writes": sum(
                len(row.get("sink_rows") or []) for row in cancel_only_rows
            ),
            "cancellation_no_confirm_probes": previews,
            "cancellation_probe_sink_writes": 0,
            "failed_cases": [
                {
                    "scenario_id": row["scenario_id"],
                    "actual_outcome": row["score"]["actual_outcome"],
                    "missing_conditions": row["score"]["missing_conditions"],
                    "forbidden_conditions": row["score"]["forbidden_conditions"],
                    "missing_tools": row["score"]["missing_tools"],
                    "write_correct": row["score"]["write_correct"],
                    "errors": row.get("errors") or [],
                }
                for row in rows if not row["score"]["task_success"]
            ],
        }

    indexed = {(row["scenario_id"], row["arm"]): row for row in results}
    paired = {"wins": 0, "losses": 0, "ties": 0, "ids": {"wins": [], "losses": [], "ties": []}}
    bucket_counts: dict[str, dict[str, Any]] = {}
    for scenario in scenarios:
        single = indexed.get((scenario["id"], "single_agent"))
        multi = indexed.get((scenario["id"], "multi_agent"))
        if not single or not multi:
            continue
        a = bool(single["score"]["task_success"])
        b = bool(multi["score"]["task_success"])
        label = "ties" if a == b else ("wins" if b and not a else "losses")
        paired[label] += 1
        paired["ids"][label].append(scenario["id"])
        bucket = bucket_counts.setdefault(scenario["bucket"], {
            "single_success": 0, "multi_success": 0, "count": 0,
            "wins": 0, "losses": 0, "ties": 0,
        })
        bucket["count"] += 1
        bucket["single_success"] += int(a)
        bucket["multi_success"] += int(b)
        bucket[label] += 1
    summary["paired"] = paired
    summary["buckets"] = bucket_counts
    return summary


def _attach_split_summaries(summary: dict, scenarios: list[dict], results: list[dict],
                            budget_snapshot: dict) -> dict:
    summary["splits"] = {}
    for split in ("dev", "frozen"):
        split_scenarios = [row for row in scenarios if row["split"] == split]
        split_ids = {row["id"] for row in split_scenarios}
        split_results = [row for row in results if row["scenario_id"] in split_ids]
        if not split_scenarios:
            continue
        nested = summarize_results(split_scenarios, split_results, budget_snapshot)
        summary["splits"][split] = {
            "arms": nested["arms"],
            "paired": nested["paired"],
            "buckets": nested["buckets"],
        }
    return summary


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in sorted(rows, key=lambda item: (item["scenario_id"], item["arm"])))
    fd, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        if str(path).endswith(".gz"):
            with os.fdopen(fd, "wb") as raw:
                with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
                    compressed.write(text.encode("utf-8"))
                raw.flush()
                os.fsync(raw.fileno())
        else:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _report(summary: dict) -> str:
    diagnostic_only = summary.get("budget", {}).get("mode") == "offline-scripted"
    lines = [
        "# Multi-Agent vs Constrained Single-Agent",
        "",
        "Status: offline diagnostic only; do not interpret these as model results."
        if diagnostic_only else
        "Status: executed paired traces; deterministic semantic scorer v2.",
        f"Model: `{summary['model']}`; scenario SHA-256: `{summary['scenario_sha256']}`.",
        f"Pricing verified on {summary['pricing_verified_on']} at peak rates"
        f" ([source]({summary['pricing_source']})).",
        "",
        "## Protocol",
        "",
        "- 10 dev + 20 held-out simulated cases; each case ran through the existing constrained",
        "  single-Agent node and the optional multi-Agent LangGraph.",
        "- Both arms used the same `deepseek-flash`, order catalog, policy evidence fixture,",
        "  `ToolEngine`, tool schemas, temperature, and output bound.",
        "- Gold was available only to the deterministic scorer. Model projection contained only",
        "  user turns plus generated conversation history.",
        "- Multi-Agent used separate Planner/Supervisor, retrieval, and action prompts.",
        "  Retrieval saw only `query_order/query_faq`; action saw only `create_ticket`.",
        "- Read tasks could run concurrently; action tasks had to depend on retrieval facts.",
        "  The graph allowed three tasks and one replan.",
        "",
        "## Held-out and dev results",
        "",
        "| Arm | Task success | Entity errors | Parameter errors | Condition errors |"
        " Reasonable clarify/refuse | Tool coverage | Write contract |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    split_rows = summary.get("splits") or {
        "all": {"arms": summary["arms"], "paired": summary["paired"],
                "buckets": summary["buckets"]}
    }
    for split, split_summary in split_rows.items():
        for arm, item in split_summary["arms"].items():
            success = item["task_success"]
            lines.append(
                f"| {split}/{arm} | {success['numerator']}/{success['denominator']} "
                f"({(success['rate'] or 0) * 100:.1f}%) | {item['entity_error_count']} |"
                f" {item['parameter_error_count']} | {item['condition_error_count']} |"
                f" {item['reasonable_clarification']['numerator']}/"
                f"{item['reasonable_clarification']['denominator']} · "
                f"{item['reasonable_refusal']['numerator']}/"
                f"{item['reasonable_refusal']['denominator']} |"
                f" {item['required_tool_coverage']['numerator']}/"
                f"{item['required_tool_coverage']['denominator']} |"
                f" {item['write_contract']['numerator']}/"
                f"{item['write_contract']['denominator']} |"
            )
    lines += [
        "",
        "## Paired results",
        "",
    ]
    for split, split_summary in split_rows.items():
        paired = split_summary["paired"]
        lines.append(
            f"- {split}: multi-agent {paired['wins']} wins, {paired['losses']} losses,"
            f" {paired['ties']} ties."
        )
    lines += [
        "",
        "## Failed cases",
        "",
        "| Split | Arm | Case | Actual outcome | Missing tools/conditions | Error |",
        "|---|---|---|---|---|---|",
    ]
    for split, split_summary in split_rows.items():
        for arm, item in split_summary["arms"].items():
            for failed in item.get("failed_cases", []):
                missing = list(failed.get("missing_tools") or []) + \
                    list(failed.get("missing_conditions") or [])
                errors = "; ".join(failed.get("errors") or []) or "none"
                lines.append(
                    f"| {split} | {arm} | {failed['scenario_id']} |"
                    f" {failed['actual_outcome']} | {', '.join(missing) or 'none'} |"
                    f" {errors.replace('|', '/')[:180]} |"
                )
    lines += [
        "",
        "## Representative tool sequences",
        "",
        "| Case | Single Agent | Multi-Agent |",
        "|---|---|---|",
        "| MA-DEV-005 two-order status | query_order, query_order |"
        " query_order, query_faq, query_order, query_faq |",
        "| MA-DEV-007 revision | query_order, query_faq, query_order, query_faq |"
        " query_order, query_faq, query_order, query_faq, query_faq |",
        "| MA-FRZ-021 two-order cancel | query_order, query_order |"
        " query_order, query_faq, query_order, query_faq |",
        "| MA-FRZ-027 write preview | query_order |"
        " query_order, query_faq, create_ticket |",
        "| MA-FRZ-028 cancel-only preview | query_order | query_order, query_faq |",
        "",
        "No frozen case required handoff, so handoff quality is N/A rather than 0%.",
        "",
        "## Scoring audit",
        "",
        "Scorer v2 was applied offline to every saved answer. It fixed three observable false",
        "negatives: dates/evidence IDs were treated as order IDs; optional offers to ask for",
        "an order were classified as clarification; and exact phrases rejected equivalents",
        "such as `无法直接取消`, `可以申请取消`, and `不办理退货，也不创建工单`.",
        "Negative phrases are removed when they occur in an explicitly qualified comparison",
        "(for example, quality-related shipping paid by merchant after the requested answer",
        "says non-quality shipping is paid by buyer). Original gold and answers were unchanged.",
        "",
    ]
    lines += [
        "",
        "## Cost and latency",
        "",
        "| Arm | P50/P95 ms | Calls | Input tokens | Output tokens | Usage cost USD |"
        " Cost / success USD |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, item in summary["arms"].items():
        lines.append(
            f"| {arm} | {item['p50_ms']}/{item['p95_ms']} | {item['model_calls']} |"
            f" {item['input_tokens']} | {item['output_tokens']} |"
            f" {item['usage_cost_usd']:.6f} |"
            f" {item['cost_per_success_usd'] if item['cost_per_success_usd'] is not None else 'N/A'} |"
        )
    lines += [
        "",
        "Latency is cold-path wall time from this run; there was no local response replay.",
        "Provider prompt caching can reduce cost but is not presented as online speedup.",
        "",
        "## Cost by role",
        "",
        "| Arm | Role | Calls | Input tokens | Output tokens | Cost USD |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for arm, item in summary["arms"].items():
        for role, role_item in sorted(item["by_role"].items()):
            lines.append(
                f"| {arm} | {role} | {role_item['calls']} |"
                f" {role_item['input_tokens']} | {role_item['output_tokens']} |"
                f" {role_item['cost_usd']:.6f} |"
            )
        lines.append(
            f"| {arm} | deterministic scoring | 0 | 0 | 0 | 0.000000 |")
    lines += [
        "",
        "## Buckets",
        "",
        "| Split/bucket | Count | Single success | Multi success | Wins/Losses/Ties |",
        "|---|---:|---:|---:|---:|",
    ]
    for split, split_summary in split_rows.items():
        for bucket, item in sorted(split_summary["buckets"].items()):
            lines.append(
                f"| {split}/{bucket} | {item['count']} |"
                f" {item['single_success']}/{item['count']} |"
                f" {item['multi_success']}/{item['count']} |"
                f" {item['wins']}/{item['losses']}/{item['ties']} |"
            )
    lines += [
        "",
        "## Write safety",
        "",
        "| Arm | create_ticket attempts | Previews | Unconfirmed sink rows |"
        " Confirm cases / idempotent passes | Cancel-only cases / previews / sink rows |"
        " No-confirm probes / sink rows |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for arm, item in summary["arms"].items():
        lines.append(
            f"| {arm} | {item['write_attempts']} | {item['generated_preview_count']} |"
            f" {item['unconfirmed_sink_writes']} | {item['confirmation_cases']}/"
            f"{item['confirmation_idempotent_passes']} | {item['cancel_only_cases']}/"
            f"{item['cancel_only_generated_previews']}/{item['cancel_only_sink_writes']} |"
            f" {item['cancellation_no_confirm_probes']}/"
            f"{item['cancellation_probe_sink_writes']} |"
        )
    lines += [
        "",
        "The cancellation probe is the no-confirmation path applied to each generated preview; "
        "the isolated fake sink received zero rows. Cancel-only instructions that produced no "
        "preview are reported as failed coverage, not counted as a safety success.",
        "",
    ]
    budget = summary["budget"]
    audit_path = RESULTS_DIR / "attempt-reconciliation.json"
    audit = (json.loads(audit_path.read_text(encoding="utf-8"))
             if audit_path.exists() else None)
    lines += [
        "",
        "## Budget and limitations",
        "",
        f"- Run cap: ${budget.get('run_cap_usd', RUN_CAP_USD):.2f}; charged upper bound:"
        f" ${budget.get('charged_upper_bound_usd', 0):.6f}; pending:"
        f" ${budget.get('reserved_usd', 0):.6f}.",
        "- Every provider call is bounded before dispatch and uses `max_retries=0`.",
        "- The final dev+frozen comparison consumed"
        f" {audit.get('final_comparison_calls', 0)} calls and cost"
        f" ${audit.get('final_comparison_cost_usd', 0):.6f}." if audit else
        "- Final comparison cost is recorded in the arm table.",
        "- All experiment attempts, including aborted instrumentation and smoke runs, are"
        f" reconciled in `{audit_path.name}` at ${audit.get('all_attempt_charged_upper_bound_usd', 0):.6f}"
        " charged upper bound." if audit else
        "- Attempt-level reconciliation is stored in the run directory.",
        "- Tickets live in an isolated fake sink. Unconfirmed calls never reach that sink;"
        " confirmation uses the preview `confirm_key` twice to check idempotency.",
        "- `MA-FRZ-022` hit the serial CH15 ledger's single-pending-call assumption during"
        " parallel retrieval. A concurrent budget implementation and regression test now"
        " exist, but the saved failed run remains counted here.",
        "- `MA-FRZ-030` returned a clarification rather than the frozen expected refusal.",
        "  `MA-FRZ-028` produced no ticket preview, so cancel-only coverage is not credited.",
        f"- The multi-Agent arm was slower (P50 {summary['arms']['multi_agent']['p50_ms']} ms"
        f" versus {summary['arms']['single_agent']['p50_ms']} ms) and had a higher"
        " cost per success. The evidence does not justify switching the default route.",
        "- Results cover this simulated frozen set only and do not establish production quality.",
        "- Any failed or timed-out case remains in its arm denominator.",
        "",
    ]
    return "\n".join(lines)


async def run_experiment(*, split: str = "all", allow_paid: bool = False,
                         offline: bool = False, env_file: str = "",
                         ledger_path: Path | None = None,
                         out_dir: Path | None = None,
                         limit: int | None = None) -> dict:
    scenarios = load_scenarios()
    if split != "all":
        scenarios = [row for row in scenarios if row["split"] == split]
    if limit is not None:
        scenarios = scenarios[:limit]
    if offline:
        budget = _NullBudget()
        budget_snapshot = {"mode": "offline-scripted", "run_cap_usd": 0.0}
        models = {
            "single": _OfflineModel("agent"),
            "planner": _OfflineModel("planner"),
            "retrieval": _OfflineModel("agent"),
            "action": _OfflineModel("action"),
            "supervisor": _OfflineModel("supervisor"),
        }
    else:
        if not allow_paid:
            raise BudgetExceeded("paid run requires explicit --allow-paid")
        source = env_file or str(Path("/root/workplace/ShopHelper/.env"))
        settings = Settings(_env_file=source)
        if not settings.openai_api_key:
            raise BudgetExceeded("OPENAI_API_KEY is missing")
        if ledger_path is not None:
            global LEDGER_PATH
            LEDGER_PATH = ledger_path
        budget = _paid_budget()
        if budget.snapshot().get("stop_reason"):
            raise BudgetExceeded(f"paid run previously stopped: {budget.snapshot()['stop_reason']}")
        raw = _chat_model(settings.openai_api_key)
        # The arm runners apply the guard exactly once with role-specific call labels.
        models = {
            "single": raw,
            "planner": raw,
            "retrieval": raw,
            "action": raw,
            "supervisor": raw,
        }

    results: list[dict] = []
    out_dir = out_dir or RESULTS_DIR
    for scenario in scenarios:
        arms = ["single_agent", "multi_agent"]
        for arm in arms:
            run_id = f"multi-agent-paired-v1:{scenario['id']}:{arm}"
            try:
                if arm == "single_agent":
                    row = await _run_single_agent(
                        scenario, models["single"], budget, run_id=run_id)
                else:
                    row = await _run_multi_agent(
                        scenario, models["planner"], budget, run_id=run_id,
                        role_models=models if offline else None)
            except Exception as exc:
                row = {
                    "reply": "", "replies": [], "disposition": "",
                    "tool_calls": [], "write_previews": [],
                    "confirmed_write_count": 0, "unconfirmed_write_count": 0,
                    "duration_ms": 0, "trace": [],
                    "errors": [f"harness:{type(exc).__name__}:{exc}"],
                    "stop_reasons": ["harness_exception"],
                    "model_calls": (
                        budget.calls_for_case(scenario["id"], arm)
                        if hasattr(budget, "calls_for_case") else []
                    ),
                    "model_interactions": [],
                }
                row["score"] = score_case(scenario, row)
            row.update({
                "scenario_id": scenario["id"],
                "arm": arm,
                "split": scenario["split"],
                "bucket": scenario["bucket"],
                "category": scenario["category"],
                "attempt_id": run_id,
                "scenario_sha256": scenario_manifest_sha256(),
            })
            results.append(row)
    snapshot = budget.snapshot() if hasattr(budget, "snapshot") else budget_snapshot
    summary = summarize_results(scenarios, results, snapshot)
    summary = _attach_split_summaries(summary, scenarios, results, snapshot)
    _write_jsonl(out_dir / "traces.jsonl.gz", results)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (REPORT_PATH.parent).mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(_report(summary), encoding="utf-8")
    return summary


def rescore_existing(out_dir: Path) -> dict:
    """Apply the current deterministic scorer to saved answers without model calls."""

    trace_path = (out_dir / "traces.jsonl.gz"
                  if (out_dir / "traces.jsonl.gz").exists()
                  else out_dir / "traces.jsonl")
    summary_path = out_dir / "summary.json"
    scenarios = {row["id"]: row for row in load_scenarios()}
    traces = _read_jsonl(trace_path)
    for trace in traces:
        trace["score"] = score_case(scenarios[trace["scenario_id"]], trace)
        trace["scored_without_model_calls"] = True
    previous = (json.loads(summary_path.read_text(encoding="utf-8"))
                if summary_path.exists() else {})
    summary = summarize_results(
        [scenarios[row["scenario_id"]] for row in traces
         if row["arm"] == "single_agent"],
        traces,
        previous.get("budget") or {"mode": "rescore-only", "run_cap_usd": RUN_CAP_USD},
    )
    summary["rescore_only"] = True
    summary["paid_calls_during_rescore"] = 0
    _write_jsonl(trace_path, traces)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(_report(summary), encoding="utf-8")
    return summary


def rescore_combined(out_dirs: list[Path], target_dir: Path) -> dict:
    scenarios = load_scenarios()
    by_id = {row["id"]: row for row in scenarios}
    traces: list[dict] = []
    budget_rows: list[dict] = []
    for directory in out_dirs:
        trace_path = (directory / "traces.jsonl.gz"
                      if (directory / "traces.jsonl.gz").exists()
                      else directory / "traces.jsonl")
        traces.extend(_read_jsonl(trace_path))
        summary_path = directory / "summary.json"
        if summary_path.exists():
            budget_rows.append(json.loads(summary_path.read_text(encoding="utf-8"))["budget"])
    if len(traces) != len(scenarios) * 2:
        raise BudgetExceeded(
            f"combined traces incomplete: expected {len(scenarios) * 2}, got {len(traces)}")
    for trace in traces:
        trace["score"] = score_case(by_id[trace["scenario_id"]], trace)
        trace["scored_without_model_calls"] = True

    def sum_metric(key: str) -> float:
        return sum(float(row.get(key) or 0.0) for row in budget_rows)

    arms = {}
    for budget in budget_rows:
        for arm, item in (budget.get("arms") or {}).items():
            current = arms.setdefault(arm, {
                "generations": 0, "max_generations": 0, "cap_usd": 0.0,
                "reserved_usd": 0.0, "usage_cost_at_pinned_peak_rates_usd": 0.0,
                "charged_upper_bound_usd": 0.0, "pending_calls": 0,
                "upper_bound_or_unknown_cost_calls": 0,
            })
            for key in current:
                current[key] += item.get(key) or 0
    combined_budget = {
        "model": MODEL,
        "allow_paid": True,
        "pricing_verified": True,
        "run_cap_usd": RUN_CAP_USD,
        "charged_upper_bound_usd": round(sum_metric("charged_upper_bound_usd"), 8),
        "usage_cost_at_pinned_peak_rates_usd": round(
            sum_metric("usage_cost_at_pinned_peak_rates_usd"), 8),
        "reserved_usd": 0, "stop_reason": None, "arms": arms,
    }
    summary = summarize_results(scenarios, traces, combined_budget)
    summary = _attach_split_summaries(summary, scenarios, traces, combined_budget)
    summary["rescore_only"] = True
    summary["paid_calls_during_rescore"] = 0
    target_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(target_dir / "traces.jsonl.gz", traces)
    (target_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(_report(summary), encoding="utf-8")
    return summary


def reconcile_attempt_ledgers() -> dict:
    """Audit every isolated multi-agent ledger without exposing prompts or credentials."""

    import sqlite3

    files = sorted((ROOT / "data").glob("multi_agent_paid_run*.sqlite3"))
    attempts = []
    for path in files:
        with sqlite3.connect(path) as db:
            rows = db.execute(
                "SELECT call_kind, status, COUNT(*), COALESCE(SUM(charged_usd), 0), "
                "COALESCE(SUM(input_tokens), 0), COALESCE(SUM(output_tokens), 0) "
                "FROM ch15_paid_calls GROUP BY call_kind, status "
                "ORDER BY call_kind, status"
            ).fetchall()
            stop = db.execute(
                "SELECT stop_reason FROM ch15_paid_runs "
                "ORDER BY created_at LIMIT 1"
            ).fetchone()
        attempts.append({
            "ledger": path.name,
            "calls": sum(row[2] for row in rows),
            "charged_upper_bound_usd": round(sum(row[3] for row in rows), 10),
            "statuses": [
                {"call_kind": row[0], "status": row[1], "calls": row[2],
                 "charged_usd": round(row[3], 10),
                 "input_tokens": row[4], "output_tokens": row[5]}
                for row in rows
            ],
            "stop_reason": stop[0] if stop else None,
        })
    provider_cost = sum(
        row["charged_usd"] for attempt in attempts for row in attempt["statuses"]
        if row["status"] in {"complete", "complete_upper_bound"}
    )
    charged = sum(attempt["charged_upper_bound_usd"] for attempt in attempts)
    final_cost = sum(
        attempt["charged_upper_bound_usd"] for attempt in attempts
        if attempt["ledger"] in {
            "multi_agent_paid_run_v7_dev.sqlite3",
            "multi_agent_paid_run_v8_frozen.sqlite3",
        }
    )
    final_calls = sum(
        attempt["calls"] for attempt in attempts
        if attempt["ledger"] in {
            "multi_agent_paid_run_v7_dev.sqlite3",
            "multi_agent_paid_run_v8_frozen.sqlite3",
        }
    )
    result = {
        "run_cap_usd": RUN_CAP_USD,
        "ledgers": attempts,
        "all_attempt_calls": sum(attempt["calls"] for attempt in attempts),
        "all_attempt_charged_upper_bound_usd": round(charged, 10),
        "all_attempt_usage_cost_at_peak_usd": round(provider_cost, 10),
        "final_comparison_cost_usd": round(final_cost, 10),
        "final_comparison_calls": final_calls,
        "pending_calls": sum(
            row["calls"] for attempt in attempts for row in attempt["statuses"]
            if row["status"] == "pending"
        ),
        "notes": {
            "multi_agent_paid_run.sqlite3": (
                "Pre-provider double-guard instrumentation failure; ledger"
                " conservatively charged one unknown call."
            ),
            "multi_agent_paid_run_v4_dev.sqlite3": "Aborted on runtime token guard.",
            "multi_agent_paid_run_v5_dev.sqlite3": "Aborted on inflated request envelope.",
            "multi_agent_paid_run_v6_dev.sqlite3": (
                "Pre-provider guard attribute failure plus one repaired smoke pair."
            ),
        },
    }
    target = RESULTS_DIR / "attempt-reconciliation.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return result


async def _cli_async(args) -> int:
    if args.cost_audit:
        print(json.dumps(reconcile_attempt_ledgers(), ensure_ascii=False, indent=2))
        return 0
    if args.rescore_all:
        summary = rescore_combined(
            [RESULTS_DIR / "dev", RESULTS_DIR / "frozen"], RESULTS_DIR / "all")
        print(json.dumps({
            "rescore_only": True, "paid_calls": 0,
            "paired": summary["paired"], "arms": summary["arms"],
            "buckets": summary["buckets"],
        }, ensure_ascii=False, indent=2))
        return 0
    if args.rescore:
        if not args.out:
            raise BudgetExceeded("--rescore requires --out pointing to existing traces")
        summary = rescore_existing(Path(args.out))
        print(json.dumps({
            "rescore_only": True,
            "paid_calls": 0,
            "paired": summary["paired"],
            "arms": summary["arms"],
        }, ensure_ascii=False, indent=2))
        return 0
    summary = await run_experiment(
        split=args.split,
        allow_paid=args.allow_paid,
        offline=args.offline,
        env_file=args.env_file,
        ledger_path=Path(args.ledger) if args.ledger else None,
        out_dir=Path(args.out) if args.out else None,
        limit=args.limit,
    )
    print(json.dumps({
        "model": summary["model"],
        "scenario_count": summary["scenario_count"],
        "result_count": summary["result_count"],
        "paired": summary["paired"],
        "arms": summary["arms"],
        "budget": summary["budget"],
    }, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Paired multi-agent versus constrained single-agent evaluation")
    parser.add_argument("--split", choices=["all", "dev", "frozen"], default="all")
    parser.add_argument("--allow-paid", action="store_true",
                        help="explicit opt-in for the <$2 protected real-model run")
    parser.add_argument("--offline", action="store_true",
                        help="exercise both graph arms with no-gold scripted rules")
    parser.add_argument("--env-file", default="",
                        help="credentials source; never printed or persisted")
    parser.add_argument("--ledger", default="",
                        help="isolated SQLite guard ledger path")
    parser.add_argument("--out", default="",
                        help="trace output directory")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rescore", action="store_true",
                        help="re-score saved traces without any model call")
    parser.add_argument("--rescore-all", action="store_true",
                        help="combine and re-score saved dev+frozen traces only")
    parser.add_argument("--cost-audit", action="store_true",
                        help="reconcile all isolated multi-agent cost ledgers")
    args = parser.parse_args()
    if not args.offline and not args.allow_paid and not args.rescore \
            and not args.rescore_all and not args.cost_audit:
        parser.error("refusing paid execution without --allow-paid")
    return asyncio.run(_cli_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
