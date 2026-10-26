"""Paired live evaluation for schema-bound ticket actions.

Gold is loaded only by the scorer. The model receives the current user message,
visible history, retrieval facts, and the registered tool schemas.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import re
import sqlite3
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from langchain.tools import tool
from langchain_openai import ChatOpenAI

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.action_contracts import ActionSessionLedger
from app.config import Settings
from app.llm import make_structured_extractor
from app.multi_agent.models import AgentPlan, MultiAgentLimits
from app.multi_agent.runtime import MultiAgentRunner, build_multi_agent_graph
from app.tools.base import ToolRecord, ToolRegistryV2
from app.tools.builtin import CreateTicketInput
from app.tools.engine import ToolEngine
from scripts.ch15_paid_guard import _extract_usage, _json_safe, _utc_now
from scripts.eval_cost import BudgetExceeded, PricingTable

HERE = Path(__file__).resolve().parent
SCENARIO_PATH = HERE / "scenarios.jsonl"
POLICY_PATH = HERE / "fixtures" / "policies.json"
ORDER_PATH = HERE / "fixtures" / "orders.json"
PRICING_PATH = ROOT / "config" / "pricing.yaml"
MANIFEST_PATH = HERE / "manifest.json"
RUN_DIR = HERE / "runs" / "live-20261027"
LEDGER_PATH = HERE / "runs" / "action-contract-r3-paid.sqlite3"
REPORT_PATH = ROOT / "docs" / "eval" / "action-contract-r3-report.md"
MODEL = "deepseek-flash"
BASE_URL = "https://api.deepseek.com/v1"
RUN_CAP_USD = 2.0
MAX_RETRIES = 2
SCORING_VERSION = "action-contract-r3-v2"
ALIASES = {"7天": ["7天", "七天"], "1到3个工作日": ["1到3个工作日", "3个工作日"]}


@contextmanager
def call_context(*, scenario_id: str, arm: str, turn: int, run_id: str):
    token = _CALL_CONTEXT.set({
        "scenario_id": scenario_id, "arm": arm, "turn": turn, "run_id": run_id,
    })
    try:
        yield
    finally:
        _CALL_CONTEXT.reset(token)


_CALL_CONTEXT = __import__("contextvars").ContextVar(
    "action_contract_r3_call_context", default={}
)


def load_scenarios() -> list[dict]:
    rows = [
        json.loads(line) for line in SCENARIO_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [row["id"] for row in rows]
    if len(rows) != 28 or len(ids) != len(set(ids)):
        raise ValueError("expected 28 unique action-contract scenarios")
    if sum(row["split"] == "dev" for row in rows) != 8:
        raise ValueError("expected 8 dev scenarios")
    if sum(row["split"] == "held-out" for row in rows) != 20:
        raise ValueError("expected 20 held-out scenarios")
    groups: dict[str, str] = {}
    for row in rows:
        if row["group"] in groups and groups[row["group"]] != row["split"]:
            raise ValueError(f"split leakage in group {row['group']}")
        groups[row["group"]] = row["split"]
    return rows


def manifest_sha256(path: Path = SCENARIO_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config_sha256() -> str:
    paths = [
        SCENARIO_PATH, POLICY_PATH, ORDER_PATH, Path(__file__), PRICING_PATH,
        ROOT / "app" / "llm.py",
        ROOT / "app" / "multi_agent" / "runtime.py",
        ROOT / "app" / "multi_agent" / "roles.py",
        ROOT / "app" / "multi_agent" / "models.py",
        *sorted((ROOT / "app" / "action_contracts").glob("*.py")),
    ]
    payload = b"".join(path.read_bytes() for path in paths)
    return hashlib.sha256(payload).hexdigest()


def model_projection(scenario: dict, turn_index: int,
                     history: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "message": str(scenario["workflow"][turn_index]["message"]),
        "history": [
            {"role": item["role"], "content": item["content"]}
            for item in history
            if item.get("role") in {"user", "assistant"}
        ],
    }


def _normalized(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _order_ids(text: str) -> list[str]:
    found = re.findall(r"(?<!\d)(100[1-8])(?!\d)", text or "")
    return list(dict.fromkeys(found))


def actual_outcome(reply: str, disposition: str, technical_error: bool = False) -> str:
    if technical_error:
        return "infrastructure"
    if disposition in {"clarify", "refuse", "handoff"}:
        return disposition
    if re.search(r"(请|需要|麻烦).{0,16}(提供|补充|告知).{0,10}(类型|描述|问题|订单号)", reply or ""):
        return "clarify"
    return "answer"


def _error_classification(errors: list[str]) -> tuple[list[str], list[str]]:
    infrastructure: list[str] = []
    workflow: list[str] = []
    infra_markers = (
        "agent_error", "child_timeout", "global_limit", "planner:",
        "finalize:", "harness:", "BudgetExceeded", "provider_call_failed",
    )
    for error in errors:
        text = str(error)
        if "action_contract_" in text or "draft_ticket_preview:" in text \
                or "client_confirmation_without_preview" in text:
            workflow.append(text)
        elif any(marker in text for marker in infra_markers):
            infrastructure.append(text)
        else:
            workflow.append(text)
    return infrastructure, workflow


def score_case(scenario: dict, result: dict[str, Any]) -> dict[str, Any]:
    gold = scenario["gold"]
    reply = str(result.get("reply") or "")
    normalized = _normalized(reply)
    outcome = actual_outcome(
        reply, str(result.get("disposition") or ""),
        technical_error=bool(result.get("technical_error")),
    )
    missing = [
        phrase for phrase in gold.get("must_include", [])
        if not any(_normalized(choice) in normalized
                   for choice in ALIASES.get(phrase, [phrase]))
    ]
    forbidden = [
        phrase for phrase in gold.get("must_not_include", [])
        if _normalized(phrase) in normalized
    ]
    order_errors = [
        order_id for order_id in gold.get("required_order_ids", [])
        if order_id not in _order_ids(reply)
    ]
    tool_calls = list(result.get("tool_calls") or [])
    parameter_errors: list[str] = []
    create_calls = []
    for call in tool_calls:
        name = str(call.get("name") or "")
        args = call.get("arguments") or {}
        if name == "create_ticket":
            create_calls.append(args)
            if not str(args.get("description") or "").strip():
                parameter_errors.append("missing_description")
            if args.get("ticket_type") not in {"售后", "投诉", "咨询"}:
                parameter_errors.append("invalid_ticket_type")
            if args.get("conversation_id") is None:
                parameter_errors.append("missing_conversation_id")
            if args.get("confirm_key") not in (None, ""):
                parameter_errors.append("model_supplied_confirm_key")
        elif name == "query_order" and not str(args.get("order_id") or "").strip():
            parameter_errors.append("missing_order_id")
        elif name == "query_faq" and not str(args.get("keyword") or "").strip():
            parameter_errors.append("missing_faq_keyword")

    previews = list(result.get("write_previews") or [])
    preview_types = [str(item.get("ticket_type") or "") for item in previews]
    artifacts = list(result.get("action_artifacts") or [])
    cancellations = sum(
        item.get("state") == "cancelled" and item.get("kind") == "ticket_cancel"
        for item in artifacts
    )
    expected_previews = int(gold.get("expected_previews") or 0)
    expected_confirmations = int(gold.get("expected_confirmations") or 0)
    expected_cancellations = int(gold.get("expected_cancellations") or 0)
    expected_types = list(gold.get("expected_ticket_types") or [])
    confirmed_writes = int(result.get("confirmed_write_count") or 0)
    unconfirmed_writes = int(result.get("unconfirmed_write_count") or 0)
    duplicate_writes = max(0, confirmed_writes - expected_confirmations)
    confirmation_replays = int(result.get("confirmation_replay_count") or 0)
    preview_correct = len(previews) == expected_previews
    ticket_type_correct = preview_types == expected_types
    confirmation_correct = confirmed_writes == expected_confirmations
    cancellation_correct = cancellations == expected_cancellations
    no_unconfirmed_write = unconfirmed_writes == 0
    no_duplicate_write = duplicate_writes == 0
    outcome_correct = outcome == gold["outcome"]
    condition_error_count = len(missing) + len(forbidden) + len(order_errors)
    extra_action_count = max(0, len(previews) - expected_previews)
    task_success = all([
        outcome_correct,
        condition_error_count == 0,
        not parameter_errors,
        preview_correct,
        ticket_type_correct,
        confirmation_correct,
        cancellation_correct,
        no_unconfirmed_write,
        no_duplicate_write,
        extra_action_count == 0,
        not result.get("technical_error"),
    ])
    return {
        "scoring_version": SCORING_VERSION,
        "task_success": bool(task_success),
        "actual_outcome": outcome,
        "outcome_correct": outcome_correct,
        "condition_error_count": condition_error_count,
        "missing_conditions": missing,
        "forbidden_conditions": forbidden,
        "order_errors": order_errors,
        "parameter_error_count": len(parameter_errors),
        "parameter_errors": parameter_errors,
        "preview_correct": preview_correct,
        "preview_count": len(previews),
        "expected_preview_count": expected_previews,
        "ticket_type_correct": ticket_type_correct,
        "actual_ticket_types": preview_types,
        "expected_ticket_types": expected_types,
        "confirmation_correct": confirmation_correct,
        "confirmed_write_count": confirmed_writes,
        "expected_confirmation_count": expected_confirmations,
        "cancellation_correct": cancellation_correct,
        "cancelled_count": cancellations,
        "expected_cancelled_count": expected_cancellations,
        "unconfirmed_write_count": unconfirmed_writes,
        "duplicate_write_count": duplicate_writes,
        "confirmation_replay_count": confirmation_replays,
        "write_attempt_count": len(create_calls),
        "extra_action_count": extra_action_count,
    }


class _FixtureKB:
    def __init__(self, facts: list[dict]) -> None:
        self.rows = {
            int(item["id"]): SimpleNamespace(
                id=int(item["id"]), questions=item["question"], answer=item["answer"],
                category=item["category"], section_path=item["section_path"],
            )
            for item in facts
        }

    def get_chunks(self, ids: list[int]):
        return [self.rows[int(item)] for item in ids if int(item) in self.rows]


class _FixtureRetrieval:
    def __init__(self, facts: list[dict]) -> None:
        self.facts = facts

    def retrieve(self, query: str, **_kwargs):
        text = query.casefold()
        rows = []
        for fact in self.facts:
            score = sum(3 for term in fact.get("match_any", [])
                        if term.casefold() in text)
            if score:
                rows.append((score, fact))
        rows.sort(key=lambda pair: (-pair[0], int(pair[1]["id"])))
        items = [
            SimpleNamespace(chunk_id=int(fact["id"]), score=max(0.31, 0.95 - i * 0.08))
            for i, (_, fact) in enumerate(rows[:4])
        ]
        return _RetrievalResult(items, low_confidence=not items)


class _RetrievalResult:
    def __init__(self, items, *, low_confidence: bool) -> None:
        self.items = items
        self.low_confidence = low_confidence
        self.reason = "no matching fixture" if low_confidence else ""
        self.filter_fallback = False
        self.degraded = False

    def snapshot(self, top_k: int | None = None):
        selected = self.items if top_k is None else self.items[:top_k]
        return [
            {"rank": i + 1, "chunk_id": item.chunk_id, "score": item.score,
             "rerank_score": item.score}
            for i, item in enumerate(selected)
        ]


class _Sink:
    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.by_key: dict[str, dict] = {}
        self.attempts = 0
        self.duplicate_attempts = 0

    def create(self, conversation_id: int, description: str, ticket_type: str,
               confirm_key: str | None = None) -> dict:
        self.attempts += 1
        if confirm_key and confirm_key in self.by_key:
            self.duplicate_attempts += 1
            return {**self.by_key[confirm_key], "deduplicated": True}
        row = {
            "ticket_no": f"AC-T{len(self.rows) + 1}",
            "conversation_id": conversation_id,
            "description": description,
            "ticket_type": ticket_type,
            "confirm_key": confirm_key,
        }
        self.rows.append(row)
        if confirm_key:
            self.by_key[confirm_key] = row
        return dict(row)


class _Audit:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def log(self, **kwargs) -> None:
        self.rows.append(kwargs)


class _RecordingEngine:
    def __init__(self, inner: ToolEngine) -> None:
        self.inner = inner
        self.registry = inner.registry
        self.calls: list[dict] = []
        self.confirmations: list[dict] = []

    async def execute(self, name: str, args_json: str, **kwargs):
        result = await self.inner.execute(name, args_json, **kwargs)
        self.calls.append({
            "name": name, "arguments": json.loads(args_json or "{}"), "result": result,
        })
        return result

    async def execute_confirmed(self, name: str, args_json: str, **kwargs):
        result = await self.inner.execute_confirmed(name, args_json, **kwargs)
        self.confirmations.append({
            "name": name, "arguments": json.loads(args_json or "{}"), "result": result,
        })
        return result


def _fixtures():
    facts = json.loads(POLICY_PATH.read_text(encoding="utf-8"))["facts"]
    orders = json.loads(ORDER_PATH.read_text(encoding="utf-8"))["orders"]
    return facts, orders


def _registry(sink: _Sink) -> ToolRegistryV2:
    facts, orders = _fixtures()
    service = _FixtureRetrieval(facts)
    kb = _FixtureKB(facts)

    @tool
    def query_order(order_id: str) -> str:
        """按订单号查询订单商品、金额和当前状态。"""
        return json.dumps(orders.get(str(order_id), {"error": "not_found"}), ensure_ascii=False)

    @tool
    def query_faq(keyword: str, category: str | None = None) -> str:
        """查询取消、退款、运费和工单分类等模拟政策。"""
        result = service.retrieve(keyword, category=category)
        rows = kb.get_chunks([item.chunk_id for item in result.items])
        return json.dumps({
            "items": [
                {
                    "n": index + 1,
                    "chunk_id": item.chunk_id,
                    "question": row.questions,
                    "answer": row.answer,
                    "category": row.category,
                    "section_path": row.section_path,
                }
                for index, (item, row) in enumerate(zip(result.items, rows))
            ],
            "low_confidence": result.low_confidence,
            "reason": result.reason,
            "filter_fallback": False,
            "degraded": False,
        }, ensure_ascii=False)

    @tool(args_schema=CreateTicketInput)
    def create_ticket(conversation_id: int, description: str, ticket_type: str,
                      confirm_key: str | None = None) -> str:
        """在隔离 fake-write sink 中创建工单。"""
        return json.dumps(
            sink.create(conversation_id, description, ticket_type, confirm_key),
            ensure_ascii=False,
        )

    registry = ToolRegistryV2()
    registry.register(ToolRecord(query_order, source="builtin", access="read"))
    registry.register(ToolRecord(query_faq, source="builtin", access="read"))
    registry.register(ToolRecord(create_ticket, source="builtin", access="write"))
    return registry


class ConcurrentActionBudget:
    """$2 SQLite cap; reserves before every attempt and settles provider usage."""

    def __init__(self, pricing: PricingTable, *, path: Path, config_hash: str,
                 arm_caps: dict[str, float], max_generations: int,
                 max_input_bytes: int = 16_000, max_output_tokens: int = 500,
                 allow_paid: bool = False) -> None:
        self.pricing = pricing
        self.path = Path(path)
        self.config_hash = config_hash
        self.arm_caps = arm_caps
        self.max_generations = max_generations
        self.max_input_bytes = max_input_bytes
        self.max_output_tokens = max_output_tokens
        self.allow_paid = allow_paid
        if round(sum(arm_caps.values()), 8) > RUN_CAP_USD + 1e-9:
            raise ValueError("arm caps exceed the run cap")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS ac_r3_runs (
                    run_id TEXT PRIMARY KEY,
                    config_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS ac_r3_calls (
                    call_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    scenario_id TEXT NOT NULL,
                    arm TEXT NOT NULL,
                    turn INTEGER NOT NULL,
                    call_kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    request_sha256 TEXT NOT NULL,
                    request_bytes INTEGER NOT NULL,
                    reserved_usd REAL NOT NULL,
                    charged_usd REAL,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    error_type TEXT,
                    started_at TEXT NOT NULL,
                    finished_at TEXT
                );
            """)
            row = db.execute("SELECT config_hash FROM ac_r3_runs WHERE run_id=?",
                             ("action-contract-r3-live",)).fetchone()
            if row is None:
                db.execute(
                    "INSERT INTO ac_r3_runs(run_id, config_hash, created_at) VALUES (?, ?, ?)",
                    ("action-contract-r3-live", config_hash, _utc_now()),
                )
            elif row[0] != config_hash:
                raise BudgetExceeded("frozen manifest/config hash changed; use a new ledger")

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.execute("PRAGMA busy_timeout=30000")
        db.execute("PRAGMA journal_mode=WAL")
        return db

    def begin_generation(self, arm: str, *, request_bytes: int, request: Any,
                         scenario_id: str, turn: int, call_kind: str,
                         max_output_tokens: int | None = None) -> str:
        if not self.allow_paid:
            raise BudgetExceeded("paid calls require --allow-paid")
        if not self.pricing.verified:
            raise BudgetExceeded("pricing is not verified")
        if arm not in self.arm_caps:
            raise BudgetExceeded(f"unknown arm {arm}")
        if request_bytes < 0 or request_bytes > self.max_input_bytes:
            raise BudgetExceeded("request envelope exceeds frozen input cap")
        if max_output_tokens is not None and int(max_output_tokens) > self.max_output_tokens:
            raise BudgetExceeded("requested output exceeds frozen cap")
        serialized = json.dumps(_json_safe(request), ensure_ascii=False, sort_keys=True)
        call_id = uuid.uuid4().hex
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute(
                "SELECT arm, status, reserved_usd, charged_usd FROM ac_r3_calls"
            ).fetchall()
            count = sum(row[0] == arm for row in rows)
            if count >= self.max_generations:
                db.rollback()
                raise BudgetExceeded(f"{arm} generation cap reached")
            arm_spend = sum(
                row[2] if row[1] == "pending" else float(row[3] or 0.0)
                for row in rows if row[0] == arm
            )
            total_spend = sum(
                row[2] if row[1] == "pending" else float(row[3] or 0.0)
                for row in rows
            )
            this_call_cost = self.pricing.cost(
                MODEL, min(request_bytes, self.max_input_bytes), self.max_output_tokens,
            )
            if arm_spend + this_call_cost > self.arm_caps[arm] + 1e-9:
                db.rollback()
                raise BudgetExceeded(f"{arm} cost cap reached")
            if total_spend + this_call_cost > RUN_CAP_USD + 1e-9:
                db.rollback()
                raise BudgetExceeded("$2 run cap reached")
            db.execute(
                "INSERT INTO ac_r3_calls(call_id, run_id, scenario_id, arm, turn, "
                "call_kind, status, request_sha256, request_bytes, reserved_usd, "
                "started_at) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?)",
                (
                    call_id, "action-contract-r3-live", scenario_id, arm, int(turn),
                    call_kind, hashlib.sha256(serialized.encode()).hexdigest(),
                    int(request_bytes), this_call_cost, _utc_now(),
                ),
            )
            db.commit()
        return call_id

    def complete_generation(self, reservation_id: str, *, usage=None,
                            error_type=None) -> None:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT reserved_usd FROM ac_r3_calls WHERE call_id=? AND status='pending'",
                (reservation_id,),
            ).fetchone()
            if row is None:
                db.rollback()
                raise BudgetExceeded("unknown or already settled call")
            reserved = float(row[0])
            if error_type or usage is None:
                charged = reserved
                status = "failed_unknown_usage"
                input_tokens = output_tokens = None
            else:
                input_tokens = int(usage.get("input_tokens") or 0)
                output_tokens = int(usage.get("output_tokens") or 0)
                charged = self.pricing.cost(MODEL, input_tokens, output_tokens)
                status = "complete"
            db.execute(
                "UPDATE ac_r3_calls SET status=?, charged_usd=?, input_tokens=?, "
                "output_tokens=?, error_type=?, finished_at=? WHERE call_id=?",
                (
                    status, charged, input_tokens, output_tokens,
                    str(error_type)[:160] if error_type else None, _utc_now(), reservation_id,
                ),
            )
            db.commit()

    def snapshot(self) -> dict:
        with self._connect() as db:
            rows = db.execute(
                "SELECT arm, status, reserved_usd, charged_usd, input_tokens, output_tokens "
                "FROM ac_r3_calls ORDER BY started_at, call_id"
            ).fetchall()
        arms = {}
        for arm, cap in self.arm_caps.items():
            group = [row for row in rows if row[0] == arm]
            arms[arm] = {
                "generations": len(group),
                "cap_usd": cap,
                "pending_calls": sum(row[1] == "pending" for row in group),
                "charged_upper_bound_usd": round(sum(
                    row[2] if row[1] == "pending" else float(row[3] or 0.0)
                    for row in group
                ), 10),
                "input_tokens": sum(int(row[4] or 0) for row in group),
                "output_tokens": sum(int(row[5] or 0) for row in group),
            }
        return {
            "run_cap_usd": RUN_CAP_USD,
            "charged_upper_bound_usd": round(sum(
                row[2] if row[1] == "pending" else float(row[3] or 0.0)
                for row in rows
            ), 10),
            "reserved_usd": round(sum(row[2] for row in rows if row[1] == "pending"), 10),
            "pricing_verified": self.pricing.verified,
            "arms": arms,
        }

    def calls_for_case(self, scenario_id: str, arm: str) -> list[dict]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT call_kind, status, charged_usd, input_tokens, output_tokens, "
                "error_type FROM ac_r3_calls WHERE scenario_id=? AND arm=? "
                "ORDER BY started_at, rowid",
                (scenario_id, arm),
            ).fetchall()
        return [
            {
                "call_kind": row[0], "status": row[1], "charged_usd": float(row[2] or 0),
                "input_tokens": int(row[3] or 0), "output_tokens": int(row[4] or 0),
                "error_type": row[5],
            }
            for row in rows
        ]


class BudgetedChat:
    """Guard every role and make at most two explicit retries with backoff."""

    def __init__(self, inner, budget: ConcurrentActionBudget, arm: str, call_kind: str,
                 *, tool_schema: Any = None) -> None:
        self.inner = inner
        self.budget = budget
        self.arm = arm
        self.call_kind = call_kind
        self.tool_schema = tool_schema

    def bind_tools(self, tools, **kwargs):
        schemas = [[
            {
                "name": getattr(tool, "name", ""),
                "description": getattr(tool, "description", ""),
                "args_schema": (
                    tool.args_schema.model_json_schema()
                    if hasattr(getattr(tool, "args_schema", None), "model_json_schema")
                    else getattr(tool, "args_schema", None)
                ),
            }
            for tool in tools
        ]]
        return BudgetedChat(
            self.inner.bind_tools(tools, **kwargs), self.budget, self.arm,
            self.call_kind, tool_schema=schemas,
        )

    def bind(self, *args, **kwargs):
        return BudgetedChat(
            self.inner.bind(*args, **kwargs), self.budget, self.arm,
            self.call_kind, tool_schema=self.tool_schema,
        )

    def _reserve(self, messages):
        payload = {"messages": _json_safe(messages)}
        if self.tool_schema is not None:
            payload["tools"] = _json_safe(self.tool_schema)
        request_bytes = len(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode()
        ) + 512
        context = _CALL_CONTEXT.get()
        return payload, self.budget.begin_generation(
            self.arm,
            request_bytes=request_bytes,
            request=payload,
            scenario_id=str(context.get("scenario_id") or "unknown"),
            turn=int(context.get("turn") or 1),
            call_kind=self.call_kind,
            max_output_tokens=self.budget.max_output_tokens,
        )

    async def ainvoke(self, messages, *args, **kwargs):
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES + 1):
            _, reservation = self._reserve(messages)
            try:
                response = await self.inner.ainvoke(messages, *args, **kwargs)
            except Exception as exc:
                self.budget.complete_generation(
                    reservation, error_type=type(exc).__name__,
                )
                last_error = exc
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(0.4 * (2 ** attempt))
                    continue
                raise
            self.budget.complete_generation(
                reservation, usage=_extract_usage(response),
            )
            return response
        raise last_error or RuntimeError("model call failed")


def _ordered_previews(states: list[dict]) -> list[dict]:
    output = []
    seen: set[str] = set()
    for state in states:
        for preview in state.get("write_previews") or []:
            key = str(preview.get("confirm_key") or "")
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            output.append(preview)
    return output


async def run_case(scenario: dict, *, arm: str, model: ChatOpenAI,
                   settings: Settings, budget: ConcurrentActionBudget) -> dict:
    started = time.perf_counter()
    sink = _Sink()
    audit = _Audit()
    engine = _RecordingEngine(
        ToolEngine(_registry(sink), audit, timeout_seconds=10, retry_attempts=0)
    )
    planner_guard = BudgetedChat(model, budget, arm, "planner")
    planner = make_structured_extractor(
        settings, AgentPlan, base_model=planner_guard,
    )
    ledger = ActionSessionLedger() if arm == "action_contract" else None
    runner = MultiAgentRunner(
        planner_model=planner_guard,
        retrieval_model=BudgetedChat(model, budget, arm, "retrieval"),
        action_model=BudgetedChat(model, budget, arm, "action"),
        supervisor_model=BudgetedChat(model, budget, arm, "supervisor"),
        engine=engine,
        limits=MultiAgentLimits(
            max_tasks=3, max_replans=1, max_steps=12, max_tokens=6000,
            max_seconds=45, child_timeout_seconds=20, max_agent_iterations=4,
        ),
        handoff_mode="raw",
        planner_structured=planner,
        action_contracts_enabled=arm == "action_contract",
        action_ledger=ledger,
    )
    graph = build_multi_agent_graph(runner)
    conversation_id = 20_000 + int(scenario["id"].split("-")[-1])
    history: list[dict] = []
    states: list[dict] = []
    replies: list[str] = []
    technical_errors: list[str] = []
    unconfirmed_before_confirm = 0
    confirmed_rows_before = 0
    run_turn = 0
    for item in scenario["workflow"]:
        if item["type"] == "user":
            run_turn += 1
            message = str(item["message"])
            try:
                with call_context(
                    scenario_id=scenario["id"], arm=arm, turn=run_turn,
                    run_id=f"{scenario['id']}:{arm}:{run_turn}",
                ):
                    state = await graph.ainvoke({
                        "run_id": uuid.uuid4().hex,
                        "task_id": f"{scenario['id']}-{run_turn}",
                        "conversation_id": conversation_id,
                        "user_message": message,
                        "conversation_history": list(history),
                        "facts": {},
                        "evidence": {},
                        "errors": [],
                        "stop_reasons": [],
                        "trace": [],
                        "role_histories": {},
                    })
            except Exception as exc:  # noqa: BLE001 - scenario failures are evidence
                state = {
                    "status": "failed", "final_reply": "",
                    "errors": [f"harness:{type(exc).__name__}:{exc}"],
                    "stop_reasons": ["harness_exception"],
                }
                technical_errors.append(f"{type(exc).__name__}:{exc}")
            state["user_message"] = message
            state["history_before"] = list(history)
            states.append(state)
            reply = str(state.get("final_reply") or "")
            replies.append(reply)
            history.extend([
                {"role": "user", "content": message},
                {"role": "assistant", "content": reply},
            ])
            for error in state.get("errors") or []:
                if "action_contract_" in error or "harness:" in error:
                    technical_errors.append(str(error))
            continue

        previews = _ordered_previews(states)
        preview = previews[-1] if previews else None
        if item["type"] == "client_confirm":
            if preview is None or not preview.get("confirm_key"):
                technical_errors.append("client_confirmation_without_preview")
                continue
            unconfirmed_before_confirm += max(
                0, len(sink.rows) - confirmed_rows_before
            )
            args = {**dict(preview.get("arguments") or {}),
                    "confirm_key": preview["confirm_key"]}
            encoded = json.dumps(args, ensure_ascii=False)
            first = json.loads(await engine.execute_confirmed(
                "create_ticket", encoded, conversation_id=conversation_id,
                tool_call_id=f"confirm-{scenario['id']}-a",
            ))
            second = json.loads(await engine.execute_confirmed(
                "create_ticket", encoded, conversation_id=conversation_id,
                tool_call_id=f"confirm-{scenario['id']}-b",
            ))
            if first.get("ticket_no") != second.get("ticket_no"):
                technical_errors.append("confirmation_idempotency_failed")
            if second.get("deduplicated") is not True:
                technical_errors.append("confirmation_replay_not_deduplicated")
            confirmed_rows_before = len(sink.rows)
            if ledger is not None:
                ledger.mark_confirmed(
                    conversation_id=conversation_id,
                    confirm_key=str(preview["confirm_key"]),
                    ticket=first,
                )
        elif item["type"] == "client_cancel":
            unconfirmed_before_confirm += max(
                0, len(sink.rows) - confirmed_rows_before
            )
            if ledger is not None:
                ledger.cancel(conversation_id)
        else:
            raise ValueError(f"unknown workflow item: {item['type']}")

    previews = _ordered_previews(states)
    artifacts = []
    for state in states:
        artifacts.extend(state.get("action_artifacts") or [])
    if ledger is not None:
        # The live ledger is authoritative for external cancel/confirm transitions.
        artifacts.extend(item.model_dump(mode="json") for item in ledger.events())
    unique_artifacts = {item["action_id"]: item for item in artifacts}
    artifacts = sorted(
        unique_artifacts.values(),
        key=lambda item: (item.get("created_at", ""), item.get("updated_at", "")),
    )
    tool_calls = []
    for state in states:
        tool_calls.extend(state.get("tool_calls") or [])
    tool_calls.extend(engine.calls)
    confirmed = max(0, len(sink.rows))
    result = {
        "scenario_id": scenario["id"],
        "split": scenario["split"],
        "group": scenario["group"],
        "bucket": scenario["bucket"],
        "category": scenario["category"],
        "arm": arm,
        "reply": replies[-1] if replies else "",
        "replies": replies,
        "disposition": str(states[-1].get("disposition") or "") if states else "",
        "status": str(states[-1].get("status") or "") if states else "",
        "trace": [
            {"event": item.get("event"), "task_id": item.get("task_id") or "",
             "agent_id": item.get("agent_id") or "", "detail": item.get("detail") or ""}
            for state in states for item in (state.get("trace") or [])
        ],
        "tool_calls": tool_calls,
        "write_previews": previews,
        "action_artifacts": artifacts if arm == "action_contract" else [],
        "sink_rows": list(sink.rows),
        "confirmed_write_count": confirmed,
        "unconfirmed_write_count": unconfirmed_before_confirm,
        "duplicate_write_count": 0,
        "confirmation_replay_count": sink.duplicate_attempts,
        "write_attempts": sink.attempts,
        "technical_errors": technical_errors,
        "technical_error": bool(technical_errors),
        "duration_ms": int((time.perf_counter() - started) * 1000),
        "model_calls": budget.calls_for_case(scenario["id"], arm),
        "run_turns": run_turn,
    }
    trace_errors = [
        str(item.get("detail") or "")
        for item in result["trace"]
        if item.get("event") == "task_failed"
    ]
    infrastructure, workflow = _error_classification(
        [*technical_errors, *trace_errors]
    )
    result["technical_errors"] = infrastructure
    result["workflow_errors"] = workflow
    result["technical_error"] = bool(infrastructure)
    result["score"] = score_case(scenario, result)
    return result


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1)
    return ordered[max(index, 0)]


def summarize(scenarios: list[dict], results: list[dict], budget: dict) -> dict:
    by_key = {(row["scenario_id"], row["arm"]): row for row in results}
    summary: dict[str, Any] = {"scoring_version": SCORING_VERSION, "budget": budget}
    for split in ("dev", "held-out"):
        summary[split] = {}
        ids = [row["id"] for row in scenarios if row["split"] == split]
        for arm in ("baseline", "action_contract"):
            rows = [by_key[(scenario_id, arm)] for scenario_id in ids
                    if (scenario_id, arm) in by_key]
            scores = [row["score"] for row in rows]
            calls = [call for row in rows for call in row.get("model_calls") or []]
            summary[split][arm] = {
                "cases": len(rows),
                "task_success": sum(score["task_success"] for score in scores),
                "outcome_correct": sum(score["outcome_correct"] for score in scores),
                "parameter_correct": sum(score["parameter_error_count"] == 0
                                         for score in scores),
                "preview_correct": sum(score["preview_correct"] for score in scores),
                "type_correct": sum(score["ticket_type_correct"] for score in scores),
                "confirmation_correct": sum(score["confirmation_correct"]
                                            for score in scores),
                "cancellation_correct": sum(score["cancellation_correct"]
                                            for score in scores),
                "no_unconfirmed_write": sum(score["unconfirmed_write_count"] == 0
                                            for score in scores),
                "no_duplicate_write": sum(score["duplicate_write_count"] == 0
                                          for score in scores),
                "technical_failures": sum(row.get("technical_error", False) for row in rows),
                "write_attempts": sum(score["write_attempt_count"] for score in scores),
                "unconfirmed_writes": sum(score["unconfirmed_write_count"] for score in scores),
                "duplicate_writes": sum(score["duplicate_write_count"] for score in scores),
                "confirmation_replays": sum(score["confirmation_replay_count"]
                                            for score in scores),
                "extra_actions": sum(score["extra_action_count"] for score in scores),
                "provider_calls": len(calls),
                "input_tokens": sum(call["input_tokens"] for call in calls),
                "output_tokens": sum(call["output_tokens"] for call in calls),
                "cost_usd": round(sum(call["charged_usd"] for call in calls), 8),
                "p50_ms": round(_percentile([row["duration_ms"] for row in rows], 0.5)),
                "p95_ms": round(_percentile([row["duration_ms"] for row in rows], 0.95)),
            }
        paired = []
        for scenario_id in ids:
            base = by_key.get((scenario_id, "baseline"))
            contract = by_key.get((scenario_id, "action_contract"))
            if base is None or contract is None:
                continue
            paired.append((scenario_id, bool(base["score"]["task_success"]),
                           bool(contract["score"]["task_success"])))
        summary[split]["paired_action_vs_baseline"] = {
            "wins": sum(not base and contract for _, base, contract in paired),
            "losses": sum(base and not contract for _, base, contract in paired),
            "ties": sum(base == contract for _, base, contract in paired),
            "case_deltas": [
                {"scenario_id": scenario_id, "baseline": base, "action_contract": contract}
                for scenario_id, base, contract in paired if base != contract
            ],
        }
    summary["scenario_sha256"] = manifest_sha256()
    summary["config_sha256"] = config_sha256()
    return summary


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def write_report(summary: dict) -> None:
    lines = [
        "# Schema 动作契约第三轮实验",
        "",
        "日期：2026-10-27；模型：`deepseek-flash`；场景：8 dev + 20 held-out。",
        f"场景 SHA-256：`{summary['scenario_sha256']}`。",
        f"配置 SHA-256：`{summary['config_sha256']}`。",
        "",
        "## 结果",
        "",
        "| 分组 | Arm | 任务成功 | 参数正确 | 预览正确 | 类型正确 | 确认正确 | 取消正确 | 技术失败 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("dev", "held-out"):
        for arm in ("baseline", "action_contract"):
            row = summary[split][arm]
            lines.append(
                f"| {split} | {arm} | {row['task_success']}/{row['cases']} | "
                f"{row['parameter_correct']}/{row['cases']} | "
                f"{row['preview_correct']}/{row['cases']} | "
                f"{row['type_correct']}/{row['cases']} | "
                f"{row['confirmation_correct']}/{row['cases']} | "
                f"{row['cancellation_correct']}/{row['cases']} | "
                f"{row['technical_failures']} |"
            )
    lines += [
        "",
        "| 分组 | Arm | 未确认写 | 重复写 | 额外动作 | 写尝试 | Provider 调用 | 费用 | P50/P95 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("dev", "held-out"):
        for arm in ("baseline", "action_contract"):
            row = summary[split][arm]
            lines.append(
                f"| {split} | {arm} | {row['unconfirmed_writes']} | "
                f"{row['duplicate_writes']} | {row['extra_actions']} | "
                f"{row['write_attempts']} | {row['provider_calls']} | "
                f"${row['cost_usd']:.6f} | {row['p50_ms']:.0f}/{row['p95_ms']:.0f} ms |"
            )
    for split in ("dev", "held-out"):
        paired = summary[split]["paired_action_vs_baseline"]
        lines += [
            "",
            (
                f"{split} 配对（action_contract 对 baseline）："
                f"`{paired['wins']}胜/{paired['losses']}负/{paired['ties']}平`。"
            ),
        ]
    lines += [
        "",
        "## 口径",
        "",
        "- 只有结构化 `write_previews` / `action_artifacts` 计为预览、取消或确认；模型文本不算任务完成。",
        "- 未确认写为客户端确认之前的 fake sink 落库数；字段为零只表示本样本未观察到写入，不等同安全率 100%。",
        "- 确认阶段使用同一 `confirm_key` 连续提交两次，验证响应丢失重放；同键只允许一个 fake sink 行。",
        "- 所有 provider 调用经过独立 SQLite 账本，调用前原子预留，实际 usage 结算；失败和重试均计费。",
        "- 延迟为端到端图执行时间，不包含外部确认后的固有限制；无缓存重放。",
        "",
        "## 未验证",
        "",
        "- 样本为模拟客服任务，不代表真实客服 ROI 或线上成功率。",
        "- HTTP API 未在本分支接线，运行时契约通过独立 harness 验证；集成补丁见 `dev-notes/action-contract-r3.md`。",
        "- 业务事实仅用隔离 fixture，未写共享 MySQL/Milvus。",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rescore_existing(source_dir: Path) -> dict:
    """Reclassify an existing frozen run without model calls; preserve raw results."""
    source_dir = Path(source_dir)
    original_summary = json.loads(
        (source_dir / "summary.json").read_text(encoding="utf-8")
    )
    rows = [
        json.loads(line)
        for line in (source_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    scenarios = load_scenarios()
    by_id = {item["id"]: item for item in scenarios}
    for row in rows:
        if "confirmation_replay_count" not in row:
            # v1 stored same-key response-loss retries in this field. They are
            # expected idempotency evidence, not duplicate committed writes.
            row["confirmation_replay_count"] = int(row.get("duplicate_write_count") or 0)
            row["duplicate_write_count"] = 0
        trace_errors = [
            str(item.get("detail") or "")
            for item in row.get("trace") or []
            if item.get("event") == "task_failed"
        ]
        infrastructure, workflow = _error_classification([
            *(row.get("technical_errors") or []), *trace_errors,
        ])
        row["technical_errors"] = infrastructure
        row["workflow_errors"] = workflow
        row["technical_error"] = bool(infrastructure)
        row["score"] = score_case(by_id[row["scenario_id"]], row)
    summary = summarize(scenarios, rows, original_summary["budget"])
    summary["rescore_only"] = True
    summary["paid_calls_during_rescore"] = 0
    summary["source_config_sha256"] = original_summary.get("config_sha256")
    summary["rescore_code_sha256"] = config_sha256()
    summary["original_scoring_version"] = original_summary.get("scoring_version")
    summary["rescored_scoring_version"] = SCORING_VERSION
    (source_dir / "results-rescored.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in rows),
        encoding="utf-8",
    )
    _write_json(source_dir / "summary-rescored.json", summary)
    return summary


async def run_experiment(*, allow_paid: bool, env_file: str = "",
                         split: str = "all", run_dir: Path | None = None,
                         ledger_path: Path | None = None,
                         write_markdown: bool = True) -> dict:
    if not allow_paid:
        raise BudgetExceeded("paid run requires --allow-paid")
    target_dir = Path(run_dir or RUN_DIR)
    target_ledger = Path(ledger_path or LEDGER_PATH)
    if target_dir.exists() or target_ledger.exists():
        raise BudgetExceeded("live run directory or ledger already exists; refusing overwrite")
    settings = Settings(
        _env_file=env_file or "/root/workplace/ShopHelper/.env",
        openai_model=MODEL,
        openai_base_url=BASE_URL,
        chat_temperature=0,
        extract_temperature=0,
        llm_max_retries=0,
        max_output_tokens=500,
        structured_output_mode="auto",
    )
    if settings.openai_model != MODEL:
        raise BudgetExceeded("model must remain deepseek-flash")
    pricing = PricingTable.load(PRICING_PATH)
    budget = ConcurrentActionBudget(
        pricing, path=target_ledger, config_hash=config_sha256(),
        arm_caps={"baseline": 0.95, "action_contract": 0.95},
        max_generations=500, allow_paid=True,
    )
    raw = ChatOpenAI(
        model=MODEL, api_key=settings.openai_api_key, base_url=BASE_URL,
        temperature=0, max_tokens=500, max_retries=0, timeout=45,
        stream_usage=True, extra_body={"thinking": {"type": "disabled"}},
    )
    scenarios = load_scenarios()
    if split != "all":
        scenarios = [item for item in scenarios if item["split"] == split]
    results = []
    for scenario in scenarios:
        for arm in ("baseline", "action_contract"):
            result = await run_case(
                scenario, arm=arm, model=raw, settings=settings, budget=budget,
            )
            results.append(result)
    summary = summarize(scenarios, results, budget.snapshot())
    target_dir.mkdir(parents=True, exist_ok=False)
    _write_json(target_dir / "manifest.json", {
        "scenario_sha256": manifest_sha256(),
        "config_sha256": config_sha256(),
        "model": MODEL,
        "scoring_version": SCORING_VERSION,
        "split_counts": {
            "dev": sum(item["split"] == "dev" for item in scenarios),
            "held-out": sum(item["split"] == "held-out" for item in scenarios),
        },
        "max_retries": MAX_RETRIES,
        "run_cap_usd": RUN_CAP_USD,
    })
    (target_dir / "results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in results),
        encoding="utf-8",
    )
    _write_json(target_dir / "summary.json", summary)
    if write_markdown:
        write_report(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-paid", action="store_true")
    parser.add_argument("--env-file", default="")
    parser.add_argument("--split", choices=("all", "dev", "held-out"), default="all")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--rescore-existing", type=Path)
    args = parser.parse_args()
    if args.rescore_existing is not None:
        summary = rescore_existing(args.rescore_existing)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    summary = asyncio.run(run_experiment(
        allow_paid=args.allow_paid, env_file=args.env_file, split=args.split,
        run_dir=args.run_dir, ledger_path=args.ledger,
        write_markdown=args.run_dir is None,
    ))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
