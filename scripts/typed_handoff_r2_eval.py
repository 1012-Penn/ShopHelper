"""Isolated paired evaluation for raw vs typed multi-agent handoffs.

The typed branch and the raw branch use the same model, planner prompt, tool
fixtures and limits. Only ``handoff_mode`` changes.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import inspect
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from langchain.tools import tool
from langchain_openai import ChatOpenAI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.evidence_contracts import call_cache_key
from app.multi_agent.models import MultiAgentLimits
from app.multi_agent.runtime import MultiAgentRunner, build_multi_agent_graph
from app.tools.base import ToolRecord, ToolRegistryV2
from app.tools.engine import ToolEngine
from scripts.ch15_paid_guard import (
    ArmLimit,
    _extract_usage,
    paid_call_context,
)
from scripts.eval_cost import PricingTable
from scripts.eval_multi_agent import (
    ConcurrentPaidRunBudget,
    _message_record,
    _tool_contract,
)
from tests.eval.typed_handoff_r2.scoring import (
    KNOWN_ORDER_IDS,
    SCORING_VERSION,
    score_case,
)

EVAL_ROOT = ROOT / "tests" / "eval" / "typed_handoff_r2"
SCENARIO_PATH = EVAL_ROOT / "scenarios.jsonl"
SCORING_PATH = EVAL_ROOT / "scoring.py"
CONFIG_PATH = EVAL_ROOT / "config.json"
ORDERS_PATH = EVAL_ROOT / "fixtures" / "orders.json"
POLICIES_PATH = EVAL_ROOT / "fixtures" / "policies.json"
RESULTS_ROOT = EVAL_ROOT / "runs" / "live-20261027"
LEDGER_PATH = EVAL_ROOT / "runs" / "typed-handoff-r2-paid.sqlite3"
ROOT_ENV = Path("/root/workplace/ShopHelper/.env")
MODEL = "deepseek-flash"
BASE_URL = "https://api.deepseek.com"
RUN_CAP_USD = 2.0


def model_visible_turn(scenario: dict[str, Any], turn_index: int) -> dict[str, str]:
    """Only user-visible input crosses this boundary."""

    return {"message": str(scenario["turns"][turn_index])}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_scenarios(path: Path = SCENARIO_PATH) -> list[dict[str, Any]]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("duplicate scenario id")
    group_splits: dict[str, set[str]] = {}
    for row in rows:
        group_splits.setdefault(row["group"], set()).add(row["split"])
    if any(len(values) != 1 for values in group_splits.values()):
        raise ValueError("business/source group crosses dev and held-out splits")
    if sum(row["split"] == "dev" for row in rows) != 10:
        raise ValueError("dev split must contain exactly 10 scenarios")
    if sum(row["split"] == "held-out" for row in rows) != 20:
        raise ValueError("held-out split must contain exactly 20 scenarios")
    for row in rows:
        for index in range(len(row["turns"])):
            if set(model_visible_turn(row, index)) != {"message"}:
                raise ValueError("gold leaked into model projection")
    return rows


def result_manifest() -> dict[str, Any]:
    files = {
        "scenarios": SCENARIO_PATH,
        "scoring": SCORING_PATH,
        "config": CONFIG_PATH,
        "orders": ORDERS_PATH,
        "policies": POLICIES_PATH,
        "runtime": ROOT / "app" / "multi_agent" / "runtime.py",
        "models": ROOT / "app" / "multi_agent" / "models.py",
        "contracts": ROOT / "app" / "evidence_contracts" / "models.py",
        "ledger": ROOT / "app" / "evidence_contracts" / "ledger.py",
        "cache": ROOT / "app" / "evidence_contracts" / "cache.py",
        "extraction": ROOT / "app" / "evidence_contracts" / "extraction.py",
    }
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scenario_count": 30,
        "split": {"dev": 10, "held-out": 20},
        "scoring_version": SCORING_VERSION,
        "sha256": {name: _sha(path) for name, path in files.items()},
        "model": MODEL,
        "arms": ["raw", "typed"],
        "only_semantic_variable": "MultiAgentRunner.handoff_mode",
        "run_cap_usd": RUN_CAP_USD,
        "stopping_rule": "one complete paired run; preserve every failed attempt",
    }


def _pricing() -> PricingTable:
    return PricingTable({
        "verified": True,
        "source_url": "https://api-docs.deepseek.com/quick_start/pricing/",
        "version": "2026-10-27",
        "currency": "USD",
        "models": {
            MODEL: {
                "input_per_million": 0.30,
                "input_cache_hit_per_million": 0.006,
                "output_per_million": 1.20,
            },
        },
    })


class TypedHandoffBudget(ConcurrentPaidRunBudget):
    """Allows the pre-agreed two bounded harness retries without hiding failed cost."""

    def complete_generation(self, reservation_id, *, usage=None, response=None,
                            error_type=None, duration_ms=None) -> None:
        if reservation_id is None or not error_type:
            super().complete_generation(
                reservation_id, usage=usage, response=response,
                error_type=error_type, duration_ms=duration_ms,
            )
            return
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT status, reserved_usd FROM ch15_paid_calls "
                "WHERE call_id=? AND experiment_id=?",
                (reservation_id, self.experiment_id),
            ).fetchone()
            if row is None or row[0] != "pending":
                db.rollback()
                raise RuntimeError("budget reservation is missing or already settled")
            db.execute(
                "UPDATE ch15_paid_calls SET status='failed_attempt', charged_usd=?, "
                "error_type=?, response_json=?, finished_at=?, duration_ms=? "
                "WHERE call_id=?",
                (
                    float(row[1]), str(error_type)[:160],
                    json.dumps(_message_record(response), ensure_ascii=False)
                    if response is not None else None,
                    datetime.now(timezone.utc).isoformat(),
                    int(duration_ms or 0), reservation_id,
                ),
            )
            db.commit()


class BudgetedChatModel:
    """Exact payload guard with at most two 429-only retries."""

    def __init__(self, inner: Any, budget: Any, arm: str, call_kind: str, *,
                 tool_schema: Any = None, structured_result: bool = False,
                 max_retries: int = 2,
                 retry_sleep: Callable[[int], Awaitable[None]] | None = None) -> None:
        self.inner = inner
        self.budget = budget
        self.arm = arm
        self.call_kind = call_kind
        self.tool_schema = tool_schema
        self.structured_result = structured_result
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep or _async_sleep

    def _copy(self, inner: Any, **overrides) -> "BudgetedChatModel":
        return BudgetedChatModel(
            inner, self.budget, self.arm, self.call_kind,
            tool_schema=overrides.get("tool_schema", self.tool_schema),
            structured_result=overrides.get(
                "structured_result", self.structured_result),
            max_retries=self.max_retries, retry_sleep=self.retry_sleep,
        )

    def _reserve(self, value: Any) -> str:
        request = {"messages": _message_record(value)}
        if self.tool_schema is not None:
            request["tools"] = _tool_contract(self.tool_schema)
        encoded = json.dumps(
            request, ensure_ascii=False, sort_keys=True, default=str)
        context = _call_context().get()
        return self.budget.begin_generation(
            self.arm, request_bytes=len(encoded.encode("utf-8")) + 512,
            scenario_id=str(context.get("scenario_id") or "offline"),
            turn=int(context.get("turn") or 1), call_kind=self.call_kind,
            request=request, max_output_tokens=self.budget.max_output_tokens,
        )

    def _details(self, result: Any):
        if not self.structured_result:
            return result, _extract_usage(result), result, None
        if not isinstance(result, dict):
            return result, _extract_usage(result), result, None
        raw = result.get("raw")
        parsed = result.get("parsed")
        return parsed, _extract_usage(raw) or _extract_usage(result), result, \
            result.get("parsing_error")

    async def ainvoke(self, value: Any, *args, **kwargs):
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            reservation = self._reserve(value)
            started = time.perf_counter()
            try:
                result = await self.inner.ainvoke(value, *args, **kwargs)
            except Exception as exc:
                last_error = exc
                self.budget.complete_generation(
                    reservation, error_type=type(exc).__name__,
                    duration_ms=int((time.perf_counter() - started) * 1000),
                )
                if _is_429(exc) and attempt < self.max_retries:
                    delay = self.retry_sleep(attempt + 1)
                    if inspect.isawaitable(delay):
                        await delay
                    continue
                raise
            returned, usage, response, parsing_error = self._details(result)
            self.budget.complete_generation(
                reservation, usage=usage, response=response,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )
            if parsing_error is not None:
                raise parsing_error
            return returned
        raise last_error or RuntimeError("model call failed")

    def bind_tools(self, tools, **kwargs):
        return self._copy(
            self.inner.bind_tools(tools, **kwargs),
            tool_schema=[self.tool_schema, [_tool_contract(item) for item in tools]],
        )

    def bind(self, *args, **kwargs):
        return self._copy(self.inner.bind(*args, **kwargs))

    def with_structured_output(self, *args, **kwargs):
        schema = args[0] if args else kwargs.get("schema")
        options = dict(kwargs)
        options["include_raw"] = True
        return self._copy(
            self.inner.with_structured_output(*args, **options),
            tool_schema=[item for item in (self.tool_schema, schema) if item is not None],
            structured_result=True,
        )


async def _async_sleep(attempt: int) -> None:
    await asyncio.sleep(float(attempt))


def _is_429(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    return status == 429 or "429" in str(exc) or "rate limit" in str(exc).casefold()


def _call_context():
    from scripts.ch15_paid_guard import _CALL_CONTEXT

    return _CALL_CONTEXT


def _load_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    if not ROOT_ENV.exists():
        raise RuntimeError("OPENAI_API_KEY is not set and root .env is missing")
    from dotenv import dotenv_values

    values = dotenv_values(ROOT_ENV)
    key = str(values.get("OPENAI_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is unavailable")
    return key


def _orders() -> dict[str, dict[str, Any]]:
    return json.loads(ORDERS_PATH.read_text(encoding="utf-8"))


def _policies() -> list[dict[str, Any]]:
    return list(json.loads(POLICIES_PATH.read_text(encoding="utf-8"))["items"])


class FakeSink:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.by_key: dict[str, dict[str, Any]] = {}

    def __call__(self, conversation_id: int, description: str, ticket_type: str,
                 confirm_key: str | None = None) -> dict[str, Any]:
        if confirm_key and confirm_key in self.by_key:
            return dict(self.by_key[confirm_key])
        row = {
            "ticket_no": f"TH-{len(self.rows) + 1}",
            "conversation_id": conversation_id,
            "description": description,
            "ticket_type": ticket_type,
            "confirm_key": confirm_key,
        }
        self.rows.append(row)
        if confirm_key:
            self.by_key[confirm_key] = row
        return dict(row)


class MemoryAudit:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def log(self, **kwargs) -> None:
        self.rows.append(kwargs)


def _registry(sink: FakeSink) -> ToolRegistryV2:
    orders = _orders()
    policies = _policies()

    @tool
    def query_order(order_id: str) -> str:
        """Query an isolated simulated order by exact order ID."""
        row = dict(orders.get(str(order_id), {"error": "order_not_found"}))
        if "error" not in row:
            row["valid_until"] = (
                datetime.now(timezone.utc) + timedelta(seconds=300)
            ).isoformat()
        return json.dumps(row, ensure_ascii=False)

    @tool
    def query_faq(keyword: str) -> str:
        """Retrieve isolated simulated policy evidence for a keyword."""
        terms = str(keyword or "").casefold()
        scored = []
        for item in policies:
            score = sum(
                2 for match in item.get("match_any", [])
                if str(match).casefold() in terms
            )
            if score:
                scored.append((score, item))
        scored.sort(key=lambda pair: (-pair[0], str(pair[1]["chunk_id"])))
        items = []
        for _, item in scored[:4]:
            items.append({key: value for key, value in item.items() if key != "match_any"})
        return json.dumps({
            "items": items,
            "low_confidence": not items,
            "keyword": keyword,
        }, ensure_ascii=False)

    @tool
    def create_ticket(conversation_id: int, description: str, ticket_type: str,
                      confirm_key: str | None = None) -> str:
        """Preview or create a ticket in the isolated simulated sink."""
        return json.dumps(
            sink(conversation_id, description, ticket_type, confirm_key),
            ensure_ascii=False,
        )

    registry = ToolRegistryV2()
    registry.register(ToolRecord(query_order, source="builtin", access="read"))
    registry.register(ToolRecord(query_faq, source="builtin", access="read"))
    registry.register(ToolRecord(create_ticket, source="builtin", access="write"))
    return registry


class RecordingEngine:
    def __init__(self, inner: ToolEngine) -> None:
        self.inner = inner
        self.registry = inner.registry
        self.calls: list[dict[str, Any]] = []
        self.confirmations: list[dict[str, Any]] = []

    async def execute(self, name: str, args_json: str, **kwargs):
        result = await self.inner.execute(name, args_json, **kwargs)
        self.calls.append({
            "name": name,
            "arguments": json.loads(args_json or "{}"),
            "result": result,
        })
        return result

    async def execute_confirmed(self, name: str, args_json: str, **kwargs):
        result = await self.inner.execute_confirmed(name, args_json, **kwargs)
        self.confirmations.append({
            "name": name,
            "arguments": json.loads(args_json or "{}"),
            "result": result,
        })
        return result


def _action_visible_claims(model_attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    visible = []
    seen = set()
    for attempt in model_attempts:
        if attempt.get("call_kind") != "action":
            continue
        for message in (attempt.get("request") or {}).get("messages") or []:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if not isinstance(content, str) or not content.lstrip().startswith("{"):
                continue
            try:
                payload = json.loads(content)
            except json.JSONDecodeError:
                continue
            for claim in payload.get("evidence_claims") or []:
                claim_id = claim.get("claim_id")
                if claim_id and claim_id not in seen:
                    seen.add(claim_id)
                    visible.append(claim)
    return visible


async def _confirm_writes(engine: RecordingEngine, sink: FakeSink,
                          previews: list[dict], write_mode: str) -> tuple[int, int]:
    if write_mode != "confirm_once" or not previews:
        return 0, len(sink.rows)
    preview = previews[-1]
    args = dict(preview.get("arguments") or {})
    if not preview.get("confirm_key"):
        return 0, len(sink.rows)
    args["confirm_key"] = preview["confirm_key"]
    encoded = json.dumps(args, ensure_ascii=False)
    before = len(sink.rows)
    await engine.execute_confirmed(
        "create_ticket", encoded, conversation_id=args.get("conversation_id"))
    await engine.execute_confirmed(
        "create_ticket", encoded, conversation_id=args.get("conversation_id"))
    confirmed = max(0, len(sink.rows) - before)
    return confirmed, len(sink.rows)


async def run_case(scenario: dict[str, Any], *, arm: str, model: Any,
                   budget: TypedHandoffBudget) -> dict[str, Any]:
    sink = FakeSink()
    audit = MemoryAudit()
    engine = RecordingEngine(ToolEngine(
        _registry(sink), audit, timeout_seconds=10, retry_attempts=0))
    roles = {
        name: BudgetedChatModel(model, budget, arm, name)
        for name in ("planner", "retrieval", "action", "supervisor")
    }
    runner = MultiAgentRunner(
        planner_model=roles["planner"], retrieval_model=roles["retrieval"],
        action_model=roles["action"], supervisor_model=roles["supervisor"],
        engine=engine,
        limits=MultiAgentLimits(
            max_seconds=45, child_timeout_seconds=20, max_steps=12,
            max_tokens=16000),
        handoff_mode=arm,
    )
    graph = build_multi_agent_graph(runner)
    history: list[dict[str, str]] = []
    replies: list[str] = []
    states: list[dict[str, Any]] = []
    run_id = f"th-{scenario['id']}-{arm}"
    started = time.perf_counter()
    error = ""
    try:
        for turn_index, visible in enumerate(scenario["turns"], start=1):
            with paid_call_context(
                    scenario_id=scenario["id"], arm=arm,
                    turn=turn_index, run_id=f"{run_id}:turn-{turn_index}"):
                state = await graph.ainvoke({
                    "run_id": f"{run_id}:turn-{turn_index}",
                    "task_id": f"{scenario['id']}:turn-{turn_index}",
                    "user_message": visible,
                    "conversation_history": list(history),
                    "facts": {}, "evidence": {}, "errors": [],
                    "stop_reasons": [], "trace": [], "role_histories": {},
                })
            states.append(state)
            reply = str(state.get("final_reply") or "")
            replies.append(reply)
            history.extend([
                {"role": "user", "content": visible},
                {"role": "assistant", "content": reply},
            ])
    except Exception as exc:
        error = f"{type(exc).__name__}:{exc}"

    previews = list(states[-1].get("write_previews", [])) if states else []
    confirmed, total_rows = await _confirm_writes(
        engine, sink, previews, scenario["gold"]["write_mode"])
    duration_ms = int((time.perf_counter() - started) * 1000)
    attempts = budget.calls_for_case(scenario["id"], arm)
    action_claims = _action_visible_claims(attempts)
    tool_calls = [
        {"name": call["name"], "arguments": call["arguments"]}
        for call in engine.calls
    ]
    requested_calls = [
        call for state in states for call in state.get("tool_calls", [])
    ]
    cache_stats = (
        states[-1].get("tool_cache_stats", {}) if states else {}
    )
    result = {
        "scenario_id": scenario["id"],
        "split": scenario["split"],
        "group": scenario["group"],
        "category": scenario["category"],
        "arm": arm,
        "reply": replies[-1] if replies else "",
        "replies": replies,
        "disposition": states[-1].get("disposition", "") if states else "",
        "tool_calls": tool_calls,
        "requested_tool_calls": requested_calls,
        "duplicate_tool_calls": _duplicate_tool_call_count(requested_calls),
        "write_previews": previews,
        "confirmed_write_count": confirmed,
        "unconfirmed_write_count": max(0, total_rows - confirmed),
        "action_visible_claims": action_claims,
        "evidence_claims": [
            claim for state in states for claim in state.get("evidence_claims", [])
        ],
        "handoffs": [item for state in states for item in state.get("handoffs", [])],
        "tool_cache_stats": cache_stats,
        "duration_ms": duration_ms,
        "model_attempts": attempts,
        "errors": [error] if error else [
            item for state in states for item in state.get("errors", [])
        ],
        "fanout": {
            "attempted_tool_calls": len(requested_calls),
            "executed_tool_calls": len(engine.calls),
            "deduplicated_tool_calls": int(cache_stats.get("deduplicated", 0)),
        },
    }
    result["score"] = score_case(scenario, result)
    if error:
        result["score"]["task_success"] = False
    return result


def _duplicate_tool_call_count(calls: list[dict[str, Any]]) -> int:
    seen = set()
    duplicates = 0
    for call in calls:
        key = call_cache_key(
            str(call.get("name") or ""),
            dict(call.get("arguments") or {}),
        )
        duplicates += int(key in seen)
        seen.add(key)
    return duplicates


def _budget() -> TypedHandoffBudget:
    return TypedHandoffBudget(
        _pricing(), path=LEDGER_PATH,
        experiment_id="typed-handoff-r2-20261027-v1",
        config_hash=result_manifest()["sha256"]["config"],
        model=MODEL,
        arm_limits={
            "raw": ArmLimit(max_generations=180, cap_usd=0.98),
            "typed": ArmLimit(max_generations=180, cap_usd=0.98),
        },
        run_cap_usd=RUN_CAP_USD, max_input_tokens=16000,
        max_output_tokens=500, allow_paid=True,
    )


def _quantile(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def summarize(results: list[dict[str, Any]],
              budget_snapshot: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "model": MODEL,
        "scoring_version": SCORING_VERSION,
        "scenario_count": 30,
        "manifest": result_manifest(),
        "budget": budget_snapshot,
        "arms": {}, "splits": {}, "paired": {}, "groups": {},
    }
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        by_arm.setdefault(result["arm"], []).append(result)

    scenarios = {row["id"]: row for row in load_scenarios()}

    def rates(rows: list[dict[str, Any]], *, arm: str) -> dict[str, Any]:
        attempts = [item for row in rows for item in row.get("model_attempts", [])]
        successes = sum(bool(row["score"]["task_success"]) for row in rows)
        input_tokens = sum(int(item.get("input_tokens") or 0) for item in attempts)
        output_tokens = sum(int(item.get("output_tokens") or 0) for item in attempts)
        cost = sum(float(item.get("charged_usd") or 0.0) for item in attempts)
        durations = [int(row["duration_ms"]) for row in rows]
        attempt_durations = [int(item.get("duration_ms") or 0) for item in attempts]
        handoff_cases = sum(
            bool(scenarios[row["scenario_id"]]["gold"].get(
                "required_handoff_predicates"))
            for row in rows
        )
        return {
            "n": len(rows),
            "success": successes,
            "task_success_rate": round(successes / len(rows), 4) if rows else None,
            "entity_error_count": sum(row["score"]["entity_error_count"] for row in rows),
            "condition_error_count": sum(
                row["score"]["condition_error_count"] for row in rows),
            "source_missing_count": sum(
                row["score"]["source_missing_count"] for row in rows),
            "handoff_omission_count": (
                sum(row["score"]["handoff_omission_count"] for row in rows)
                if arm == "typed" else None
            ),
            "handoff_case_count": handoff_cases,
            "duplicate_tool_calls": sum(row["duplicate_tool_calls"] for row in rows),
            "attempted_tool_calls": sum(
                row["fanout"]["attempted_tool_calls"] for row in rows),
            "executed_tool_calls": sum(
                row["fanout"]["executed_tool_calls"] for row in rows),
            "cache_deduplicated_tool_calls": sum(
                int(row["tool_cache_stats"].get("deduplicated", 0))
                for row in rows),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "model_calls": len(attempts),
            "cost_usd": round(cost, 8),
            "cost_per_success_usd": round(cost / successes, 8) if successes else None,
            "p50_case_ms": _quantile(durations, 0.50),
            "p95_case_ms": _quantile(durations, 0.95),
            "p95_model_attempt_ms": _quantile(attempt_durations, 0.95),
            "provider_failures": sum(
                bool(item.get("error_type")) for item in attempts),
        }

    for arm, rows in by_arm.items():
        summary["arms"][arm] = rates(rows, arm=arm)
    for split in ("dev", "held-out"):
        summary["splits"][split] = {
            arm: rates([row for row in rows if row["split"] == split], arm=arm)
            for arm, rows in by_arm.items()
        }
    for group in sorted({row["group"] for row in results}):
        summary["groups"][group] = {
            arm: rates([row for row in rows if row["group"] == group], arm=arm)
            for arm, rows in by_arm.items()
        }

    by_pair = {(row["scenario_id"], row["arm"]): row for row in results}
    for split in ("dev", "held-out", "all"):
        wins = losses = ties = 0
        ids = {
            row["scenario_id"] for row in results
            if split == "all" or row["split"] == split
        }
        for scenario_id in ids:
            raw = by_pair.get((scenario_id, "raw"))
            typed = by_pair.get((scenario_id, "typed"))
            if raw is None or typed is None:
                continue
            raw_ok = raw["score"]["task_success"]
            typed_ok = typed["score"]["task_success"]
            wins += int(typed_ok and not raw_ok)
            losses += int(raw_ok and not typed_ok)
            ties += int(raw_ok == typed_ok)
        summary["paired"][split] = {"typed_wins": wins, "typed_losses": losses,
                                    "ties": ties}
    return summary


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_gzip_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Typed Handoff R2 结果",
        "",
        f"模型：`{summary['model']}`；题集：10 dev + 20 held-out；"
        "两臂只改变 `handoff_mode`。",
        "",
        "| split | arm | success | entity errors | condition errors | "
        "source missing | duplicate calls | input tokens | cost/success | P95 ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("dev", "held-out"):
        for arm in ("raw", "typed"):
            row = summary["splits"][split][arm]
            lines.append(
                f"| {split} | {arm} | {row['success']}/{row['n']} | "
                f"{row['entity_error_count']} | {row['condition_error_count']} | "
                f"{row['source_missing_count']} | {row['duplicate_tool_calls']} | "
                f"{row['input_tokens']} | "
                f"{row['cost_per_success_usd']} | {row['p95_case_ms']} |"
            )
    lines.extend([
        "",
        "配对结果（typed 对 raw）：",
        "",
        "| split | W | L | T |",
        "|---|---:|---:|---:|",
    ])
    for split, row in summary["paired"].items():
        lines.append(
            f"| {split} | {row['typed_wins']} | {row['typed_losses']} | {row['ties']} |")
    lines.extend([
        "",
        "费用账本上限："
        f"${summary['budget']['run_cap_usd']:.2f}；"
        f"已结算/预留上限 ${summary['budget']['charged_upper_bound_usd']:.6f}。",
        "",
        "P95 是按完整案例耗时统计的真实 API 路径；未包含缓存重放冒充在线延迟。",
        "",
    ])
    return "\n".join(lines)


async def run_live(*, allow_paid: bool) -> dict[str, Any]:
    if not allow_paid:
        raise RuntimeError("live run requires explicit --allow-paid")
    if RESULTS_ROOT.exists() and any(RESULTS_ROOT.iterdir()):
        raise RuntimeError(f"results already exist; refuse to overwrite: {RESULTS_ROOT}")
    if LEDGER_PATH.exists():
        raise RuntimeError(
            "isolated ledger already exists; this experiment allows one paid run")
    scenarios = load_scenarios()
    manifest = result_manifest()
    RESULTS_ROOT.mkdir(parents=True, exist_ok=False)
    (RESULTS_ROOT / "run-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    model = ChatOpenAI(
        model=MODEL, api_key=_load_api_key(), base_url=BASE_URL,
        temperature=0, max_tokens=500, max_retries=0, timeout=45,
        stream_usage=True, extra_body={"thinking": {"type": "disabled"}},
    )
    budget = _budget()
    results = []
    for index, scenario in enumerate(scenarios, start=1):
        for arm in ("raw", "typed"):
            result = await run_case(
                scenario, arm=arm, model=model, budget=budget)
            results.append(result)
            print(
                f"[{index:02d}/30][{arm}] {scenario['id']} "
                f"success={result['score']['task_success']} "
                f"cost={sum(float(item.get('charged_usd') or 0) for item in result['model_attempts']):.6f}",
                flush=True,
            )
    snapshot = budget.snapshot()
    summary = summarize(results, snapshot)
    _write_jsonl(RESULTS_ROOT / "results.jsonl", results)
    _write_gzip_jsonl(RESULTS_ROOT / "traces.jsonl.gz", results)
    (RESULTS_ROOT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULTS_ROOT / "summary.md").write_text(
        render_report(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-paid", action="store_true")
    args = parser.parse_args()
    summary = asyncio.run(run_live(allow_paid=args.allow_paid))
    print(json.dumps({
        "paired": summary["paired"],
        "budget": summary["budget"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
