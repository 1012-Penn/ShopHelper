"""Pre-call caps and durable usage accounting for the CH15 model comparison.

Each accepted generation reserves its worst-case cost at the configured
input/output envelope before the provider is called. The persistent ledger
survives process restarts and treats an unresolved call as a full reservation.
"""
from __future__ import annotations

import contextvars
import hashlib
import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.eval_cost import BudgetExceeded, PricingTable

MAX_INPUT_TOKENS = 65_536
MAX_OUTPUT_TOKENS = 2_000
RUN_CAP_USD = 17.40


@dataclass(frozen=True)
class ArmLimit:
    max_generations: int
    cap_usd: float


ARM_LIMITS = {
    "fixed": ArmLimit(max_generations=378, cap_usd=9.20),
    "autonomous": ArmLimit(max_generations=336, cap_usd=8.20),
}

_CALL_CONTEXT = contextvars.ContextVar("ch15_paid_call_context", default={})


@contextmanager
def paid_call_context(*, scenario_id: str, arm: str, turn: int, run_id: str):
    """Attach stable case metadata to every model call made by the graph."""
    token = _CALL_CONTEXT.set({"scenario_id": scenario_id, "arm": arm,
                               "turn": int(turn), "run_id": run_id})
    try:
        yield
    finally:
        _CALL_CONTEXT.reset(token)


def validate_paid_settings(settings: Any) -> None:
    """Reject settings that invalidate the published call-count envelope."""
    if int(getattr(settings, "llm_max_retries", -1)) != 0:
        raise BudgetExceeded("CH15付费对照要求 llm_max_retries=0，禁止隐藏重试")
    if bool(getattr(settings, "intent_escalation_enabled", True)):
        raise BudgetExceeded("CH15付费对照关闭意图升级，禁止额外模型调用")
    if int(getattr(settings, "max_agent_steps", MAX_OUTPUT_TOKENS)) > 6:
        raise BudgetExceeded("CH15付费对照的 Agent 步数上限为 6")
    if int(getattr(settings, "max_output_tokens", MAX_OUTPUT_TOKENS)) > MAX_OUTPUT_TOKENS:
        raise BudgetExceeded("CH15付费对照的输出 token 上限为 2000")


class PaidRunBudget:
    """In-memory CH15 run/arm envelopes; does not read or alter CH11 ledger.

    One instance must be shared by both serial arms. State is not durable across
    process restarts, so a resumed paid run must reconcile provider spend first.
    """

    def __init__(
        self,
        pricing: PricingTable,
        *,
        model: str,
        arm_limits: dict[str, ArmLimit] = ARM_LIMITS,
        run_cap_usd: float = RUN_CAP_USD,
        max_input_tokens: int = MAX_INPUT_TOKENS,
        max_output_tokens: int = MAX_OUTPUT_TOKENS,
        allow_paid: bool = False,
    ) -> None:
        self.pricing = pricing
        self.model = model
        self.arm_limits = dict(arm_limits)
        self.run_cap_usd = float(run_cap_usd)
        self.max_input_tokens = int(max_input_tokens)
        self.max_output_tokens = int(max_output_tokens)
        self.allow_paid = bool(allow_paid)
        self.max_call_cost_usd = pricing.cost(
            model, self.max_input_tokens, self.max_output_tokens)
        self._generations = {arm: 0 for arm in self.arm_limits}
        self._reserved_usd = {arm: 0.0 for arm in self.arm_limits}

        if self.run_cap_usd <= 0 or self.max_input_tokens <= 0 or self.max_output_tokens <= 0:
            raise ValueError("CH15付费预算和 token 上限必须为正数")
        if sum(limit.cap_usd for limit in self.arm_limits.values()) > self.run_cap_usd:
            raise ValueError("各组费用上限之和超过 CH15 总费用上限")
        for arm, limit in self.arm_limits.items():
            if limit.max_generations <= 0 or limit.cap_usd <= 0:
                raise ValueError(f"{arm} 组调用数和费用上限必须为正数")
            if self.max_call_cost_usd * limit.max_generations > limit.cap_usd + 1e-9:
                raise ValueError(f"{arm} 组硬调用上限无法落入费用封顶")

    def reserve_generation(self, arm: str, *, request_bytes: int) -> float:
        """Reserve one maximum-cost generation before dispatching it upstream."""
        if not self.allow_paid:
            raise BudgetExceeded("付费调用默认关闭；需要显式 allow_paid=True")
        if not self.pricing.verified:
            raise BudgetExceeded("价格未核实，拒绝付费调用")
        if arm not in self.arm_limits:
            raise BudgetExceeded(f"未知 CH15 评测组: {arm}")
        if request_bytes < 0 or request_bytes > self.max_input_tokens:
            raise BudgetExceeded(
                f"请求输入上限: {request_bytes} > {self.max_input_tokens} bytes/token-envelope")

        limit = self.arm_limits[arm]
        next_count = self._generations[arm] + 1
        if next_count > limit.max_generations:
            raise BudgetExceeded(f"{arm} 组调用次数上限 {limit.max_generations} 已达")
        next_arm_cost = self._reserved_usd[arm] + self.max_call_cost_usd
        next_run_cost = sum(self._reserved_usd.values()) + self.max_call_cost_usd
        if next_arm_cost > limit.cap_usd + 1e-9:
            raise BudgetExceeded(f"{arm} 组硬费用上限 ${limit.cap_usd:.2f} 将被突破")
        if next_run_cost > self.run_cap_usd + 1e-9:
            raise BudgetExceeded(f"CH15 总费用上限 ${self.run_cap_usd:.2f} 将被突破")

        self._generations[arm] = next_count
        self._reserved_usd[arm] = next_arm_cost
        return self.max_call_cost_usd

    def begin_generation(self, arm: str, **kwargs) -> None:
        """Compatibility hook used by GuardedChatModel before every provider call."""
        self.reserve_generation(arm, request_bytes=int(kwargs.get("request_bytes", 0)))
        return None

    def complete_generation(self, reservation_id, *, usage=None, response=None,
                            error_type=None, duration_ms=None) -> None:
        """In-memory guard has no durable call record; production runner uses SQLite."""

    def snapshot(self) -> dict:
        return {
            "model": self.model,
            "allow_paid": self.allow_paid,
            "pricing_verified": self.pricing.verified,
            "run_cap_usd": self.run_cap_usd,
            "reserved_usd": round(sum(self._reserved_usd.values()), 8),
            "max_call_cost_usd": round(self.max_call_cost_usd, 10),
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens": self.max_output_tokens,
            "arms": {
                arm: {
                    "generations": self._generations[arm],
                    "max_generations": limit.max_generations,
                    "cap_usd": limit.cap_usd,
                    "reserved_usd": round(self._reserved_usd[arm], 8),
                }
                for arm, limit in self.arm_limits.items()
            },
        }


class PersistentPaidRunBudget(PaidRunBudget):
    """SQLite-backed CH15 cap and per-call ledger, isolated from the CH11 ledger.

    Call reservations are committed before dispatch. A process restart with a
    pending call fails closed; it must be reconciled before the run can continue.
    Completed calls with provider usage are charged from usage tokens. Missing or
    uncertain usage is charged at the full maximum reservation.
    """

    def __init__(self, pricing, *, path: Path, experiment_id: str, config_hash: str,
                 **kwargs) -> None:
        super().__init__(pricing, **kwargs)
        self.path = Path(path)
        self.experiment_id = str(experiment_id)
        self.config_hash = str(config_hash)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS ch15_paid_runs (
                    experiment_id TEXT PRIMARY KEY,
                    config_hash TEXT NOT NULL,
                    model TEXT NOT NULL,
                    run_cap_usd REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    stop_reason TEXT
                );
                CREATE TABLE IF NOT EXISTS ch15_paid_calls (
                    call_id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    scenario_id TEXT NOT NULL,
                    arm TEXT NOT NULL,
                    turn INTEGER NOT NULL,
                    call_kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    request_bytes INTEGER NOT NULL,
                    input_sha256 TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    response_json TEXT,
                    reserved_usd REAL NOT NULL,
                    charged_usd REAL,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    cache_hit_tokens INTEGER,
                    cache_miss_tokens INTEGER,
                    error_type TEXT,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    duration_ms INTEGER,
                    FOREIGN KEY(experiment_id) REFERENCES ch15_paid_runs(experiment_id)
                );
                CREATE INDEX IF NOT EXISTS idx_ch15_paid_calls_case
                    ON ch15_paid_calls(experiment_id, scenario_id, arm, turn);
            """)
            run_columns = {row[1] for row in db.execute(
                "PRAGMA table_info(ch15_paid_runs)").fetchall()}
            if "stop_reason" not in run_columns:
                db.execute("ALTER TABLE ch15_paid_runs ADD COLUMN stop_reason TEXT")
            call_columns = {row[1] for row in db.execute(
                "PRAGMA table_info(ch15_paid_calls)").fetchall()}
            if "duration_ms" not in call_columns:
                db.execute("ALTER TABLE ch15_paid_calls ADD COLUMN duration_ms INTEGER")
            if "cache_hit_tokens" not in call_columns:
                db.execute("ALTER TABLE ch15_paid_calls ADD COLUMN cache_hit_tokens INTEGER")
            if "cache_miss_tokens" not in call_columns:
                db.execute("ALTER TABLE ch15_paid_calls ADD COLUMN cache_miss_tokens INTEGER")
            row = db.execute(
                "SELECT config_hash, model, run_cap_usd FROM ch15_paid_runs WHERE experiment_id=?",
                (self.experiment_id,),
            ).fetchone()
            if row is None:
                db.execute(
                    "INSERT INTO ch15_paid_runs(experiment_id, config_hash, model, run_cap_usd, "
                    "created_at, stop_reason) VALUES (?, ?, ?, ?, ?, NULL)",
                    (self.experiment_id, self.config_hash, self.model, self.run_cap_usd,
                     _utc_now()),
                )
            elif row != (self.config_hash, self.model, self.run_cap_usd):
                raise BudgetExceeded("CH15 既有持久账本的配置/model/总额与本次运行不一致")

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.execute("PRAGMA busy_timeout=30000")
        db.execute("PRAGMA journal_mode=WAL")
        return db

    def close(self) -> None:
        # Connections are scoped per transaction; retained for uniform lifecycle API.
        return None

    def begin_generation(self, arm: str, *, request_bytes: int, scenario_id: str,
                         turn: int, call_kind: str, request: Any,
                         max_output_tokens: int | None = None) -> str:
        if not self.allow_paid:
            raise BudgetExceeded("付费调用默认关闭；需要显式 allow_paid=True")
        if not self.pricing.verified:
            raise BudgetExceeded("价格未核实，拒绝付费调用")
        if arm not in self.arm_limits:
            raise BudgetExceeded(f"未知 CH15 评测组: {arm}")
        if request_bytes < 0 or request_bytes > self.max_input_tokens:
            raise BudgetExceeded(
                f"请求输入上限: {request_bytes} > {self.max_input_tokens} bytes/token-envelope")
        if max_output_tokens is not None and int(max_output_tokens) > self.max_output_tokens:
            raise BudgetExceeded("请求输出 token 上限超过 CH15 冻结值")
        if not scenario_id or turn <= 0 or not call_kind:
            raise BudgetExceeded("CH15 持久费用记录缺少 case/turn/component 元数据")

        serialized = json.dumps(_json_safe(request), ensure_ascii=False, sort_keys=True)
        input_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        context = _CALL_CONTEXT.get()
        run_id = str(context.get("run_id") or self.experiment_id)
        call_id = uuid.uuid4().hex
        now = _utc_now()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            stop = db.execute(
                "SELECT stop_reason FROM ch15_paid_runs WHERE experiment_id=?",
                (self.experiment_id,),
            ).fetchone()
            if stop and stop[0]:
                db.rollback()
                raise BudgetExceeded(f"CH15 付费运行已因 provider 错误停止: {stop[0]}")
            pending = db.execute(
                "SELECT call_id FROM ch15_paid_calls WHERE experiment_id=? AND status='pending' LIMIT 1",
                (self.experiment_id,),
            ).fetchone()
            if pending:
                db.rollback()
                raise BudgetExceeded(
                    f"CH15 账本存在未结算调用 {pending[0]}；先核实 provider usage 再恢复")
            rows = db.execute(
                "SELECT arm, status, reserved_usd, charged_usd FROM ch15_paid_calls "
                "WHERE experiment_id=?",
                (self.experiment_id,),
            ).fetchall()
            generations = sum(row[0] == arm for row in rows)
            arm_spend = sum((row[2] if row[1] == "pending" else (row[3] or 0.0))
                            for row in rows if row[0] == arm)
            total_spend = sum((row[2] if row[1] == "pending" else (row[3] or 0.0))
                              for row in rows)
            limit = self.arm_limits[arm]
            if generations + 1 > limit.max_generations:
                db.rollback()
                raise BudgetExceeded(f"{arm} 组调用次数上限 {limit.max_generations} 已达")
            if arm_spend + self.max_call_cost_usd > limit.cap_usd + 1e-9:
                db.rollback()
                raise BudgetExceeded(f"{arm} 组硬费用上限 ${limit.cap_usd:.2f} 将被突破")
            if total_spend + self.max_call_cost_usd > self.run_cap_usd + 1e-9:
                db.rollback()
                raise BudgetExceeded(f"CH15 总费用上限 ${self.run_cap_usd:.2f} 将被突破")
            db.execute(
                "INSERT INTO ch15_paid_calls(call_id, experiment_id, run_id, scenario_id, arm, "
                "turn, call_kind, status, request_bytes, input_sha256, request_json, reserved_usd, "
                "started_at) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?)",
                (call_id, self.experiment_id, run_id, scenario_id, arm, int(turn), call_kind,
                 request_bytes, input_hash, serialized, self.max_call_cost_usd, now),
            )
            db.commit()
        return call_id

    def complete_generation(self, reservation_id, *, usage=None, response=None,
                            error_type=None, duration_ms=None) -> None:
        if reservation_id is None:
            return
        input_tokens = _usage_int(usage, "input_tokens") if usage is not None else None
        output_tokens = _usage_int(usage, "output_tokens") if usage is not None else None
        cache_hit_tokens = _usage_int(usage, "cache_hit_tokens") if usage is not None else None
        cache_miss_tokens = _usage_int(usage, "cache_miss_tokens") if usage is not None else None
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT status, reserved_usd FROM ch15_paid_calls WHERE call_id=? AND experiment_id=?",
                (reservation_id, self.experiment_id),
            ).fetchone()
            if row is None or row[0] != "pending":
                db.rollback()
                raise BudgetExceeded("CH15 调用记录不存在或已经结算")
            reserved = float(row[1])
            if error_type or input_tokens is None or output_tokens is None:
                charged = reserved
                status = "failed_unknown_usage" if error_type else "usage_missing"
            elif cache_hit_tokens is not None and cache_miss_tokens is not None \
                    and cache_hit_tokens + cache_miss_tokens == input_tokens:
                entry = self.pricing.models.get(self.model, {})
                if cache_hit_tokens and entry.get("input_cache_hit_per_million") is None:
                    charged = self.pricing.cost(self.model, input_tokens, output_tokens)
                    status = "complete_upper_bound"
                else:
                    hit_price = float(entry.get("input_cache_hit_per_million")
                                      or entry.get("input_per_million") or 0.0)
                    miss_price = float(entry.get("input_per_million") or 0.0)
                    output_price = float(entry.get("output_per_million") or 0.0)
                    charged = ((cache_hit_tokens * hit_price + cache_miss_tokens * miss_price
                                + output_tokens * output_price) / 1_000_000)
                    status = "complete"
            else:
                charged = self.pricing.cost(self.model, input_tokens, output_tokens)
                status = "complete_upper_bound"
            response_json = (json.dumps(_json_safe(response), ensure_ascii=False, sort_keys=True)
                             if response is not None else None)
            db.execute(
                "UPDATE ch15_paid_calls SET status=?, charged_usd=?, input_tokens=?, "
                "output_tokens=?, cache_hit_tokens=?, cache_miss_tokens=?, error_type=?, "
                "response_json=?, finished_at=?, duration_ms=? "
                "WHERE call_id=?",
                (status, charged, input_tokens, output_tokens, cache_hit_tokens, cache_miss_tokens,
                 (str(error_type)[:160] if error_type else None), response_json, _utc_now(),
                 int(duration_ms or 0), reservation_id),
            )
            if error_type:
                db.execute(
                    "UPDATE ch15_paid_runs SET stop_reason=? WHERE experiment_id=? "
                    "AND stop_reason IS NULL",
                    (f"provider_call_failed:{str(error_type)[:120]}", self.experiment_id),
                )
            db.commit()

    def snapshot(self) -> dict:
        with self._connect() as db:
            stop = db.execute(
                "SELECT stop_reason FROM ch15_paid_runs WHERE experiment_id=?",
                (self.experiment_id,),
            ).fetchone()
            rows = db.execute(
                "SELECT arm, status, reserved_usd, charged_usd FROM ch15_paid_calls "
                "WHERE experiment_id=? ORDER BY started_at, call_id",
                (self.experiment_id,),
            ).fetchall()
        arms = {}
        for arm, limit in self.arm_limits.items():
            group = [row for row in rows if row[0] == arm]
            arms[arm] = {
                "generations": len(group),
                "max_generations": limit.max_generations,
                "cap_usd": limit.cap_usd,
                "reserved_usd": round(sum(row[2] for row in group
                                           if row[1] == "pending"), 8),
                "usage_cost_at_pinned_peak_rates_usd": round(sum(
                    row[3] or 0.0 for row in group if row[1] == "complete"), 8),
                "charged_upper_bound_usd": round(sum(
                    row[2] if row[1] == "pending" else (row[3] or 0.0)
                    for row in group), 8),
                "pending_calls": sum(row[1] == "pending" for row in group),
                "upper_bound_or_unknown_cost_calls": sum(row[1] in {
                    "complete_upper_bound", "failed_unknown_usage", "usage_missing"}
                    for row in group),
            }
        total = sum(item["charged_upper_bound_usd"] for item in arms.values())
        return {
            "model": self.model,
            "experiment_id": self.experiment_id,
            "config_hash": self.config_hash,
            "allow_paid": self.allow_paid,
            "pricing_verified": self.pricing.verified,
            "run_cap_usd": self.run_cap_usd,
            "charged_upper_bound_usd": round(total, 8),
            "usage_cost_at_pinned_peak_rates_usd": round(sum(
                row[3] or 0.0 for row in rows if row[1] == "complete"), 8),
            "stop_reason": stop[0] if stop else None,
            "reserved_usd": round(sum(row[2] for row in rows if row[1] == "pending"), 8),
            "max_call_cost_usd": round(self.max_call_cost_usd, 10),
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens": self.max_output_tokens,
            "arms": arms,
        }

    def calls_for_case(self, scenario_id: str, arm: str) -> list[dict]:
        with self._connect() as db:
            db.row_factory = sqlite3.Row
            rows = db.execute(
                "SELECT * FROM ch15_paid_calls WHERE experiment_id=? AND scenario_id=? AND arm=? "
                "ORDER BY started_at, turn, rowid",
                (self.experiment_id, scenario_id, arm),
            ).fetchall()
        output = []
        for row in rows:
            payload = dict(row)
            payload["request"] = json.loads(payload.pop("request_json"))
            raw_response = payload.pop("response_json")
            payload["response"] = json.loads(raw_response) if raw_response else None
            payload["reserved_usd"] = round(float(payload["reserved_usd"]), 10)
            payload["charged_usd"] = round(float(payload["charged_usd"] or 0.0), 10)
            output.append(payload)
        return output


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "args_schema"):
        # LangChain StructuredTool is itself a Pydantic model with callable
        # fields, so its model_json_schema is not provider input. Serialize the
        # exposed tool contract (name/description/argument schema) instead.
        schema = getattr(value, "args_schema", None)
        schema_builder = getattr(schema, "model_json_schema", None)
        if callable(schema_builder):
            schema = schema_builder()
        return {
            "name": getattr(value, "name", ""),
            "description": getattr(value, "description", ""),
            "args_schema": _json_safe(schema),
        }
    schema_builder = getattr(value, "model_json_schema", None)
    if callable(schema_builder):
        return _json_safe(schema_builder())
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _json_safe(model_dump())
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


def _encoded_request_size(value: Any) -> int:
    try:
        encoded = json.dumps(_json_safe(value), ensure_ascii=False).encode("utf-8")
    except (TypeError, ValueError):
        encoded = repr(value).encode("utf-8", errors="replace")
    # Reserve room for provider message framing and tool-call metadata.
    return len(encoded) + 512


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _usage_int(usage: Any, key: str) -> int | None:
    if not isinstance(usage, dict):
        return None
    value = usage.get(key)
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _extract_usage(value: Any) -> dict | None:
    candidates = [getattr(value, "usage_metadata", None),
                  getattr(value, "response_metadata", None)]
    if isinstance(value, dict):
        candidates.extend([value.get("usage_metadata"), value.get("response_metadata"),
                           value.get("usage")])
    for item in candidates:
        if not isinstance(item, dict):
            continue
        input_tokens = item.get("input_tokens", item.get("prompt_tokens"))
        output_tokens = item.get("output_tokens", item.get("completion_tokens"))
        if input_tokens is not None and output_tokens is not None:
            normalized = {"input_tokens": int(input_tokens),
                          "output_tokens": int(output_tokens)}
            hit = item.get("prompt_cache_hit_tokens")
            miss = item.get("prompt_cache_miss_tokens")
            details = item.get("input_token_details") or item.get("prompt_tokens_details") or {}
            if hit is None and isinstance(details, dict):
                hit = details.get("cache_read", details.get("cached_tokens"))
                if hit is None:
                    hit = next((v for k, v in details.items()
                                if k.endswith("cache_read") and v is not None), None)
            if hit is not None:
                normalized["cache_hit_tokens"] = int(hit)
                if miss is None:
                    miss = max(0, int(input_tokens) - int(hit))
            if miss is not None:
                normalized["cache_miss_tokens"] = int(miss)
            return normalized
        token_usage = item.get("token_usage")
        if isinstance(token_usage, dict):
            nested = _extract_usage(token_usage)
            if nested is not None:
                return nested
    return None


class GuardedChatModel:
    """Guard LangChain-style invoke/stream entrypoints used by the CH15 graph."""

    def __init__(self, inner, budget: PaidRunBudget, arm: str, call_kind: str,
                 *, tool_schema: Any = None, structured_result: bool = False) -> None:
        self.inner = inner
        self.budget = budget
        self.arm = arm
        self.call_kind = call_kind
        self.tool_schema = tool_schema
        self.structured_result = structured_result

    def _reserve(self, value: Any):
        request_bytes = _encoded_request_size(value)
        request = {"messages": _json_safe(value)}
        if self.tool_schema is not None:
            request_bytes += max(0, _encoded_request_size(self.tool_schema) - 512)
            request["tools"] = _json_safe(self.tool_schema)
        context = _CALL_CONTEXT.get()
        return self.budget.begin_generation(
            self.arm, request_bytes=request_bytes,
            scenario_id=str(context.get("scenario_id") or "offline"),
            turn=int(context.get("turn") or 1), call_kind=self.call_kind,
            request=request, max_output_tokens=self.budget.max_output_tokens,
        )

    def _finish(self, reservation_id, *, usage=None, response=None, error_type=None,
                duration_ms=None):
        self.budget.complete_generation(reservation_id, usage=usage, response=response,
                                        error_type=error_type, duration_ms=duration_ms)

    def _validate_options(self, options: dict) -> None:
        pending = [options]
        while pending:
            current = pending.pop()
            for key, value in current.items():
                if key in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
                    if value is not None and int(value) > self.budget.max_output_tokens:
                        raise BudgetExceeded(
                            f"请求输出 token 上限: {key}={value} > "
                            f"{self.budget.max_output_tokens}")
                elif isinstance(value, dict):
                    pending.append(value)

    def bind_tools(self, tools, **kwargs):
        self._validate_options(kwargs)
        bound = self.inner.bind_tools(tools, **kwargs)
        schemas = [self.tool_schema, tools]
        return GuardedChatModel(bound, self.budget, self.arm, self.call_kind,
                                tool_schema=[schema for schema in schemas if schema is not None])

    def bind(self, *args, **kwargs):
        self._validate_options(kwargs)
        return GuardedChatModel(self.inner.bind(*args, **kwargs), self.budget,
                                self.arm, self.call_kind, tool_schema=self.tool_schema)

    def with_structured_output(self, *args, **kwargs):
        structured_schema = args[0] if args else kwargs.get("schema")
        schemas = [schema for schema in (self.tool_schema, structured_schema) if schema is not None]
        options = dict(kwargs)
        # LangChain's parsed model often discards AIMessage usage metadata. Keep
        # the raw provider message for ledger accounting, then return only the
        # parsed value to the graph to preserve its existing contract.
        options["include_raw"] = True
        return GuardedChatModel(self.inner.with_structured_output(*args, **options),
                                self.budget, self.arm, self.call_kind,
                                tool_schema=schemas or None, structured_result=True)

    def _result_details(self, result):
        if not self.structured_result or not isinstance(result, dict):
            return result, _extract_usage(result), result, None
        raw = result.get("raw")
        parsed = result.get("parsed")
        error = result.get("parsing_error")
        return parsed, _extract_usage(raw) or _extract_usage(result), result, error

    async def ainvoke(self, value, *args, **kwargs):
        self._validate_options(kwargs)
        reservation = self._reserve(value)
        started = time.perf_counter()
        try:
            result = await self.inner.ainvoke(value, *args, **kwargs)
        except BaseException as exc:
            self._finish(reservation, error_type=type(exc).__name__,
                         duration_ms=int((time.perf_counter() - started) * 1000))
            raise
        if self.structured_result and isinstance(result, dict):
            returned, usage, response, parsing_error = self._result_details(result)
        else:
            returned, usage, response, parsing_error = result, _extract_usage(result), result, None
        self._finish(reservation, usage=usage, response=response,
                     duration_ms=int((time.perf_counter() - started) * 1000))
        if parsing_error is not None:
            raise parsing_error
        return returned

    async def astream(self, value, *args, **kwargs):
        self._validate_options(kwargs)
        reservation = self._reserve(value)
        started = time.perf_counter()
        usage = None
        output = []
        try:
            async for item in self.inner.astream(value, *args, **kwargs):
                usage = _extract_usage(item) or usage
                output.append(_json_safe(item))
                yield item
        except BaseException as exc:
            self._finish(reservation, usage=usage, response=output,
                         error_type=type(exc).__name__,
                         duration_ms=int((time.perf_counter() - started) * 1000))
            raise
        self._finish(reservation, usage=usage, response=output,
                     duration_ms=int((time.perf_counter() - started) * 1000))

    def invoke(self, value, *args, **kwargs):
        self._validate_options(kwargs)
        reservation = self._reserve(value)
        started = time.perf_counter()
        try:
            result = self.inner.invoke(value, *args, **kwargs)
        except BaseException as exc:
            self._finish(reservation, error_type=type(exc).__name__,
                         duration_ms=int((time.perf_counter() - started) * 1000))
            raise
        if self.structured_result and isinstance(result, dict):
            returned, usage, response, parsing_error = self._result_details(result)
        else:
            returned, usage, response, parsing_error = result, _extract_usage(result), result, None
        self._finish(reservation, usage=usage, response=response,
                     duration_ms=int((time.perf_counter() - started) * 1000))
        if parsing_error is not None:
            raise parsing_error
        return returned

    def stream(self, value, *args, **kwargs):
        self._validate_options(kwargs)
        reservation = self._reserve(value)
        started = time.perf_counter()
        usage = None
        output = []
        try:
            for item in self.inner.stream(value, *args, **kwargs):
                usage = _extract_usage(item) or usage
                output.append(_json_safe(item))
                yield item
        except BaseException as exc:
            self._finish(reservation, usage=usage, response=output,
                         error_type=type(exc).__name__,
                         duration_ms=int((time.perf_counter() - started) * 1000))
            raise
        self._finish(reservation, usage=usage, response=output,
                     duration_ms=int((time.perf_counter() - started) * 1000))


def guard_graph_models(*, budget: PaidRunBudget, arm: str, agent, resolver,
                       intent, expander, resolver_schema=None,
                       resolver_options: dict | None = None,
                       expander_schema=None,
                       expander_options: dict | None = None) -> dict[str, GuardedChatModel]:
    """Wrap every LLM generation source injected into the CH15 graph.

    Optional schemas are applied after wrapping so they are included in each
    request envelope and raw provider usage remains available to the ledger.
    Returning a complete mapping prevents an entry or Agent model from escaping
    the guard.
    """
    components = {
        "model": (agent, "agent"),
        "resolver": (resolver, "resolve"),
        "intent_model": (intent, "intent"),
        "expander": (expander, "expand"),
    }
    missing = [name for name, (model, _kind) in components.items() if model is None]
    if missing:
        raise BudgetExceeded("CH15 付费图缺少待守卫模型: " + ", ".join(missing))
    guarded = {
        name: GuardedChatModel(model, budget, arm, kind)
        for name, (model, kind) in components.items()
    }
    if resolver_schema is not None:
        guarded["resolver"] = guarded["resolver"].with_structured_output(
            resolver_schema, **(resolver_options or {}))
    if expander_schema is not None:
        guarded["expander"] = guarded["expander"].with_structured_output(
            expander_schema, **(expander_options or {}))
    return guarded
