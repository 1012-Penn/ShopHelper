"""Independent, fail-closed provider budget for answer-plan R3."""
from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from app.history import estimate_tokens


class BudgetExceeded(RuntimeError):
    pass


class AnswerPlanBudgetLedger:
    def __init__(self, path: str | Path, *, limit_usd: float = 2.0) -> None:
        if not 0 < float(limit_usd) <= 2.0:
            raise ValueError("answer-plan R3 protection cap must be in (0, 2.0] USD")
        self.path = str(path)
        self.limit_usd = float(limit_usd)
        self._lock = threading.RLock()
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS calls (
                    reservation_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL DEFAULT '',
                    split TEXT NOT NULL DEFAULT '',
                    arm TEXT NOT NULL DEFAULT '',
                    kind TEXT NOT NULL DEFAULT '',
                    model TEXT NOT NULL,
                    estimated_usd REAL NOT NULL,
                    actual_usd REAL,
                    status TEXT NOT NULL,
                    usage_json TEXT NOT NULL DEFAULT '{}',
                    error TEXT,
                    latency_ms REAL,
                    attempts INTEGER NOT NULL DEFAULT 1,
                    reserved_at TEXT NOT NULL,
                    settled_at TEXT
                )
                """
            )

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=5, check_same_thread=False)
        connection.execute("PRAGMA busy_timeout = 5000")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _committed(self, connection: sqlite3.Connection) -> float:
        row = connection.execute(
            """
            SELECT COALESCE(SUM(
                CASE WHEN status = 'pending' THEN estimated_usd ELSE actual_usd END
            ), 0.0) FROM calls
            """
        ).fetchone()
        return float(row[0] or 0.0)

    def reserve(
        self,
        estimated_usd: float,
        *,
        model: str,
        kind: str,
        case_id: str = "",
        split: str = "",
        arm: str = "",
    ) -> str:
        amount = max(0.0, float(estimated_usd))
        reservation_id = uuid.uuid4().hex
        with self._lock, self._connection() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                committed = self._committed(connection)
                if committed + amount > self.limit_usd + 1e-12:
                    raise BudgetExceeded(
                        f"answer-plan budget exceeded: "
                        f"{committed + amount:.8f} > {self.limit_usd:.2f}"
                    )
                connection.execute(
                    """
                    INSERT INTO calls (
                        reservation_id, case_id, split, arm, kind, model,
                        estimated_usd, status, reserved_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)
                    """,
                    (
                        reservation_id,
                        case_id,
                        split,
                        arm,
                        kind,
                        model,
                        amount,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        return reservation_id

    def settle(
        self,
        reservation_id: str,
        *,
        actual_usd: float,
        usage: dict,
        status: str = "settled",
        error: str | None = None,
        latency_ms: float | None = None,
        attempts: int = 1,
    ) -> None:
        amount = max(0.0, float(actual_usd))
        with self._lock, self._connection() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT estimated_usd, status FROM calls WHERE reservation_id = ?",
                    (reservation_id,),
                ).fetchone()
                if row is None:
                    raise KeyError(f"unknown reservation: {reservation_id}")
                if row[1] != "pending":
                    raise RuntimeError(f"reservation already settled: {reservation_id}")
                connection.execute(
                    """
                    UPDATE calls SET actual_usd = ?, status = ?, usage_json = ?,
                        error = ?, latency_ms = ?, attempts = ?, settled_at = ?
                    WHERE reservation_id = ?
                    """,
                    (
                        amount,
                        status,
                        __import__("json").dumps(
                            usage or {}, ensure_ascii=False, sort_keys=True
                        ),
                        error,
                        latency_ms,
                        int(attempts),
                        datetime.now(timezone.utc).isoformat(),
                        reservation_id,
                    ),
                )
                total = self._committed(connection)
                if total > self.limit_usd + 1e-12:
                    raise BudgetExceeded(
                        f"actual usage exceeded answer-plan cap: "
                        f"{total:.8f} > {self.limit_usd:.2f}"
                    )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def committed_usd(self) -> float:
        with self._lock, self._connection() as connection:
            return self._committed(connection)

    def pending_usd(self) -> float:
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT COALESCE(SUM(estimated_usd), 0.0) "
                "FROM calls WHERE status = 'pending'"
            ).fetchone()
            return float(row[0] or 0.0)

    def assert_no_pending(self) -> None:
        pending = self.pending_usd()
        if pending:
            raise BudgetExceeded(
                f"prior answer-plan run left pending reservations: ${pending:.8f}"
            )

    def records(self) -> list[dict]:
        import json

        with self._lock, self._connection() as connection:
            rows = connection.execute(
                """
                SELECT reservation_id, case_id, split, arm, kind, model,
                       estimated_usd, actual_usd, status, usage_json, error,
                       latency_ms, attempts, reserved_at, settled_at
                FROM calls ORDER BY reserved_at, reservation_id
                """
            ).fetchall()
        return [
            {
                "reservation_id": row[0],
                "case_id": row[1],
                "split": row[2],
                "arm": row[3],
                "kind": row[4],
                "model": row[5],
                "estimated_usd": row[6],
                "actual_usd": row[7],
                "status": row[8],
                "usage": json.loads(row[9] or "{}"),
                "error": row[10],
                "latency_ms": row[11],
                "attempts": row[12],
                "reserved_at": row[13],
                "settled_at": row[14],
            }
            for row in rows
        ]


def _usage(response) -> tuple[int, int, int]:
    usage = getattr(response, "usage_metadata", None) or {}
    if not usage:
        metadata = getattr(response, "response_metadata", None) or {}
        usage = metadata.get("token_usage") or metadata.get("usage") or {}
    input_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
    details = usage.get("input_token_details") or {}
    cached = int(
        details.get("cache_read")
        or details.get("cached_tokens")
        or usage.get("prompt_cache_hit_tokens")
        or 0
    )
    return input_tokens, output_tokens, cached


class BudgetedModel:
    """Count every provider attempt before and after it runs."""

    def __init__(
        self,
        inner,
        *,
        model: str,
        pricing,
        ledger: AnswerPlanBudgetLedger,
        max_tokens: int = 1200,
        max_retries: int = 2,
        backoff_seconds: float = 0.5,
    ) -> None:
        self.inner = inner
        self.model = model
        self.pricing = pricing
        self.ledger = ledger
        self.max_tokens = max(1, int(max_tokens))
        self.max_retries = max(0, min(2, int(max_retries)))
        self.backoff_seconds = max(0.0, float(backoff_seconds))
        self._current_context = {"case_id": "", "split": "", "arm": "", "kind": "llm"}
        self.calls: list[dict] = []

    @contextmanager
    def context(self, **values):
        previous = dict(self._current_context)
        self._current_context.update(values)
        try:
            yield self
        finally:
            self._current_context = previous

    def invoke(self, request, config=None, **kwargs):
        input_tokens = max(1, estimate_tokens(str(request)))
        estimate = self.pricing.cost(self.model, input_tokens, self.max_tokens)
        last_error = None
        for attempt in range(self.max_retries + 1):
            context = dict(self._current_context)
            reservation_id = self.ledger.reserve(
                estimate,
                model=self.model,
                kind=str(context.get("kind") or "llm"),
                case_id=str(context.get("case_id") or ""),
                split=str(context.get("split") or ""),
                arm=str(context.get("arm") or ""),
            )
            started = time.perf_counter()
            try:
                response = self.inner.invoke(request, config=config, **kwargs)
            except BaseException as exc:
                latency = (time.perf_counter() - started) * 1000
                self.ledger.settle(
                    reservation_id,
                    actual_usd=estimate,
                    usage={},
                    status="provider_error",
                    error=f"{type(exc).__name__}: {exc}"[:500],
                    latency_ms=latency,
                    attempts=attempt + 1,
                )
                self.calls.append({
                    **context,
                    "status": "provider_error",
                    "attempt": attempt + 1,
                    "cost_usd": estimate,
                    "latency_ms": latency,
                })
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(self.backoff_seconds * (2 ** attempt))
                    continue
                raise
            latency = (time.perf_counter() - started) * 1000
            actual_input, output_tokens, cached = _usage(response)
            usage_missing = actual_input == 0 and output_tokens == 0
            if usage_missing:
                actual = estimate
                usage = {"estimated_only": True, "input_tokens": input_tokens}
            else:
                actual = self.pricing.cost(
                    self.model, actual_input, output_tokens, cached
                )
                usage = {
                    "input_tokens": actual_input,
                    "output_tokens": output_tokens,
                    "cached_input_tokens": cached,
                }
            status = "usage_missing" if usage_missing else "succeeded"
            self.ledger.settle(
                reservation_id,
                actual_usd=actual,
                usage=usage,
                status=status,
                latency_ms=latency,
                attempts=attempt + 1,
            )
            self.calls.append({
                **context,
                "status": status,
                "attempt": attempt + 1,
                "cost_usd": actual,
                "usage": usage,
                "latency_ms": latency,
            })
            return response
        raise last_error or RuntimeError("provider call failed")

    def with_structured_output(self, *_args, **_kwargs):
        raise RuntimeError(
            "BudgetedModel only supports the shared JSON structured-output path"
        )
