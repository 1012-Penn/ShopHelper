"""Independent, fail-closed budget ledger for claim-grounding calls.

Reservations and settlements live in SQLite. ``BEGIN IMMEDIATE`` makes the
limit check atomic across concurrent callers, and a pending reservation keeps
its conservative cost counted until the provider usage is settled.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class BudgetExceeded(RuntimeError):
    pass


class ClaimBudgetLedger:
    def __init__(self, path: str | Path, *, total_limit_usd: float) -> None:
        self.path = str(path)
        if total_limit_usd <= 0:
            raise ValueError("total_limit_usd 必须大于 0")
        self.total_limit_usd = float(total_limit_usd)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self.path,
            timeout=5,
            check_same_thread=False,
        )
        self._connection.execute("PRAGMA busy_timeout = 5000")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS reservations (
                reservation_id TEXT PRIMARY KEY,
                reserved_usd REAL NOT NULL,
                actual_usd REAL,
                state TEXT NOT NULL,
                model TEXT NOT NULL,
                kind TEXT NOT NULL,
                note TEXT NOT NULL DEFAULT '',
                usage_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                settled_at TEXT
            )
            """
        )
        self._connection.commit()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def reserve(
        self, reserved_usd: float, *, model: str, kind: str, note: str = ""
    ) -> str:
        amount = max(0.0, float(reserved_usd))
        reservation_id = uuid.uuid4().hex
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                row = self._connection.execute(
                    """
                    SELECT COALESCE(SUM(
                        CASE WHEN state = 'pending' THEN reserved_usd ELSE actual_usd END
                    ), 0.0)
                    FROM reservations
                    WHERE state IN ('pending', 'settled')
                    """
                ).fetchone()
                committed = float(row[0] or 0.0)
                if committed + amount > self.total_limit_usd + 1e-12:
                    raise BudgetExceeded(
                        f"claim grounding 预算超限: "
                        f"{committed + amount:.6f} > {self.total_limit_usd:.6f}"
                    )
                self._connection.execute(
                    """
                    INSERT INTO reservations (
                        reservation_id, reserved_usd, actual_usd, state,
                        model, kind, note, created_at
                    ) VALUES (?, ?, NULL, 'pending', ?, ?, ?, ?)
                    """,
                    (reservation_id, amount, model, kind, note, self._now()),
                )
                self._connection.commit()
            except BaseException:
                self._connection.rollback()
                raise
        return reservation_id

    def settle(
        self,
        reservation_id: str,
        *,
        actual_usd: float,
        usage: dict[str, Any] | None = None,
        note: str | None = None,
    ) -> None:
        amount = max(0.0, float(actual_usd))
        if amount > self.total_limit_usd + 1e-12:
            raise BudgetExceeded("单次结算金额超过实验保护上限")
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                row = self._connection.execute(
                    "SELECT reserved_usd, state, note FROM reservations "
                    "WHERE reservation_id = ?",
                    (reservation_id,),
                ).fetchone()
                if row is None:
                    raise KeyError(f"未知 reservation: {reservation_id}")
                if row[1] != "pending":
                    raise RuntimeError(f"reservation 已结算: {reservation_id}")
                final_note = row[2] if note is None else note
                self._connection.execute(
                    """
                    UPDATE reservations
                    SET actual_usd = ?, state = 'settled', usage_json = ?,
                        note = ?, settled_at = ?
                    WHERE reservation_id = ?
                    """,
                    (
                        amount,
                        json.dumps(usage or {}, ensure_ascii=False, sort_keys=True),
                        final_note,
                        self._now(),
                        reservation_id,
                    ),
                )
                total = self._connection.execute(
                    """
                    SELECT COALESCE(SUM(
                        CASE WHEN state = 'pending' THEN reserved_usd ELSE actual_usd END
                    ), 0.0)
                    FROM reservations
                    WHERE state IN ('pending', 'settled')
                    """
                ).fetchone()[0]
                if float(total) > self.total_limit_usd + 1e-12:
                    raise BudgetExceeded(
                        f"实际结算后预算超限: {total:.6f} > {self.total_limit_usd:.6f}"
                    )
                self._connection.commit()
            except BaseException:
                self._connection.rollback()
                raise

    def total_usd(self) -> float:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT COALESCE(SUM(
                    CASE WHEN state = 'pending' THEN reserved_usd ELSE actual_usd END
                ), 0.0)
                FROM reservations
                WHERE state IN ('pending', 'settled')
                """
            ).fetchone()
        return float(row[0] or 0.0)

    def records(self) -> list[dict[str, Any]]:
        with self._lock:
            cursor = self._connection.execute(
                """
                SELECT reservation_id, reserved_usd, actual_usd, state, model,
                       kind, note, usage_json, created_at, settled_at
                FROM reservations ORDER BY created_at, reservation_id
                """
            )
            rows = cursor.fetchall()
        return [
            {
                "reservation_id": row[0],
                "reserved_usd": row[1],
                "actual_usd": row[2],
                "state": row[3],
                "model": row[4],
                "kind": row[5],
                "note": row[6],
                "usage": json.loads(row[7] or "{}"),
                "created_at": row[8],
                "settled_at": row[9],
            }
            for row in rows
        ]


def usage_cost(usage: dict[str, Any], pricing: dict[str, float]) -> float:
    input_tokens = max(0, int(usage.get("input_tokens") or 0))
    output_tokens = max(0, int(usage.get("output_tokens") or 0))
    cached = min(
        max(0, int(usage.get("cached_input_tokens") or 0)),
        input_tokens,
    )
    miss = input_tokens - cached
    return (
        miss / 1_000_000 * float(pricing.get("input_per_million") or 0.0)
        + cached
        / 1_000_000
        * float(
            pricing.get("input_cache_hit_per_million")
            or pricing.get("input_per_million")
            or 0.0
        )
        + output_tokens / 1_000_000 * float(pricing.get("output_per_million") or 0.0)
    )
