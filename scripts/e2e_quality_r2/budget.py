"""Concurrent-safe $2 protection ledger and exact-call response cache."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


class BudgetExceeded(RuntimeError):
    """A provider call would exceed the independent protection cap."""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _atomic_json_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


class R2Budget:
    def __init__(self, path: Path, *, limit_usd: float = 2.0) -> None:
        if float(limit_usd) > 2.0:
            raise BudgetExceeded("E2E R2 protection cap must not exceed $2.00")
        self.path = Path(path)
        self.limit_usd = float(limit_usd)
        if not self.path.exists():
            _atomic_json_write(self.path, {
                "schema_version": 1,
                "limit_usd": self.limit_usd,
                "records": [],
                "pending": {},
            })

    def _load(self) -> dict:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BudgetExceeded(f"budget ledger unreadable; refusing paid work: {exc}") from exc
        if float(payload.get("limit_usd", -1)) != self.limit_usd:
            raise BudgetExceeded("budget ledger cap changed; start a new run ledger")
        return payload

    @staticmethod
    def _settled(payload: dict) -> float:
        return sum(float(row.get("actual_usd") or 0.0)
                   for row in payload.get("records") or [])

    @staticmethod
    def _pending(payload: dict) -> float:
        return sum(float(row.get("estimated_usd") or 0.0)
                   for row in (payload.get("pending") or {}).values())

    def reserve(self, context: dict, estimated_usd: float) -> str:
        estimate = max(float(estimated_usd), 0.0)
        with _locked(self.path):
            payload = self._load()
            committed = self._settled(payload) + self._pending(payload)
            if committed + estimate > self.limit_usd + 1e-12:
                raise BudgetExceeded(
                    f"budget reservation would exceed cap: "
                    f"{committed + estimate:.8f} > {self.limit_usd:.2f}"
                )
            attempt_id = uuid.uuid4().hex
            payload.setdefault("pending", {})[attempt_id] = {
                "attempt_id": attempt_id,
                "estimated_usd": estimate,
                "context": dict(context),
                "reserved_at": _utcnow(),
            }
            _atomic_json_write(self.path, payload)
            return attempt_id

    def settle(
        self,
        attempt_id: str,
        *,
        actual_usd: float,
        status: str,
        usage: dict | None = None,
        error: str | None = None,
    ) -> None:
        actual = max(float(actual_usd), 0.0)
        with _locked(self.path):
            payload = self._load()
            pending = (payload.get("pending") or {}).pop(attempt_id, None)
            if pending is None:
                raise BudgetExceeded(f"unknown or already settled attempt: {attempt_id}")
            if self._settled(payload) + actual > self.limit_usd + 1e-12:
                raise BudgetExceeded("actual provider usage exceeded the protection cap")
            payload.setdefault("records", []).append({
                "attempt_id": attempt_id,
                "estimated_usd": float(pending.get("estimated_usd") or 0.0),
                "actual_usd": actual,
                "context": pending.get("context") or {},
                "status": status,
                "usage": usage or {},
                "error": error,
                "reserved_at": pending.get("reserved_at"),
                "settled_at": _utcnow(),
            })
            _atomic_json_write(self.path, payload)

    def used_usd(self) -> float:
        with _locked(self.path):
            return self._settled(self._load())

    def pending_usd(self) -> float:
        with _locked(self.path):
            return self._pending(self._load())

    def committed_usd(self) -> float:
        with _locked(self.path):
            payload = self._load()
            return self._settled(payload) + self._pending(payload)

    def records(self) -> list[dict]:
        with _locked(self.path):
            return list(self._load().get("records") or [])

    def assert_no_pending(self) -> None:
        pending = self.pending_usd()
        if pending:
            raise BudgetExceeded(f"prior run left pending provider reservations: ${pending:.8f}")

    def summary(self) -> dict:
        with _locked(self.path):
            payload = self._load()
            return {
                "limit_usd": self.limit_usd,
                "settled_usd": round(self._settled(payload), 8),
                "pending_usd": round(self._pending(payload), 8),
                "calls": len(payload.get("records") or []),
            }


class ExactCallCache:
    """Exact request cache. Keys include namespace, split/turn and every schema input."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            _atomic_json_write(self.path, {"schema_version": 1, "entries": {}})

    @staticmethod
    def _stable(value) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def key(
        self,
        *,
        namespace: str,
        model: str,
        request,
        schema=None,
        tools=None,
        split: str,
        turn: int,
    ) -> str:
        payload = {
            "namespace": namespace,
            "model": model,
            "request": request,
            "schema": schema,
            "tools": tools,
            "split": split,
            "turn": int(turn),
        }
        return hashlib.sha256(self._stable(payload).encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict | None:
        with _locked(self.path):
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"exact call cache unreadable: {exc}") from exc
            row = (payload.get("entries") or {}).get(key)
            return dict(row) if row is not None else None

    def put(self, key: str, payload: dict) -> None:
        with _locked(self.path):
            try:
                current = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"exact call cache unreadable: {exc}") from exc
            current.setdefault("entries", {})[key] = payload
            _atomic_json_write(self.path, current)
