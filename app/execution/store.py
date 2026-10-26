"""持久 run/事件/会话租约存储；所有方法同步，调用方按需放入线程池。"""
from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from app.execution.models import ChatRun, ChatRunEvent, ChatSessionLease, ChatToolAttempt
from app.execution.locks import write_lock_for


class RequestConflict(Exception):
    """同一 client_request_id 携带了不同请求。"""


class SessionBusy(Exception):
    """同一会话已有有效执行租约。"""

    def __init__(self, run_id: str) -> None:
        super().__init__(f"session busy: {run_id}")
        self.run_id = run_id


class UnsafeResume(Exception):
    """旧 run 已进入写工具，结果未知时不能自动重放。"""


class ResumeConflict(Exception):
    """run 当前状态不允许显式恢复。"""


class LeaseLost(Exception):
    """当前执行 owner 已失去 run 的租约，必须停止写状态。"""


@dataclass(frozen=True)
class ReservedRun:
    run_id: str
    created: bool
    status: str


@dataclass(frozen=True)
class PersistedEvent:
    id: int
    seq: int
    event: dict


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _request_hash(payload: dict) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ExecutionStore:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._factory = session_factory

    @contextmanager
    def _write_session(self):
        with write_lock_for(self._factory):
            with self._factory() as session:
                yield session

    def reserve_run(self, *, conversation_id: int, client_request_id: str | None,
                    request_payload: dict, owner_token: str,
                    lease_seconds: int) -> ReservedRun:
        request_hash = _request_hash(request_payload)
        now = _utcnow()
        with self._write_session() as session:
            existing = self._find_by_client_request(
                session, conversation_id, client_request_id)
            if existing is not None:
                if existing.request_hash != request_hash:
                    raise RequestConflict("client_request_id payload mismatch")
                return ReservedRun(existing.run_id, False, existing.status)

            lease = session.scalar(
                select(ChatSessionLease)
                .where(ChatSessionLease.conversation_id == conversation_id)
                .with_for_update()
            )
            if lease is not None and lease.lease_expires_at > now:
                raise SessionBusy(lease.run_id)
            if lease is not None:
                self._mark_lease_owner_interrupted(session, lease.run_id, now)

            run_id = str(uuid4())
            run = ChatRun(
                run_id=run_id,
                conversation_id=conversation_id,
                client_request_id=client_request_id,
                request_hash=request_hash,
                request_json=request_payload,
                status="queued",
                owner_token=owner_token,
            )
            session.add(run)
            if lease is None:
                session.add(ChatSessionLease(
                    conversation_id=conversation_id,
                    run_id=run_id,
                    owner_token=owner_token,
                    lease_expires_at=now + timedelta(seconds=lease_seconds),
                ))
            else:
                lease.run_id = run_id
                lease.owner_token = owner_token
                lease.lease_expires_at = now + timedelta(seconds=lease_seconds)
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
                existing = self._find_by_client_request(
                    session, conversation_id, client_request_id)
                if existing is not None:
                    if existing.request_hash != request_hash:
                        raise RequestConflict("client_request_id payload mismatch")
                    return ReservedRun(existing.run_id, False, existing.status)
                active = session.get(ChatSessionLease, conversation_id)
                if active is not None:
                    raise SessionBusy(active.run_id)
                raise
            return ReservedRun(run_id, True, run.status)

    @staticmethod
    def _find_by_client_request(session, conversation_id: int,
                                client_request_id: str | None) -> ChatRun | None:
        if client_request_id is None:
            return None
        return session.scalar(
            select(ChatRun).where(
                ChatRun.conversation_id == conversation_id,
                ChatRun.client_request_id == client_request_id,
            )
        )

    @staticmethod
    def _mark_lease_owner_interrupted(session, run_id: str, now: datetime) -> None:
        run = session.get(ChatRun, run_id)
        if run is None or run.status not in {"queued", "running", "cancelling"}:
            return
        if run.write_state in {"started", "submitted"}:
            run.write_state = "unknown"
            attempts = session.scalars(
                select(ChatToolAttempt).where(
                    ChatToolAttempt.run_id == run_id,
                    ChatToolAttempt.status.in_(("started", "submitted")),
                )
            ).all()
            for attempt in attempts:
                attempt.status = "unknown"
                attempt.finished_at = now
        resume_safe = run.write_state == "none"
        run.status = "interrupted"
        run.error_message = (
            "lease_expired" if resume_safe else "lease_expired_write_outcome_unknown"
        )
        run.finished_at = now
        run.last_event_seq += 1
        session.add(ChatRunEvent(
            run_id=run_id,
            seq=run.last_event_seq,
            event_type="interrupted",
            payload={
                "type": "interrupted",
                "reason": "lease_expired" if resume_safe else "write_outcome_unknown",
                "resume_safe": resume_safe,
            },
        ))

    def recover_expired_lease(self, conversation_id: int) -> bool:
        now = _utcnow()
        with self._write_session() as session:
            lease = session.scalar(
                select(ChatSessionLease)
                .where(ChatSessionLease.conversation_id == conversation_id)
                .with_for_update()
            )
            if lease is None or lease.lease_expires_at > now:
                return False
            self._mark_lease_owner_interrupted(session, lease.run_id, now)
            session.delete(lease)
            session.commit()
            return True

    def resume_run(self, run_id: str, *, owner_token: str, lease_seconds: int) -> ReservedRun:
        now = _utcnow()
        with self._write_session() as session:
            run = session.scalar(
                select(ChatRun).where(ChatRun.run_id == run_id).with_for_update())
            if run is None:
                raise KeyError(run_id)
            if run.status != "interrupted":
                raise ResumeConflict(run.status)
            if run.write_state != "none":
                raise UnsafeResume(run.write_state)
            lease = session.get(ChatSessionLease, run.conversation_id)
            if lease is not None and lease.run_id != run_id and lease.lease_expires_at > now:
                raise SessionBusy(lease.run_id)
            if lease is None:
                session.add(ChatSessionLease(
                    conversation_id=run.conversation_id,
                    run_id=run_id,
                    owner_token=owner_token,
                    lease_expires_at=now + timedelta(seconds=lease_seconds),
                ))
            else:
                lease.run_id = run_id
                lease.owner_token = owner_token
                lease.lease_expires_at = now + timedelta(seconds=lease_seconds)
            run.status = "queued"
            run.attempt += 1
            run.owner_token = owner_token
            run.error_message = None
            run.finished_at = None
            run.last_event_seq += 1
            session.add(ChatRunEvent(
                run_id=run_id,
                seq=run.last_event_seq,
                event_type="resumed",
                payload={"type": "resumed", "run_id": run_id, "attempt": run.attempt},
            ))
            session.commit()
            return ReservedRun(run_id, True, run.status)

    def renew_lease(self, run_id: str, owner_token: str, *, lease_seconds: int) -> bool:
        now = _utcnow()
        with self._write_session() as session:
            lease = session.scalar(
                select(ChatSessionLease).where(
                    ChatSessionLease.run_id == run_id,
                    ChatSessionLease.owner_token == owner_token,
                ).with_for_update()
            )
            if lease is None:
                return False
            lease.lease_expires_at = now + timedelta(seconds=lease_seconds)
            session.commit()
            return True

    def lease_owned_by(self, run_id: str, owner_token: str) -> bool:
        now = _utcnow()
        with self._factory() as session:
            lease = session.scalar(
                select(ChatSessionLease).where(
                    ChatSessionLease.run_id == run_id,
                    ChatSessionLease.owner_token == owner_token,
                )
            )
            return lease is not None and lease.lease_expires_at > now

    def mark_running(self, run_id: str, owner_token: str) -> None:
        with self._write_session() as session:
            run = session.scalar(
                select(ChatRun).where(ChatRun.run_id == run_id).with_for_update())
            if run is None:
                raise KeyError(run_id)
            lease = session.scalar(
                select(ChatSessionLease).where(
                    ChatSessionLease.run_id == run_id,
                    ChatSessionLease.owner_token == owner_token,
                ).with_for_update()
            )
            if lease is None or run.owner_token != owner_token:
                raise LeaseLost("run owner changed before start")
            run.status = "running"
            if run.started_at is None:
                run.started_at = _utcnow()
            session.commit()

    def append_event(self, run_id: str, event: dict, *,
                     owner_token: str | None = None) -> int:
        with self._write_session() as session:
            run = session.scalar(
                select(ChatRun).where(ChatRun.run_id == run_id).with_for_update())
            if run is None:
                raise KeyError(run_id)
            if owner_token is not None:
                lease = session.scalar(
                    select(ChatSessionLease).where(
                        ChatSessionLease.run_id == run_id,
                        ChatSessionLease.owner_token == owner_token,
                    ).with_for_update()
                )
                if (run.owner_token != owner_token or lease is None
                        or lease.lease_expires_at <= _utcnow()):
                    raise LeaseLost("run owner changed")
            run.last_event_seq += 1
            row = ChatRunEvent(
                run_id=run_id,
                seq=run.last_event_seq,
                event_type=str(event.get("type") or "message")[:64],
                payload=dict(event),
            )
            session.add(row)
            session.commit()
            return int(row.id)

    def start_tool_attempt(self, run_id: str, *, tool_call_id: str, tool_name: str,
                           access: str, arguments: dict | None,
                           idempotency_key: str | None,
                           executes: bool = True) -> None:
        with self._write_session() as session:
            run = session.scalar(
                select(ChatRun).where(ChatRun.run_id == run_id).with_for_update())
            if run is None:
                raise KeyError(run_id)
            session.add(ChatToolAttempt(
                run_id=run_id,
                tool_call_id=tool_call_id[:64],
                tool_name=tool_name[:128],
                access=access,
                status="started",
                idempotency_key=idempotency_key,
                arguments=arguments,
            ))
            if access == "write" and executes:
                run.write_state = "started"
            session.commit()

    def finish_tool_attempt(self, run_id: str, tool_call_id: str, *, status: str,
                            result_text: str | None = None) -> None:
        now = _utcnow()
        with self._write_session() as session:
            attempt = session.scalar(
                select(ChatToolAttempt).where(
                    ChatToolAttempt.run_id == run_id,
                    ChatToolAttempt.tool_call_id == tool_call_id,
                ).with_for_update()
            )
            if attempt is None:
                raise KeyError((run_id, tool_call_id))
            attempt.status = status
            attempt.result_text = result_text
            attempt.finished_at = now
            if attempt.access == "write":
                run = session.get(ChatRun, run_id)
                if run is not None and run.write_state in {"started", "submitted"}:
                    run.write_state = "succeeded" if status == "succeeded" else "unknown"
            session.commit()

    def read_events(self, run_id: str, after_id: int = 0) -> list[PersistedEvent]:
        with self._write_session() as session:
            rows = session.scalars(
                select(ChatRunEvent)
                .where(ChatRunEvent.run_id == run_id, ChatRunEvent.id > after_id)
                .order_by(ChatRunEvent.id)
            ).all()
            return [
                PersistedEvent(id=int(row.id), seq=int(row.seq), event=dict(row.payload))
                for row in rows
            ]

    def finish_run(self, run_id: str, *, status: str, result: dict | None = None,
                   error: str | None = None, usage: dict | None = None,
                   duration_ms: int = 0,
                   owner_token: str | None = None) -> None:
        usage = usage or {}
        now = _utcnow()
        with self._write_session() as session:
            run = session.scalar(
                select(ChatRun).where(ChatRun.run_id == run_id).with_for_update())
            if run is None:
                raise KeyError(run_id)
            if owner_token is not None:
                lease = session.scalar(
                    select(ChatSessionLease).where(
                        ChatSessionLease.run_id == run_id,
                        ChatSessionLease.owner_token == owner_token,
                    ).with_for_update()
                )
                if (run.owner_token != owner_token or lease is None
                        or lease.lease_expires_at <= now):
                    raise LeaseLost("run owner changed")
            run.status = status
            run.result_json = result
            run.error_message = error
            run.input_tokens = int(usage.get("input_tokens", 0) or 0)
            run.output_tokens = int(usage.get("output_tokens", 0) or 0)
            run.total_tokens = int(usage.get("total_tokens", 0) or 0)
            run.duration_ms = int(duration_ms)
            run.finished_at = now
            session.execute(delete(ChatSessionLease).where(
                ChatSessionLease.run_id == run_id))
            session.commit()

    def get_run(self, run_id: str) -> dict:
        with self._factory() as session:
            run = session.get(ChatRun, run_id)
            if run is None:
                raise KeyError(run_id)
            return {
                "run_id": run.run_id,
                "conversation_id": int(run.conversation_id),
                "client_request_id": run.client_request_id,
                "status": run.status,
                "request": run.request_json,
                "result": run.result_json,
                "error": run.error_message,
                "write_state": run.write_state,
                "attempt": int(run.attempt),
                "last_event_seq": int(run.last_event_seq),
                "usage": {
                    "input_tokens": int(run.input_tokens),
                    "output_tokens": int(run.output_tokens),
                    "total_tokens": int(run.total_tokens),
                },
                "duration_ms": int(run.duration_ms),
                "created_at": run.created_at.isoformat() if run.created_at else None,
                "started_at": run.started_at.isoformat() if run.started_at else None,
                "finished_at": run.finished_at.isoformat() if run.finished_at else None,
            }
