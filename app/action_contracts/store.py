"""Conversation-scoped pending action ledger used by the optional runtime."""

from __future__ import annotations

from datetime import datetime, timezone
from threading import RLock
from typing import Any

from app.action_contracts.models import ActionArtifact


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ActionSessionLedger:
    """One pending action per conversation; history remains observable."""

    def __init__(self) -> None:
        self._pending: dict[int | None, ActionArtifact] = {}
        self._events: list[ActionArtifact] = []
        self._lock = RLock()

    def pending(self, conversation_id: int | None) -> ActionArtifact | None:
        with self._lock:
            item = self._pending.get(conversation_id)
            return item.model_copy(deep=True) if item is not None else None

    def latest_for_confirm_key(self, confirm_key: str) -> ActionArtifact | None:
        """Return the newest observable artifact for a server-issued key."""
        if not confirm_key:
            return None
        with self._lock:
            for item in reversed(self._events):
                if item.confirm_key == confirm_key:
                    return item.model_copy(deep=True)
        return None

    def put_pending(self, artifact: ActionArtifact) -> list[ActionArtifact]:
        """A revised preview supersedes, but never confirms, the prior preview."""
        with self._lock:
            events: list[ActionArtifact] = []
            previous = self._pending.get(artifact.conversation_id)
            if previous is not None:
                cancelled = previous.model_copy(update={
                    "state": "cancelled",
                    "error_code": "superseded_by_revision",
                    "error": "用户修改了描述，旧预览已作废且未提交。",
                    "updated_at": _now(),
                })
                events.append(cancelled.model_copy(deep=True))
                self._events.append(cancelled)
            self._pending[artifact.conversation_id] = artifact.model_copy(deep=True)
            self._events.append(artifact.model_copy(deep=True))
            events.append(artifact.model_copy(deep=True))
            return events

    def cancel(self, conversation_id: int | None) -> list[ActionArtifact]:
        with self._lock:
            previous = self._pending.pop(conversation_id, None)
            if previous is None:
                artifact = ActionArtifact(
                    kind="ticket_cancel",
                    state="cancelled",
                    conversation_id=conversation_id,
                    error_code="no_pending_action",
                    error="当前没有待确认的工单预览。",
                )
            else:
                artifact = previous.model_copy(update={
                    "state": "cancelled",
                    "error_code": "",
                    "error": "",
                    "updated_at": _now(),
                })
            self._events.append(artifact.model_copy(deep=True))
            return [artifact]

    def no_action(self, conversation_id: int | None) -> ActionArtifact:
        artifact = ActionArtifact(
            kind="none",
            state="cancelled",
            conversation_id=conversation_id,
            error_code="no_pending_action",
            error="本轮不是写入动作；未生成工单预览。",
        )
        with self._lock:
            self._events.append(artifact.model_copy(deep=True))
        return artifact

    def mark_confirmed(
        self,
        *,
        conversation_id: int | None,
        confirm_key: str,
        ticket: dict[str, Any],
    ) -> ActionArtifact:
        """Called only after the client-confirmed tool engine returns."""
        with self._lock:
            pending = self._pending.get(conversation_id)
            if pending is None or pending.confirm_key != confirm_key:
                raise ValueError("confirmation key does not match a pending preview")
            confirmed = pending.model_copy(update={
                "state": "confirmed",
                "write_committed": True,
                "ticket_no": str(ticket.get("ticket_no") or ""),
                "updated_at": _now(),
            })
            self._pending.pop(conversation_id, None)
            self._events.append(confirmed.model_copy(deep=True))
            return confirmed

    def events(self) -> list[ActionArtifact]:
        with self._lock:
            return [item.model_copy(deep=True) for item in self._events]
