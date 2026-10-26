"""Serializable artifacts for user-visible actions.

Only structured artifacts count as previews, confirmations, or cancellations.
Model prose is deliberately not an executable action.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


ActionIntentKind = Literal["none", "ticket_preview", "ticket_confirm", "ticket_cancel"]
ActionArtifactState = Literal[
    "preview", "awaiting_confirmation", "confirmed", "cancelled", "failed",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class NormalizationRecord(BaseModel):
    field: str
    original: Any
    normalized: Any
    reason: str


class ActionArgumentDecision(BaseModel):
    ok: bool
    arguments: dict[str, Any] = Field(default_factory=dict)
    original_arguments: dict[str, Any] = Field(default_factory=dict)
    normalizations: list[NormalizationRecord] = Field(default_factory=list)
    feedback: str = ""
    needs_clarification: bool = False


class ActionIntent(BaseModel):
    kind: ActionIntentKind = "none"
    source: Literal["current_turn", "pending_state"] = "current_turn"
    reason: str = ""
    explicit: bool = False


class ActionArtifact(BaseModel):
    action_id: str = Field(default_factory=lambda: uuid4().hex)
    kind: ActionIntentKind
    state: ActionArtifactState
    conversation_id: int | None = None
    tool_name: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)
    original_arguments: dict[str, Any] = Field(default_factory=dict)
    normalizations: list[NormalizationRecord] = Field(default_factory=list)
    confirm_key: str | None = None
    preview: dict[str, Any] | None = None
    write_committed: bool = False
    ticket_no: str | None = None
    task_id: str | None = None
    agent_id: str | None = None
    error_code: str = ""
    error: str = ""
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)


class ActionContractPolicy(BaseModel):
    max_repairs: int = Field(default=1, ge=0, le=1)
    allow_model_confirmation: Literal[False] = False
    allow_model_confirm_key: Literal[False] = False


class CompletionVerdict(BaseModel):
    satisfied: bool
    code: str
    feedback: str = ""
    needs_clarification: bool = False
