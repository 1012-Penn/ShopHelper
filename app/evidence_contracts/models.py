"""Serializable evidence and handoff records.

Claims may only be created from a concrete tool observation. Model drafts are not a
source type, which prevents a polished summary from becoming an asserted fact.
"""

from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


UNKNOWN = "unknown"
Certainty = Literal["verified", "unverified", "unknown"]
HandoffKind = Literal["role_handoff", "human_handoff", "tool_call"]


class EvidenceSource(BaseModel):
    """Exact tool observation retained for audit."""

    tool_name: str = Field(min_length=1, max_length=128)
    arguments: dict[str, Any] = Field(default_factory=dict)
    raw_result: str = Field(min_length=1)
    tool_call_id: str = Field(default=UNKNOWN, max_length=128)
    chunk_id: str = Field(default=UNKNOWN, max_length=128)


class EvidenceClaim(BaseModel):
    claim_id: str = Field(min_length=1, max_length=128)
    run_id: str = Field(min_length=1, max_length=128)
    task_id: str = Field(min_length=1, max_length=128)
    subject_ids: list[str] = Field(default_factory=list)
    predicate: str = Field(min_length=1, max_length=128)
    value: Any
    unit: str = Field(default=UNKNOWN, max_length=64)
    scope: str = Field(default=UNKNOWN, max_length=128)
    policy_version: str = Field(default=UNKNOWN, max_length=64)
    knowledge_as_of: str = Field(default=UNKNOWN, max_length=64)
    observed_at: str = Field(min_length=1)
    valid_from: str = Field(default=UNKNOWN, max_length=64)
    valid_until: str = Field(default=UNKNOWN, max_length=64)
    observed_sequence: int = Field(ge=0)
    certainty: Certainty = "unknown"
    source: EvidenceSource
    superseded_by: str | None = None
    superseded_at: str | None = None

    @model_validator(mode="after")
    def validate_source(self) -> "EvidenceClaim":
        if not self.source.tool_name.strip() or not self.source.raw_result.strip():
            raise ValueError("source tool_name and raw_result are required")
        if self.certainty == "verified" and not self.subject_ids:
            raise ValueError("verified claim requires an explicit subject scope")
        return self


class TaskHandoff(BaseModel):
    """An explicit transfer record; it never grants tools or confirmation rights."""

    handoff_id: str = Field(default_factory=lambda: uuid4().hex)
    run_id: str = Field(min_length=1, max_length=128)
    kind: HandoffKind
    from_agent: str = Field(max_length=64)
    to_agent: str = Field(max_length=64)
    task_id: str = Field(default=UNKNOWN, max_length=128)
    objective: str = Field(default=UNKNOWN, max_length=600)
    dependency_task_ids: list[str] = Field(default_factory=list)
    subject_ids: list[str] = Field(default_factory=list)
    evidence_claim_ids: list[str] = Field(default_factory=list)
    observed_at: str = Field(default=UNKNOWN, max_length=64)
    reason: str = Field(default="", max_length=500)

    @model_validator(mode="after")
    def validate_role_transfer(self) -> "TaskHandoff":
        if self.kind == "role_handoff" and (
                self.task_id == UNKNOWN or self.objective == UNKNOWN):
            raise ValueError("role handoff requires task_id and objective")
        return self

    @classmethod
    def role(cls, *, run_id: str, from_agent: str, to_agent: str,
             task, subject_ids: list[str], observed_at: str) -> "TaskHandoff":
        return cls(
            run_id=run_id, kind="role_handoff", from_agent=from_agent,
            to_agent=to_agent, task_id=task.task_id, objective=task.objective,
            dependency_task_ids=list(task.depends_on), subject_ids=subject_ids,
            observed_at=observed_at,
        )

    @classmethod
    def tool(cls, *, run_id: str, from_agent: str, task_id: str,
             tool_name: str, observed_at: str) -> "TaskHandoff":
        return cls(
            run_id=run_id, kind="tool_call", from_agent=from_agent,
            to_agent=f"tool:{tool_name}", task_id=task_id,
            observed_at=observed_at,
        )

    @classmethod
    def human(cls, *, run_id: str, from_agent: str, reason: str,
              observed_at: str) -> "TaskHandoff":
        return cls(
            run_id=run_id, kind="human_handoff", from_agent=from_agent,
            to_agent="human", reason=reason, observed_at=observed_at,
        )
