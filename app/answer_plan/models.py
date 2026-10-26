from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Disposition = Literal["answer", "clarify", "limited_help", "refuse"]
ClaimStatus = Literal["supported", "unsupported", "uncertain"]
PlanStatus = Literal["pass", "fail", "uncertain"]


def _non_empty(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("value must not be empty")
    return value


class PolicyScope(BaseModel):
    version: str = ""
    effective_from: str | None = None
    effective_to: str | None = None


class EvidenceCandidate(BaseModel):
    evidence_id: str
    text: str
    source_id: str
    source_title: str = ""
    entities: list[str] = Field(default_factory=list)
    required_conditions: list[str] = Field(default_factory=list)
    condition_aliases: dict[str, list[str]] = Field(default_factory=dict)
    policy_scope: PolicyScope | None = None

    @field_validator("evidence_id", "text", "source_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        return _non_empty(value)


class EvidenceBinding(BaseModel):
    evidence_id: str
    quote: str
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, ge=0)

    @field_validator("evidence_id", "quote")
    @classmethod
    def _binding_text(cls, value: str) -> str:
        return _non_empty(value)


class PlanClaim(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    claim_id: str
    text: str
    subject: str = ""
    value: str = ""
    unit: str = ""
    dates: list[str] = Field(default_factory=list)
    policy_version: str = ""
    necessary_conditions: list[str] = Field(default_factory=list)
    evidence: list[EvidenceBinding] = Field(default_factory=list)

    @field_validator("claim_id", "text")
    @classmethod
    def _claim_text(cls, value: str) -> str:
        return _non_empty(value)


class AnswerPlan(BaseModel):
    disposition: Disposition = "answer"
    claims: list[PlanClaim] = Field(default_factory=list)
    clarifying_question: str = ""
    limitations: list[str] = Field(default_factory=list)


class PlanIssue(BaseModel):
    code: str
    message: str
    claim_id: str = ""
    severity: Literal["error", "warning"] = "error"


class ClaimValidation(BaseModel):
    claim: PlanClaim
    status: ClaimStatus
    issues: list[PlanIssue] = Field(default_factory=list)


class PlanValidation(BaseModel):
    plan: AnswerPlan
    status: PlanStatus
    claims: list[ClaimValidation] = Field(default_factory=list)
    issues: list[PlanIssue] = Field(default_factory=list)
    evidence: list[EvidenceCandidate] = Field(default_factory=list)

    @property
    def supported_claims(self) -> list[PlanClaim]:
        return [row.claim for row in self.claims if row.status == "supported"]


class RenderedCitation(BaseModel):
    evidence_id: str
    source_id: str
    source_title: str = ""


class RenderedAnswer(BaseModel):
    text: str
    citations: dict[int, dict[str, str]] = Field(default_factory=dict)
    used_evidence_ids: list[str] = Field(default_factory=list)


class AnswerPlanResult(BaseModel):
    plan: AnswerPlan
    validation: PlanValidation
    rendered: RenderedAnswer
    correction_attempts: int = 0
    planner_calls: int = 0
    trace: list[str] = Field(default_factory=list)
