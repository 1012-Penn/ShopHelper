from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

GroundingStatus = Literal["supported", "unsupported", "uncertain"]


class EvidenceCandidate(BaseModel):
    evidence_id: str
    text: str

    @field_validator("evidence_id", "text")
    @classmethod
    def _not_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("evidence_id 和 text 不能为空")
        return value


class ClaimInput(BaseModel):
    claim: str
    subject: str = ""
    model_number: str = ""
    order_id: str = ""
    as_of_date: str | None = None
    policy_version: str = ""
    unit: str = ""
    evidence: list[EvidenceCandidate] = Field(default_factory=list)

    @field_validator("claim")
    @classmethod
    def _claim_not_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("claim 不能为空")
        return value

    def evidence_ids(self) -> set[str]:
        return {item.evidence_id for item in self.evidence}


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class GroundingResult(BaseModel):
    status: GroundingStatus
    reason: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    backend: str = "deterministic"
    model_version: str = ""
    usage: TokenUsage = Field(default_factory=TokenUsage)
    baseline_status: GroundingStatus | None = None
    backend_called: bool = False
    cache_hit: bool = False
    latency_ms: float | None = None
    attempts: int = 0
