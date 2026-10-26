from __future__ import annotations

from typing import Protocol

from app.claim_grounding.contradictions import deterministic_status
from app.claim_grounding.models import ClaimInput, GroundingResult
from app.claim_grounding.v31 import v31_rule_status


class GroundingBackend(Protocol):
    def ground(self, claim: ClaimInput) -> GroundingResult: ...


class ClaimGrounder:
    def __init__(
        self,
        *,
        backend: GroundingBackend | None = None,
        only_uncertain: bool = True,
    ) -> None:
        self.backend = backend
        self.only_uncertain = bool(only_uncertain)

    def ground(self, claim: ClaimInput) -> GroundingResult:
        if not claim.evidence:
            return GroundingResult(
                status="uncertain",
                reason="没有候选证据",
                baseline_status="uncertain",
            )

        deterministic, detail = deterministic_status(claim)
        if deterministic is not None:
            return GroundingResult(
                status=deterministic,
                reason=detail,
                evidence_ids=[item.evidence_id for item in claim.evidence],
                backend="deterministic",
                baseline_status=deterministic,
            )

        baseline = v31_rule_status(claim)
        should_call = self.backend is not None and (
            baseline == "uncertain" or not self.only_uncertain
        )
        if not should_call:
            reason = (
                "v3.1 规则已确定"
                if baseline != "uncertain"
                else "v3.1 无法可靠判断,语义 backend 未启用"
            )
            return GroundingResult(
                status=baseline,
                reason=reason,
                evidence_ids=[item.evidence_id for item in claim.evidence],
                baseline_status=baseline,
            )

        try:
            semantic = self.backend.ground(claim)
        except Exception as exc:  # noqa: BLE001 - backend failure must fail safe.
            return GroundingResult(
                status="uncertain",
                reason=f"semantic backend failure:{type(exc).__name__}:{exc}",
                evidence_ids=[item.evidence_id for item in claim.evidence],
                backend="structured_api",
                baseline_status=baseline,
                backend_called=True,
            )
        semantic.baseline_status = baseline
        semantic.backend_called = True
        return semantic
