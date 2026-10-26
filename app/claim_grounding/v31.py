"""Read-only adapter to the existing rag-quality-v3.1 claim rule."""

from __future__ import annotations

from app.claim_grounding.models import ClaimInput, GroundingStatus


def v31_rule_status(claim: ClaimInput) -> GroundingStatus:
    if not claim.evidence:
        return "uncertain"
    from scripts.eval_checks import _sentence_support_status

    evidence = "\n".join(item.text for item in claim.evidence)
    status = _sentence_support_status(claim.claim, evidence)
    if status in {"supported", "unsupported", "uncertain"}:
        return status  # type: ignore[return-value]
    return "uncertain"
