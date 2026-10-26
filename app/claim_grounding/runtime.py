"""Optional production adapter for selective claim grounding."""

from __future__ import annotations

import time
from pathlib import Path

from app.claim_grounding.answer_adapter import ClaimGroundingAnswerSemantic
from app.claim_grounding.backends import ProviderResponse, StructuredGroundingBackend
from app.claim_grounding.budget import ClaimBudgetLedger
from app.claim_grounding.grounder import ClaimGrounder
from app.config import Settings
from app.llm import make_extract_model


def _usage(response) -> dict[str, int]:
    usage = getattr(response, "usage_metadata", None) or {}
    details = usage.get("input_token_details") or {}
    return {
        "input_tokens": int(usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "cached_input_tokens": int(details.get("cache_read") or 0),
    }


def build_claim_grounding_semantic(settings: Settings, *, model=None):
    """Build the opt-in verifier adapter; unsupported config fails before serving."""

    ledger_path = settings.claim_grounding_ledger_path.strip()
    if not ledger_path:
        raise ValueError(
            "claim_grounding_ledger_path must be set when claim grounding is enabled"
        )
    Path(ledger_path).parent.mkdir(parents=True, exist_ok=True)
    ledger = ClaimBudgetLedger(
        ledger_path,
        total_limit_usd=settings.claim_grounding_max_cost_usd,
    )
    raw_model = model or make_extract_model(settings)

    def caller(prompt: str) -> ProviderResponse:
        started = time.perf_counter()
        response = raw_model.invoke(prompt)
        return ProviderResponse(
            text=str(getattr(response, "content", response)),
            usage=_usage(response),
            model_version=settings.openai_model,
            latency_ms=(time.perf_counter() - started) * 1000,
            raw=response,
        )

    backend = StructuredGroundingBackend(
        caller=caller,
        ledger=ledger,
        pricing={
            "input_per_million": 0.30,
            "input_cache_hit_per_million": 0.006,
            "output_per_million": 1.20,
        },
        model=settings.openai_model,
        reserve_usd=settings.claim_grounding_reserve_usd,
        max_retries=2,
    )
    return ClaimGroundingAnswerSemantic(
        ClaimGrounder(backend=backend, only_uncertain=True)
    )
