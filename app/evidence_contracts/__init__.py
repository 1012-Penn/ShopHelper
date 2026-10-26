"""Typed, source-backed evidence and handoff contracts for the multi-agent runtime."""

from app.evidence_contracts.cache import (
    CacheStats,
    ToolCacheResult,
    ToolResultCache,
    call_cache_key,
)
from app.evidence_contracts.extraction import (
    claims_from_tool_result,
    subjects_from_arguments,
    subjects_from_text,
)
from app.evidence_contracts.ledger import EvidenceLedger
from app.evidence_contracts.models import (
    UNKNOWN,
    Certainty,
    EvidenceClaim,
    EvidenceSource,
    HandoffKind,
    TaskHandoff,
)

__all__ = [
    "UNKNOWN",
    "CacheStats",
    "Certainty",
    "EvidenceClaim",
    "EvidenceLedger",
    "EvidenceSource",
    "HandoffKind",
    "TaskHandoff",
    "ToolCacheResult",
    "ToolResultCache",
    "claims_from_tool_result",
    "call_cache_key",
    "subjects_from_arguments",
    "subjects_from_text",
]
