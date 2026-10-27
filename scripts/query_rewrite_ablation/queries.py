"""Construct raw, legacy and guarded queries through the existing contracts."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from app.prompts import RESOLVE_PROMPT
from app.query_guard import (
    audit_history_resolution,
    audit_query_rewrite,
    safe_resolve_prompt,
)
from app.rewrite import REWRITE_PROMPT, SAFE_REWRITE_PROMPT, QueryRewrite


@dataclass(frozen=True)
class VariantQuery:
    name: str
    text: str
    resolved: str
    normalized: str
    synonyms: tuple[str, ...] = ()
    guard_violations: tuple[str, ...] = ()
    dropped_synonyms: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


def render_history(context: list[dict]) -> str:
    labels = {"user": "用户", "assistant": "客服"}
    return "\n".join(
        f"{labels.get(str(turn.get('role')), str(turn.get('role')))}：{turn.get('content', '')}"
        for turn in context
    )


def _resolved_text(result) -> tuple[str, bool]:
    if isinstance(result, dict):
        return str(result.get("resolved") or "").strip(), bool(result.get("changed"))
    return str(getattr(result, "resolved", "") or "").strip(), bool(
        getattr(result, "changed", False)
    )


def _rewrite_value(result) -> QueryRewrite:
    if isinstance(result, QueryRewrite):
        return result
    if isinstance(result, dict):
        return QueryRewrite(**result)
    return QueryRewrite(
        normalized=str(getattr(result, "normalized", "") or ""),
        synonyms=list(getattr(result, "synonyms", []) or []),
    )


def _safe_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:240]


def build_query_variants(
    case: dict,
    *,
    legacy_resolve: Callable[[str], object],
    legacy_rewrite: Callable[[str], object],
    safe_resolve: Callable[[str], object],
    safe_rewrite: Callable[[str], object],
) -> dict[str, VariantQuery]:
    """Build three query texts; only the safe path applies audit fallbacks."""
    current = str(case["query"]).strip()
    history = render_history(list(case.get("conversation_context") or []))
    raw = VariantQuery("raw", current, current, current)

    errors: list[str] = []
    try:
        legacy_resolved, _ = _resolved_text(
            legacy_resolve(RESOLVE_PROMPT.format(history=history or "(无)", query=current))
        )
    except Exception as exc:  # noqa: BLE001 - model failures become trace data.
        errors.append(f"legacy_resolve:{_safe_error(exc)}")
        legacy_resolved = current
    legacy_resolved = legacy_resolved or current
    resolve_audit = audit_history_resolution(current, history, legacy_resolved)
    try:
        legacy_rewrite_result = _rewrite_value(
            legacy_rewrite(REWRITE_PROMPT.format(query=legacy_resolved))
        )
        legacy_normalized = legacy_rewrite_result.normalized.strip() or legacy_resolved
        legacy_synonyms = tuple(
            item.strip() for item in legacy_rewrite_result.synonyms
            if item and item.strip()
        )[:3]
    except Exception as exc:  # noqa: BLE001 - model failures become trace data.
        errors.append(f"legacy_rewrite:{_safe_error(exc)}")
        legacy_normalized, legacy_synonyms = legacy_resolved, ()
    rewrite_audit = audit_query_rewrite(legacy_resolved, legacy_normalized)
    legacy = VariantQuery(
        "legacy",
        legacy_normalized,
        legacy_resolved,
        legacy_normalized,
        legacy_synonyms,
        resolve_audit.violations + rewrite_audit.violations,
        errors=tuple(errors),
    )

    errors = []
    try:
        safe_resolved, _ = _resolved_text(
            safe_resolve(safe_resolve_prompt(current, history or "(无)"))
        )
    except Exception as exc:  # noqa: BLE001 - model failures become trace data.
        errors.append(f"safe_resolve:{_safe_error(exc)}")
        safe_resolved = current
    safe_resolved = safe_resolved or current
    resolve_audit = audit_history_resolution(current, history, safe_resolved)
    safe_resolved = resolve_audit.text
    try:
        safe_rewrite_result = _rewrite_value(
            safe_rewrite(SAFE_REWRITE_PROMPT.format(query=safe_resolved))
        )
        candidate = safe_rewrite_result.normalized.strip() or safe_resolved
        candidates = [
            item.strip() for item in safe_rewrite_result.synonyms
            if item and item.strip()
        ][:3]
    except Exception as exc:  # noqa: BLE001 - model failures become trace data.
        errors.append(f"safe_rewrite:{_safe_error(exc)}")
        candidate, candidates = safe_resolved, []
    rewrite_audit = audit_query_rewrite(safe_resolved, candidate)
    dropped: list[str] = []
    accepted_synonyms: list[str] = []
    for synonym in candidates:
        synonym_audit = audit_query_rewrite(safe_resolved, synonym)
        if synonym_audit.accepted:
            accepted_synonyms.append(synonym)
        else:
            dropped.append(synonym)
    if rewrite_audit.accepted:
        normalized = candidate
    else:
        normalized = safe_resolved
        accepted_synonyms = []
    safe = VariantQuery(
        "safe",
        normalized,
        safe_resolved,
        normalized,
        tuple(accepted_synonyms),
        resolve_audit.violations + rewrite_audit.violations,
        tuple(dropped),
        tuple(errors),
    )
    return {"raw": raw, "legacy": legacy, "safe": safe}
