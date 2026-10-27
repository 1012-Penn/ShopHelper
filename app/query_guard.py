"""Explicit-constraint guard for history resolution and retrieval query rewriting.

The guard is intentionally lexical: it protects identifiers, dates, numbers,
policy versions and polarity before a semantic rewrite is allowed to alter the
query.  A failed audit falls back to the exact prior text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.prompts import RESOLVE_PROMPT

_ENTITY_ANCHOR = re.compile(
    r"(?<![A-Za-z0-9])(?:ORD-\d{4,6}|SH-[A-Za-z0-9]+)(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_DATE_ANCHOR = re.compile(r"(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)")
_VERSION_ANCHOR = re.compile(r"(?<![A-Za-z0-9])v\d+(?![A-Za-z0-9])", re.IGNORECASE)
_NUMBER_ANCHOR = re.compile(
    r"(?<!\d)\d+(?:\.\d+)?\s*(?:工作日|小时|个月|mah|元|天|月|年|米|键|折)?",
    re.IGNORECASE,
)
_CHINESE_NUMBER_ANCHOR = re.compile(
    r"[零一二两三四五六七八九十百]+(?:工作日|小时|个月|元|天|月|年|米|键|折)"
)
_NEUTRAL_QUESTION = re.compile(r"能不能|可不可以|会不会|是不是|有没有|能否|可否|是否")
_NEGATIVE = re.compile(
    r"不支持|不能|不可|不予|不是|不要|不需要|不收|无需|无须|没有|没|未|别|禁止|非|不"
)
_COREFERENCE = re.compile(
    r"它|这个|那个|这款|那款|这单|那单|这笔|那笔|上面|刚才|前面|该订单|该商品"
)


@dataclass(frozen=True)
class QueryGuardDecision:
    """One audited query and the exact reasons a candidate was rejected."""

    accepted: bool
    text: str
    violations: tuple[str, ...]
    input_anchors: tuple[str, ...]
    output_anchors: tuple[str, ...]


def _canonical_anchor(value: str) -> str:
    compact = re.sub(r"\s+", "", value)
    if re.match(r"^(?:ORD|SH)-", compact, re.IGNORECASE):
        return compact.upper()
    if re.fullmatch(r"v\d+", compact, re.IGNORECASE):
        return compact.casefold()
    return compact.casefold()


def protected_anchors(text: str) -> tuple[str, ...]:
    """Return explicit facts a paraphrase must not silently replace or remove."""
    values: list[str] = []
    for pattern in (
        _DATE_ANCHOR,
        _ENTITY_ANCHOR,
        _VERSION_ANCHOR,
        _CHINESE_NUMBER_ANCHOR,
        _NUMBER_ANCHOR,
    ):
        values.extend(match.group(0) for match in pattern.finditer(str(text or "")))
    return tuple(dict.fromkeys(_canonical_anchor(value) for value in values))


def _has_negative_polarity(text: str) -> bool:
    neutral_stripped = _NEUTRAL_QUESTION.sub("", str(text or ""))
    return bool(_NEGATIVE.search(neutral_stripped))


def has_negative_polarity(text: str) -> bool:
    """Public diagnostic form; neutral ``能不能/是否`` questions stay non-negative."""
    return _has_negative_polarity(text)


def _audit(
    original: str,
    candidate: str,
    *,
    allowed_extra_anchors: tuple[str, ...] = (),
    allow_history_negative: bool = False,
) -> QueryGuardDecision:
    original = str(original or "").strip()
    candidate = str(candidate or "").strip()
    input_anchors = protected_anchors(original)
    output_anchors = protected_anchors(candidate)
    allowed = set(input_anchors) | {
        _canonical_anchor(value) for value in allowed_extra_anchors
    }
    missing = sorted(set(input_anchors) - set(output_anchors))
    added = sorted(set(output_anchors) - allowed)
    violations = [f"missing_anchor:{anchor}" for anchor in missing]
    violations.extend(f"new_anchor:{anchor}" for anchor in added)
    original_negative = _has_negative_polarity(original)
    candidate_negative = _has_negative_polarity(candidate)
    if original_negative != candidate_negative and not (
        not original_negative and candidate_negative and allow_history_negative
    ):
        violations.append("polarity_changed")
    accepted = bool(candidate) and not violations
    return QueryGuardDecision(
        accepted=accepted,
        text=candidate if accepted else original,
        violations=tuple(violations),
        input_anchors=input_anchors,
        output_anchors=output_anchors,
    )


def audit_query_rewrite(original: str, candidate: str) -> QueryGuardDecision:
    """A rewrite may normalize wording but may not add unsupported query facts."""
    return _audit(original, candidate)


def audit_history_resolution(
    current_message: str,
    history_text: str,
    candidate: str,
) -> QueryGuardDecision:
    """A history completion may add only anchors actually present in the history."""
    return _audit(
        current_message,
        candidate,
        allowed_extra_anchors=protected_anchors(history_text),
        allow_history_negative=_has_negative_polarity(history_text),
    )


def needs_history_resolution(query: str) -> bool:
    """Conservative marker used to report when history resolution is plausibly needed."""
    return bool(_COREFERENCE.search(str(query or "")))


def safe_resolve_prompt(query: str, history_text: str) -> str:
    """Existing resolver contract plus precedence and invariant instructions."""
    instructions = (
        "补充硬约束:最新消息优先于历史;必须保留最新消息中的否定、改口、例外、"
        "型号、订单号、日期、政策版本、数字与单位。不得把型号、订单、日期或版本"
        "替换成历史出现过的另一项。取消、退货、运费或优惠问题还必须显式补全历史中"
        "相关的订单状态、商品型号、会员等级或地区条件。"
        "无法安全补全时原样返回最新消息且 changed=false。"
    )
    return RESOLVE_PROMPT.format(
        history=f"{history_text or '(无)'}\n{instructions}",
        query=query,
    )
