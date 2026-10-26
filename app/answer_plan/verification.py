"""Deterministic evidence binding checks.

The checks reject clear conflicts and missing applicability conditions. They do
not pretend that lexical overlap proves semantic entailment; claims with an
unanchored value remain uncertain.
"""
from __future__ import annotations

import re
from datetime import date

from app.answer_plan.models import (
    AnswerPlan,
    ClaimValidation,
    EvidenceBinding,
    EvidenceCandidate,
    PlanClaim,
    PlanIssue,
    PlanValidation,
)

_ENTITY_RE = re.compile(r"\b(?:SH|ORD)-[A-Za-z0-9]+\b", re.IGNORECASE)
_DATE_RE = re.compile(r"(?<!\d)(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})日?")
_NUMBER_TOKEN = r"(?:\d+(?:\.\d+)?|[零〇一二三四五六七八九十百两]+)"
_NUMBER_UNIT_RE = re.compile(
    rf"(?<!\w)({_NUMBER_TOKEN})\s*(元|天|小时|年|个月|月|mAh|键|米|个工作日|折)",
    re.IGNORECASE,
)
_NUMERIC_VALUE_RE = re.compile(rf"^(?:{_NUMBER_TOKEN})(?:\.\d+)?$")
_VALUE_WITH_UNIT_RE = re.compile(
    rf"^({_NUMBER_TOKEN})\s*(元|天|小时|年|个月|月|mAh|键|米|个工作日|折)$",
    re.IGNORECASE,
)
_PUNCT_RE = re.compile(r"[\s，。；;,：:、（）()【】\[\]]+")
_MAX_QUOTE_CHARS = 320


def _norm(value: str) -> str:
    return _PUNCT_RE.sub("", str(value or "")).casefold()


def _date(value: str) -> date | None:
    match = _DATE_RE.fullmatch(str(value or "").strip())
    if not match:
        return None
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None


def _issue(claim: PlanClaim, code: str, message: str) -> PlanIssue:
    return PlanIssue(code=code, message=message, claim_id=claim.claim_id)


def _binding(
    claim: PlanClaim,
    binding: EvidenceBinding,
    candidate: EvidenceCandidate,
) -> tuple[EvidenceBinding | None, PlanIssue | None]:
    text = candidate.text
    if binding.start is None or binding.end is None:
        start = text.find(binding.quote)
        if start < 0:
            return None, _issue(
                claim,
                "quote_not_found",
                f"原文 quote 不在 evidence {candidate.evidence_id} 中",
            )
        return EvidenceBinding(
            evidence_id=binding.evidence_id,
            quote=binding.quote,
            start=start,
            end=start + len(binding.quote),
        ), None
    if (
        binding.end <= binding.start
        or binding.end > len(text)
        or text[binding.start:binding.end] != binding.quote
    ):
        return None, _issue(
            claim,
            "span_mismatch",
            f"quote/span 与 evidence {candidate.evidence_id} 的原文不一致",
        )
    return binding, None


def _validate_claim(
    claim: PlanClaim,
    candidates: dict[str, EvidenceCandidate],
) -> ClaimValidation:
    issues: list[PlanIssue] = []
    normalized: list[EvidenceBinding] = []
    bound_candidates: list[EvidenceCandidate] = []
    if not claim.evidence:
        issues.append(_issue(claim, "evidence_missing", "重要 claim 未绑定证据"))

    for binding in claim.evidence:
        candidate = candidates.get(binding.evidence_id)
        if candidate is None:
            issues.append(
                _issue(
                    claim,
                    "evidence_id_missing",
                    f"引用的 evidence ID 不存在:{binding.evidence_id}",
                )
            )
            continue
        if len(binding.quote) > _MAX_QUOTE_CHARS or (
            len(candidate.text) > 160
            and len(binding.quote) / max(1, len(candidate.text)) > 0.85
        ):
            issues.append(
                _issue(claim, "quote_too_large", "quote 近似整段证据，不能作为精确绑定")
            )
            continue
        checked, problem = _binding(claim, binding, candidate)
        if problem is not None:
            issues.append(problem)
            continue
        normalized.append(checked)
        bound_candidates.append(candidate)

    claim_entities = {
        item.upper() for item in _ENTITY_RE.findall(claim.text)
    }
    if claim.subject:
        claim_entities.update(
            item.upper() for item in _ENTITY_RE.findall(claim.subject)
        )
    if claim_entities:
        evidence_entities = {
            item.upper()
            for candidate in bound_candidates
            for item in candidate.entities
        }
        searchable = "\n".join(
            candidate.text for candidate in bound_candidates
        ).upper()
        if evidence_entities and not claim_entities.issubset(evidence_entities):
            issues.append(
                _issue(
                    claim,
                    "subject_conflict",
                    "claim 主体与绑定证据不一致:"
                    f"{sorted(claim_entities)} vs {sorted(evidence_entities)}",
                )
            )
        elif not evidence_entities and any(
            entity not in searchable for entity in claim_entities
        ):
            issues.append(
                _issue(claim, "subject_not_in_evidence", "claim 主体未出现在绑定证据中")
            )

    quote_text = "\n".join(binding.quote for binding in normalized)
    normalized_quote = _norm(quote_text)
    normalized_claim = _norm(claim.text)
    claim_value = claim.value.strip()
    claim_unit = claim.unit.strip()
    embedded = _VALUE_WITH_UNIT_RE.fullmatch(claim_value)
    if embedded and not claim_unit:
        claim_value, claim_unit = embedded.group(1), embedded.group(2)
    if claim_value and (
        claim_unit or _NUMERIC_VALUE_RE.fullmatch(claim_value)
    ):
        target = f"{claim_value}{claim_unit}"
        anchors = {
            (number.replace(" ", ""), unit.casefold())
            for number, unit in _NUMBER_UNIT_RE.findall(quote_text)
        }
        if (claim_value.replace(" ", ""), claim_unit.casefold()) not in anchors:
            same_unit = [
                number for number, unit in anchors
                if claim_unit and unit.casefold() == claim_unit.casefold()
            ]
            same_number = [
                unit for number, unit in anchors
                if number == claim_value.replace(" ", "")
            ]
            if same_unit or same_number:
                issues.append(
                    _issue(
                        claim,
                        "number_unit_conflict",
                        f"claim 数值/单位冲突:{target} 不在绑定 quote 中，"
                        f"证据为 {same_unit or same_number}",
                    )
                )
            else:
                issues.append(
                    _issue(
                        claim,
                        "claim_not_anchored",
                        f"quote 未锚定 claim 数值/单位:{target}",
                    )
                )
    elif normalized_claim and normalized_claim not in normalized_quote:
        # Without a structured value, require the claim itself to be grounded by
        # the quote. This is still deterministic support, not semantic truth.
        claim_parts = [part for part in re.split(r"[，。；;,]", claim.text) if len(part) >= 4]
        if not any(_norm(part) in normalized_quote for part in claim_parts):
            issues.append(
                _issue(claim, "claim_not_anchored", "绑定 quote 未锚定 claim 的实质陈述")
            )

    for candidate in bound_candidates:
        for condition in candidate.required_conditions:
            claim_conditions = _norm(" ".join(claim.necessary_conditions))
            aliases = [
                condition,
                *(candidate.condition_aliases.get(condition) or []),
            ]
            if not any(
                _norm(alias) in normalized_claim + claim_conditions
                for alias in aliases
            ):
                issues.append(
                    _issue(
                        claim,
                        "condition_omitted",
                        f"必要条件未被保留:{condition}",
                    )
                )

    if claim.policy_version:
        for candidate in bound_candidates:
            scope = candidate.policy_scope
            if scope and scope.version and scope.version.casefold() != claim.policy_version.casefold():
                issues.append(
                    _issue(
                        claim,
                        "policy_version_conflict",
                        f"政策版本与证据不一致:{claim.policy_version} vs {scope.version}",
                    )
                )
            if scope:
                start = _date(scope.effective_from or "")
                end = _date(scope.effective_to or "")
                for value in claim.dates:
                    observed = _date(value)
                    if observed and (
                        (start and observed < start) or (end and observed > end)
                    ):
                        issues.append(
                            _issue(
                                claim,
                                "policy_date_conflict",
                                f"政策适用日期超出证据范围:{value}",
                            )
                        )

    if any(issue.code in {
        "quote_not_found",
        "span_mismatch",
        "quote_too_large",
        "evidence_missing",
        "evidence_id_missing",
        "subject_conflict",
        "subject_not_in_evidence",
        "number_unit_conflict",
        "condition_omitted",
        "policy_version_conflict",
        "policy_date_conflict",
    } for issue in issues):
        status = "unsupported"
    elif any(issue.code in {"claim_not_anchored"} for issue in issues):
        status = "uncertain"
    else:
        status = "supported"

    # Persist normalized spans so the renderer and saved contract use offsets
    # that were actually verified, not model-provided approximations.
    for binding in normalized:
        for original in claim.evidence:
            if original.evidence_id == binding.evidence_id and original.quote == binding.quote:
                original.start = binding.start
                original.end = binding.end
                break
    return ClaimValidation(claim=claim, status=status, issues=issues)


def validate_answer_plan(
    plan: AnswerPlan,
    candidates: list[EvidenceCandidate],
) -> PlanValidation:
    by_id = {candidate.evidence_id: candidate for candidate in candidates}
    claims = [_validate_claim(claim, by_id) for claim in plan.claims]
    issues = [issue for claim in claims for issue in claim.issues]
    if not claims:
        status = (
            "pass"
            if plan.disposition in {"clarify", "refuse", "limited_help"}
            else "uncertain"
        )
    elif any(claim.status == "unsupported" for claim in claims):
        status = "fail"
    elif any(claim.status == "uncertain" for claim in claims):
        status = "uncertain"
    else:
        status = "pass"
    return PlanValidation(
        plan=plan,
        status=status,
        claims=claims,
        issues=issues,
        evidence=list(candidates),
    )
