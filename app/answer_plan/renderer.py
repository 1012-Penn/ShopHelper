from __future__ import annotations

import re

from app.answer_plan.models import (
    PlanValidation,
    RenderedAnswer,
)

_CITATION_RE = re.compile(r"\s*\[\d+\]")


def _clean_claim(text: str) -> str:
    return _CITATION_RE.sub("", str(text or "")).strip().rstrip("。！？!?；;")


def _with_conditions(text: str, conditions: list[str]) -> str:
    missing = [
        condition.strip()
        for condition in conditions
        if condition.strip() and condition.strip() not in text
    ]
    if not missing:
        return text
    return f"{text}（适用条件：{'；'.join(missing)}）"


def render_answer_plan(validation: PlanValidation) -> RenderedAnswer:
    plan = validation.plan
    if validation.status != "pass":
        supported = [
            row.claim for row in validation.claims if row.status == "supported"
        ]
        if supported:
            lines = ["目前只能提供有限帮助。"]
            citation_numbers: dict[str, int] = {}
            citations: dict[int, dict[str, str]] = {}
            for claim in supported:
                numbers = []
                for binding in claim.evidence:
                    if binding.evidence_id not in citation_numbers:
                        number = len(citation_numbers) + 1
                        citation_numbers[binding.evidence_id] = number
                    numbers.append(citation_numbers[binding.evidence_id])
                marker = "".join(f"[{number}]" for number in dict.fromkeys(numbers))
                text = _clean_claim(claim.text)
                text = _with_conditions(text, claim.necessary_conditions)
            lines.append(
                f"{text}{(' ' + marker) if marker else ''}。"
            )
            unresolved = [
                claim.claim for claim in validation.claims
                if claim.status != "supported"
            ]
            lines.append(
                f"其余 {len(unresolved)} 项证据不足，引用无法核验，"
                "未纳入确定回答。"
            )
            return RenderedAnswer(
                text="\n".join(lines),
                citations=citations,
                used_evidence_ids=list(citation_numbers),
            )
        detail = "引用无法核验或证据不足"
        counts = {
            status: sum(row.status == status for row in validation.claims)
            for status in ("unsupported", "uncertain")
        }
        return RenderedAnswer(
            text=(
                "目前只能提供有限帮助。"
                f"{detail}（unsupported={counts['unsupported']}, "
                f"uncertain={counts['uncertain']}），"
                "请补充商品型号、订单号或更具体的业务信息。"
            )
        )

    if plan.disposition == "clarify":
        question = plan.clarifying_question.strip() or "请补充商品型号或订单号。"
        return RenderedAnswer(text=f"需要先确认：{question}")
    if plan.disposition == "limited_help" and plan.clarifying_question.strip():
        limitations = "；".join(plan.limitations)
        suffix = f"（{limitations}）" if limitations else ""
        return RenderedAnswer(
            text=f"需要先确认：{plan.clarifying_question.strip()}{suffix}"
        )
    if plan.disposition == "refuse":
        reason = "；".join(plan.limitations) or "现有证据不足"
        return RenderedAnswer(
            text=f"抱歉，我无法依据现有资料确认该信息（{reason}）。"
        )

    lines: list[str] = []
    citations: dict[int, dict[str, str]] = {}
    by_id = {
        candidate.evidence_id: candidate
        for candidate in []
    }
    del by_id
    citation_numbers: dict[str, int] = {}
    for row in validation.claims:
        if row.status != "supported":
            continue
        claim = row.claim
        numbers = []
        for binding in claim.evidence:
            if binding.evidence_id not in citation_numbers:
                citation_numbers[binding.evidence_id] = len(citation_numbers) + 1
            numbers.append(citation_numbers[binding.evidence_id])
        marker = "".join(f"[{number}]" for number in dict.fromkeys(numbers))
        text = _clean_claim(claim.text)
        lines.append(f"{text}{(' ' + marker) if marker else ''}。")
        if claim.necessary_conditions:
            lines.append(f"适用条件：{'；'.join(claim.necessary_conditions)}。")
    if plan.limitations:
        lines.append(f"补充说明：{'；'.join(plan.limitations)}。")
    by_id = {candidate.evidence_id: candidate for candidate in validation.evidence}
    for evidence_id, number in citation_numbers.items():
        candidate = by_id[evidence_id]
        citations[number] = {
            "evidence_id": evidence_id,
            "source_id": candidate.source_id,
            "source_title": candidate.source_title,
        }
    return RenderedAnswer(
        text="\n".join(lines),
        citations=citations,
        used_evidence_ids=list(citation_numbers),
    )


def render_with_candidates(
    validation: PlanValidation,
    candidates: list,
) -> RenderedAnswer:
    rendered = render_answer_plan(validation)
    by_id = {candidate.evidence_id: candidate for candidate in candidates}
    citations: dict[int, dict[str, str]] = {}
    for number, evidence_id in enumerate(rendered.used_evidence_ids, 1):
        candidate = by_id[evidence_id]
        citations[number] = {
            "evidence_id": evidence_id,
            "source_id": candidate.source_id,
            "source_title": candidate.source_title,
        }
    return RenderedAnswer(
        text=rendered.text,
        citations=citations,
        used_evidence_ids=list(rendered.used_evidence_ids),
    )
