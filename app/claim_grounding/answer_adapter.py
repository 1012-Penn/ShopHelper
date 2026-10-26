"""Adapter for AnswerVerifier's existing ``semantic.invoke(prompt)`` contract."""

from __future__ import annotations

import re

from app.answer_verifier import AnswerVerdict, _sentence_claims
from app.claim_grounding.models import ClaimInput, EvidenceCandidate

_MODEL_RE = re.compile(r"\bSH-[A-Za-z0-9]+\b", re.IGNORECASE)
_ORDER_RE = re.compile(r"\bORD-[A-Za-z0-9]+\b", re.IGNORECASE)
_DATE_RE = re.compile(r"(?<!\d)(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})日?")
_VERSION_RE = re.compile(r"\bv([0-9]+)\b", re.IGNORECASE)
_UNIT_RE = re.compile(r"(元|小时|天|工作日|个月|月|年|mAh|Wh|kg|米)")


def _section(prompt: str, start: str, end: str | None = None) -> str:
    begin = prompt.find(start)
    if begin < 0:
        return ""
    begin += len(start)
    if end is None:
        return prompt[begin:].strip()
    finish = prompt.find(end, begin)
    return prompt[begin : finish if finish >= 0 else None].strip()


class ClaimGroundingAnswerSemantic:
    def __init__(self, grounder) -> None:
        self.grounder = grounder

    def invoke(self, prompt: str) -> AnswerVerdict:
        answer = _section(prompt, "【候选答案】")
        evidence_block = _section(prompt, "【编号证据】", "【候选答案】")
        evidence: dict[int, EvidenceCandidate] = {}
        for line in evidence_block.splitlines():
            match = re.match(r"^\s*\[(\d+)\]\s*(.*)$", line)
            if match:
                evidence[int(match.group(1))] = EvidenceCandidate(
                    evidence_id=match.group(1),
                    text=match.group(2),
                )

        results = []
        issues = []
        for sentence, citation_numbers in _sentence_claims(answer):
            candidates = [
                evidence[number] for number in citation_numbers if number in evidence
            ]
            models = _MODEL_RE.findall(sentence)
            orders = _ORDER_RE.findall(sentence)
            dates = _DATE_RE.findall(sentence)
            versions = _VERSION_RE.findall(sentence)
            units = _UNIT_RE.findall(sentence)
            claim = ClaimInput(
                claim=sentence,
                model_number=models[0].upper() if models else "",
                order_id=orders[0].upper() if orders else "",
                as_of_date=(
                    f"{dates[0][0]}-{int(dates[0][1]):02d}-{int(dates[0][2]):02d}"
                    if dates
                    else None
                ),
                policy_version=f"v{versions[0]}" if versions else "",
                unit=units[0] if units else "",
                evidence=candidates,
            )
            result = self.grounder.ground(claim)
            results.append(result.status)
            if result.status in {"unsupported", "uncertain"}:
                issues.append(f"{result.status}:{result.reason}")

        if not results:
            return AnswerVerdict(
                status="uncertain",
                reason="候选答案没有可判断的引用 claim",
                issues=["no_cited_claim"],
            )
        if "unsupported" in results:
            return AnswerVerdict(
                status="fail",
                reason="至少一个 claim 与候选证据冲突",
                issues=issues,
            )
        if all(status == "supported" for status in results):
            return AnswerVerdict(
                status="pass",
                reason="全部引用 claim 均得到证据支持",
                issues=[],
            )
        return AnswerVerdict(
            status="uncertain",
            reason="仍有 claim 无法可靠判断",
            issues=issues,
        )
