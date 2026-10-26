"""Independent final-answer verification and one bounded correction.

This checker intentionally does not consume ``EvidenceVerdict.answered``. A
retriever can find enough material while the generated candidate still binds it
to the wrong order, date, product, number or policy version.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Literal

from pydantic import BaseModel

from app.guard import REFUSAL_MARKER
from app.prompts import ANSWER_VERIFIER_PROMPT

_CITE_RE = re.compile(r"\[(\d+)\]")
_ENTITY_RE = re.compile(r"(?:SH|ORD)-[A-Za-z0-9]+", re.I)
_DATE_RE = re.compile(r"(?<!\d)(\d{4})\s*[-/年]\s*(\d{1,2})\s*[-/月]\s*(\d{1,2})\s*日?")


class AnswerVerdict(BaseModel):
    status: Literal["pass", "fail", "uncertain"]
    reason: str
    issues: list[str] = []


@dataclass(frozen=True)
class VerificationResult:
    status: Literal["pass", "fail", "uncertain", "refuse"]
    reason: str
    issues: list[str] = field(default_factory=list)
    final_answer: str = ""
    revision_calls: int = 0


def _sentence_claims(answer: str) -> list[tuple[str, list[int]]]:
    owned = re.sub(r"([。！？!?；;])(\s*(?:\[\d+\])+)", r"\2\1", answer)
    claims = []
    for sentence in re.split(r"(?<=[。！？!?；;])", owned):
        numbers = [int(value) for value in _CITE_RE.findall(sentence)]
        if sentence.strip():
            claims.append((sentence.strip(), numbers))
    return claims


def _evidence_block(evidence: list[dict]) -> str:
    return "\n".join(
        f"[{item.get('n')}] {item.get('section_path', '')} "
        f"问:{item.get('question', '')} 答:{item.get('answer', '')}"
        for item in evidence
    )


def _dates(text: str) -> list[tuple[int, int, int]]:
    return [
        (int(match.group(1)), int(match.group(2)), int(match.group(3)))
        for match in _DATE_RE.finditer(text)
    ]


def _order_dates(text: str) -> dict[str, tuple[int, int, int]]:
    rows: dict[str, tuple[int, int, int]] = {}
    for entity in _ENTITY_RE.findall(text):
        if not entity.upper().startswith("ORD-"):
            continue
        pattern = re.compile(
            rf"{re.escape(entity)}[^\n|]{{0,40}}?\|\s*"
            r"(\d{4})\s*[-/]\s*(\d{1,2})\s*[-/]\s*(\d{1,2})",
            flags=re.I,
        )
        match = pattern.search(text)
        if match:
            date = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
            rows[entity.upper()] = date
    return rows


class AnswerVerifier:
    """Combine hard deterministic conflicts with an independent semantic verdict."""

    def __init__(self, semantic=None, *, max_revisions: int = 1) -> None:
        self._semantic = semantic
        self._max_revisions = max(0, min(1, int(max_revisions)))

    def _deterministic_issues(self, answer: str, evidence: list[dict]) -> list[str]:
        by_n = {int(item["n"]): item for item in evidence if "n" in item}
        all_text = _evidence_block(evidence)
        known_orders = _order_dates(all_text)
        issues: list[str] = []
        claims = _sentence_claims(answer)
        for sentence, citations in claims:
            if citations and not all(n in by_n for n in citations):
                missing = sorted(set(citations) - set(by_n))
                issues.append(f"引用编号不存在:{missing}")
                continue
            cited_text = "\n".join(
                f"{by_n[n].get('question', '')}\n{by_n[n].get('answer', '')}"
                for n in citations if n in by_n
            )
            answer_orders = {item.upper() for item in _ENTITY_RE.findall(sentence)
                             if item.upper().startswith("ORD-")}
            answer_dates = _dates(sentence)
            for order_id in answer_orders:
                expected = known_orders.get(order_id)
                if expected and answer_dates and expected not in answer_dates:
                    issues.append(
                        f"{order_id} 的签收日期与订单证据不一致:"
                        f"答案={answer_dates[0]}, 证据={expected}"
                    )
                cited_orders = {item.upper() for item in _ENTITY_RE.findall(cited_text)
                                if item.upper().startswith("ORD-")}
                if cited_orders and order_id not in cited_orders:
                    issues.append(f"{order_id} 的事实引用了另一订单的证据:{sorted(cited_orders)}")

            sentence_numbers = set(re.findall(
                r"\d+(?:\.\d+)?", _CITE_RE.sub("", sentence)
            ))
            cited_numbers = set(re.findall(r"\d+(?:\.\d+)?", cited_text))
            unexpected_numbers = sentence_numbers - cited_numbers
            if citations and unexpected_numbers:
                issues.append(f"答案数值未能由所引证据确认:{sorted(unexpected_numbers)}")

        lower = answer.lower()
        answer_dates = _dates(answer)
        if "v2" in lower and any(item < (2026, 6, 1) for item in answer_dates):
            issues.append("v2 被错误用于 2026-06-01 之前的签收日期")
        if "v1" in lower and any(item >= (2026, 6, 1) for item in answer_dates):
            issues.append("v1 被错误用于 2026-06-01 或之后的签收日期")
        if "v2" in lower and "15" not in all_text:
            issues.append("答案引用 v2,但证据未给出 15 天窗口")
        if "v1" in lower and "7" not in all_text:
            issues.append("答案引用 v1,但证据未给出 7 天窗口")
        return list(dict.fromkeys(issues))

    def verify(self, answer: str, evidence: list[dict], *, query: str = "") -> VerificationResult:
        if not answer.strip():
            return VerificationResult("uncertain", "候选答案为空")
        issues = self._deterministic_issues(answer, evidence)
        query_orders = {item.upper() for item in _ENTITY_RE.findall(query)
                        if item.upper().startswith("ORD-")}
        answer_orders = {item.upper() for item in _ENTITY_RE.findall(answer)
                         if item.upper().startswith("ORD-")}
        known_orders = _order_dates(_evidence_block(evidence))
        introduced = answer_orders - query_orders if query.strip() else set()
        if introduced:
            if not query_orders:
                issues.append(
                    f"答案引入了问题未提供的订单:{','.join(sorted(introduced))}"
                )
            else:
                issues.append(
                    "答案订单与当前问题不一致:"
                    f"问题={','.join(sorted(query_orders))}, "
                    f"答案={','.join(sorted(introduced))}"
                )
        query_dates = _dates(query)
        answer_date_values = _dates(answer)
        for order_id in sorted(query_orders & set(known_orders)):
            expected_date = known_orders[order_id]
            if query_dates and expected_date not in query_dates:
                corrected = (
                    expected_date in answer_date_values
                    and all(item in answer_date_values for item in query_dates)
                    and any(term in answer for term in ("不是", "实际", "订单详情", "以订单"))
                )
                if not corrected:
                    issues.append(
                        f"{order_id} 的用户提供的签收日期与订单证据不一致:"
                        f"用户={query_dates[0]}, 证据={expected_date}; 答案未显式纠正"
                    )
        if issues:
            return VerificationResult("fail", "确定性实体/数值/条件检查失败", issues)
        if answer.lstrip().startswith(REFUSAL_MARKER) and self._semantic is None:
            return VerificationResult("pass", "安全拒答")
        if self._semantic is None:
            return VerificationResult("uncertain", "无法完成独立语义复核")

        response = self._semantic.invoke(ANSWER_VERIFIER_PROMPT.format(
            query=query, evidence=_evidence_block(evidence), answer=answer
        ))
        if isinstance(response, dict):
            response = AnswerVerdict(**response)
        status = str(getattr(response, "status", "uncertain"))
        if status not in {"pass", "fail", "uncertain"}:
            status = "uncertain"
        return VerificationResult(
            status, str(getattr(response, "reason", "") or ""),
            [str(item) for item in getattr(response, "issues", []) or []],
        )

    def review(
        self,
        answer: str,
        evidence: list[dict],
        *,
        query: str = "",
        revise: Callable[[str, list[str]], str] | None = None,
    ) -> VerificationResult:
        checked = self.verify(answer, evidence, query=query)
        if checked.status == "pass":
            return VerificationResult(
                "pass", checked.reason, checked.issues, answer, revision_calls=0
            )
        if checked.status != "fail" or revise is None or self._max_revisions == 0:
            return self.safe_result(checked, revision_calls=0)

        revised = str(revise(answer, checked.issues) or "").strip()
        final_checked = self.verify(revised, evidence, query=query)
        if final_checked.status == "pass":
            return VerificationResult(
                "pass", final_checked.reason, final_checked.issues, revised,
                revision_calls=1,
            )
        return self.safe_result(final_checked, revision_calls=1)

    @staticmethod
    def safe_result(checked: VerificationResult, *, revision_calls: int) -> VerificationResult:
        detail = checked.reason or (checked.issues[0] if checked.issues else "答案未通过复核")
        answer = (
            f"{REFUSAL_MARKER}候选答案未通过最终证据复核({detail[:80]}),"
            "我暂时无法给出可靠答案,请提供订单号或转人工核实。"
        )
        return VerificationResult(
            "refuse", detail, checked.issues, answer, revision_calls=revision_calls
        )
