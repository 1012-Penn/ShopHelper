"""Input-boundary and layered deterministic scoring for E2E quality R2."""
from __future__ import annotations

import json
import math
import re
import unicodedata
from typing import Any

from scripts.eval_checks import (
    citation_support_rate,
    condition_coverage,
    forbidden_claims_found,
    parse_citations,
)
from app.guard import REFUSAL_MARKER

FORBIDDEN_KEYS = {
    "expected_behavior",
    "gold_facts",
    "required_conditions",
    "forbidden_claims",
    "answer_key",
    "reference_answer",
    "scoring",
    "rubric",
    "required_evidence_terms",
    "expected_tools",
    "clarifying_slots",
}
ANSWER_BEHAVIORS = {"answer", "answer_with_conditions", "verify_then_answer"}


class InputBoundaryError(RuntimeError):
    """A scoring label or gold field leaked into the model-facing projection."""


def assert_model_input_safe(projection: object, case: dict) -> None:
    def walk(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                normalized = str(key).strip().lower()
                if normalized in FORBIDDEN_KEYS:
                    raise InputBoundaryError(f"forbidden model-input field at {path}.{key}")
                walk(child, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")
        elif isinstance(value, str):
            for fact in case.get("facts") or []:
                fact_id = str(fact.get("fact_id") or "").strip()
                if len(fact_id) >= 4 and fact_id in value:
                    raise InputBoundaryError(f"gold fact id leaked at {path}")

    walk(projection, "$")


def _json_object(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _evidence_text(item: dict) -> str:
    pieces = [
        item.get("category"),
        item.get("section_path"),
        item.get("question"),
        item.get("answer"),
        item.get("text"),
    ]
    return "\n".join(str(piece) for piece in pieces if piece not in (None, "")).strip()


def resolve_citation_evidence(state: dict, tool_calls: list[dict]) -> dict[int, dict]:
    mapped: dict[int, dict] = {}
    offset = 0
    for index, item in enumerate(state.get("evidence") or [], start=1):
        if not isinstance(item, dict):
            continue
        n = int(item.get("n") or index)
        mapped[n] = {**item, "text": _evidence_text(item)}
        offset = max(offset, n)
    for call in tool_calls:
        result = _json_object(call.get("result"))
        for item in result.get("items") or []:
            if not isinstance(item, dict):
                continue
            local_n = int(item.get("n") or 1)
            n = offset + local_n
            mapped[n] = {
                **item,
                "n": n,
                "local_n": local_n,
                "tool_name": call.get("name"),
                "text": _evidence_text(item),
            }
        offset += len(result.get("items") or [])
    return mapped


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(text)).lower().split())


def classify_observed_behavior(answer: str) -> str:
    """Classify answer shape without treating an optional upsell as clarification.

    A complete answer may end with "需要我帮您介绍其他型号吗". The shared
    historical classifier sees the trailing question and marks it clarify. R2
    requires a short leading request for a missing slot before choosing clarify.
    """
    text = str(answer or "").strip()
    normalized = _normalize(text)
    marker = _normalize(REFUSAL_MARKER)
    if normalized.startswith(marker):
        return "refuse"
    refusal = (
        "无法回答", "无法确认", "无法判断", "无法根据", "无法依据", "无法提供",
        "无法给出", "无法告知", "无法协助", "查不到", "证据不足", "暂时无法",
    )
    if normalized.startswith(("抱歉", "很抱歉")) and any(term in normalized for term in refusal):
        return "refuse"
    if any(term in normalized for term in refusal[:8]):
        return "refuse"
    asks_for_slot = (
        text.startswith(("请提供", "请告知", "请告诉我", "请问", "麻烦提供")),
        any(term in normalized for term in (
            "订单号", "订单编号", "哪个型号", "哪款", "具体商品", "会员等级", "收货地区",
        )),
    )
    if ("?" in text or "？" in text) and len(text) <= 100 and any(asks_for_slot):
        return "clarify"
    return "answer"


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[。！？!?；;\n])", text) if part.strip()]


_UNIT_ALIASES = {
    "元": ("元", "块"),
    "块": ("元", "块"),
    "小时": ("小时", "h"),
    "天": ("天", "日"),
    "日": ("天", "日"),
    "年": ("年",),
    "个月": ("个月", "月"),
    "月": ("个月", "月"),
    "米": ("米",),
    "克": ("克", "g"),
    "kg": ("kg", "千克"),
    "%": ("%", "％"),
}


def _quantity_in_sentence(sentence: str, value, unit: str) -> bool:
    if isinstance(value, bool):
        yes = ("支持", "可以", "能", "适用", "参与", "享受", "计入")
        no = ("不支持", "不可以", "不能", "不适用", "不参与", "不享受", "不计入")
        terms = yes if value else no
        return any(term in _normalize(sentence) for term in terms)
    if not isinstance(value, (int, float)):
        return _normalize(value) in _normalize(sentence)
    normalized = _normalize(sentence).replace(",", "")
    aliases = _UNIT_ALIASES.get(_normalize(unit), (_normalize(unit),)) if unit else ()
    if aliases:
        unit_pattern = "(?:" + "|".join(re.escape(alias) for alias in aliases) + ")"
    else:
        unit_pattern = r"(?=$|[^\d.])"
    for match in re.finditer(r"(-?\d+(?:\.\d+)?)\s*" + unit_pattern, normalized):
        if math.isclose(float(match.group(1)), float(value), abs_tol=1e-9):
            return True
    return False


def _fact_passes(answer: str, fact: dict) -> tuple[bool, str]:
    accepted = [str(item) for item in fact.get("accepted_phrases") or []]
    entity = str(fact.get("entity") or "")
    require_entity = bool(fact.get("require_entity"))
    for sentence in _sentences(answer):
        if require_entity and entity and _normalize(entity) not in _normalize(sentence):
            continue
        if any(_normalize(phrase) in _normalize(sentence) for phrase in accepted):
            return True, sentence
        if _quantity_in_sentence(sentence, fact.get("value"), str(fact.get("unit") or "")):
            return True, sentence
    return False, ""


def _expected_entity_terms(case: dict) -> list[str]:
    terms = []
    for fact in case.get("facts") or []:
        if fact.get("require_entity") and fact.get("entity"):
            terms.append(str(fact["entity"]))
    for term in case.get("required_entities") or []:
        if term not in terms:
            terms.append(str(term))
    return terms


def _tool_match(call: dict, expected: dict) -> bool:
    if call.get("name") != expected.get("name"):
        return False
    args = _json_object(call.get("args"))
    for key, value in (expected.get("args_contains") or {}).items():
        if str(args.get(key)) != str(value):
            return False
    return not bool(call.get("is_error"))


def score_case(case: dict, outcome: dict) -> dict:
    answer = str(outcome.get("answer") or "")
    expected = str(case.get("expected_behavior") or "answer")
    actual_behavior = classify_observed_behavior(answer)
    technical = [str(item) for item in outcome.get("technical_failures") or []]
    failure_labels: list[str] = []
    if technical:
        failure_labels = ["technical_failure"]
        return {
            "case_id": case.get("id"),
            "expected_behavior": expected,
            "actual_behavior": actual_behavior,
            "behavior": {"passed": False, "expected": expected, "actual": actual_behavior},
            "technical_failure": True,
            "technical_failures": technical,
            "task_success": False,
            "review_required": False,
            "failure_labels": failure_labels,
        }

    if expected == "clarify":
        behavior_passed = actual_behavior == "clarify"
    elif expected == "refuse":
        behavior_passed = actual_behavior == "refuse"
    elif expected == "verify_then_answer":
        markers = case.get("verification_markers") or []
        behavior_passed = actual_behavior == "answer" and (
            not markers or any(str(marker) in answer for marker in markers)
        )
    else:
        behavior_passed = actual_behavior == "answer"
    if not behavior_passed:
        failure_labels.append(f"behavior_{actual_behavior}_expected_{expected}")

    facts_rows = []
    for fact in case.get("facts") or []:
        passed, sentence = _fact_passes(answer, fact)
        facts_rows.append({
            "fact_id": fact.get("fact_id"),
            "passed": passed,
            "sentence": sentence,
        })
    facts_passed = all(row["passed"] for row in facts_rows)
    if not facts_passed:
        failure_labels.append("answer_wrong_fact")

    condition = condition_coverage(
        answer,
        list(case.get("conditions") or []),
        case.get("condition_aliases") or {},
    )
    if not condition.passed:
        failure_labels.append("answer_missing_condition")

    forbidden = forbidden_claims_found(answer, list(case.get("forbidden_claims") or []))
    if forbidden:
        failure_labels.append("forbidden_claim")

    entity_terms = _expected_entity_terms(case)
    missing_entities = [term for term in entity_terms if _normalize(term) not in _normalize(answer)]
    forbidden_entities = [
        str(term) for term in case.get("forbidden_entities") or []
        if _normalize(term) in _normalize(answer)
    ]
    entities_passed = not missing_entities and not forbidden_entities
    if not entities_passed:
        failure_labels.append("entity_mismatch")

    evidence = {
        int(n): {**item, "text": item.get("text") or _evidence_text(item)}
        for n, item in (outcome.get("evidence") or {}).items()
    }
    evidence_texts = {n: item["text"] for n, item in evidence.items()}
    citations = parse_citations(answer)
    invalid = sorted({n for _sentence, n in citations if n not in evidence})
    missing_terms = []
    for requirement in case.get("required_evidence_terms") or []:
        terms = [str(term) for term in requirement]
        found = False
        for _sentence, n in citations:
            text = _normalize(evidence.get(n, {}).get("text") or "")
            if all(_normalize(term) in text for term in terms):
                found = True
                break
        if not found:
            missing_terms.append(terms)
    support = (
        citation_support_rate(answer, evidence_texts)
        if evidence_texts and not invalid
        else None
    )
    citation_status = (support.status if support is not None else "") or "missing"
    if invalid:
        citation_status = "invalid"
    if expected in {"refuse", "clarify"} and not citations:
        citation_passed = True
        citation_status = "not_required"
    elif not bool(case.get("requires_citation", True)):
        citation_passed = not invalid
        citation_status = "tool_only"
    else:
        citation_passed = (
            bool(citations)
            and not invalid
            and not missing_terms
            and support is not None
            and support.status == "supported"
        )
    if invalid:
        failure_labels.append("citation_invalid")
    requires_citation = bool(case.get("requires_citation", True))
    if (requires_citation and not citations
            and expected not in {"refuse", "clarify"}):
        failure_labels.append("citation_missing")
    if missing_terms and not invalid:
        failure_labels.append("evidence_missing")
    if support is not None and citation_status == "unsupported" and not invalid:
        failure_labels.append("citation_unsupported")
    if citation_status == "uncertain":
        failure_labels.append("citation_uncertain")

    tools_passed = True
    actual_calls = list(outcome.get("tool_calls") or [])
    required_tools = list(case.get("expected_tools") or [])
    for expected_call in required_tools:
        if not any(_tool_match(call, expected_call) for call in actual_calls):
            tools_passed = False
            break
    if not tools_passed:
        failure_labels.append("tool_incomplete")

    required_nodes = [str(item) for item in case.get("required_nodes") or []]
    route_nodes = [str(item) for item in outcome.get("route_nodes") or []]
    route_passed = all(node in route_nodes for node in required_nodes)
    if not route_passed:
        failure_labels.append("route_incomplete")

    success = (
        behavior_passed
        and facts_passed
        and condition.passed
        and not forbidden
        and entities_passed
        and citation_passed
        and tools_passed
        and route_passed
    )
    return {
        "scoring_version": "e2e-quality-r2-score-v2",
        "case_id": case.get("id"),
        "expected_behavior": expected,
        "actual_behavior": actual_behavior,
        "behavior": {"passed": behavior_passed, "expected": expected, "actual": actual_behavior},
        "facts": {"passed": facts_passed, "rows": facts_rows},
        "conditions": {
            "passed": condition.passed,
            "score": condition.score,
            "detail": condition.detail,
            "measured": len(case.get("conditions") or []),
        },
        "forbidden_claims": forbidden,
        "entities": {
            "passed": entities_passed,
            "measured": len(entity_terms),
            "missing": missing_entities,
            "forbidden_present": forbidden_entities,
        },
        "citations": {
            "passed": citation_passed,
            "status": citation_status,
            "invalid_labels": invalid,
            "missing_source_terms": missing_terms,
            "labels": [n for _sentence, n in citations],
        },
        "tools": {
            "passed": tools_passed,
            "calls": actual_calls,
            "required": len(required_tools),
        },
        "route": {
            "passed": route_passed,
            "nodes": route_nodes,
            "required": len(required_nodes),
        },
        "technical_failure": False,
        "technical_failures": [],
        "task_success": success,
        "review_required": citation_status == "uncertain",
        "failure_labels": list(dict.fromkeys(failure_labels)),
    }
