"""Frozen deterministic scoring for answer-plan R3.

This module is deliberately independent from ``app.answer_plan``. It scores the
user-visible answer against frozen case facts, conditions, behavior and citation
relationships. ``uncertain`` never contributes to strict success.
"""
from __future__ import annotations

import re

_CITE_RE = re.compile(r"\[(\d+)\]")
_ENTITY_RE = re.compile(r"\b(?:SH|ORD)-[A-Za-z0-9]+\b", re.IGNORECASE)
_PUNCT_RE = re.compile(r"[\s，。；;,：:、（）()【】\[\]]+")
_NEGATION_RE = re.compile(r"不支持|不能|不可|不享|不参与|不予|不免费|无法|仅限|不是|无需")


def _norm(value: object) -> str:
    return _PUNCT_RE.sub("", str(value or "")).casefold()


def _without_citations(value: str) -> str:
    return _CITE_RE.sub("", str(value or ""))


def _sentences(answer: str) -> list[str]:
    rows = []
    for sentence in re.split(r"(?<=[。！？!?\n])", str(answer or "")):
        sentence = sentence.strip()
        if not sentence:
            continue
        if rows and not _without_citations(sentence).strip():
            rows[-1] = f"{rows[-1]} {sentence}"
        else:
            rows.append(sentence)
    return rows


def _accepted_phrases(fact: dict) -> list[str]:
    phrases = [str(item) for item in fact.get("accepted_phrases") or []]
    if fact.get("value") not in (None, ""):
        value = str(fact["value"])
        unit = str(fact.get("unit") or "")
        phrases.extend([value, f"{value}{unit}", f"{value} {unit}".strip()])
    return [item for item in dict.fromkeys(phrases) if item]


def _fact_value_matches(fact: dict, text: str) -> bool:
    norm = _norm(text)
    value = fact.get("value")
    predicate = _norm(fact.get("predicate") or "")
    if isinstance(value, bool):
        if value:
            return bool(predicate and predicate in norm and not _NEGATION_RE.search(
                re.sub(r"不参与|不适用", "", pred_context(norm, predicate))
            ))
        if not predicate or predicate not in norm:
            return False
        context = pred_context(norm, predicate)
        return bool(_NEGATION_RE.search(context))
    if value is None or value == "":
        return True
    return any(_norm(phrase) in norm for phrase in _accepted_phrases(fact))


def pred_context(text: str, predicate: str, radius: int = 14) -> str:
    index = text.find(predicate)
    if index < 0:
        return ""
    return text[max(0, index - radius):index + len(predicate) + radius]


def _fact_passes(fact: dict, answer: str) -> bool:
    text = _without_citations(answer)
    norm = _norm(text)
    entity = str(fact.get("entity") or "")
    if fact.get("require_entity") and entity and _norm(entity) not in norm:
        return False
    return _fact_value_matches(fact, text)


def _condition_passes(condition: str, aliases: dict, answer: str) -> bool:
    norm = _norm(_without_citations(answer))
    phrases = [condition, *(aliases.get(condition) or [])]
    return any(_norm(phrase) in norm for phrase in phrases)


def _classify_behavior(answer: str) -> str:
    text = str(answer or "").strip()
    norm = _norm(text)
    clarify = (
        ("请提供" in text or "请问" in text)
        and any(term in norm for term in ("型号", "sku", "订单号", "哪一笔", "哪个产品"))
    )
    if clarify:
        return "clarify"
    if (
        any(term in norm for term in ("无法确认", "证据不足", "没有足够", "不能确定"))
        or ("抱歉" in text and any(term in norm for term in ("无法", "不足", "不确定")))
    ):
        return "refuse"
    return "answer"


def _behavior_passes(expected: str, observed: str, answer: str, case: dict) -> bool:
    if expected == "clarify":
        if observed != "clarify":
            return False
        slots = set(case.get("clarifying_slots") or [])
        norm = _norm(answer)
        if "sku" in slots and not any(term in norm for term in ("型号", "sku", "商品")):
            return False
        if "order_id" in slots and not any(
            term in norm for term in ("订单号", "订单", "哪一笔")
        ):
            return False
        return True
    if expected == "refuse":
        return observed == "refuse"
    return observed == "answer"


def _fact_mentioned(fact: dict, sentence: str) -> bool:
    norm = _norm(sentence)
    entity = _norm(fact.get("entity") or "")
    predicate = _norm(fact.get("predicate") or "")
    phrases = [_norm(item) for item in _accepted_phrases(fact)]
    return bool(
        (entity and entity in norm)
        or (predicate and predicate in norm)
        or any(phrase and phrase in norm for phrase in phrases)
    )


def _citation_status(case: dict, answer: str, evidence: list[dict], mapping: dict) -> dict:
    if not case.get("requires_citation"):
        return {"status": "not_required", "invalid_labels": [], "claims": []}
    by_id = {row["evidence_id"]: row for row in evidence}
    mapping = {
        int(key): str(value)
        for key, value in (mapping or {}).items()
    }
    records = []
    invalid = set()
    unsupported = False
    uncertain = False
    covered_facts: set[str] = set()
    covered_conditions: set[str] = set()
    any_citation = False
    for original in _sentences(answer):
        labels = [int(value) for value in _CITE_RE.findall(original)]
        if not labels:
            continue
        sentence = _without_citations(original).strip()
        any_citation = True
        cited_rows = []
        for label in labels:
            evidence_id = (mapping or {}).get(label)
            row = by_id.get(str(evidence_id))
            if row is None:
                invalid.add(label)
                unsupported = True
            else:
                cited_rows.append(row)
        if not cited_rows:
            continue
        cited_text = "\n".join(row.get("text") or "" for row in cited_rows)
        cited_entities = {
            str(item).upper()
            for row in cited_rows
            for item in row.get("entities") or []
        }
        answer_entities = {
            item.upper() for item in _ENTITY_RE.findall(sentence)
        }
        misplaced_entity = answer_entities - cited_entities if cited_entities else set()
        if misplaced_entity:
            unsupported = True
            records.append({
                "sentence": sentence,
                "status": "unsupported",
                "reason": f"答案实体未由所引证据支持:{sorted(misplaced_entity)}",
            })
            continue
        mentioned = [
            fact for fact in case.get("facts") or []
            if (
                not answer_entities
                or not fact.get("entity")
                or str(fact["entity"]).upper() in answer_entities
            )
            and _fact_mentioned(fact, sentence)
        ]
        supported_here = True
        for fact in mentioned:
            fact_id = str(fact.get("fact_id") or "")
            entity = str(fact.get("entity") or "")
            entity_ok = (
                not fact.get("require_entity")
                or not entity
                or any(entity.upper() in (row.get("text") or "").upper() for row in cited_rows)
            )
            if not entity_ok or not _fact_value_matches(fact, cited_text):
                supported_here = False
            else:
                covered_facts.add(fact_id)
        mentioned_conditions = []
        for condition in case.get("conditions") or []:
            aliases = [condition, *(case.get("condition_aliases") or {}).get(condition, [])]
            if any(_norm(item) in _norm(sentence) for item in aliases):
                mentioned_conditions.append(condition)
                evidence_phrases = [condition, *[
                    alias for alias in aliases if _norm(alias) in _norm(sentence)
                ]]
                if any(_norm(item) in _norm(cited_text) for item in evidence_phrases):
                    covered_conditions.add(condition)
                else:
                    # Synonym overlap without exact evidence text remains
                    # uncertain; it is not promoted to support.
                    supported_here = False
        if not mentioned and not mentioned_conditions:
            uncertain = True
            records.append({
                "sentence": sentence,
                "status": "uncertain",
                "reason": "引用句没有可绑定的业务事实或条件",
            })
        elif supported_here:
            records.append({
                "sentence": sentence,
                "status": "supported",
                "reason": "事实/条件直接由所引证据支持",
            })
        else:
            # A clear entity mismatch was handled above. Remaining cases are
            # insufficient direct support, not proof of entailment.
            uncertain = True
            records.append({
                "sentence": sentence,
                "status": "uncertain",
                "reason": "所引证据未直接包含答案中的事实或条件",
            })
    all_fact_ids = {
        str(fact.get("fact_id") or "") for fact in case.get("facts") or []
        if fact.get("fact_id")
    }
    all_conditions = set(case.get("conditions") or [])
    if all_fact_ids - covered_facts or all_conditions - covered_conditions:
        uncertain = True
    if invalid:
        status = "unsupported"
    elif unsupported:
        status = "unsupported"
    elif not any_citation:
        status = "unsupported"
    elif uncertain:
        status = "uncertain"
    else:
        status = "supported"
    return {
        "status": status,
        "invalid_labels": sorted(invalid),
        "claims": records,
        "covered_fact_ids": sorted(covered_facts),
        "covered_conditions": sorted(covered_conditions),
    }


def score_case(case: dict, outcome: dict, evidence: list[dict]) -> dict:
    answer = str(outcome.get("answer") or "")
    fact_items = [
        {
            "fact_id": fact.get("fact_id"),
            "passed": _fact_passes(fact, answer),
        }
        for fact in case.get("facts") or []
    ]
    condition_items = [
        {
            "condition": condition,
            "passed": _condition_passes(
                condition, case.get("condition_aliases") or {}, answer
            ),
        }
        for condition in case.get("conditions") or []
    ]
    banned = [
        claim for claim in case.get("forbidden_claims") or []
        if _norm(claim) in _norm(answer)
    ]
    observed = _classify_behavior(answer)
    behavior_passed = _behavior_passes(
        str(case.get("expected_behavior") or "answer"), observed, answer, case
    )
    citations = _citation_status(
        case,
        answer,
        evidence,
        outcome.get("citation_items") or {},
    )
    facts_passed = all(row["passed"] for row in fact_items)
    conditions_passed = all(row["passed"] for row in condition_items)
    strict_success = bool(
        not outcome.get("technical_failure")
        and behavior_passed
        and facts_passed
        and conditions_passed
        and not banned
        and citations["status"] in {"supported", "not_required"}
    )
    return {
        "behavior": {
            "expected": case.get("expected_behavior"),
            "observed": observed,
            "passed": behavior_passed,
        },
        "facts": {"passed": facts_passed, "items": fact_items},
        "conditions": {"passed": conditions_passed, "items": condition_items},
        "forbidden_claims": {"passed": not banned, "matches": banned},
        "citations": citations,
        "review_required": citations["status"] == "uncertain",
        "strict_success": strict_success,
        "misrefusal": bool(
            str(case.get("expected_behavior") or "").startswith("answer")
            and observed in {"clarify", "refuse"}
        ),
    }
