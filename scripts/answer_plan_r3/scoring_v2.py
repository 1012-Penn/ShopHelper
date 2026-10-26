"""Post-run scoring audit v2 for answer-plan R3.

This version fixes three deterministic scoring defects found while diagnosing
the first held-out run. It is not an independent evaluation: it re-scores
saved answers after those answers were observed.
"""
from __future__ import annotations

import re

from scripts.answer_plan_r3.scoring import (
    _CITE_RE,
    _ENTITY_RE,
    _NEGATION_RE,
    _accepted_phrases,
    _behavior_passes,
    _classify_behavior,
    _condition_passes,
    _norm,
    _sentences,
    _without_citations,
)


def _predicate_terms(predicate: str) -> list[str]:
    compact = _norm(predicate)
    if not compact:
        return []
    pieces = [
        _norm(item)
        for item in re.split(r"后|时|的|与|和|状态", predicate)
        if len(_norm(item)) >= 2
    ]
    return list(dict.fromkeys(pieces or [compact]))


def _contexts(text: str, term: str, radius: int = 14) -> list[str]:
    return [
        text[max(0, match.start() - radius):match.end() + radius]
        for match in re.finditer(re.escape(term), text)
    ]


def _fact_value_matches_v2(fact: dict, text: str) -> bool:
    norm = _norm(text)
    value = fact.get("value")
    predicate = fact.get("predicate") or ""
    terms = _predicate_terms(predicate)
    contexts = [context for term in terms for context in _contexts(norm, term)]
    if isinstance(value, bool):
        if value:
            return bool(
                terms
                and all(term in norm for term in terms)
                and contexts
                and not any(_NEGATION_RE.search(context) for context in contexts)
            )
        return bool(
            terms
            and all(term in norm for term in terms)
            and any(_NEGATION_RE.search(context) for context in contexts)
        )
    if value is None or value == "":
        return True
    return any(_norm(phrase) in norm for phrase in _accepted_phrases(fact))


def _fact_passes_v2(fact: dict, answer: str) -> bool:
    text = _without_citations(answer)
    if fact.get("require_entity") and fact.get("entity"):
        if _norm(fact["entity"]) not in _norm(text):
            return False
    return _fact_value_matches_v2(fact, text)


def _fact_mentioned_v2(fact: dict, sentence: str) -> bool:
    norm = _norm(sentence)
    terms = _predicate_terms(fact.get("predicate") or "")
    if terms and all(term in norm for term in terms):
        return True
    if any(_norm(item) in norm for item in _accepted_phrases(fact)):
        return True
    return False


def _condition_aliases(case: dict, evidence: dict, condition: str) -> list[str]:
    values = [
        condition,
        *(case.get("condition_aliases") or {}).get(condition, []),
        *(evidence.get("condition_aliases") or {}).get(condition, []),
    ]
    return list(dict.fromkeys(item for item in values if item))


def _citation_status_v2(case: dict, answer: str, evidence: list[dict], mapping: dict) -> dict:
    if not case.get("requires_citation"):
        return {"status": "not_required", "invalid_labels": [], "claims": []}
    mapping = {
        int(key): str(value)
        for key, value in (mapping or {}).items()
    }
    by_id = {row["evidence_id"]: row for row in evidence}
    records = []
    invalid: set[int] = set()
    unsupported = False
    uncertain = False
    covered_facts: set[str] = set()
    covered_conditions: set[str] = set()
    any_citation = False

    for original in _sentences(answer):
        labels = [int(value) for value in _CITE_RE.findall(original)]
        if not labels:
            continue
        any_citation = True
        sentence = _without_citations(original).strip()
        cited_rows = []
        for label in labels:
            row = by_id.get(mapping.get(label))
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
        misplaced = answer_entities - cited_entities if cited_entities else set()
        if misplaced:
            unsupported = True
            records.append({
                "sentence": sentence,
                "status": "unsupported",
                "reason": f"答案实体未由所引证据支持:{sorted(misplaced)}",
            })
            continue

        mentioned_facts = []
        for fact in case.get("facts") or []:
            entity = str(fact.get("entity") or "").upper()
            if answer_entities and entity and entity not in answer_entities:
                continue
            if _fact_mentioned_v2(fact, sentence):
                mentioned_facts.append(fact)

        supported_here = True
        for fact in mentioned_facts:
            entity = str(fact.get("entity") or "")
            entity_in_evidence = not entity or any(
                entity.upper() in (row.get("text") or "").upper()
                for row in cited_rows
            )
            if not entity_in_evidence or not _fact_value_matches_v2(fact, cited_text):
                supported_here = False
            else:
                covered_facts.add(str(fact.get("fact_id") or ""))

        mentioned_conditions = []
        for condition in case.get("conditions") or []:
            aliases = []
            for row in cited_rows:
                aliases.extend(_condition_aliases(case, row, condition))
            aliases = list(dict.fromkeys(aliases))
            if any(_norm(alias) in _norm(sentence) for alias in aliases):
                mentioned_conditions.append(condition)
                if any(_norm(alias) in _norm(cited_text) for alias in aliases):
                    covered_conditions.add(condition)
                else:
                    supported_here = False

        if not mentioned_facts and not mentioned_conditions:
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
            uncertain = True
            records.append({
                "sentence": sentence,
                "status": "uncertain",
                "reason": "所引证据未直接包含答案中的事实或条件",
            })

    all_facts = {
        str(fact.get("fact_id") or "") for fact in case.get("facts") or []
        if fact.get("fact_id")
    }
    if all_facts - covered_facts or set(case.get("conditions") or []) - covered_conditions:
        uncertain = True
    if invalid or unsupported or not any_citation:
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


def score_case_v2(case: dict, outcome: dict, evidence: list[dict]) -> dict:
    answer = str(outcome.get("answer") or "")
    fact_items = [
        {
            "fact_id": fact.get("fact_id"),
            "passed": _fact_passes_v2(fact, answer),
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
    citations = _citation_status_v2(
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
        "scoring_version": "answer-plan-r3-score-v2-audit",
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
