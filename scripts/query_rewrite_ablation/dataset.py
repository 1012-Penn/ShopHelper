"""Materialize compact ablation seeds from the corpus fact/evidence source."""
from __future__ import annotations

import copy
from collections.abc import Iterable

from scripts.corpus_facts import FactBook


def _gold_fact(fact) -> dict:
    return {
        "fact_id": fact.fact_id,
        "key": fact.key,
        "value": fact.value,
        "unit": fact.unit,
        "conditions": list(fact.conditions),
        "effective_from": fact.effective_from.isoformat()
        if fact.effective_from is not None else None,
        "effective_to": fact.effective_to.isoformat()
        if fact.effective_to is not None else None,
    }


def build_cases(
    seeds: Iterable[dict],
    factbook: FactBook,
    *,
    split: str,
    review_status: str = "draft",
    label_version: str = "qr-v1",
) -> list[dict]:
    """Expand seeds and fail if a fact is not covered by a selected evidence unit."""
    cases: list[dict] = []
    seen_ids: set[str] = set()
    for seed in seeds:
        case = copy.deepcopy(seed)
        case_id = str(case.get("id") or "").strip()
        if not case_id or case_id in seen_ids:
            raise ValueError(f"query-rewrite case id 为空或重复:{case_id!r}")
        seen_ids.add(case_id)
        fact_ids = list(dict.fromkeys(case.pop("gold_fact_ids", [])))
        evidence_ids = list(dict.fromkeys(case.pop("evidence_ids", [])))
        if not fact_ids or not evidence_ids:
            raise ValueError(f"{case_id}: gold_fact_ids/evidence_ids 不能为空")
        missing_facts = sorted(set(fact_ids) - set(factbook.facts))
        missing_evidence = sorted(set(evidence_ids) - set(factbook.evidence))
        if missing_facts:
            raise ValueError(f"{case_id}: 未知 fact:{missing_facts}")
        if missing_evidence:
            raise ValueError(f"{case_id}: 未知 evidence:{missing_evidence}")

        selected_evidence = [factbook.evidence[evidence_id] for evidence_id in evidence_ids]
        for fact_id in fact_ids:
            if not any(fact_id in unit.fact_ids for unit in selected_evidence):
                raise ValueError(
                    f"{case_id}: fact {fact_id} 未被选中的 evidence 覆盖"
                )

        source_refs: list[str] = []
        full_case = {
            "schema_version": 2,
            "id": case_id,
            "split": split,
            "label_version": label_version,
            "bucket": str(case.pop("bucket", "query_rewrite")),
            "query": str(case.pop("query", "")).strip(),
            "conversation_context": list(case.pop("conversation_context", [])),
            "expected_behavior": str(case.pop("expected_behavior", "answer")),
            "answerable": True,
            "evidence_state": "sufficient",
            "required_slots": [],
            "clarifying_slots": [],
            "required_conditions": [],
            "condition_aliases": {},
            "slot_aliases": {},
            "verification_markers": [],
            "forbidden_claims": [],
            "expected_refusal_reason": "",
            "as_of_date": None,
            "requires_tools": [],
            "tool_fixtures": [],
            "source": "synthetic",
            "annotator": "codex",
            "reviewer": "codex",
            "review_status": review_status,
        }
        if not full_case["query"]:
            raise ValueError(f"{case_id}: query 不能为空")
        full_case["gold_facts"] = [
            _gold_fact(factbook.fact(fact_id)) for fact_id in fact_ids
        ]
        full_case["required_evidence"] = [
            {
                "doc": unit.doc,
                "section": unit.section,
                "quotes": list(unit.quotes),
                "fact_ids": list(unit.fact_ids),
            }
            for unit in selected_evidence
        ]
        for fact_id in fact_ids:
            source_refs.extend(
                factbook.fact(fact_id).evidence_ids
            )
        full_case["source_ref"] = ",".join(dict.fromkeys(source_refs))
        full_case.update(case)
        full_case["required_anchor_groups"] = [
            list(group) for group in full_case.get("required_anchor_groups", [])
        ]
        full_case["forbidden_anchors"] = list(full_case.get("forbidden_anchors", []))
        full_case["preserve_negation"] = bool(full_case.get("preserve_negation", False))
        full_case["template_id"] = str(
            full_case.get("template_id") or full_case["bucket"]
        )
        cases.append(full_case)
    return cases
