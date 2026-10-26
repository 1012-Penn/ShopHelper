from __future__ import annotations

from scripts.answer_plan_r3.rescore import rescore_row


def test_rescore_replays_plan_validation_without_new_provider_calls():
    case = {
        "id": "D1",
        "expected_behavior": "answer",
        "facts": [{
            "fact_id": "F1",
            "entity": "SH-B7",
            "predicate": "质保",
            "value": "一年",
            "accepted_phrases": ["一年"],
            "require_entity": True,
        }],
        "conditions": [],
        "condition_aliases": {},
        "forbidden_claims": [],
        "requires_citation": True,
        "clarifying_question": "",
        "clarifying_slots": [],
        "evidence_ids": ["EV-1"],
    }
    evidence = [{
        "evidence_id": "EV-1",
        "text": "SH-B7 智能手环一年质保。",
        "source_id": "catalog",
        "source_title": "商品目录",
        "entities": ["SH-B7"],
    }]
    stored = {
        "case_id": "D1",
        "split": "dev",
        "arm": "answer_plan",
        "answer": "旧的安全收口",
        "citation_items": {},
        "evidence_ids": ["EV-1"],
        "provider_calls": 2,
        "cost_usd": 0.01,
        "latency_ms": 100,
        "correction_attempts": 1,
        "technical_failure": False,
        "error": None,
        "plan": {
            "disposition": "answer",
            "claims": [{
                "claim_id": "C1",
                "text": "SH-B7 手环提供一年质保",
                "subject": "SH-B7",
                "value": "一年",
                "unit": "",
                "evidence": [{
                    "evidence_id": "EV-1",
                    "quote": "一年质保",
                }],
            }],
        },
    }

    row = rescore_row(stored, case, evidence)

    assert row["provider_calls"] == 2
    assert row["cost_usd"] == 0.01
    assert row["answer"] != stored["answer"]
    assert "一年质保 [1]" in row["answer"]
    assert row["score"]["strict_success"] is True
    assert row["rescored"] is True
