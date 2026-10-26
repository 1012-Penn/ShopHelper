from __future__ import annotations

from scripts.e2e_quality_r2.history import diagnose_history_case


def test_history_uncertain_is_scoring_review_not_model_wrong():
    case = {
        "id": "D1",
        "expected_behavior": "answer",
        "gold_facts": [{"fact_id": "F1", "value": 299, "unit": "元"}],
        "required_conditions": [],
    }
    trace = {
        "case_id": "D1",
        "answer": "SH-E300 售价 299 元 [1]。",
        "retrieved": [{"n": 1, "chunk_id": 1}],
        "deterministic_checks": {
            "gold_facts_checked": 1,
            "gold_fact_accuracy": {"score": 1.0},
            "condition_coverage": {"score": 1.0},
            "citation_support": {"status": "uncertain"},
            "failure_labels": ["citation_uncertain"],
            "behavior_pred": "answer",
        },
    }

    result = diagnose_history_case(
        case,
        trace,
        {1: {"chunk_id": 1, "text": "SH-E300 售价 299 元。"}},
        reused_by=("baseline", "controlled_loop"),
    )

    assert result["outcome"] == "scoring_uncertain"
    assert result["citation_status"] == "uncertain"
    assert result["answer_reused_by"] == ["baseline", "controlled_loop"]


def test_history_fact_mismatch_is_model_wrong_but_missing_evidence_is_wiring():
    base = {
        "id": "D2",
        "expected_behavior": "answer",
        "gold_facts": [{"fact_id": "F1", "value": 299, "unit": "元"}],
        "required_conditions": [],
    }
    trace = {
        "case_id": "D2",
        "answer": "SH-E300 售价 399 元 [1]。",
        "retrieved": [{"n": 1, "chunk_id": 1}],
        "deterministic_checks": {
            "gold_facts_checked": 1,
            "gold_fact_accuracy": {"score": 0.0},
            "condition_coverage": {"score": 1.0},
            "citation_support": {"status": "unsupported"},
            "failure_labels": ["answer_wrong_fact"],
            "behavior_pred": "answer",
        },
    }
    assert diagnose_history_case(
        base, trace, {1: {"chunk_id": 1, "text": "SH-E300 售价 299 元。"}}
    )["outcome"] == "model_wrong"

    broken = {**trace, "retrieved": [{"n": 1, "chunk_id": 999}]}
    assert diagnose_history_case(
        base, broken, {}
    )["outcome"] == "wiring_missing_evidence"


def test_history_summary_counts_the_rows_it_receives():
    from scripts.eval_e2e_quality_r2 import _history_summary

    rows = [
        {
            "outcome": "correct",
            "expected_behavior": "answer",
            "actual_behavior": "answer",
            "answer_reused_by": ["baseline", "controlled_loop"],
        }
    ]

    summary = _history_summary(rows, "dev", "baseline")

    assert summary["n"] == 1
    assert summary["outcomes"] == {"correct": 1}
    assert summary["answers_reused_in_two_or_more_modes"] == 1
