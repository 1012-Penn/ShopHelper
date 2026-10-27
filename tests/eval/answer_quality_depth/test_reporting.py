from __future__ import annotations

from scripts.answer_quality_depth.reporting import aggregate_result_rows


def _row(case_id: str, arm: str, status: str, *, split="dev", denom="evidence_answer"):
    return {
        "split": split,
        "case_id": case_id,
        "arm": arm,
        "bucket": "multi_hop",
        "denominator": denom,
        "final_status": status,
        "cost_usd": 0.1,
        "latency_ms": 100,
        "judge": {
            "verdict": {
                "answer_status": "correct" if status == "correct" else "incorrect",
                "grounding": "supported" if status == "correct" else "partly_supported",
                "citation_support": "supported" if status == "correct" else "partial",
                "unsupported_claims": [],
            }
        },
    }


def test_aggregate_keeps_uncertain_in_the_reported_denominator() -> None:
    rows = [
        _row("A", "baseline", "uncertain"),
        _row("A", "slate", "correct"),
        _row("B", "baseline", "correct"),
        _row("B", "slate", "correct"),
    ]
    result = aggregate_result_rows(
        rows,
        evidence_rows={
            "A": {"baseline_hit": False, "slate_hit": True},
            "B": {"baseline_hit": True, "slate_hit": True},
        },
    )

    answer = result["overall"]["by_denominator"]["evidence_answer"]
    assert answer["baseline"]["n"] == 2
    assert answer["baseline"]["uncertain"] == 1
    assert answer["baseline"]["strict_accuracy"] == 0.5
    assert (
        result["coverage_attribution"]["baseline_missing_slate_complete"]["outcomes"][
            "uncertain->correct"
        ]
        == 1
    )


def test_aggregate_reports_actionable_paired_deltas() -> None:
    rows = [
        _row("A", "baseline", "incorrect"),
        _row("A", "slate", "correct"),
        _row("B", "baseline", "correct", denom="clarify"),
        _row("B", "slate", "uncertain", denom="clarify"),
    ]
    result = aggregate_result_rows(rows, evidence_rows={})

    assert result["overall"]["pairwise"]["slate_vs_baseline"] == {
        "wins": 1,
        "losses": 1,
        "ties": 0,
    }


def test_fact_and_condition_rates_only_use_applicable_cases() -> None:
    rows = [
        {
            **_row("A", "baseline", "correct"),
            "facts_checked": 1,
            "facts_passed": True,
            "conditions_passed": True,
            "deterministic_checks": {"condition_coverage": {"detail": "1/1"}},
        },
        {
            **_row("B", "baseline", "incorrect"),
            "facts_checked": 0,
            "facts_passed": True,
            "conditions_passed": True,
            "deterministic_checks": {"condition_coverage": {"detail": "no_conditions"}},
        },
    ]
    result = aggregate_result_rows(rows, evidence_rows={})
    baseline = result["overall"]["by_denominator"]["evidence_answer"]["baseline"]

    assert baseline["facts_passed"] == 1
    assert baseline["facts_measured"] == 1
    assert baseline["conditions_passed"] == 1
    assert baseline["conditions_measured"] == 1
