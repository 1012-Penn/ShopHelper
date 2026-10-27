from scripts.eval_evidence_slate import summarize_rows


def test_summarize_rows_reports_paired_wins_losses_and_bucket_counts():
    rows = [
        {
            "case_id": "A",
            "bucket": "multi_hop",
            "baseline_hit": False,
            "slate_hit": True,
            "baseline_rank": None,
            "slate_rank": 4,
            "latency_ms": 100.0,
            "selection_latency_ms": 0.2,
        },
        {
            "case_id": "B",
            "bucket": "multi_hop",
            "baseline_hit": True,
            "slate_hit": True,
            "baseline_rank": 2,
            "slate_rank": 3,
            "latency_ms": 120.0,
            "selection_latency_ms": 0.3,
        },
        {
            "case_id": "C",
            "bucket": "policy_conflict",
            "baseline_hit": True,
            "slate_hit": False,
            "baseline_rank": 5,
            "slate_rank": None,
            "latency_ms": 140.0,
            "selection_latency_ms": 0.4,
        },
    ]

    summary = summarize_rows(rows)

    assert summary["n"] == 3
    assert summary["baseline_hits"] == 2
    assert summary["slate_hits"] == 2
    assert summary["wins"] == ["A"]
    assert summary["losses"] == ["C"]
    assert summary["mcnemar"]["b"] == 1
    assert summary["mcnemar"]["c"] == 1
    assert summary["by_bucket"]["multi_hop"]["wins"] == ["A"]
    assert summary["by_bucket"]["multi_hop"]["losses"] == []
