from __future__ import annotations

import json

import pytest

from scripts.answer_plan_r3.runner import (
    HeldoutAlreadyRun,
    claim_heldout_once,
    summarize_results,
)


def test_heldout_claim_is_durable_and_refuses_second_run(tmp_path):
    marker = tmp_path / "heldout-run.json"
    claim_heldout_once(
        marker,
        split="heldout",
        config_sha256="a" * 64,
        case_ids=["H1", "H2"],
    )

    with pytest.raises(HeldoutAlreadyRun):
        claim_heldout_once(
            marker,
            split="heldout",
            config_sha256="a" * 64,
            case_ids=["H1", "H2"],
        )
    assert json.loads(marker.read_text(encoding="utf-8"))["status"] == "started"


def test_summary_keeps_uncertain_out_of_success_and_reports_pairwise_deltas():
    rows = [
        {
            "case_id": "D1",
            "split": "dev",
            "arm": "free_text",
            "score": {"strict_success": False, "review_required": True,
                      "facts": {"passed": True}, "conditions": {"passed": True}},
            "provider_calls": 1,
            "cost_usd": 0.01,
            "latency_ms": 100,
            "correction_attempts": 0,
        },
        {
            "case_id": "D1",
            "split": "dev",
            "arm": "answer_plan",
            "score": {"strict_success": True, "review_required": False,
                      "facts": {"passed": True}, "conditions": {"passed": True}},
            "provider_calls": 2,
            "cost_usd": 0.02,
            "latency_ms": 200,
            "correction_attempts": 1,
        },
        {
            "case_id": "D2",
            "split": "dev",
            "arm": "free_text",
            "score": {"strict_success": True, "review_required": False,
                      "facts": {"passed": True}, "conditions": {"passed": True}},
            "provider_calls": 1,
            "cost_usd": 0.01,
            "latency_ms": 120,
            "correction_attempts": 0,
        },
        {
            "case_id": "D2",
            "split": "dev",
            "arm": "answer_plan",
            "score": {"strict_success": False, "review_required": False,
                      "facts": {"passed": False}, "conditions": {"passed": True}},
            "provider_calls": 1,
            "cost_usd": 0.02,
            "latency_ms": 220,
            "correction_attempts": 0,
        },
    ]

    summary = summarize_results(rows)

    assert summary["dev"]["free_text"]["strict_success"] == 1
    assert summary["dev"]["answer_plan"]["strict_success"] == 1
    assert summary["dev"]["answer_plan"]["review_required"] == 0
    assert summary["pairwise"]["dev"]["answer_plan_vs_free_text"] == {
        "wins": 1,
        "losses": 1,
        "ties": 0,
    }
    assert summary["dev"]["answer_plan"]["cost_usd"] == pytest.approx(0.04)
    assert summary["dev"]["answer_plan"]["provider_calls"] == 3


def test_summary_reports_citation_status_breakdown_and_token_totals():
    rows = [
        {
            "case_id": f"D{index}",
            "split": "dev",
            "arm": arm,
            "score": {
                "strict_success": False,
                "review_required": status == "uncertain",
                "behavior": {"passed": True},
                "facts": {"passed": True},
                "conditions": {"passed": True},
                "citations": {"status": status},
            },
            "provider_calls": 1,
            "cost_usd": 0.0,
            "latency_ms": 1,
            "input_tokens": 10,
            "output_tokens": 2,
        }
        for index, (arm, status) in enumerate([
            ("free_text", "supported"),
            ("answer_plan", "unsupported"),
            ("free_text", "uncertain"),
            ("answer_plan", "supported"),
        ])
    ]

    summary = summarize_results(rows)

    assert summary["dev"]["free_text"]["citation_supported"] == 1
    assert summary["dev"]["free_text"]["citation_uncertain"] == 1
    assert summary["dev"]["answer_plan"]["citation_supported"] == 1
    assert summary["dev"]["answer_plan"]["citation_unsupported"] == 1
    assert summary["dev"]["answer_plan"]["input_tokens"] == 20
    assert summary["dev"]["answer_plan"]["output_tokens"] == 4
