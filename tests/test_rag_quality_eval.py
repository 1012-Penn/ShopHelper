import json
from pathlib import Path

from app.answer_verifier import VerificationResult
from scripts.eval_rag_quality import (
    apply_answer_verifier,
    classify_answer_outcome,
    summarize_quality,
)


def _case(case_id="Q1", bucket="single_hop"):
    return {
        "id": case_id,
        "bucket": bucket,
        "expected_behavior": "answer",
        "gold_facts": [{"fact_id": "P", "value": 8, "unit": "元"}],
        "required_conditions": [],
    }


def _trace(case_id="Q1", status="supported", labels=None, blocked=False):
    return {
        "case_id": case_id,
        "bucket": "single_hop",
        "answer_status": "measured",
        "gate_blocked": blocked,
        "answer": "普通订单邮费 8 元[1]。",
        "retrieved": [{"rank": 1, "chunk_id": 1, "score": 0.9}],
        "l1": {"n": 1, "r3": 1, "r5": 1, "mrr": 1.0},
        "deterministic_checks": {
            "gold_facts_checked": 1,
            "gold_fact_accuracy": {"passed": True, "score": 1.0},
            "condition_coverage": {"passed": True, "score": 1.0},
            "citation_support": {"passed": status == "supported", "status": status},
            "forbidden_claims": [],
            "behavior_pred": "answer",
            "failure_labels": labels or [],
        },
    }


def test_uncertain_citation_is_not_hard_accuracy():
    assert classify_answer_outcome(_case(), _trace(status="uncertain")) == "uncertain"
    assert classify_answer_outcome(_case(), _trace(status="supported")) == "correct"
    assert classify_answer_outcome(
        _case(), _trace(status="supported", labels=["answer_wrong_fact"])
    ) == "incorrect"


def test_quality_summary_reports_tri_state_and_measured_denominator():
    cases = [_case("Q1"), _case("Q2"), _case("Q3")]
    traces = [
        _trace("Q1", status="supported"),
        _trace("Q2", status="supported", labels=["answer_wrong_fact"]),
        _trace("Q3", status="uncertain"),
    ]
    summary = summarize_quality(cases, traces)

    assert summary["answers"]["correct"] == 1
    assert summary["answers"]["incorrect"] == 1
    assert summary["answers"]["uncertain"] == 1
    assert summary["answers"]["accuracy_among_determinate"] == 0.5
    assert summary["answers"]["uncertain_case_ids"] == ["Q3"]
    assert summary["factual_answers"]["correct"] == 2
    assert summary["factual_answers"]["incorrect"] == 1
    assert summary["citations"]["supported"] == 2
    assert summary["citations"]["uncertain"] == 1


def test_answer_verifier_overlay_preserves_original_and_rescores_final_answer():
    class FakeVerifier:
        def review(self, answer, evidence, *, query="", revise=None):
            return VerificationResult(
                "pass", "corrected", [], "普通订单邮费 8 元[1]。", revision_calls=1
            )

    case = _case()
    trace = _trace(status="unsupported", labels=["citation_unsupported"])
    trace["answer"] = "普通订单邮费 9 元[1]。"
    result = apply_answer_verifier(
        case, trace, {1: "普通订单邮费 8 元。"}, FakeVerifier(),
        json.loads(Path("tests/eval/overlays/rag-quality-v3.json").read_text()),
    )

    assert result["original_answer"] == "普通订单邮费 9 元[1]。"
    assert result["answer"] == "普通订单邮费 8 元[1]。"
    assert result["answer_verification"]["revision_calls"] == 1
    assert result["deterministic_checks"]["gold_fact_accuracy"]["passed"] is True


def test_quality_overlay_is_versioned_and_every_case_has_a_basis():
    overlay = json.loads(
        Path("tests/eval/overlays/rag-quality-v3.json").read_text()
    )
    cases = {
        case["id"]: case
        for split in ("dev", "frozen-v1")
        for case in map(json.loads, Path(f"tests/eval/splits/{split}.jsonl").read_text().splitlines())
    }
    assert overlay["version"] == "rag-quality-v3"
    assert overlay["revision_basis"]
    for case_id, revision in overlay["cases"].items():
        assert case_id in cases
        assert revision["basis"].strip()
        fact_ids = {fact["fact_id"] for fact in cases[case_id]["gold_facts"]}
        assert set(revision.get("background_gold_fact_ids") or []) <= fact_ids
