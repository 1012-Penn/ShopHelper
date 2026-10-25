"""ch09 正式证据置信度：信号、校准和落池快照。"""
import pytest

from app.retrieval import (
    EvidenceCalibration,
    RetrievalResult,
    Retrieved,
    calibrate_evidence_thresholds,
    evidence_confidence,
)


def test_evidence_confidence_combines_top1_count_and_margin():
    result = evidence_confidence(
        [Retrieved(1, 0.92), Retrieved(2, 0.51), Retrieved(3, 0.04)],
        EvidenceCalibration(
            top1_floor=0.30,
            effective_score_floor=0.30,
            min_effective_count=2,
            margin_floor=0.20,
            combined_floor=0.60,
        ),
    )

    assert result.passed is True
    assert result.top1_relevance == 0.92
    assert result.effective_count == 2
    assert result.top1_top2_margin == 0.41
    assert result.signals["top1_relevance"] == 0.92


def test_evidence_confidence_rejects_weak_top1_even_with_candidates():
    result = evidence_confidence(
        [Retrieved(1, 0.21), Retrieved(2, 0.20)],
        EvidenceCalibration(
            top1_floor=0.30,
            effective_score_floor=0.10,
            min_effective_count=1,
            margin_floor=0.0,
            combined_floor=0.0,
        ),
    )

    assert result.passed is False
    assert result.effective_count == 2


def test_single_evidence_uses_top1_as_margin_and_snapshot_keeps_scores():
    result = RetrievalResult(items=[Retrieved(7, 0.88)])
    confidence = evidence_confidence(
        result.items,
        EvidenceCalibration(top1_floor=0.3, effective_score_floor=0.3,
                            min_effective_count=1, margin_floor=0.5,
                            combined_floor=0.1),
    )

    assert confidence.top1_top2_margin == 0.88
    assert result.snapshot(3) == [{
        "rank": 1, "chunk_id": 7, "score": 0.88, "rerank_score": 0.88,
    }]


def test_evidence_calibration_rejects_min_effective_count_zero():
    with pytest.raises(ValueError):
        EvidenceCalibration(min_effective_count=0)


def test_combined_floor_is_diagnostic_only():
    confidence = evidence_confidence(
        [Retrieved(1, 0.20)],
        EvidenceCalibration(top1_floor=0.10, effective_score_floor=0.03,
                            min_effective_count=1, margin_floor=0.0,
                            combined_floor=0.0),
    )
    assert confidence.passed is True


def _samples() -> list[dict]:
    buckets = ["A_policy", "B_model", "C_colloquial", "E_multi", "F_service"]
    samples: list[dict] = []
    for i in range(15):
        samples.append({
            "id": f"P{i:02d}", "bucket": buckets[i % 5], "label": 1,
            "evidence_retrieved": True,
            "top1_relevance": 0.80 + (i % 5) * 0.03,
            "top1_top2_margin": 0.20 + (i % 4) * 0.05,
            "effective_count": 1 + (i % 2),
        })
    for i in range(5):
        samples.append({
            "id": f"H{i:02d}", "bucket": "H_hard_negative", "label": 0,
            "evidence_retrieved": False,
            "top1_relevance": 0.10 + i * 0.05,
            "top1_top2_margin": i * 0.01,
            "effective_count": i % 2,
        })
    for i in range(5):
        samples.append({
            "id": f"J{i:02d}", "bucket": "junk", "label": 0,
            "evidence_retrieved": False,
            "top1_relevance": i * 0.01,
            "top1_top2_margin": 0.0,
            "effective_count": 0,
        })
    return samples


def test_calibration_zero_hard_false_and_full_positive_recall():
    calibration, selection, holdout, split = calibrate_evidence_thresholds(_samples())

    assert selection["status"] == "ok"
    assert selection["hard_negative_false_accept_ids"] == []
    assert selection["positive_recall"] == 1.0
    assert selection["top1_floor"] >= 0.5
    assert calibration.min_effective_count >= 1
    assert calibration.combined_floor == 0.0
    assert holdout["status"] == "ok"
    assert holdout["hard_negative_false_accept_ids"] == []
    assert set(split["calibration_ids"]).isdisjoint(split["holdout_ids"])
    assert split["calibration_ids"] and split["holdout_ids"]


def test_calibration_excludes_retrieval_miss_from_fitting():
    samples = [
        {"id": "P1", "bucket": "A_policy", "label": 1, "evidence_retrieved": True,
         "top1_relevance": 0.90, "top1_top2_margin": 0.40, "effective_count": 2},
        {"id": "P2", "bucket": "A_policy", "label": 1, "evidence_retrieved": False,
         "top1_relevance": 0.00, "top1_top2_margin": 0.00, "effective_count": 0},
        {"id": "H1", "bucket": "H_hard_negative", "label": 0, "evidence_retrieved": False,
         "top1_relevance": 0.20, "top1_top2_margin": 0.01, "effective_count": 0},
    ]
    _, selection, _, _ = calibrate_evidence_thresholds(samples, holdout_ratio=0.0)

    assert selection["retrieval_miss_ids"] == ["P2"]
    assert selection["positives"] == 1
    assert selection["positive_recall"] == 1.0
    assert selection["status"] == "ok"


def test_calibration_marks_infeasible_when_hard_negatives_overlap():
    samples = [
        {"id": f"P{i}", "bucket": "A_policy", "label": 1, "evidence_retrieved": True,
         "top1_relevance": 0.10, "top1_top2_margin": 0.0, "effective_count": 1}
        for i in range(10)
    ] + [
        {"id": f"H{i}", "bucket": "H_hard_negative", "label": 0, "evidence_retrieved": False,
         "top1_relevance": 0.90, "top1_top2_margin": 0.50, "effective_count": 3}
        for i in range(10)
    ]
    _, selection, holdout, _ = calibrate_evidence_thresholds(samples)

    assert selection["status"] == "infeasible"
    assert holdout["status"] == "infeasible"
