"""ch09 正式证据置信度：信号、校准和落池快照。"""

from app.retrieval import (
    EvidenceCalibration,
    Retrieved,
    RetrievalResult,
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


def test_calibration_prefers_threshold_that_keeps_positive_recall():
    samples = [
        {"top1_relevance": 0.90, "effective_count": 2, "top1_top2_margin": 0.40, "relevant": True},
        {"top1_relevance": 0.70, "effective_count": 1, "top1_top2_margin": 0.20, "relevant": True},
        {"top1_relevance": 0.12, "effective_count": 1, "top1_top2_margin": 0.01, "relevant": False},
        {"top1_relevance": 0.35, "effective_count": 1, "top1_top2_margin": 0.02, "relevant": False},
    ]

    calibration = calibrate_evidence_thresholds(samples, min_recall=1.0)

    assert calibration.top1_floor == 0.70
    assert calibration.source == "ch04_eval_set"
