"""证据信号口径:clarify 不参与检索正样本,也不冒充 OOD 误放。"""
from __future__ import annotations

from types import SimpleNamespace

from scripts.gen_evidence_signals import _entailment_metrics, signal_for


def test_signal_for_marks_clarify_as_non_positive():
    case = {"id": "CL01", "bucket": "clarify", "query": "多少钱？", "answerable": True}
    result = SimpleNamespace(items=[SimpleNamespace(chunk_id=1, score=0.9)])
    signal = signal_for(case, result, 0.0, {}, verdict=None)
    assert signal["label"] == 0
    assert signal["evidence_retrieved"] is False


def test_clarify_is_not_counted_as_ood_false_accept():
    signals = [
        {"id": "P1", "label": 1, "bucket": "single_hop", "entailed": True},
        {"id": "H1", "label": 0, "bucket": "H_hard_negative", "entailed": True},
        {"id": "CL1", "label": 0, "bucket": "clarify", "entailed": True},
        {"id": "J1", "label": 0, "bucket": "junk", "entailed": True},
    ]
    metrics = _entailment_metrics(signals)
    assert metrics["ood_false_accept_ids"] == ["J1"]
