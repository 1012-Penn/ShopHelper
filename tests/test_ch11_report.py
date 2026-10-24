"""分层聚合、阻断项、配对统计与报告稳定性。"""
from __future__ import annotations

from scripts.eval_report import aggregate_traces, blocking_metrics, render_report
from scripts.eval_stats import bootstrap_ci, mcnemar_exact


def _checks(**overrides):
    payload = {
        "gold_facts_checked": 1,
        "gold_fact_accuracy": {"passed": True, "score": 1.0},
        "condition_coverage": {"passed": True, "score": 1.0},
        "forbidden_claims": [],
        "citation_support": {"passed": True, "score": 1.0},
        "clarify_slot_recall": {"score": 1.0},
        "clarify_precision": {"score": 1.0},
        "unsupported_claim_rate": 0.0,
        "verification_compliant": True,
        "behavior_pred": "answer",
    }
    payload.update(overrides)
    return payload


def _trace(case_id="T1", *, bucket="single_hop", behavior="answer", gate=True,
           l4=True, forbidden=None, checks=None, labels=None):
    return {
        "case_id": case_id,
        "bucket": bucket,
        "expected_behavior": behavior,
        "l1": {
            "n": 1 if bucket not in {"hard_negative", "ood", "clarify"} else 0,
            "coverage_rank": 1,
            "r3": 1,
            "r5": 1,
            "r10": 1,
            "group_recall": 1.0,
            "single1": 1,
            "single3": 1,
            "mrr": 1.0,
            "evidence_precision": 1.0,
            "hard_negative_hit_rate": 0.0,
        },
        "gate": {"passed": gate},
        "deterministic_checks": checks,
        "forbidden_claims": forbidden or [],
        "failure_labels": labels or [],
        "l4_applicable": l4,
        "llm_calls": 1,
        "tokens_in": 10,
        "tokens_out": 5,
        "cost_estimate": 0.001,
        "cached": False,
        "latency_ms": 100.0,
    }


def test_aggregate_denominators_for_clarify_and_refuse():
    traces = [
        _trace("A", checks=_checks()),
        _trace("C", bucket="clarify", behavior="clarify",
               checks=_checks(behavior_pred="clarify"), l4=True),
        _trace("H", bucket="hard_negative", behavior="refuse", gate=False,
               checks=_checks(behavior_pred="refuse", gold_facts_checked=0)),
        _trace("TOOL", checks=_checks(), l4=False),
    ]
    agg = aggregate_traces(traces)
    assert agg["l1"]["n"] == 2  # clarify/refuse 无 required_evidence,不进 L1 分母
    assert agg["l4"]["n"] == 3  # 工具题不进入 L4 分母
    assert agg["blocking"]["hard_negative_false_accept"]["denominator"] == 1


def test_blocking_metrics_are_not_offset_by_recall():
    traces = [_trace("A", checks=_checks())]
    traces.append(_trace("H", bucket="hard_negative", behavior="refuse", gate=True,
                         checks=_checks(behavior_pred="answer"), labels=["under_refusal"]))
    agg = aggregate_traces(traces)
    assert agg["l1"]["r10"]["rate"] == 1.0
    assert blocking_metrics(agg)["hard_negative_false_accept"]["count"] == 1
    assert "total_score" not in agg
    assert "score" not in agg


def test_clarify_precision_and_forbidden_denominators():
    traces = [
        _trace("C", bucket="clarify", behavior="clarify",
               checks=_checks(behavior_pred="clarify", clarify_precision={"score": 0.5})),
        _trace("F", checks=_checks(forbidden_claims=["可以便宜"]), forbidden=["可以便宜"]),
    ]
    agg = aggregate_traces(traces)
    assert agg["l3"]["clarify_precision"] == 0.5
    assert agg["l3"]["forbidden_claim_rate"] == 1.0


def test_bootstrap_ci_deterministic_with_seed():
    values = [0.0, 0.5, 1.0, 1.0]
    assert bootstrap_ci(values, seed=1) == bootstrap_ci(values, seed=1)


def test_mcnemar_exact_simple_case():
    assert mcnemar_exact([False] * 5, [True] * 5) == {
        "b": 5, "c": 0, "n": 5, "statistic": 0, "p_value": 0.0625
    }


def test_report_keeps_historical_section_separate():
    agg = aggregate_traces([_trace()])
    contract = {
        "_sha": "abc",
        "dataset": {"split": "frozen-v1", "n": 90, "sha256": "d" * 64},
        "index": {"build_id": "b"},
        "retrieval": {"calibration_status": "stale"},
        "pricing": {"verified": False},
    }
    report = render_report(agg, contract, historical_md="旧 3 文档 R@10=1.000")
    assert "## 历史基线" in report
    assert "旧 3 文档" in report
    assert "R@10 | 1.0" in report


def test_report_reproducible_for_same_traces():
    traces = [_trace()]
    agg = aggregate_traces(traces)
    contract = {
        "_sha": "abc",
        "dataset": {"split": "frozen-v1", "n": 90, "sha256": "d" * 64},
        "index": {"build_id": "b"},
        "retrieval": {"calibration_status": "stale"},
        "pricing": {"verified": False},
    }
    assert render_report(agg, contract) == render_report(aggregate_traces(traces), contract)
