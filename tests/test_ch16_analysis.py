from scripts.ch16_analysis import (
    compare_groups,
    paired_bucket_outcomes,
    strict_answer_outcome,
    summarize_group,
)


def _case(case_id, bucket="multi_hop", expected="answer", conditions=()):
    return {
        "case_id": case_id,
        "bucket": bucket,
        "expected_behavior": expected,
        "gold_facts": [{"fact_id": "X", "value": 1}],
        "gold_facts_checked": 1,
        "required_evidence": [{"doc": "X", "quotes": ["x"]}],
        "required_conditions": list(conditions),
    }


def _trace(case_id, *, bucket="multi_hop", r3=1, r5=1, mrr=1.0,
           strict=True, blocked=False, behavior="answer", condition=1.0,
           citation=True, unsupported=0, substantive=1, checks=True,
           conditioned=False):
    deterministic = None
    if checks:
        deterministic = {
            "gold_facts_checked": 1,
            "gold_fact_accuracy": {"passed": strict, "score": float(strict)},
            "condition_coverage": {"passed": condition == 1.0, "score": condition},
            "forbidden_claims": [],
            "citation_support": {"passed": citation, "score": float(citation)},
            "verification_compliant": True,
            "behavior_pred": behavior,
            "unsupported_claims": unsupported,
            "substantive_claims": substantive,
        }
    return {
        "case_id": case_id,
        "bucket": bucket,
        "expected_behavior": "answer",
        "required_conditions": ["有效条件"] if conditioned or condition != 1.0 else [],
        "l1": {"n": 1, "r3": r3, "r5": r5, "mrr": mrr},
        "gate_blocked": blocked,
        "deterministic_checks": deterministic,
        "latency_ms": 100.0,
        "llm_calls": 1,
        "evidence_judge_calls": 0,
        "evidence_judge_cached": 0,
        "cost_actual": 0.0,
    }


def test_gate_blocked_answerable_case_counts_as_strict_failure():
    case = _case("A1")
    trace = _trace("A1", blocked=True, strict=True)

    assert strict_answer_outcome(case, trace) is False


def test_gate_block_is_strict_failure_even_without_answer_checks():
    case = _case("A1")
    trace = {
        "case_id": "A1",
        "expected_behavior": "answer",
        "gate_blocked": True,
        "deterministic_checks": None,
    }

    assert strict_answer_outcome(case, trace) is False


def test_unmeasured_answer_is_not_a_gate_blocked_failure():
    case = _case("A1")
    trace = {
        "case_id": "A1",
        "expected_behavior": "answer",
        "answer_status": "unmeasured",
        "gate_blocked": True,
        "deterministic_checks": None,
    }

    assert strict_answer_outcome(case, trace) is None


def test_strict_answer_requires_facts_conditions_citations_and_no_refusal():
    case = _case("A1", expected="answer_with_conditions", conditions=("有效条件",))
    trace = _trace("A1", condition=1.0)
    assert strict_answer_outcome(case, trace) is True

    trace["deterministic_checks"]["citation_support"]["passed"] = False
    assert strict_answer_outcome(case, trace) is False

    trace["deterministic_checks"]["citation_support"]["passed"] = True
    trace["deterministic_checks"]["behavior_pred"] = "refuse"
    assert strict_answer_outcome(case, trace) is False


def test_unmeasured_answer_is_not_counted_as_wrong_or_correct():
    assert strict_answer_outcome(_case("A1"), _trace("A1", checks=False)) is None


def test_group_summary_separates_retrieval_and_answer_metrics():
    traces = [
        _trace("A1", r3=1, r5=2, mrr=0.5, condition=1.0, conditioned=True,
               unsupported=1, substantive=2),
        _trace("A2", bucket="policy_conflict", r3=None, r5=None, mrr=None,
               strict=False, condition=0.0, citation=False, behavior="refuse",
               unsupported=0, substantive=0),
        _trace("A3", r3=None, r5=None, mrr=None, checks=False),
    ]

    cases = {f"A{i}": _case(f"A{i}", conditions=("有效条件",) if i < 3 else ())
             for i in range(1, 4)}
    summary = summarize_group(traces, cases)

    assert summary["retrieval"]["r3"] == {
        "hits": 1, "measured": 3, "eligible": 3, "unmeasured": 0, "rate": 1 / 3
    }
    assert summary["retrieval"]["mrr"] == {
        "mean": 1 / 6, "n": 3, "eligible": 3, "unmeasured": 0
    }
    assert summary["answers"]["strict_accuracy"] == {
        "correct": 1, "measured": 2, "eligible": 3, "unmeasured": 1, "rate": 0.5
    }
    assert summary["answers"]["ch11_compatible_attempted_strict_accuracy"] == {
        "correct": 1, "measured": 2, "excludes_gate_blocked": True,
        "includes_any_expected_behavior_with_gold_fact_check": True, "rate": 0.5
    }
    assert summary["answers"]["condition_coverage"] == {
        "mean": 0.5, "measured": 2, "eligible": 2, "unmeasured": 0
    }
    assert summary["answers"]["citation_supported_cases"] == {
        "supported": 1, "measured": 2, "rate": 0.5
    }
    assert summary["answers"]["unsupported_claims"] == {
        "unsupported": 1, "claims": 2, "rate": 0.5,
        "cases_with_unsupported_claims": 1, "case_ids": ["A1"]
    }
    assert summary["answers"]["over_refusal"] == {
        "refused": 1, "measured": 2, "eligible": 3, "unmeasured": 1, "rate": 0.5,
        "case_ids": ["A2"],
    }


def test_pairing_reports_retrieval_and_strict_answer_gains_and_losses():
    reference = [
        _trace("A1", r3=None, r5=5, mrr=0.2, strict=False),
        _trace("A2", bucket="policy_conflict", r3=1, r5=2, mrr=1.0, strict=True),
    ]
    candidate = [
        _trace("A1", r3=2, r5=2, mrr=0.5, strict=True),
        _trace("A2", bucket="policy_conflict", r3=None, r5=None, mrr=None, strict=False),
    ]

    paired = compare_groups(reference, candidate)

    assert paired["retrieval"]["r3"]["gained_ids"] == ["A1"]
    assert paired["retrieval"]["r3"]["lost_ids"] == ["A2"]
    assert paired["answers"]["strict"]["gained_ids"] == ["A1"]
    assert paired["answers"]["strict"]["lost_ids"] == ["A2"]


def test_bucket_pairing_preserves_multi_hop_and_policy_conflict_rows():
    reference = [
        _trace("A1", bucket="multi_hop", r3=None, strict=False),
        _trace("A2", bucket="policy_conflict", strict=False),
    ]
    candidate = [
        _trace("A1", bucket="multi_hop", r3=1, strict=True),
        _trace("A2", bucket="policy_conflict", r3=None, strict=False),
    ]

    outcome = paired_bucket_outcomes(
        {"baseline": reference, "loop": candidate}, ("multi_hop", "policy_conflict")
    )

    assert outcome["loop"]["multi_hop"]["retrieval"]["r3"]["gained_ids"] == ["A1"]
    assert outcome["loop"]["multi_hop"]["answers"]["strict"]["gained_ids"] == ["A1"]
    assert outcome["loop"]["policy_conflict"]["answers"]["strict"]["ties"] == 1


def test_latency_uses_nearest_rank_and_reports_call_counts():
    traces = [
        {**_trace("A1"), "latency_ms": 10, "latency_source": "live_api", "paid_api_calls": 1,
         "llm_calls": 1, "evidence_judge_calls": 2},
        {**_trace("A2"), "latency_ms": 20, "latency_source": "cache_or_local", "paid_api_calls": 0,
         "llm_calls": 0, "evidence_judge_calls": 0},
        {**_trace("A3"), "latency_ms": 90, "latency_source": "live_api", "paid_api_calls": 2,
         "llm_calls": 0, "evidence_judge_calls": 0, "rag_analyzer_api_calls": 1},
        {**_trace("A4"), "latency_ms": 100, "latency_source": "cache_or_local", "paid_api_calls": 1,
         "llm_calls": 0, "evidence_judge_calls": 0, "rag_analyzer_cached": 2},
    ]

    operations = summarize_group(traces)["operations"]

    assert operations["latency_ms"] == {
        "method": "nearest_rank", "n": 4, "p50": 20, "p95": 100,
        "by_source": {
            "cache_or_local": {"n": 2, "p50": 20, "p95": 100},
            "live_api": {"n": 2, "p50": 10, "p95": 90},
        },
    }
    assert operations["calls"]["paid_api_calls"] == 4
    assert operations["calls"]["answer_api_calls"] == 1
    assert operations["calls"]["evidence_judge_api_calls"] == 2
    assert operations["calls"]["gap_analyzer_api_calls"] == 1
    assert operations["calls"]["gap_analyzer_cache_hits"] == 2


def test_failure_attribution_and_unsupported_claims_cover_all_attempted_answers():
    case = _case("A1")
    traces = [
        {**_trace("A1", unsupported=1, substantive=2),
         "failure_labels": ["citation_unsupported"]},
        {**_trace("A2", unsupported=2, substantive=3),
         "expected_behavior": "clarify", "failure_labels": ["answer_missing_condition"]},
        {**_trace("A3", unsupported=9, substantive=9, blocked=True),
         "failure_labels": ["gate_blocked"]},
        {**_trace("A4", checks=False), "retrieval_status": "unmeasured",
         "answer_status": "unmeasured", "rag_status": "unmeasured",
         "retrieval_unmeasured_reason": "embedding miss",
         "answer_unmeasured_reason": "embedding miss",
         "rag_unmeasured_reason": "embedding miss"},
    ]

    summary = summarize_group(traces, {"A1": case})

    assert summary["answers"]["unsupported_claims"] == {
        "unsupported": 3, "claims": 5, "rate": 0.6,
        "cases_with_unsupported_claims": 2, "case_ids": ["A1", "A2"],
    }
    assert summary["failure_attribution"]["failure_labels"] == {
        "answer_missing_condition": {"count": 1, "case_ids": ["A2"]},
        "citation_unsupported": {"count": 1, "case_ids": ["A1"]},
        "gate_blocked": {"count": 1, "case_ids": ["A3"]},
    }
    assert summary["failure_attribution"]["unmeasured"] == {
        layer: {"count": 1, "case_ids": ["A4"], "reasons": {"embedding miss": 1}}
        for layer in ("retrieval", "answer", "rag")
    }
