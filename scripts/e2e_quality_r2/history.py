"""Zero-call diagnosis of historical RAG traces."""
from __future__ import annotations

from collections import defaultdict


def _labels(trace: dict) -> set[str]:
    checks = trace.get("deterministic_checks") or {}
    return {str(item) for item in checks.get("failure_labels") or trace.get("failure_labels") or []}


def diagnose_history_case(
    case: dict,
    trace: dict,
    evidence: dict[int, dict],
    *,
    reused_by: tuple[str, ...] = (),
) -> dict:
    checks = trace.get("deterministic_checks") or {}
    expected = str(case.get("expected_behavior") or "")
    actual = str(checks.get("behavior_pred") or trace.get("behavior_pred") or "")
    citation = str((checks.get("citation_support") or {}).get("status") or "")
    retrieved = trace.get("retrieved") or []
    missing_ids = [
        int(item["chunk_id"])
        for item in retrieved
        if int(item.get("chunk_id", -1)) not in evidence
    ]
    labels = _labels(trace)
    fact_score = float((checks.get("gold_fact_accuracy") or {}).get("score") or 0.0)
    condition_score = float((checks.get("condition_coverage") or {}).get("score") or 1.0)
    behavior_failed = bool(
        (expected == "refuse" and actual != "refuse")
        or (expected in {"answer", "answer_with_conditions", "verify_then_answer"}
            and actual == "refuse")
        or (expected == "clarify" and actual not in {"clarify", "refuse"})
    )

    if trace.get("answer_status") == "unmeasured":
        outcome = "not_applicable"
    elif missing_ids:
        outcome = "wiring_missing_evidence"
    elif behavior_failed or fact_score < 1.0 or condition_score < 1.0 or bool(
        labels & {"answer_wrong_fact", "answer_missing_condition", "forbidden_claim",
                  "over_refusal", "under_refusal", "clarify_missing_slot"}
    ):
        outcome = "model_wrong"
    elif citation == "uncertain":
        outcome = "scoring_uncertain"
    elif citation == "unsupported":
        outcome = "scoring_unsupported"
    elif expected == "refuse" and actual == "refuse":
        outcome = "correct"
    elif expected == "clarify" and actual == "clarify":
        outcome = "correct"
    elif citation == "supported":
        outcome = "correct"
    else:
        outcome = "not_applicable"

    mapped = []
    for item in retrieved:
        chunk_id = int(item.get("chunk_id", -1))
        row = evidence.get(chunk_id) or {}
        mapped.append({
            "n": int(item.get("n") or item.get("rank") or 0),
            "chunk_id": chunk_id,
            "text": row.get("text"),
        })
    return {
        "case_id": case.get("id") or trace.get("case_id"),
        "expected_behavior": expected,
        "actual_behavior": actual,
        "citation_status": citation,
        "fact_score": fact_score,
        "condition_score": condition_score,
        "failure_labels": sorted(labels),
        "missing_evidence_ids": missing_ids,
        "evidence_map": mapped,
        "outcome": outcome,
        "answer_reused_by": list(reused_by),
    }


def answer_reuse(traces_by_mode: dict[str, list[dict]]) -> dict[str, list[str]]:
    answers: dict[str, dict[str, str]] = defaultdict(dict)
    for mode, traces in traces_by_mode.items():
        for trace in traces:
            answers[str(trace["case_id"])][mode] = str(trace.get("answer") or "")
    reused: dict[str, list[str]] = {}
    for case_id, by_mode in answers.items():
        counts = defaultdict(int)
        for answer in by_mode.values():
            counts[answer] += 1
        modes = list(by_mode)
        reused[case_id] = modes if any(count > 1 for count in counts.values()) else []
    return reused
