"""Aggregate paired retrieval and query-intent outcomes."""
from __future__ import annotations

import math
from collections import Counter


def _ratio(values: list[bool]) -> dict:
    return {"numerator": sum(bool(value) for value in values),
            "denominator": len(values)}


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def _paired(left: dict[str, dict], right: dict[str, dict], metric: str) -> dict:
    shared = sorted(set(left) & set(right))
    if metric in {"complete_at_5", "complete_at_10"}:
        shared = [
            case_id for case_id in shared
            if left[case_id]["metrics"].get("coverage_applicable", True)
            and right[case_id]["metrics"].get("coverage_applicable", True)
        ]
    gains = losses = ties = 0
    for case_id in shared:
        a = bool(left[case_id]["metrics"].get(metric))
        b = bool(right[case_id]["metrics"].get(metric))
        if a and not b:
            gains += 1
        elif b and not a:
            losses += 1
        else:
            ties += 1
    return {"pairs": len(shared), "gains": gains, "losses": losses, "ties": ties}


def summarize_variant_traces(traces: list[dict]) -> dict:
    """Report explicit denominators and paired changes without percentage rounding."""
    by_variant: dict[str, list[dict]] = {}
    by_case_variant: dict[str, dict[str, dict]] = {}
    for trace in traces:
        variant = str(trace["variant"])
        by_variant.setdefault(variant, []).append(trace)
        by_case_variant.setdefault(str(trace["case_id"]), {})[variant] = trace

    variants: dict[str, dict] = {}
    for variant, rows in sorted(by_variant.items()):
        metrics = [row["metrics"] for row in rows]
        coverage_metrics = [
            metric for metric in metrics
            if metric.get("coverage_applicable", True)
        ]
        latencies = [float(row.get("latency_ms") or 0.0) for row in rows]
        costs = [float(row.get("cost_usd") or 0.0) for row in rows]
        failures = sum(bool(row.get("error")) for row in rows)
        query_errors = sum(bool(row.get("query_errors")) for row in rows)
        variants[variant] = {
            "cases": len(rows),
            "coverage_cases": len(coverage_metrics),
            "complete_at_5": _ratio([m.get("complete_at_5") for m in coverage_metrics]),
            "complete_at_10": _ratio([m.get("complete_at_10") for m in coverage_metrics]),
            "mrr_at_10": round(
                sum(float(m.get("mrr_at_10") or 0.0) for m in coverage_metrics)
                / len(coverage_metrics),
                6,
            ) if coverage_metrics else 0.0,
            "anchors_complete": _ratio([m.get("anchors_complete") for m in metrics]),
            "intent_preserved": _ratio([m.get("intent_preserved") for m in metrics]),
            "wrong_entity": sum(bool(m.get("wrong_entity")) for m in metrics),
            "latency_ms": {
                "p50": _percentile(latencies, 0.50),
                "p95": _percentile(latencies, 0.95),
            },
            "cost_estimate_usd": round(sum(costs), 8),
            "failures": failures,
            "failure_rate": _ratio([bool(row.get("error")) for row in rows]),
            "query_errors": query_errors,
            "query_error_rate": _ratio([
                bool(row.get("query_errors")) for row in rows
            ]),
        }

    comparisons = {}
    for left_name, right_name in (
        ("legacy", "raw"),
        ("safe", "raw"),
        ("safe", "legacy"),
    ):
        left = {
            case_id: variants_by_name[left_name]
            for case_id, variants_by_name in by_case_variant.items()
            if left_name in variants_by_name
        }
        right = {
            case_id: variants_by_name[right_name]
            for case_id, variants_by_name in by_case_variant.items()
            if right_name in variants_by_name
        }
        comparisons[f"{left_name}_vs_{right_name}"] = {
            "pairs": len(set(left) & set(right)),
            "complete_at_5": _paired(left, right, "complete_at_5"),
            "complete_at_10": _paired(left, right, "complete_at_10"),
            "intent_preserved": _paired(left, right, "intent_preserved"),
        }

    bucket_counts = Counter(
        str(row.get("bucket") or "") for row in traces if row["variant"] == "raw"
    )
    return {"variants": variants, "paired": comparisons,
            "bucket_counts": dict(sorted(bucket_counts.items()))}
