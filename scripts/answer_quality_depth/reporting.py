"""Aggregate answer-depth artifacts without changing the underlying responses."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path


def _p(values: Iterable[float], fraction: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    return round(ordered[max(0, math.ceil(len(ordered) * fraction) - 1)], 3)


def _arm_summary(rows: list[dict]) -> dict:
    statuses = Counter(
        str(row.get("final_status") or row.get("status")) for row in rows
    )
    n = len(rows)
    latencies = [
        float(row["latency_ms"]) for row in rows if row.get("latency_ms") is not None
    ]
    fact_rows = [row for row in rows if int(row.get("facts_checked") or 0) > 0]
    condition_rows = [
        row
        for row in rows
        if str(
            (
                (row.get("deterministic_checks") or {}).get("condition_coverage") or {}
            ).get("detail")
        )
        not in {"", "no_conditions", "None"}
    ]
    return {
        "n": n,
        "correct": statuses["correct"],
        "incorrect": statuses["incorrect"],
        "uncertain": statuses["uncertain"],
        "technical_failure": statuses["technical_failure"],
        "strict_accuracy": statuses["correct"] / n if n else None,
        "uncertainty_rate": statuses["uncertain"] / n if n else None,
        "technical_failure_rate": statuses["technical_failure"] / n if n else None,
        "facts_passed": sum(bool(row.get("facts_passed")) for row in fact_rows),
        "facts_measured": len(fact_rows),
        "conditions_passed": sum(
            bool(row.get("conditions_passed")) for row in condition_rows
        ),
        "conditions_measured": len(condition_rows),
        "behavior_passed": sum(bool(row.get("behavior_passed")) for row in rows),
        "citation_status": dict(
            sorted(
                Counter(str(row.get("citation_status") or "") for row in rows).items()
            )
        ),
        "cost_usd": round(sum(float(row.get("cost_usd") or 0.0) for row in rows), 8),
        "latency_ms": {"p50": _p(latencies, 0.50), "p95": _p(latencies, 0.95)},
    }


def _paired(rows: list[dict]) -> dict:
    by_case: dict[str, dict[str, str]] = defaultdict(dict)
    for row in rows:
        by_case[str(row["case_id"])][str(row["arm"])] = str(
            row.get("final_status") or row["status"]
        )
    wins = losses = ties = 0
    for values in by_case.values():
        if "baseline" not in values or "slate" not in values:
            continue
        if values["slate"] == "correct" and values["baseline"] != "correct":
            wins += 1
        elif values["baseline"] == "correct" and values["slate"] != "correct":
            losses += 1
        else:
            ties += 1
    return {
        "slate_vs_baseline": {
            "wins": wins,
            "losses": losses,
            "ties": ties,
        }
    }


def _section(rows: list[dict]) -> dict:
    by_denominator = {}
    for denominator in sorted({str(row["denominator"]) for row in rows}):
        by_denominator[denominator] = {
            arm: _arm_summary(
                [
                    row
                    for row in rows
                    if row["denominator"] == denominator and row["arm"] == arm
                ]
            )
            for arm in ("baseline", "slate")
        }
    by_bucket = {}
    for bucket in sorted({str(row.get("bucket") or "") for row in rows}):
        bucket_rows = [row for row in rows if str(row.get("bucket") or "") == bucket]
        by_bucket[bucket] = {
            arm: _arm_summary([row for row in bucket_rows if row["arm"] == arm])
            for arm in ("baseline", "slate")
        }
    by_evidence_state = {}
    for state in sorted({str(row.get("evidence_state") or "") for row in rows} - {""}):
        state_rows = [
            row for row in rows if str(row.get("evidence_state") or "") == state
        ]
        by_evidence_state[state] = {
            arm: _arm_summary([row for row in state_rows if row["arm"] == arm])
            for arm in ("baseline", "slate")
        }
    return {
        "by_denominator": by_denominator,
        "by_bucket": by_bucket,
        "by_evidence_state": by_evidence_state,
        "pairwise": _paired(rows),
        "judge": _judge_summary(rows),
    }


def _judge_summary(rows: list[dict]) -> dict:
    answer_rows = [row for row in rows if row["denominator"] == "evidence_answer"]
    result = {}
    for arm in ("baseline", "slate"):
        selected = [row for row in answer_rows if row["arm"] == arm]
        verdicts = [(row.get("judge") or {}).get("verdict") or {} for row in selected]
        result[arm] = {
            "answer_status": dict(
                sorted(Counter(item.get("answer_status") for item in verdicts).items())
            ),
            "grounding": dict(
                sorted(Counter(item.get("grounding") for item in verdicts).items())
            ),
            "citation_support": dict(
                sorted(
                    Counter(item.get("citation_support") for item in verdicts).items()
                )
            ),
            "rows_with_unsupported_claims": sum(
                bool(item.get("unsupported_claims")) for item in verdicts
            ),
            "unsupported_claim_count": sum(
                len(item.get("unsupported_claims") or []) for item in verdicts
            ),
        }
    return result


def aggregate_result_rows(rows: list[dict], *, evidence_rows: dict[str, dict]) -> dict:
    by_split = {}
    for split in sorted({str(row["split"]) for row in rows}):
        by_split[split] = _section([row for row in rows if str(row["split"]) == split])
    by_case: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_case[str(row["case_id"])][str(row["arm"])] = row
    coverage: dict[str, dict] = {}
    for name, predicate in (
        (
            "baseline_missing_slate_complete",
            lambda e: not bool(e["baseline_hit"]) and bool(e["slate_hit"]),
        ),
        (
            "both_complete",
            lambda e: bool(e["baseline_hit"]) and bool(e["slate_hit"]),
        ),
        (
            "baseline_complete_slate_missing",
            lambda e: bool(e["baseline_hit"]) and not bool(e["slate_hit"]),
        ),
        (
            "both_missing",
            lambda e: not bool(e["baseline_hit"]) and not bool(e["slate_hit"]),
        ),
    ):
        selected = [
            case_id
            for case_id, evidence in evidence_rows.items()
            if predicate(evidence)
        ]
        outcomes = Counter()
        answer_cases = 0
        for case_id in selected:
            pair = by_case.get(case_id) or {}
            if "baseline" not in pair or "slate" not in pair:
                continue
            if pair["baseline"]["denominator"] != "evidence_answer":
                continue
            answer_cases += 1
            outcomes[
                f"{pair['baseline'].get('final_status')}->{pair['slate'].get('final_status')}"
            ] += 1
        coverage[name] = {
            "cases": len(selected),
            "answer_cases": answer_cases,
            "outcomes": dict(sorted(outcomes.items())),
        }
    coverage_totals = {
        "eligible": sum(
            item.get("baseline_hit") is not None for item in evidence_rows.values()
        ),
        "baseline_complete": sum(
            bool(item["baseline_hit"]) for item in evidence_rows.values()
        ),
        "slate_complete": sum(
            bool(item["slate_hit"]) for item in evidence_rows.values()
        ),
    }
    if all(
        "baseline_top5_hit" in item and "slate_top5_hit" in item
        for item in evidence_rows.values()
    ):
        coverage_totals["baseline_top5_complete"] = sum(
            bool(item["baseline_top5_hit"]) for item in evidence_rows.values()
        )
        coverage_totals["slate_top5_complete"] = sum(
            bool(item["slate_top5_hit"]) for item in evidence_rows.values()
        )
    return {
        "dataset": {
            "questions": len(by_case),
            "responses": len(rows),
            "technical_failures": sum(
                bool(row.get("technical_failure")) for row in rows
            ),
            "reasoning_responses": sum(
                str(row.get("generation", {}).get("raw", "{}")).find(
                    "reasoning_content"
                )
                >= 0
                for row in rows
            ),
        },
        "splits": by_split,
        "overall": _section(rows),
        "coverage": coverage_totals,
        "coverage_attribution": coverage,
        "answer_cost_usd": round(
            sum(float(row.get("cost_usd") or 0.0) for row in rows), 8
        ),
    }


def load_results(paths: dict[str, Path]) -> tuple[list[dict], dict[str, dict]]:
    rows = []
    evidence = {}
    for split, path in paths.items():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                row["split"] = split
                rows.append(row)
    return rows, evidence
