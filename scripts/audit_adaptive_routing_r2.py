"""Zero-model analysis for the frozen adaptive-routing R2 artifacts.

This script does not import the evaluator, so post-hoc report generation does
not alter the frozen execution harness hash.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "tests" / "eval" / "adaptive_routing_r2"
RUNS_DIR = DATA_DIR / "runs"
FINAL_DIR = RUNS_DIR / "final-rescore-v3"
SESSION_CAP_USD = 2.0
ARMS = ("fixed_single", "fixed_multi", "adaptive")
LEDGERS = (
    ROOT / "data" / "adaptive_routing_r2_paid_run.sqlite3",
    ROOT / "data" / "adaptive_routing_r2_paid_run_final.sqlite3",
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def load_rows(path: Path = FINAL_DIR / "traces.jsonl.gz") -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _rate(values: list[dict]) -> dict:
    successes = sum(bool(row["success"]) for row in values)
    return {
        "numerator": successes,
        "denominator": len(values),
        "rate": round(successes / len(values), 4) if values else None,
    }


def split_summary(rows: list[dict]) -> dict:
    result: dict[str, Any] = {}
    for split in ("dev", "held-out", "all"):
        selected = rows if split == "all" else [
            row for row in rows if row["split"] == split
        ]
        arms: dict[str, Any] = {}
        for arm in ARMS:
            arm_rows = [row for row in selected if row["arm"] == arm]
            cost = sum(float(row.get("cost_usd") or 0.0) for row in arm_rows)
            success = sum(bool(row["success"]) for row in arm_rows)
            calls = [call for row in arm_rows for call in row.get("model_calls", [])]
            arms[arm] = {
                "task_success": _rate(arm_rows),
                "model_calls": len(calls),
                "input_tokens": sum(int(call.get("input_tokens") or 0) for call in calls),
                "output_tokens": sum(int(call.get("output_tokens") or 0) for call in calls),
                "cost_usd": round(cost, 10),
                "cost_per_success_usd": round(cost / success, 10) if success else None,
            }
        index = {(row["scenario_id"], row["arm"]): row for row in selected}
        diagnostics: dict[str, Any] = {}
        for other in ("fixed_single", "fixed_multi"):
            wins = losses = ties = 0
            ids: dict[str, list[str]] = {"wins": [], "losses": [], "ties": []}
            for scenario_id in sorted({row["scenario_id"] for row in selected}):
                adaptive = index[(scenario_id, "adaptive")]["success"]
                comparator = index[(scenario_id, other)]["success"]
                label = "ties" if adaptive == comparator else (
                    "wins" if adaptive else "losses"
                )
                ids[label].append(scenario_id)
                if label == "wins":
                    wins += 1
                elif label == "losses":
                    losses += 1
                else:
                    ties += 1
            diagnostics[other] = {
                "wins": wins, "losses": losses, "ties": ties, "ids": ids,
            }
        buckets: dict[str, dict[str, dict]] = {}
        for bucket in sorted({row["bucket"] for row in selected}):
            buckets[bucket] = {
                arm: _rate([
                    row for row in selected
                    if row["bucket"] == bucket and row["arm"] == arm
                ])
                for arm in ARMS
            }
        result[split] = {
            "arms": arms,
            "paired": diagnostics,
            "buckets": buckets,
        }
    return result


def cost_audit() -> dict:
    ledger_rows = []
    total_cost = 0.0
    total_calls = 0
    for path in LEDGERS:
        if not path.exists():
            raise FileNotFoundError(path)
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
            calls = db.execute(
                "SELECT COUNT(*), SUM(status='pending'), "
                "COALESCE(SUM(charged_usd), 0), "
                "COALESCE(SUM(input_tokens), 0), COALESCE(SUM(output_tokens), 0) "
                "FROM ch15_paid_calls"
            ).fetchone()
            arms = db.execute(
                "SELECT arm, COUNT(*), COALESCE(SUM(charged_usd), 0) "
                "FROM ch15_paid_calls GROUP BY arm ORDER BY arm"
            ).fetchall()
            stop = db.execute("SELECT stop_reason FROM ch15_paid_runs LIMIT 1").fetchone()
        call_count, pending, charged, input_tokens, output_tokens = calls
        total_calls += int(call_count)
        total_cost += float(charged)
        ledger_rows.append({
            "ledger": path.name,
            "calls": int(call_count),
            "pending_calls": int(pending or 0),
            "charged_usd": round(float(charged), 10),
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "stop_reason": stop[0] if stop else None,
            "arms": [
                {"arm": arm, "calls": count, "charged_usd": round(cost, 10)}
                for arm, count, cost in arms
            ],
        })
    return {
        "session_cap_usd": SESSION_CAP_USD,
        "all_attempt_calls": total_calls,
        "total_charged_usd": round(total_cost, 10),
        "remaining_usd": round(SESSION_CAP_USD - total_cost, 10),
        "within_session_cap": total_cost <= SESSION_CAP_USD + 1e-9,
        "pending_calls": sum(row["pending_calls"] for row in ledger_rows),
        "stop_reasons": [
            row["stop_reason"] for row in ledger_rows if row["stop_reason"]
        ],
        "ledgers": ledger_rows,
        "notes": [
            "The first ledger contains three dev infrastructure attempts; all are retained.",
            "The final ledger contains one frozen 34-case three-arm execution.",
            "No provider usage was inferred from response text; charged amounts come from ledger rows.",
        ],
    }


def route_audit(rows: list[dict]) -> dict:
    decisions = [
        decision
        for row in rows if row["arm"] == "adaptive"
        for decision in row.get("route_decisions", [])
    ]
    return {
        "decision_count": len(decisions),
        "multi_scenario_count": sum(
            row["route_selected"] == "multi" for row in rows if row["arm"] == "adaptive"
        ),
        "scenario_count": sum(row["arm"] == "adaptive" for row in rows),
        "selected_multi_ids": [
            row["scenario_id"] for row in rows
            if row["arm"] == "adaptive" and row["route_selected"] == "multi"
        ],
        "reason_counts": dict(sorted(Counter(
            reason for decision in decisions for reason in decision["reason_codes"]
        ).items())),
        "integer_ms_counts": dict(sorted(Counter(
            decision["routing_latency_ms"] for decision in decisions
        ).items())),
        "routing_cost_usd": round(sum(
            float(decision.get("routing_cost_usd") or 0.0)
            for decision in decisions
        ), 10),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", type=Path, default=FINAL_DIR / "traces.jsonl.gz")
    args = parser.parse_args()
    rows = load_rows(args.traces)
    analysis = {
        "result_count": len(rows),
        "technical_error_count": sum(bool(row.get("errors")) for row in rows),
        "paid_calls_during_analysis": 0,
        "split_summary": split_summary(rows),
        "route_audit": route_audit(rows),
        "cost_audit": cost_audit(),
    }
    _write_json(FINAL_DIR / "split-summary.json", analysis["split_summary"])
    _write_json(FINAL_DIR / "route-audit.json", analysis["route_audit"])
    _write_json(RUNS_DIR / "attempt-reconciliation.json", analysis["cost_audit"])
    print(json.dumps(analysis, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
