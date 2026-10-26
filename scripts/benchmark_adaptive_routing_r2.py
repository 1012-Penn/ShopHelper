"""CPU-only latency benchmark for the frozen routing rules."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from app.adaptive_routing import AdaptiveRouter
from scripts.eval_adaptive_routing_r2 import load_scenarios, routing_input_for


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT / "tests" / "eval" / "adaptive_routing_r2" / "runs"
    / "final-rescore-v3" / "routing-benchmark.json"
)


def main() -> int:
    router = AdaptiveRouter()
    inputs = [
        routing_input_for(scenario, turn_index=turn_index)
        for scenario in load_scenarios()
        for turn_index in range(len(scenario["turns"]))
    ]
    measurements_us: list[float] = []
    for _ in range(5_000):
        for routing_input in inputs:
            started = time.perf_counter_ns()
            router.route(routing_input)
            measurements_us.append((time.perf_counter_ns() - started) / 1_000)
    measurements_us.sort()

    def percentile(percent: int) -> float:
        index = max(0, (percent * len(measurements_us) + 99) // 100 - 1)
        return round(measurements_us[min(index, len(measurements_us) - 1)], 3)

    result = {
        "benchmark": "rules-only local CPU",
        "paid_calls": 0,
        "decision_count": len(inputs),
        "iterations_per_decision": 5_000,
        "sample_count": len(measurements_us),
        "p50_us": percentile(50),
        "p95_us": percentile(95),
        "p99_us": percentile(99),
        "mean_us": round(statistics.fmean(measurements_us), 3),
        "scope": (
            "Python feature extraction and RoutingDecision construction only; "
            "does not include model or tool execution."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
