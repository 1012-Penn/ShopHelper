"""Compare two ablation trace files without contacting any model or index."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.query_rewrite_ablation.metrics import summarize_variant_traces

COMPARE_FIELDS = ("constructed_query", "metrics", "retrieved")


def _read(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def compare_run_files(baseline: Path, replay: Path) -> dict:
    left = {
        (row["case_id"], row["variant"]): row for row in _read(baseline)
    }
    right = {
        (row["case_id"], row["variant"]): row for row in _read(replay)
    }
    mismatches: list[dict] = []
    for key in sorted(set(left) | set(right)):
        if key not in left or key not in right:
            mismatches.append({
                "case_id": key[0],
                "variant": key[1],
                "fields": ["missing_trace"],
            })
            continue
        changed = [
            field for field in COMPARE_FIELDS
            if left[key].get(field) != right[key].get(field)
        ]
        if changed:
            mismatches.append({
                "case_id": key[0],
                "variant": key[1],
                "fields": changed,
            })
    baseline_summary = summarize_variant_traces(list(left.values()))
    replay_summary = summarize_variant_traces(list(right.values()))
    aggregate_fields = ("variants", "paired", "bucket_counts")
    aggregate_changed = [
        field for field in aggregate_fields
        if baseline_summary.get(field) != replay_summary.get(field)
    ]
    return {
        "ok": not mismatches and not aggregate_changed,
        "baseline_traces": len(left),
        "replay_traces": len(right),
        "mismatches": mismatches,
        "aggregate_changed": aggregate_changed,
        "summary_equal": not aggregate_changed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="verify query-rewrite ablation replay")
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    result = compare_run_files(args.baseline, args.replay)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
