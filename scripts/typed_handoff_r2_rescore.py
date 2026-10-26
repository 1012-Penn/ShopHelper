"""Offline semantic rescore for the preserved typed-handoff R2 traces."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.typed_handoff_r2_eval import (  # noqa: E402
    RESULTS_ROOT,
    _write_gzip_jsonl,
    _write_jsonl,
    load_scenarios,
    render_report,
    result_manifest,
    summarize,
)
from tests.eval.typed_handoff_r2.scoring import SCORING_VERSION, score_case  # noqa: E402


SOURCE_RESULTS = RESULTS_ROOT / "results.jsonl"
SOURCE_SUMMARY = RESULTS_ROOT / "summary-v1.json"
OUTPUT = RESULTS_ROOT / "v3-final"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if not SOURCE_RESULTS.exists() or not SOURCE_SUMMARY.exists():
        raise RuntimeError("preserved v1 results are missing")
    if OUTPUT.exists():
        raise RuntimeError("v2 rescore already exists; preserve prior artifacts")
    OUTPUT.mkdir(parents=True)
    scenarios = {row["id"]: row for row in load_scenarios()}
    source_rows = [
        json.loads(line) for line in SOURCE_RESULTS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rescored = []
    changes = []
    for source in source_rows:
        row = json.loads(json.dumps(source, ensure_ascii=False))
        old_score = dict(row["score"])
        row["score"] = score_case(scenarios[row["scenario_id"]], row)
        rescored.append(row)
        if old_score["task_success"] != row["score"]["task_success"]:
            changes.append({
                "scenario_id": row["scenario_id"],
                "arm": row["arm"],
                "old_success": old_score["task_success"],
                "new_success": row["score"]["task_success"],
                "old_errors": {
                    key: old_score.get(key) for key in (
                        "condition_error_count", "entity_error_count",
                        "source_missing_count",
                    )
                },
                "new_errors": {
                    key: row["score"].get(key) for key in (
                        "condition_error_count", "entity_error_count",
                        "source_missing_count",
                    )
                },
            })
    source_summary = json.loads(SOURCE_SUMMARY.read_text(encoding="utf-8"))
    summary = summarize(rescored, source_summary["budget"])
    summary["rescoring"] = {
        "model_calls": 0,
        "source_results": str(SOURCE_RESULTS.relative_to(ROOT)),
        "source_results_sha256": _sha(SOURCE_RESULTS),
        "source_scoring_version": "typed-handoff-r2-v1",
        "new_scoring_version": SCORING_VERSION,
        "new_scoring_sha256": _sha(
            ROOT / "tests" / "eval" / "typed_handoff_r2" / "scoring.py"),
        "reason": (
            "v1/v2 treated correct expired-policy background and opposite-case "
            "qualifications as forbidden claims, misparsed markdown or bullet "
            "relations, and counted negated order mentions as entity use"
        ),
        "score_changes": changes,
    }
    _write_jsonl(OUTPUT / "results.jsonl", rescored)
    _write_gzip_jsonl(OUTPUT / "traces.jsonl.gz", rescored)
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT / "summary.md").write_text(render_report(summary), encoding="utf-8")
    (OUTPUT / "manifest.json").write_text(
        json.dumps(result_manifest(), ensure_ascii=False, indent=2),
        encoding="utf-8")
    (OUTPUT / "score-changes.json").write_text(
        json.dumps(changes, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "paired": summary["paired"],
        "changed_rows": len(changes),
        "budget": summary["budget"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
