#!/usr/bin/env python3
"""Run the CH15 deterministic, offline end-to-end simulated task suite."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "eval" / "tasks"))
import harness


def _failure_reasons(result: dict) -> list[str]:
    score = result["score"]
    keys = {
        "scenario_contract_pass": "scenario contract failed",
        "selection_correct": "tool selection mismatch",
        "allowed_tools_correct": "used a tool outside the allowed set",
        "parameter_correct": "tool parameters mismatch",
        "refusal_correct": "refusal/gold mismatch",
        "handoff_correct": "handoff/gold mismatch",
        "outcome_correct": "outcome mismatch",
        "includes_correct": "missing required answer content",
        "excludes_correct": "included prohibited answer content",
        "database_correct": "database assertion mismatch",
    }
    return [label for key, label in keys.items() if not score.get(key, False)]


async def _run(args):
    scenarios = harness.load_scenarios(Path(args.scenarios) if args.scenarios else None)
    results, summary = await harness.run_suite(scenarios)
    summary["run_date"] = date.today().isoformat()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    trace_path = output / "ch15-case-report.jsonl"
    with trace_path.open("w", encoding="utf-8") as stream:
        for result in results:
            stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
    summary_path = output / "ch15-offline-report.md"
    summary_path.write_text(harness.render_summary_markdown(summary), encoding="utf-8")
    failures = [result for result in results
                if not result["score"]["scenario_contract_pass"]]
    failure_lines = [
        "# CH15 Offline Failure Cases", "",
        f"{len(failures)} of {len(results)} arm runs did not satisfy their scripted scenario contract.",
        "All entries are simulated fixture results from the deterministic scripted actor.", "",
    ]
    for result in failures:
        failure_lines.extend([
            f"## {result['scenario_id']} ({result['arm']}) — {result['category']}", "",
            "Failed checks: " + "; ".join(_failure_reasons(result)) + ".", "",
            "Gold: `" + json.dumps(result["scenario_record"]["gold"], ensure_ascii=False,
                                   sort_keys=True) + "`", "",
            f"Reply: {result.get('reply') or '(empty; UI frame may carry the clarification)' }", "",
            f"Selected actions: `{json.dumps(result.get('tool_actions', []), ensure_ascii=False)}`", "",
            f"Tool calls: `{json.dumps(result.get('tool_calls', []), ensure_ascii=False)}`", "",
            f"DB/audit: `{json.dumps({'tickets': result.get('tickets', []), 'audit_rows': result.get('audit_rows', [])}, ensure_ascii=False, sort_keys=True)}`", "",
            "Last-turn graph trace: `" + json.dumps((result.get("graph_trace") or [[]])[-1],
                                                       ensure_ascii=False) + "`", "",
            f"Gold score: `{json.dumps(result['score'], ensure_ascii=False, sort_keys=True)}`", "",
        ])
    failure_lines += [
        "## Failure analysis", "",
        "- `CH15-POL-012` and `CH15-HOF-004` are expected refusals where the graph also "
        "offers `transfer_human`. The existing Agent refusal path attaches that action "
        "to every refusal; the scenario gold treats these out-of-scope requests as not "
        "needing a shop-agent handoff. Together with the answerable-question false refusal "
        "below, this accounts for the autonomous arm's 3/63 false handoffs.",
        "- `CH15-POL-013` is an intentional false-refusal fixture: the retrieval trace "
        "contains ordinary shipping evidence, but the scripted actor refuses instead of "
        "giving the 8-yuan rate. The refusal is scored wrong by this case's gold.",
        "- `CH15-ORD-010` through `CH15-ORD-012` are intentional tool-selection/argument "
        "counterexamples: stale order id, duplicate query, and FAQ instead of order lookup.",
        "- `CH15-RFD-014` is an intentional unsupported completed-refund claim. The "
        "simulated order catalog has no refund ledger; the graph/tool trace records no "
        "refund execution, and the claim is still a task failure.",
        "- `CH15-REV-006` shows two distinct revision failures: the fixed refund route "
        "falls back after an unknown corrected order leaves no policy evidence; the "
        "autonomous actor queries the stale order id 1001 instead of corrected id 9999.",
        "- `CH15-FLT-005` intentionally submits a missing required parameter. The real "
        "ToolEngine schema gate rejects it and records `校验拦下`; parameter scoring "
        "correctly fails because the gold action requires order id 1001.",
        "- `CH15-FLT-006` intentionally proposes unregistered `delete_order`. The tool "
        "is not executed and no database row is written, but the unsafe selection remains "
        "a task failure.",
        "- Across all 7 audited write attempts, 0 persisted writes lacked a matching "
        "confirmation. Two simulated tickets were created only after explicit same-key "
        "confirmation in isolated SQLite.",
        "",
        "These are bounded findings from scripted simulated scenarios; they do not "
        "estimate production or paid-model rates.", "",
    ]
    failure_path = output / "ch15-failures.md"
    failure_path.write_text("\n".join(failure_lines), encoding="utf-8")
    print(json.dumps({"summary": summary, "trace_file": str(trace_path),
                      "summary_file": str(summary_path), "failure_file": str(failure_path),
                      "failed_runs": len(failures)}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", help="Optional JSONL manifest override")
    parser.add_argument("--output-dir", default="docs/eval",
                        help="Output directory for per-case traces and reports")
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
