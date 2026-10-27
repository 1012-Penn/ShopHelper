"""CLI for the single-variable answer-quality depth experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.answer_quality_depth.data import load_depth_cases
from scripts.answer_quality_depth.runner import (
    DEFAULT_PRICING,
    estimate_run,
    freeze_dev_config,
    prepare_evidence,
    rejudge_existing,
    repair_generation_truncations,
    repair_judge_failures,
    run_experiment,
)
from scripts.answer_quality_depth.scoring import (
    apply_semantic_status,
    load_scoring_overlay,
    score_response,
    summarize_scores,
)

ROOT = Path(__file__).resolve().parent.parent


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _rescore(run_dir: Path) -> dict:
    cases = {
        str(case["id"]): case for rows in load_depth_cases().values() for case in rows
    }
    source = run_dir / "results.jsonl"
    rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    overlay = load_scoring_overlay()
    rescored = []
    for row in rows:
        candidate = dict(row)
        score = score_response(
            cases[str(row["case_id"])],
            str(row.get("answer") or ""),
            list(row.get("evidence") or []),
            overlay=overlay,
            technical_failure=row.get("technical_failure"),
        )
        score = apply_semantic_status(
            score,
            (row.get("judge") or {}).get("verdict"),
        )
        candidate["legacy_score"] = row.get("score")
        candidate["score"] = score
        candidate.update(score)
        rescored.append(candidate)
    output = run_dir / "results-rescored.jsonl"
    with output.open("w", encoding="utf-8") as stream:
        for row in rescored:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = summarize_scores(rescored)
    _write_json(run_dir / "summary-rescored.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="答案质量深度对照")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare-evidence")
    prepare.add_argument(
        "--split",
        required=True,
        choices=("dev", "frozen-v1", "retrieval-slate-holdout-v1"),
    )
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--top-k", type=int, default=10)
    prepare.add_argument("--credentials", type=Path, required=True)

    estimate = sub.add_parser("estimate")
    estimate.add_argument(
        "--split",
        required=True,
        choices=("dev", "frozen-v1", "retrieval-slate-holdout-v1"),
    )
    estimate.add_argument("--evidence", type=Path, required=True)
    estimate.add_argument("--top-k", type=int, default=10)

    freeze = sub.add_parser("freeze")
    freeze.add_argument("--dev-evidence", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--top-k", type=int, default=10)
    freeze.add_argument("--max-cost-usd", type=float, default=2.0)

    run = sub.add_parser("run")
    run.add_argument(
        "--split",
        required=True,
        choices=("dev", "frozen-v1", "retrieval-slate-holdout-v1"),
    )
    run.add_argument("--evidence", type=Path, required=True)
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--credentials", type=Path, required=True)
    run.add_argument("--ledger", type=Path, required=True)
    run.add_argument("--config", type=Path)
    run.add_argument("--top-k", type=int, default=10)
    run.add_argument("--allow-paid", action="store_true")
    run.add_argument("--max-cost-usd", type=float, default=2.0)

    rescore = sub.add_parser("rescore")
    rescore.add_argument("--run-dir", type=Path, required=True)

    repair = sub.add_parser("repair-judge")
    repair.add_argument("--run-dir", type=Path, required=True)
    repair.add_argument("--credentials", type=Path, required=True)
    repair.add_argument("--ledger", type=Path, required=True)
    repair.add_argument("--allow-paid", action="store_true")
    repair.add_argument("--max-cost-usd", type=float, default=2.0)

    rejudge = sub.add_parser("rejudge")
    rejudge.add_argument("--source-run-dir", type=Path, required=True)
    rejudge.add_argument("--output-run-dir", type=Path, required=True)
    rejudge.add_argument("--evidence", type=Path, required=True)
    rejudge.add_argument("--credentials", type=Path, required=True)
    rejudge.add_argument("--ledger", type=Path, required=True)
    rejudge.add_argument("--allow-paid", action="store_true")
    rejudge.add_argument("--max-cost-usd", type=float, default=2.0)

    repair_generation = sub.add_parser("repair-generation")
    repair_generation.add_argument("--source-run-dir", type=Path, required=True)
    repair_generation.add_argument("--output-run-dir", type=Path, required=True)
    repair_generation.add_argument("--evidence", type=Path, required=True)
    repair_generation.add_argument("--credentials", type=Path, required=True)
    repair_generation.add_argument("--ledger", type=Path, required=True)
    repair_generation.add_argument("--top-k", type=int, default=10)
    repair_generation.add_argument("--allow-paid", action="store_true")
    repair_generation.add_argument("--max-cost-usd", type=float, default=2.0)

    sub.add_parser("report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare-evidence":
        result = prepare_evidence(
            args.split,
            output_path=args.output,
            top_k=args.top_k,
            credential_env_file=args.credentials,
        )
    elif args.command == "estimate":
        result = estimate_run(
            args.split,
            evidence_path=args.evidence,
            top_k=args.top_k,
            pricing_path=DEFAULT_PRICING,
        )
    elif args.command == "freeze":
        result = freeze_dev_config(
            dev_evidence_path=args.dev_evidence,
            output_path=args.output,
            top_k=args.top_k,
            max_cost_usd=args.max_cost_usd,
        )
    elif args.command == "run":
        frozen = (
            json.loads(args.config.read_text(encoding="utf-8"))
            if args.config is not None
            else None
        )
        result = run_experiment(
            args.split,
            evidence_path=args.evidence,
            run_dir=args.run_dir,
            credentials_path=args.credentials,
            ledger_path=args.ledger,
            pricing_path=DEFAULT_PRICING,
            top_k=args.top_k,
            max_cost_usd=args.max_cost_usd,
            allow_paid=args.allow_paid,
            frozen_config=frozen,
        )
    elif args.command == "rescore":
        result = _rescore(args.run_dir)
    elif args.command == "repair-judge":
        result = repair_judge_failures(
            run_dir=args.run_dir,
            credentials_path=args.credentials,
            ledger_path=args.ledger,
            max_cost_usd=args.max_cost_usd,
            allow_paid=args.allow_paid,
        )
    elif args.command == "rejudge":
        result = rejudge_existing(
            source_run_dir=args.source_run_dir,
            output_run_dir=args.output_run_dir,
            evidence_path=args.evidence,
            credentials_path=args.credentials,
            ledger_path=args.ledger,
            max_cost_usd=args.max_cost_usd,
            allow_paid=args.allow_paid,
        )
    elif args.command == "repair-generation":
        result = repair_generation_truncations(
            source_run_dir=args.source_run_dir,
            output_run_dir=args.output_run_dir,
            evidence_path=args.evidence,
            credentials_path=args.credentials,
            ledger_path=args.ledger,
            top_k=args.top_k,
            max_cost_usd=args.max_cost_usd,
            allow_paid=args.allow_paid,
        )
    elif args.command == "report":
        from scripts.answer_quality_depth.report import materialize

        result = materialize()
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
