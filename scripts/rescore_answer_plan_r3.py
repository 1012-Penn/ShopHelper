"""CLI: rescore saved answer-plan R3 trajectories without provider calls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.answer_plan_r3.rescore import rescore_run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("tests/eval/answer_plan_r3"),
    )
    parser.add_argument("--scoring-version", choices=("v1", "v2"), default="v1")
    args = parser.parse_args()
    summary = rescore_run(
        result_path=args.results,
        data_root=args.data,
        output_path=args.output,
        scoring_version=args.scoring_version,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
