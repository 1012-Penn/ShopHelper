"""CLI for the frozen answer-plan R3 paired experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.answer_plan_r3.runner import run

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "tests" / "eval" / "answer_plan_r3"
RUNS = DATA / "runs"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "heldout"), required=True)
    parser.add_argument("--allow-paid", action="store_true")
    parser.add_argument("--max-cost-usd", type=float, default=2.0)
    parser.add_argument("--ledger", type=Path, default=RUNS / "answer-plan-r3-paid.sqlite3")
    parser.add_argument(
        "--credentials",
        type=Path,
        default=Path("/root/workplace/ShopHelper/.env"),
    )
    args = parser.parse_args()
    result = run(
        split=args.split,
        allow_paid=args.allow_paid,
        max_cost_usd=args.max_cost_usd,
        data_root=DATA,
        output_root=RUNS,
        ledger_path=args.ledger,
        credentials_path=args.credentials,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
