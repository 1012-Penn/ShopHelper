"""CLI for the frozen fact-scope R3 evaluation."""
from __future__ import annotations

import argparse
import asyncio
import json

from tests.eval.fact_scope_r3.harness import freeze_manifest, preflight, run_split


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--run", choices=("dev", "heldout"))
    parser.add_argument("--allow-paid", action="store_true")
    args = parser.parse_args()

    if args.freeze:
        payload = freeze_manifest()
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return
    if args.preflight:
        print(json.dumps(preflight(), ensure_ascii=False, sort_keys=True))
        return
    if not args.allow_paid:
        raise SystemExit("付费运行需要显式 --allow-paid")
    print(asyncio.run(run_split(args.run)))


if __name__ == "__main__":
    main()
