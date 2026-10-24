"""评测数据校验 CLI。

用法:
    python -m scripts.validate_eval_data --split all \
        --check-evidence --check-facts --check-buckets

退出码:0 通过;4 数据错误(schema/证据/事实/分桶/registry)。
"""
from __future__ import annotations

import argparse
import sys

from scripts.corpus_facts import CorpusDataError, load_factbook, validate_factbook
from scripts.corpus_manifest import DOCS_DIR, CorpusManifestError
from scripts.eval_dataset import (
    EXPECTED_FROZEN,
    EvalDataError,
    behavior_counts,
    bucket_counts,
    cross_split_issues,
    load_registry,
    load_split,
    split_entries,
    split_sha256,
    validate_cases,
)

DATA_EXIT_CODE = 4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="校验评测数据集 v2")
    parser.add_argument(
        "--split",
        default="all",
        help="要校验的 split 名,或 all(默认;未注册的 dev/frozen-v1 会跳过并提示)",
    )
    parser.add_argument("--check-evidence", action="store_true", help="校验证据逐字与 chunk 可达")
    parser.add_argument("--check-facts", action="store_true", help="校验 gold_facts 与 FactBook 一致")
    parser.add_argument("--check-buckets", action="store_true", help="校验分桶分布")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        registry = load_registry()
    except EvalDataError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return DATA_EXIT_CODE

    entries = split_entries(registry)
    if args.split == "all":
        targets = sorted(entries)
        for name in ("dev", "frozen-v1"):
            if name not in entries:
                print(f"[skip] {name}: 尚未注册")
    else:
        if args.split not in entries:
            print(f"[error] split 未注册: {args.split}", file=sys.stderr)
            return DATA_EXIT_CODE
        targets = [args.split]

    factbook = None
    if args.check_facts:
        try:
            factbook = load_factbook()
            fact_issues = validate_factbook(factbook)
        except (CorpusDataError, CorpusManifestError) as exc:
            print(f"[error] 事实库无法校验: {exc}", file=sys.stderr)
            return DATA_EXIT_CODE
        if fact_issues:
            for issue in fact_issues:
                print(f"[factbook] {issue}", file=sys.stderr)
            return DATA_EXIT_CODE

    cases_by_split: dict[str, list[dict]] = {}
    failed = False
    for split in targets:
        try:
            cases = load_split(split)
            expected = EXPECTED_FROZEN if (args.check_buckets and split == "frozen-v1") else None
            validate_cases(
                cases,
                split=split,
                bucket_counts=expected,
                docs_dir=DOCS_DIR if args.check_evidence else None,
                factbook=factbook,
            )
        except (EvalDataError, CorpusDataError, CorpusManifestError) as exc:
            print(f"[error] {split}:\n{exc}", file=sys.stderr)
            failed = True
            continue

        cases_by_split[split] = cases
        print(
            f"[ok] {split}: {len(cases)} 题, sha256={split_sha256(cases)[:12]},"
            f" buckets={bucket_counts(cases)}, behaviors={behavior_counts(cases)}"
        )

    cross_issues = cross_split_issues(cases_by_split)
    for issue in cross_issues:
        print(f"[error] {issue}", file=sys.stderr)
    if cross_issues:
        failed = True

    return DATA_EXIT_CODE if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
