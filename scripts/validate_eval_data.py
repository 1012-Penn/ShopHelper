"""评测数据校验 CLI。

用法:
    python -m scripts.validate_eval_data --split all \
        --check-evidence --check-facts --check-buckets

退出码:0 通过;4 数据错误(schema/证据/事实/分桶/registry)。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from scripts.corpus_facts import CorpusDataError, load_factbook, validate_factbook
from scripts.corpus_manifest import DOCS_DIR, CorpusManifestError, corpus_sha256, load_manifest
from scripts.eval_dataset import (
    EXPECTED_FROZEN,
    EvalDataError,
    behavior_counts,
    bucket_counts,
    cross_split_issues,
    dataset_sha256,
    load_registry,
    load_split,
    split_entries,
    split_sha256,
    validate_cases,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT / "tests" / "eval" / "manifests" / "frozen-v1.json"
DESIGN_DOC = ROOT / "docs" / "superpowers" / "specs" / "2026-10-24-ch11-evaluation-baseline-design.md"
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
    parser.add_argument("--freeze", action="store_true", help="冻结 frozen-v1 并写 manifest")
    parser.add_argument("--confirm-reviewer", default="", help="冻结复核人;必须显式给出")
    parser.add_argument("--write-manifest", action="store_true", help="写 frozen-v1 manifest")
    parser.add_argument("--manifest-out", type=Path, default=DEFAULT_MANIFEST)
    return parser


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_frozen_manifest(cases: list[dict], *, reviewer: str) -> dict:
    entry = split_entries(load_registry())["frozen-v1"]
    dataset_path = ROOT / entry["path"]
    manifest = load_manifest()
    factbook = load_factbook()
    return {
        "schema_version": 1,
        "split": "frozen-v1",
        "label_version": entry.get("label_version", "v2"),
        "n": len(cases),
        "file_sha256": dataset_sha256(dataset_path),
        "canonical_sha256": split_sha256(cases),
        "buckets": bucket_counts(cases),
        "behaviors": behavior_counts(cases),
        "multi_turn": sum(1 for case in cases if case.get("conversation_context")),
        "tool_count": sum(1 for case in cases if case.get("requires_tools")),
        "corpus_version": manifest.corpus_version,
        "corpus_sha256": corpus_sha256(DOCS_DIR, manifest),
        "facts_sha256": factbook.sha256(),
        "design_doc_sha256": _sha256_file(DESIGN_DOC) if DESIGN_DOC.exists() else "",
        "reviewer": reviewer,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "leakage": None,
    }


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

    if args.freeze or args.write_manifest:
        if failed:
            print("[error] 校验未通过,不写冻结 manifest", file=sys.stderr)
            return DATA_EXIT_CODE
        if args.split not in {"frozen-v1", "all"}:
            print("[error] --freeze/--write-manifest 只支持 frozen-v1", file=sys.stderr)
            return DATA_EXIT_CODE
        if not args.confirm_reviewer.strip():
            print("[error] 冻结必须提供 --confirm-reviewer", file=sys.stderr)
            return DATA_EXIT_CODE
        cases = cases_by_split.get("frozen-v1")
        if cases is None:
            print("[error] frozen-v1 未加载", file=sys.stderr)
            return DATA_EXIT_CODE
        if any(case.get("review_status") != "frozen" for case in cases):
            print("[error] frozen-v1 存在未标 frozen 的题", file=sys.stderr)
            return DATA_EXIT_CODE
        manifest = build_frozen_manifest(cases, reviewer=args.confirm_reviewer.strip())
        args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.manifest_out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(args.manifest_out)
        print(f"[frozen] {len(cases)} 题 -> {args.manifest_out}")

    return DATA_EXIT_CODE if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
