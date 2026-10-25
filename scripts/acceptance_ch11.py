"""ch11 离线验收:语料/数据集/泄漏/预算/离线 harness;可选快照与测试。"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "tests" / "eval" / "manifests" / "frozen-v1.json"
OLD_SHA = {
    "tests/eval/ch04_eval_set.jsonl": "5852485baf591d52ceef794950ba47071c0e98ca005a6b0ced8dfa1571b3a74b",
    "tests/eval/ch09_junk_set.jsonl": "d4bb3e03321eace9eace7e7bf1269fa98aefd0cdc9140fbf669d46aa41c95846",
}


def _check_corpus() -> tuple[bool, str]:
    from scripts.build_kb import chunk_corpus
    from scripts.corpus_facts import load_factbook
    from scripts.corpus_manifest import expected_docs

    chunks = chunk_corpus()
    fact_count = load_factbook().count()
    total = sum(len(items) for items in chunks.values())
    ok = (
        len(chunks) == expected_docs()
        and 120 <= total <= 180
        and 150 <= fact_count <= 250
    )
    return ok, f"docs={len(chunks)} chunks={total} facts={fact_count}"


def _check_frozen() -> tuple[bool, str]:
    from scripts.eval_dataset import (
        EXPECTED_FROZEN,
        bucket_counts,
        dataset_sha256,
        load_split,
        split_sha256,
    )

    cases = load_split("frozen-v1")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    dataset = ROOT / "tests" / "eval" / "splits" / "frozen-v1.jsonl"
    ok = (
        len(cases) == 90
        and bucket_counts(cases) == EXPECTED_FROZEN
        and manifest["file_sha256"] == dataset_sha256(dataset)
        and manifest["canonical_sha256"] == split_sha256(cases)
    )
    return ok, f"n={len(cases)} buckets={bucket_counts(cases)}"


def _check_regression() -> tuple[bool, str]:
    from scripts.eval_dataset import dataset_sha256, load_split

    cases = load_split("regression-v1")
    by_id = {case["id"]: case for case in cases}
    ok = (
        len(cases) == 60
        and by_id["C04"]["expected_behavior"] == "answer_with_conditions"
        and by_id["C05"]["expected_behavior"] == "verify_then_answer"
        and all(dataset_sha256(ROOT / path) == sha for path, sha in OLD_SHA.items())
    )
    return ok, f"n={len(cases)} C04/C05 labels present; old files unchanged"


def _check_leakage() -> tuple[bool, str]:
    from scripts.leakage_scan import scan_all

    findings = scan_all()
    blocking = [item for item in findings if item.severity == "blocking"]
    return not blocking, f"blocking={len(blocking)} warning={sum(1 for f in findings if f.severity == 'warning')}"


def _check_budget() -> tuple[bool, str]:
    from scripts.eval_cost import BudgetLedger

    ledger = BudgetLedger(ROOT / "data" / "ch11_budget.json")
    return ledger.total_usd() <= ledger.total_limit_usd, f"${ledger.total_usd():.4f}/${ledger.total_limit_usd:.2f}"


def _check_offline_harness() -> tuple[bool, str]:
    from scripts.eval_ch11 import main

    with tempfile.TemporaryDirectory() as tmp:
        code = main(["--mode", "offline", "--split", "frozen-v1", "--out", tmp])
    return code == 0, "eval_ch11 --mode offline frozen-v1"


def _check_snapshot(snapshot: Path | None) -> tuple[bool, str]:
    if snapshot is None:
        return False, "未提供 --snapshot;快照一致性是必验项(可用 --skip-snapshot 显式跳过)"
    from app.config import Settings
    from app.db import make_engine, make_session_factory
    from app.kb import KnowledgeBaseStore
    from app.vector_store import KnowledgeVectorStore
    from scripts.corpus_manifest import DOCS_DIR, corpus_sha256, load_manifest
    from scripts.eval_snapshot import verify_consistency

    settings = Settings()
    factory = make_session_factory(make_engine(settings))
    kb = KnowledgeBaseStore(factory)
    report = verify_consistency(
        kb,
        KnowledgeVectorStore(str(snapshot), dim=settings.embedding_dim),
        corpus_sha256=corpus_sha256(DOCS_DIR, load_manifest()),
        embedding_model=settings.embedding_model,
        dim=settings.embedding_dim,
    )
    build_manifest = ROOT / "data" / "build_manifest.json"
    build_ok = False
    if build_manifest.exists():
        build_ok = (
            json.loads(build_manifest.read_text(encoding="utf-8")).get("build_id")
            == report.build_id
        )
    ok = report.ok and build_ok
    detail = json.dumps(report.as_dict(), ensure_ascii=False)
    return ok, detail + f" build_manifest_match={build_ok}"


def _check_tests() -> tuple[bool, str]:
    targets = [
        "tests/test_corpus_manifest.py",
        "tests/test_corpus_facts.py",
        "tests/test_knowledge_docs_ch11.py",
        "tests/test_eval_dataset.py",
        "tests/test_ch11_regression_overlay.py",
        "tests/test_ch11_leakage.py",
        "tests/test_ch11_eval_metrics.py",
        "tests/test_ch11_checks.py",
        "tests/test_ch11_cost.py",
        "tests/test_ch11_cache.py",
        "tests/test_ch11_evidence_signals.py",
        "tests/test_ch11_contract.py",
        "tests/test_ch11_snapshot.py",
        "tests/test_ch11_harness.py",
        "tests/test_ch11_report.py",
    ]
    result = subprocess.run(
        [sys.executable, "-m", "pytest", *targets, "-q"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    tail = (result.stdout or "").strip().splitlines()[-1:] or [""]
    return result.returncode == 0, tail[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ch11 验收")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-snapshot", action="store_true")
    args = parser.parse_args(argv)

    checks = {
        "corpus": _check_corpus,
        "frozen-v1": _check_frozen,
        "regression-v1": _check_regression,
        "leakage": _check_leakage,
        "budget": _check_budget,
        "offline-harness": _check_offline_harness,
    }
    if not args.skip_snapshot:
        checks["snapshot"] = lambda: _check_snapshot(args.snapshot)
    if not args.skip_tests:
        checks["tests"] = _check_tests

    failed = 0
    for name, check in checks.items():
        try:
            ok, detail = check()
        except Exception as exc:
            ok, detail = False, f"exception: {exc}"
        print(f"[{'ok' if ok else 'fail'}] {name}: {detail}")
        if not ok:
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
