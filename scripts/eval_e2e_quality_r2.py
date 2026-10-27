"""CLI for historical RAG diagnosis and the production-path E2E R2 baseline."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import os
import sqlite3
import threading
import time
from collections import Counter
from pathlib import Path

from scripts.e2e_quality_r2.budget import ExactCallCache, R2Budget
from scripts.e2e_quality_r2.data import load_frozen_cases
from scripts.e2e_quality_r2.harness import (
    TraceCapture,
    build_instrumented_app,
    prepare_private_runtime,
    reclassify_captured_costs,
    run_split,
    summarize_results,
)
from scripts.e2e_quality_r2.history import answer_reuse, diagnose_history_case
from scripts.e2e_quality_r2.scoring import assert_model_input_safe, score_case
from scripts.eval_cost import PricingTable
from scripts.eval_dataset import load_split
from scripts.eval_prompts import EVAL_ANSWER_PROMPT, prompt_sha256

ROOT = Path(__file__).resolve().parent.parent
CASES_ROOT = ROOT / "tests" / "eval" / "e2e_quality_r2"
SOURCE_RUNS = ROOT / "tests" / "eval" / "rag_ablation" / "runs"
RESCORED_RUNS = ROOT / "tests" / "eval" / "rag_quality" / "v3_1" / "runs"
SOURCE_SNAPSHOT = Path(
    "/root/.codex/worktrees/ch16-rag-ablation/ShopHelper/data/eval-snapshots/"
    "ebfd49478b37ef387c8b124319a450ed73a86ae2a4b9f4fc28d6a52b2ccaa0ba.db"
)
SOURCE_METADATA_DB = Path(
    "/root/.codex/worktrees/ch16-rag-ablation/ShopHelper/data/ch12_eval.db"
)
SOURCE_CACHE = Path(
    "/root/.codex/worktrees/ch16-rag-ablation/ShopHelper/data/"
    "eval-caches-ch16-live-2026-10-26"
)
EXPECTED_SNAPSHOT_SHA256 = (
    "b6b1fa237d03d2c0f8ddcfe8e003f13c4cb584f0764ac04c2d57ae834bbdca73"
)
PRICING = ROOT / "tests" / "eval" / "rag_ablation" / "pricing-snapshot-2026-10-26.yaml"
DEFAULT_RESULTS = CASES_ROOT / "results" / "baseline-v1"
DEFAULT_SESSION_BUDGET = ROOT / "data" / "e2e-quality-r2" / "session-budget.json"


class _CallCappedBudget(R2Budget):
    """Cap calls for this run; the ledger's USD cap remains shared.

    Separate processes using the same ledger have separate call caps. They
    must not interpret ``max_calls`` as a ledger-wide concurrent-call limit.
    """
    def __init__(self, path: Path, *, limit_usd: float, max_calls: int) -> None:
        super().__init__(path, limit_usd=limit_usd)
        self.max_calls = int(max_calls)
        self._new_calls = 0
        self._call_lock = threading.Lock()

    def reserve(self, context: dict, estimated_usd: float) -> str:
        with self._call_lock:
            if self._new_calls >= self.max_calls:
                raise RuntimeError(f"provider-call stop rule reached: {self.max_calls}")
            attempt_id = super().reserve(context, estimated_usd)
            self._new_calls += 1
            return attempt_id


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _metadata_evidence(path: Path) -> dict[int, dict]:
    connection = sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "select id, category, questions, answer, section_path "
            "from knowledge_chunks order by id"
        ).fetchall()
    finally:
        connection.close()
    return {
        int(chunk_id): {
            "chunk_id": int(chunk_id),
            "category": str(category or ""),
            "questions": str(questions or ""),
            "answer": str(answer or ""),
            "section_path": str(section or ""),
            "text": "\n".join(
                str(value) for value in (category, questions, answer)
                if value not in (None, "")
            ),
        }
        for chunk_id, category, questions, answer, section in rows
    }


def _vector_texts(snapshot: Path) -> dict[int, str]:
    import pyarrow.parquet as pq

    files = sorted(Path(snapshot).glob(
        "collections/*/partitions/*/data/*.parquet"
    ))
    rows = {}
    for path in files:
        for row in pq.read_table(path, columns=["id", "text"]).to_pylist():
            rows[int(row["id"])] = str(row["text"])
    return rows


def _history_summary(rows: list[dict], split: str, mode: str) -> dict:
    by_outcome = Counter(row["outcome"] for row in rows)
    behaviors = Counter(
        f"{row['expected_behavior']}->{row['actual_behavior']}" for row in rows
    )
    reused = sum(bool(row.get("answer_reused_by")) for row in rows)
    return {
        "split": split,
        "mode": mode,
        "n": len(rows),
        "outcomes": dict(sorted(by_outcome.items())),
        "behaviors": dict(sorted(behaviors.items())),
        "answers_reused_in_two_or_more_modes": reused,
    }


def _answer_equivalence(traces_by_mode: dict[str, list[dict]]) -> dict:
    by_mode = {
        mode: {str(trace["case_id"]): str(trace.get("answer") or "") for trace in traces}
        for mode, traces in traces_by_mode.items()
    }
    modes = ("baseline", "decompose_once", "controlled_loop")
    case_ids = sorted(set().union(*(set(rows) for rows in by_mode.values())))
    pairs = {}
    for left_index, left in enumerate(modes):
        for right in modes[left_index + 1:]:
            pairs[f"{left}__{right}"] = sum(
                by_mode[left].get(case_id) == by_mode[right].get(case_id)
                for case_id in case_ids
            )
    pairs["all_three"] = sum(
        by_mode["baseline"].get(case_id)
        == by_mode["decompose_once"].get(case_id)
        == by_mode["controlled_loop"].get(case_id)
        for case_id in case_ids
    )
    return pairs


def diagnose_history(args) -> dict:
    metadata = _metadata_evidence(args.source_metadata_db)
    vector_texts = _vector_texts(args.source_snapshot)
    for chunk_id, text in vector_texts.items():
        metadata.setdefault(chunk_id, {"chunk_id": chunk_id, "text": text})
        metadata[chunk_id].setdefault("text", text)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    all_rows = []
    summaries = []
    equivalence = {}
    for split in ("dev", "frozen-v1"):
        cases = {str(case["id"]): case for case in load_split(split)}
        traces_by_mode = {
            mode: _read_jsonl(
                RESCORED_RUNS / split / "rag" / mode / "rescore.jsonl"
            )
            for mode in ("baseline", "decompose_once", "controlled_loop")
        }
        equivalence[split] = _answer_equivalence(traces_by_mode)
        reused = answer_reuse(traces_by_mode)
        for mode, traces in traces_by_mode.items():
            rows = []
            for trace in traces:
                case_id = str(trace["case_id"])
                row = diagnose_history_case(
                    cases[case_id],
                    trace,
                    metadata,
                    reused_by=tuple(reused.get(case_id) or ()),
                )
                row["split"] = split
                row["mode"] = mode
                row["original_answer"] = trace.get("answer")
                row["original_failure_labels"] = trace.get("failure_labels") or []
                row["llm_calls_recorded"] = int(trace.get("llm_calls") or 0)
                row["was_exact_cache_replay"] = bool(trace.get("cached"))
                rows.append(row)
                all_rows.append(row)
            summaries.append(_history_summary(rows, split, mode))
    _write_jsonl(output / "history-diagnosis.jsonl", all_rows)
    aggregate = {
        "schema_version": 1,
        "model_calls": 0,
        "historical_traces_are_read_only": True,
        "trace_source": "tests/eval/rag_quality/v3_1/runs/**/rag/**/rescore.jsonl",
        "answer_prompt": {
            "path": "scripts/eval_prompts.py",
            "sha256": prompt_sha256(EVAL_ANSWER_PROMPT),
            "contains_expected_behavior": "expected_behavior" in EVAL_ANSWER_PROMPT
            or "{behavior}" in EVAL_ANSWER_PROMPT,
            "template": EVAL_ANSWER_PROMPT,
        },
        "response_cache_key_source_sha256": hashlib.sha256(
            inspect.getsource(__import__(
                "scripts.eval_cache", fromlist=["ResponseCache"]
            ).ResponseCache.response_key).encode("utf-8")
        ).hexdigest(),
        "summaries": summaries,
        "exact_answer_equivalence": equivalence,
        "verified_dev_before": {
            "source": "tests/eval/rag_quality/v3_1/verified/dev/aggregate.json",
            **(
                json.loads(
                    (ROOT / "tests" / "eval" / "rag_quality" / "v3_1"
                     / "verified" / "dev" / "aggregate.json").read_text(encoding="utf-8")
                ).get("groups", {}).get("baseline", {}).get("before", {})
            ),
        },
        "overall_outcomes": dict(sorted(
            Counter(row["outcome"] for row in all_rows).items()
        )),
    }
    _write_json(output / "history-aggregate.json", aggregate)
    return aggregate


def _run_contract(
    *,
    splits: list[str],
    cases: dict[str, list[dict]],
    arm: str,
    answer_verification_enabled: bool,
    structured_mode: str,
    budget_limit_usd: float = 2.0,
    case_ids: list[str] | None = None,
    max_provider_calls: int | None = None,
    max_wall_seconds: float | None = None,
    session_budget: Path | None = None,
) -> dict:
    case_manifest = json.loads((CASES_ROOT / "manifest.json").read_text(encoding="utf-8"))
    pricing = PricingTable.load(PRICING)
    return {
        "schema_version": 1,
        "run_version": "e2e-quality-r2-v1",
        "arm": arm,
        "splits": splits,
        "case_counts": {split: len(cases[split]) for split in splits},
        "case_manifest": str((CASES_ROOT / "manifest.json").relative_to(ROOT)),
        "case_manifest_sha256": _sha_file(CASES_ROOT / "manifest.json"),
        "case_files": case_manifest["files"],
        "scorer_sha256": _sha_file(ROOT / "scripts" / "e2e_quality_r2" / "scoring.py"),
        "harness_sha256": _sha_file(ROOT / "scripts" / "e2e_quality_r2" / "harness.py"),
        "history_sha256": _sha_file(ROOT / "scripts" / "e2e_quality_r2" / "history.py"),
        "model": "deepseek-flash",
        "model_temperature": "production default",
        "answer_verification_enabled": answer_verification_enabled,
        "structured_output_mode": structured_mode,
        "retrieval_route": "hybrid_rerank",
        "rag_gap_trigger_enabled": False,
        "pricing": {
            "path": str(PRICING.relative_to(ROOT)),
            "sha256": _sha_file(PRICING),
            "version": pricing.version,
            "verified": pricing.verified,
        },
        "snapshot_expected_sha256": EXPECTED_SNAPSHOT_SHA256,
        "budget_limit_usd": budget_limit_usd,
        "case_ids": case_ids,
        "max_provider_calls": max_provider_calls,
        "max_wall_seconds": max_wall_seconds,
        "session_budget_path": str(session_budget) if session_budget else None,
        "heldout_policy": "run once after freeze; no held-out-driven tuning",
    }


async def _paid_run(args) -> dict:
    if not args.allow_paid:
        raise RuntimeError("--run requires --allow-paid")
    if args.max_cost_usd > 2.0:
        raise RuntimeError("E2E R2 protection cap must not exceed $2.00")
    if args.arm == "answer-verification" and args.splits != "dev":
        raise RuntimeError("optional answer verification is limited to dev")
    structured_mode = "json-fallback" if args.arm == "structured-json" else "production"
    splits = [part.strip() for part in args.splits.split(",") if part.strip()]
    if not splits or any(split not in {"dev", "heldout"} for split in splits):
        raise RuntimeError("splits must be dev, heldout, or dev,heldout")
    cases = load_frozen_cases(CASES_ROOT)
    requested_ids = [
        part.strip() for part in str(args.case_ids or "").split(",") if part.strip()
    ]
    if requested_ids:
        available_ids = {
            str(case["id"])
            for split_cases in cases.values()
            for case in split_cases
        }
        unknown = [case_id for case_id in requested_ids if case_id not in available_ids]
        if unknown:
            raise RuntimeError(f"unknown case ids: {','.join(unknown)}")
        for split in splits:
            cases[split] = [
                case for case in cases[split] if str(case["id"]) in requested_ids
            ]
        if not any(cases[split] for split in splits):
            raise RuntimeError("selected case ids do not belong to selected splits")
    output = Path(args.output)
    runtime_root = (
        Path(args.runtime_root)
        if args.runtime_root is not None
        else ROOT / "data" / "e2e-quality-r2" / f"{args.arm}-v1"
    )
    attempt_root = output / f"attempt-{int(args.attempt):02d}"
    if attempt_root.exists() and any(attempt_root.iterdir()):
        raise RuntimeError(f"attempt output already exists: {attempt_root}")
    attempt_root.mkdir(parents=True, exist_ok=True)

    contract = _run_contract(
        splits=splits,
        cases=cases,
        arm=args.arm,
        answer_verification_enabled=args.arm == "answer-verification",
        structured_mode=structured_mode,
        budget_limit_usd=args.max_cost_usd,
        case_ids=requested_ids,
        max_provider_calls=int(args.max_provider_calls),
        max_wall_seconds=float(args.max_wall_seconds),
        session_budget=Path(args.session_budget),
    )
    _write_json(attempt_root / "run-contract.json", contract)
    if not (output / "run-contract.json").exists():
        _write_json(output / "run-contract.json", contract)

    runtime = prepare_private_runtime(
        source_snapshot=Path(args.source_snapshot),
        source_metadata_db=Path(args.source_metadata_db),
        expected_snapshot_sha256=EXPECTED_SNAPSHOT_SHA256,
        destination=runtime_root / "private-runtime",
        source_cache_dir=Path(args.source_cache_dir),
    )
    budget = _CallCappedBudget(
        Path(args.session_budget),
        limit_usd=args.max_cost_usd,
        max_calls=int(args.max_provider_calls),
    )
    budget.assert_no_pending()
    exact_cache = ExactCallCache(runtime_root / "exact-calls.json")
    pricing = PricingTable.load(PRICING)
    capture = TraceCapture()
    capture.arm = args.arm
    namespace = "|".join((
        str(contract["run_version"]),
        contract["case_manifest_sha256"],
        contract["scorer_sha256"],
        contract["harness_sha256"],
        contract["model"],
        args.arm,
    ))
    app, settings = build_instrumented_app(
        paths=runtime,
        credential_env_file=Path(args.credential_env_file),
        pricing=pricing,
        budget=budget,
        capture=capture,
        exact_cache=exact_cache,
        namespace=namespace,
        answer_verification_enabled=args.arm == "answer-verification",
        structured_mode=structured_mode,
    )
    retrieval_integration = {
        "evidence_slate_enabled": settings.retrieval_evidence_slate_enabled,
        "rerank_pool_k": settings.retrieval_rerank_pool_k,
        "retrieval_cache_enabled": settings.retrieval_cache_enabled,
        "retrieval_cache_corpus_version": settings.retrieval_cache_corpus_version,
        "query_rewrite_enabled": settings.query_rewrite_enabled,
        "query_guard_enabled": settings.query_guard_enabled,
    }
    if args.require_rag_depth_integration and not all((
        settings.retrieval_evidence_slate_enabled,
        settings.retrieval_cache_enabled,
        bool(settings.retrieval_cache_corpus_version.strip()),
        settings.query_rewrite_enabled,
        settings.query_guard_enabled,
    )):
        raise RuntimeError("RAG-depth smoke requires slate/cache/rewrite-guard wiring")
    contract["retrieval_integration"] = retrieval_integration
    _write_json(attempt_root / "run-contract.json", contract)
    _write_json(output / "run-contract.json", contract)

    split_results = {}
    run_started = time.monotonic()
    for split in splits:
        remaining_seconds = float(args.max_wall_seconds) - (
            time.monotonic() - run_started
        )
        if remaining_seconds <= 0:
            raise RuntimeError(f"wall-clock stop rule reached: {args.max_wall_seconds}s")
        results = await asyncio.wait_for(
            run_split(
                cases[split],
                app=app,
                capture=capture,
                budget=budget,
                output_root=attempt_root / split,
            ),
            timeout=remaining_seconds,
        )
        summary = summarize_results(results)
        summary["model_calls"] = {
            "calls": sum(len(row.get("model_calls") or []) for row in results),
            "cache_hits": sum(
                bool(call.get("cache_hit"))
                for row in results for call in row.get("model_calls") or []
            ),
            "by_kind": dict(sorted(Counter(
                str(call.get("kind") or "")
                for row in results for call in row.get("model_calls") or []
            ).items())),
        }
        split_results[split] = summary
        _write_json(attempt_root / split / "aggregate.json", summary)
    aggregate = {
        "schema_version": 1,
        "contract": contract,
        "splits": split_results,
        "budget": budget.summary(),
        "model": settings.openai_model,
        "answer_verification_enabled": settings.answer_verification_enabled,
        "retrieval_route": settings.retrieval_route,
        "retrieval_integration": retrieval_integration,
    }
    _write_json(attempt_root / "aggregate.json", aggregate)
    return aggregate


def rescore(args) -> dict:
    cases = load_frozen_cases(CASES_ROOT)
    lookup = {
        str(case["id"]): case
        for split_cases in cases.values()
        for case in split_cases
    }
    source = Path(args.results)
    rows = _read_jsonl(source / "cases.jsonl")
    updated = []
    for row in rows:
        case = lookup[row["case_id"]]
        for call in row.get("model_calls") or []:
            assert_model_input_safe(call.get("input"), case)
        score = score_case(case, {
            "answer": row.get("answer"),
            "behavior": (row.get("score") or {}).get("actual_behavior"),
            "evidence": row.get("evidence_map") or {},
            "tool_calls": row.get("tool_calls") or [],
            "technical_failures": row.get("technical_failures") or [],
            "route_nodes": ((row.get("score") or {}).get("route") or {}).get("nodes") or [],
        })
        updated_row = {**row, "score": score, "task_success": score["task_success"],
                       "review_required": score["review_required"],
                       "failure_labels": score["failure_labels"]}
        updated_row = reclassify_captured_costs(
            updated_row,
            PricingTable.load(PRICING),
        )
        updated.append(updated_row)
    _write_jsonl(source / "cases-rescored.jsonl", updated)
    _write_jsonl(
        source / "manual-review-v2.jsonl",
        [
            {**row, "manual_review_status": "pending"}
            for row in updated
            if row["review_required"] or row["failure_labels"]
        ],
    )
    summary = summarize_results(updated)
    summary["scoring_version"] = "e2e-quality-r2-score-v2"
    summary["cost_source"] = "captured_calls_reclassified_v2"
    _write_json(source / "rescore-contract.json", {
        "schema_version": 1,
        "scoring_version": "e2e-quality-r2-score-v2",
        "source_cases_sha256": _sha_file(source / "cases.jsonl"),
        "rescored_cases_sha256": _sha_file(source / "cases-rescored.jsonl"),
        "scoring_source_sha256": _sha_file(
            ROOT / "scripts" / "e2e_quality_r2" / "scoring.py"
        ),
        "harness_source_sha256": _sha_file(
            ROOT / "scripts" / "e2e_quality_r2" / "harness.py"
        ),
        "order_fixture_sha256": _sha_file(
            CASES_ROOT / "fixtures" / "orders.json"
        ),
        "model_calls_checked": sum(
            len(row.get("model_calls") or []) for row in rows
        ),
        "model_input_boundary_violations": 0,
        "provider_calls": 0,
        "heldout_regenerated": False,
        "original_cases_preserved": True,
    })
    _write_json(source / "aggregate-rescored.json", summary)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ShopHelper E2E quality baseline R2")
    parser.add_argument("--diagnose-history", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--rescore", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--source-snapshot", type=Path, default=SOURCE_SNAPSHOT)
    parser.add_argument("--source-metadata-db", type=Path, default=SOURCE_METADATA_DB)
    parser.add_argument("--source-cache-dir", type=Path, default=SOURCE_CACHE)
    parser.add_argument("--credential-env-file", type=Path,
                        default=Path("/root/workplace/ShopHelper/.env"))
    parser.add_argument("--splits", default="dev,heldout")
    parser.add_argument("--arm", choices=("baseline", "structured-json", "answer-verification"),
                        default="baseline")
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--max-cost-usd", type=float, default=2.0)
    parser.add_argument("--session-budget", type=Path, default=DEFAULT_SESSION_BUDGET)
    parser.add_argument("--case-ids", default=None,
                        help="comma-separated fixed case ids for a bounded smoke")
    parser.add_argument("--max-provider-calls", type=int, default=1000,
                        help="maximum calls in this run, not across concurrent processes")
    parser.add_argument("--max-wall-seconds", type=float, default=1800.0)
    parser.add_argument("--require-rag-depth-integration", action="store_true",
                        help="require the opt-in RAG depth features for integration smoke")
    parser.add_argument("--allow-paid", action="store_true")
    parser.add_argument("--results", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.diagnose_history:
        result = diagnose_history(args)
        print(json.dumps({
            "mode": "history-diagnosis",
            "model_calls": 0,
            "stats": result["overall_outcomes"],
            "output": str(Path(args.output) / "history-diagnosis.jsonl"),
        }, ensure_ascii=False))
        return 0
    if args.run:
        result = asyncio.run(_paid_run(args))
        print(json.dumps({
            "mode": "paid-run",
            "splits": result["splits"],
            "budget": result["budget"],
            "output": str(Path(args.output) / f"attempt-{args.attempt:02d}"),
        }, ensure_ascii=False))
        return 0
    if args.rescore:
        if args.results is None:
            parser.error("--rescore requires --results")
        result = rescore(args)
        print(json.dumps({"mode": "rescore", **result}, ensure_ascii=False))
        return 0
    parser.error("choose --diagnose-history, --run, or --rescore")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
