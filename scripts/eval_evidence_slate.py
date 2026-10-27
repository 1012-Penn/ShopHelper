"""Evaluate query-conditioned evidence-slate selection without answer generation.

The experiment uses one real hybrid retrieval + cross-encoder pass per case. The
baseline is the first five entries of that same rerank output; the candidate
reserves slots for explicit query facets and then returns five entries. This
isolates the selection layer without changing the corpus, rewrite cache, or
rerank model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from scripts.ch11_harness import HarnessConfig
from scripts.eval_dataset import dataset_sha256
from scripts.eval_metrics import coverage_rank, evidence_group_recall
from scripts.eval_stats import mcnemar_exact

ROOT = Path(__file__).resolve().parent.parent
REFERENCE_MODEL = "deepseek-flash"
CODE_PATHS = (
    "app/config.py",
    "app/evidence_slate.py",
    "app/retrieval.py",
)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return round(ordered[index], 3)


def summarize_rows(rows: list[dict]) -> dict:
    measured = [row for row in rows if row.get("technical_error") is None]
    baseline = [bool(row.get("baseline_hit")) for row in measured]
    candidate = [bool(row.get("slate_hit")) for row in measured]
    paired = mcnemar_exact(baseline, candidate)
    latencies = [float(row.get("latency_ms") or 0.0) for row in measured]
    selection_latencies = [
        float(row.get("selection_latency_ms") or 0.0) for row in measured
    ]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in measured:
        grouped[str(row.get("bucket") or "")].append(row)

    by_bucket = {}
    for bucket, bucket_rows in sorted(grouped.items()):
        by_bucket[bucket] = {
            "n": len(bucket_rows),
            "baseline_hits": sum(bool(row.get("baseline_hit")) for row in bucket_rows),
            "slate_hits": sum(bool(row.get("slate_hit")) for row in bucket_rows),
            "wins": [
                row["case_id"] for row in bucket_rows
                if not row.get("baseline_hit") and row.get("slate_hit")
            ],
            "losses": [
                row["case_id"] for row in bucket_rows
                if row.get("baseline_hit") and not row.get("slate_hit")
            ],
        }

    def mean(field: str) -> float | None:
        values = [
            float(row[field]) for row in measured
            if row.get(field) is not None
        ]
        return round(sum(values) / len(values), 6) if values else None

    return {
        "n": len(measured),
        "technical_errors": len(rows) - len(measured),
        "baseline_hits": sum(baseline),
        "slate_hits": sum(candidate),
        "wins": [
            row["case_id"] for row in measured
            if not row.get("baseline_hit") and row.get("slate_hit")
        ],
        "losses": [
            row["case_id"] for row in measured
            if row.get("baseline_hit") and not row.get("slate_hit")
        ],
        "mcnemar": paired,
        "baseline_mrr_at_10": mean("baseline_mrr_at_10"),
        "slate_mrr_at_10": mean("slate_mrr_at_10"),
        "baseline_group_recall_at_5": mean("baseline_group_recall_at_5"),
        "slate_group_recall_at_5": mean("slate_group_recall_at_5"),
        "pool_complete": sum(bool(row.get("pool_complete")) for row in measured),
        "latency_ms": {
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
        },
        "selection_latency_ms": {
            "p50": _percentile(selection_latencies, 0.50),
            "p95": _percentile(selection_latencies, 0.95),
        },
        "by_bucket": by_bucket,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_state() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "", "dirty": None}


def _configure_environment(credential_file: Path | None, sqlite_db: Path) -> None:
    from dotenv import dotenv_values

    credentials = dotenv_values(credential_file) if credential_file else {}
    for name in (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "EMBEDDING_API_KEY",
        "RERANK_API_KEY",
    ):
        value = credentials.get(name)
        if value:
            os.environ[name] = str(value)
    os.environ.setdefault("OPENAI_API_KEY", "offline-cache-replay-only")
    os.environ["DATABASE_URL"] = f"sqlite:///{sqlite_db.resolve()}"
    os.environ["OPENAI_MODEL"] = REFERENCE_MODEL
    os.environ["LLM_MAX_RETRIES"] = "0"
    os.environ["HYBRID_CANDIDATES"] = "50"
    os.environ["RERANK_TOP_K"] = "10"
    os.environ["RETRIEVAL_EVIDENCE_SLATE_ENABLED"] = "true"
    os.environ["RETRIEVAL_RERANK_POOL_K"] = "50"


def _load_cases(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run(args) -> dict:
    from scripts.eval_ch11 import _build_context
    from scripts.eval_cost import AttemptLedger, BudgetExceeded

    split_path = args.cases or ROOT / "tests" / "eval" / "splits" / f"{args.split}.jsonl"
    cases = _load_cases(split_path)
    if not cases:
        raise ValueError(f"题集为空:{split_path}")
    args.out.mkdir(parents=True, exist_ok=True)
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    args.attempt_ledger.parent.mkdir(parents=True, exist_ok=True)

    _configure_environment(args.credential_env_file, args.sqlite_db)
    config = HarnessConfig(
        split=args.split,
        mode="retrieval",
        strategy="hybrid_rerank",
        max_cost_usd=args.max_cost_usd,
        allow_paid=bool(args.allow_paid),
        rag_mode="baseline",
        temperature=0.0,
    )
    context_args = SimpleNamespace(
        snapshot=args.snapshot,
        cache_dir=args.cache_dir,
        pricing=args.pricing,
        ledger=args.ledger,
        allow_paid=bool(args.allow_paid),
        no_rewrite=False,
        rewrite_cache=args.rewrite_cache,
        rewrite_cache_mode="read",
    )
    try:
        ctx, _, source_contract = _build_context(config, context_args, cases)
    except BudgetExceeded:
        raise
    ctx.cost.attempt_ledger = AttemptLedger(args.attempt_ledger)
    ctx.cost.max_cost_usd = args.max_cost_usd
    ctx.cost.ledger.total_limit_usd = args.max_cost_usd

    contract = {
        "schema_version": 1,
        "experiment": "query_conditioned_evidence_slate",
        "selection_version": "evidence-slate-v1",
        "split": args.split,
        "dataset": {
            "path": str(split_path),
            "sha256": dataset_sha256(split_path),
            "n": len(cases),
            "evidence_cases": sum(bool(case.get("required_evidence")) for case in cases),
        },
        "code_sha256": {
            path: _sha256(ROOT / path) for path in CODE_PATHS
        },
        "selection_config": {
            "rerank_pool_k": 50,
            "final_top_k": 10,
            "uses_gold": False,
        },
        "source_contract": source_contract,
        "budget": {
            "max_cost_usd": args.max_cost_usd,
            "allow_paid": bool(args.allow_paid),
            "ledger": str(args.ledger),
            "attempt_ledger": str(args.attempt_ledger),
        },
        "git": _git_state(),
    }
    (args.out / "contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    rows = []
    results_path = args.out / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as stream:
        for index, case in enumerate(cases, 1):
            case_id = str(case["id"])
            gold = ctx.gold_by_case.get(case_id, [])
            if not gold:
                continue
            before_attempts = int(ctx.cost.api_attempts)
            before_paid_attempts = int(ctx.cost.paid_api_attempts)
            before_spent = float(ctx.cost.spent_usd)
            started = time.perf_counter()
            try:
                result = ctx.service.retrieve(
                    case["query"],
                    strategy="hybrid_rerank",
                    prepared=ctx.prepared.get(case_id),
                )
                latency_ms = round((time.perf_counter() - started) * 1000, 3)
                baseline_ids = [item.chunk_id for item in result.rerank_candidates[:10]]
                slate_ids = [item.chunk_id for item in result.items[:10]]
                pool_ids = [item.chunk_id for item in result.rerank_candidates]
                baseline_rank = coverage_rank(baseline_ids, gold, cap=5)
                slate_rank = coverage_rank(slate_ids, gold, cap=5)
                baseline_top10_rank = coverage_rank(baseline_ids, gold, cap=10)
                slate_top10_rank = coverage_rank(slate_ids, gold, cap=10)
                pool_rank = coverage_rank(pool_ids, gold, cap=len(pool_ids))
                row = {
                    "case_id": case_id,
                    "bucket": case.get("bucket", ""),
                    "query": case["query"],
                    "baseline_top5": baseline_ids[:5],
                    "slate_top5": slate_ids[:5],
                    "baseline_rank": baseline_rank,
                    "slate_rank": slate_rank,
                    "baseline_top10_rank": baseline_top10_rank,
                    "slate_top10_rank": slate_top10_rank,
                    "pool_rank": pool_rank,
                    "baseline_hit": baseline_rank is not None,
                    "slate_hit": slate_rank is not None,
                    "baseline_mrr_at_10": (
                        1.0 / baseline_top10_rank
                        if baseline_top10_rank is not None else 0.0
                    ),
                    "slate_mrr_at_10": (
                        1.0 / slate_top10_rank
                        if slate_top10_rank is not None else 0.0
                    ),
                    "baseline_group_recall_at_5": round(
                        evidence_group_recall(baseline_ids, gold, cap=5), 6
                    ),
                    "slate_group_recall_at_5": round(
                        evidence_group_recall(slate_ids, gold, cap=5), 6
                    ),
                    "pool_size": len(pool_ids),
                    "pool_complete": pool_rank is not None,
                    "selection_facets": result.selection_facets,
                    "selection_latency_ms": result.selection_latency_ms,
                    "latency_ms": latency_ms,
                    "api_attempts": int(ctx.cost.api_attempts) - before_attempts,
                    "paid_api_attempts": (
                        int(ctx.cost.paid_api_attempts) - before_paid_attempts
                    ),
                    "cost_usd": round(
                        float(ctx.cost.spent_usd) - before_spent, 9
                    ),
                    "technical_error": None,
                }
            except Exception as exc:  # keep every failed case auditable
                row = {
                    "case_id": case_id,
                    "bucket": case.get("bucket", ""),
                    "query": case.get("query", ""),
                    "technical_error": f"{type(exc).__name__}: {exc}"[:500],
                    "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                }
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            print(
                f"[{index}/{len(cases)}] {case_id} "
                f"baseline={row.get('baseline_hit')} slate={row.get('slate_hit')} "
                f"error={bool(row.get('technical_error'))}",
                flush=True,
            )

    summary = summarize_rows(rows)
    summary.update({
        "split": args.split,
        "selection_version": contract["selection_version"],
        "cost": ctx.cost.snapshot(),
        "cache": {
            "embedding": ctx.service._embedder.stats(),
            "rerank": ctx.service._reranker.cache.stats(),
        },
    })
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检索证据面选择对照")
    parser.add_argument("--split", required=True)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=ROOT / "data" / "eval-snapshots"
        / "ebfd49478b37ef387c8b124319a450ed73a86ae2a4b9f4fc28d6a52b2ccaa0ba.db",
    )
    parser.add_argument("--sqlite-db", type=Path, default=ROOT / "data" / "ch12_eval.db")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "eval-caches")
    parser.add_argument(
        "--pricing",
        type=Path,
        default=ROOT / "tests/eval/rag_ablation/pricing-snapshot-2026-10-26.yaml",
    )
    parser.add_argument("--rewrite-cache", type=Path)
    parser.add_argument("--credential-env-file", type=Path)
    parser.add_argument("--allow-paid", action="store_true")
    parser.add_argument("--max-cost-usd", type=float, default=2.0)
    parser.add_argument("--ledger", type=Path, default=None)
    parser.add_argument("--attempt-ledger", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.ledger is None:
        args.ledger = args.out / "cost-ledger.json"
    if args.attempt_ledger is None:
        args.attempt_ledger = args.out / "attempts.jsonl"
    try:
        summary = run(args)
    except Exception as exc:
        print(f"评测中止:{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
