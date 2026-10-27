"""Bounded real-model retrieval ablation for history resolution and safe rewriting."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path
from types import SimpleNamespace

from app.config import Settings
from app.llm import make_extract_model, make_structured_extractor
from app.prompts import ResolvedQuestion
from app.query_guard import has_negative_polarity
from app.retrieval import PreparedQuery
from app.rewrite import QueryRewrite, bm25_query_text
from scripts.corpus_facts import load_factbook
from scripts.corpus_manifest import DOCS_DIR, corpus_sha256, load_manifest
from scripts.eval_cost import (
    AttemptLedger,
    AuditedChat,
    BudgetExceeded,
    BudgetLedger,
    CallResultJournal,
    CostTracker,
    PricingTable,
)
from scripts.eval_dataset import (
    dataset_sha256,
    load_cases_from_path,
    split_sha256,
    validate_cases,
)
from scripts.eval_metrics import (
    coverage_rank,
    evidence_group_recall,
    mrr_at,
    resolve_gold_chunks,
)
from scripts.eval_snapshot import tree_sha256
from scripts.query_rewrite_ablation.dataset import build_cases
from scripts.query_rewrite_ablation.metrics import summarize_variant_traces
from scripts.query_rewrite_ablation.queries import VariantQuery, build_query_variants

ROOT = Path(__file__).resolve().parents[2]
RUN_COST_LIMIT_USD = 2.0
RETRIEVAL_STRATEGY = "hybrid_rerank"
SEED_PATH = ROOT / "tests" / "eval" / "query_rewrite_ablation"
SNAPSHOT_PATH = (
    ROOT / "data" / "eval-snapshots"
    / "ebfd49478b37ef387c8b124319a450ed73a86ae2a4b9f4fc28d6a52b2ccaa0ba.db"
)
SQLITE_PATH = ROOT / "data" / "ch12_eval.db"
CACHE_DIR = ROOT / "data" / "eval-caches"
RUN_DATA_DIR = ROOT / "data" / "query-rewrite-ablation"
DEFAULT_PRICING = ROOT / "tests" / "eval" / "rag_ablation" / "pricing-snapshot-2026-10-26.yaml"


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(text or "")).casefold().split())


def _contains(text: str, anchor: str) -> bool:
    return _normalize(anchor) in _normalize(text)


def _anchor_is_rejected(text: str, anchor: str) -> bool:
    """An explicitly negated distractor is a correction, not an accidental entity."""
    normalized = _normalize(text)
    wanted = _normalize(anchor)
    for match in re.finditer(re.escape(wanted), normalized):
        prefix = normalized[max(0, match.start() - 16):match.start()]
        clause = re.split(r"[，,。；;！？!?]", prefix)[-1]
        if not re.search(r"(?:不|别|非|不要|不是|而非|排除|去掉).{0,10}$", clause):
            return False
    return bool(wanted)


def query_intent_metrics(case: dict, query: str) -> dict:
    """Deterministic constraint audit independent of retrieved evidence."""
    groups = list(case.get("required_anchor_groups") or [])
    group_results = [
        any(_contains(query, alias) for alias in group)
        for group in groups
    ]
    forbidden = list(case.get("forbidden_anchors") or [])
    wrong_entity = any(
        _contains(query, anchor) and not _anchor_is_rejected(query, anchor)
        for anchor in forbidden
    )
    negation_ok = (
        has_negative_polarity(query)
        if case.get("preserve_negation")
        else True
    )
    return {
        "anchors_complete": all(group_results),
        "anchor_groups": group_results,
        "wrong_entity": wrong_entity,
        "negation_preserved": negation_ok,
        "intent_preserved": all(group_results) and not wrong_entity and negation_ok,
    }


class RetryingStructured:
    """Exact prompt calls with one bounded repair retry and per-stage accounting."""

    def __init__(self, extractor, cost: CostTracker, model: str, kind: str,
                 *, max_attempts: int = 2) -> None:
        self.extractor = extractor
        self.cost = cost
        self.model = model
        self.kind = kind
        self.max_attempts = max(1, int(max_attempts))
        self.calls = 0
        self.failures = 0
        self.cost_usd = 0.0
        self.latency_ms = 0.0

    def invoke(self, prompt: str):
        last_error: BaseException | None = None
        for attempt in range(1, self.max_attempts + 1):
            request = prompt if attempt == 1 else (
                prompt + "\n上次输出未通过结构校验。请修正并只输出符合 JSON Schema 的对象。"
            )
            before_cost = float(self.cost.spent_usd)
            started = time.perf_counter()
            try:
                self.cost.guard(
                    max(1, len(request) // 2) + 1200,
                    self.model,
                    kind=self.kind,
                    request_text=request,
                )
                result = self.extractor.invoke(request)
                self.calls += 1
                self.latency_ms += (time.perf_counter() - started) * 1000
                self.cost_usd += float(self.cost.spent_usd) - before_cost
                return result
            except BudgetExceeded:
                raise
            except Exception as exc:
                self.failures += 1
                self.latency_ms += (time.perf_counter() - started) * 1000
                self.cost_usd += float(self.cost.spent_usd) - before_cost
                last_error = exc
        assert last_error is not None
        raise last_error

    def snapshot(self) -> dict:
        return {
            "calls": self.calls,
            "failures": self.failures,
            "cost_usd": round(self.cost_usd, 8),
            "latency_ms": round(self.latency_ms, 3),
        }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _file_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_state() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip())
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": "", "dirty": True}


def load_cases(seed_path: Path, split: str) -> list[dict]:
    factbook = load_factbook()
    seeds = load_cases_from_path(seed_path)
    review_status = "frozen" if split.startswith("frozen") else "reviewed"
    cases = build_cases(
        seeds,
        factbook,
        split=split,
        review_status=review_status,
    )
    validate_cases(
        cases,
        split=f"query-rewrite-{split}",
        docs_dir=DOCS_DIR,
        factbook=factbook,
    )
    return cases


def estimate_run(cases: list[dict]) -> dict:
    """Deterministic upper planning estimate; not a bill and not an authorization."""
    stage_calls = len(cases) * 4 * 2  # legacy+safe resolve/rewrite, one repair each
    input_tokens_per_call = 3000
    output_tokens_per_call = 320
    pricing = PricingTable.load(DEFAULT_PRICING)
    model = "deepseek-flash"
    query_cost = stage_calls * pricing.cost(
        model, input_tokens_per_call, output_tokens_per_call
    )
    retrieval_cost = len(cases) * 3 * pricing.cost(
        "BAAI/bge-m3", 3000, 0
    )
    retrieval_cost += len(cases) * 3 * pricing.cost(
        "BAAI/bge-reranker-v2-m3", 12000, 0
    )
    total = query_cost + retrieval_cost
    return {
        "case_count": len(cases),
        "variants": ["raw", "legacy", "safe"],
        "max_logical_query_calls_before_retry": len(cases) * 4,
        "max_attempts_with_one_repair": stage_calls,
        "estimate_usd": round(total, 6),
        "hard_limit_usd": RUN_COST_LIMIT_USD,
        "within_limit": total <= RUN_COST_LIMIT_USD,
        "notes": "upper-bound planning estimate; cache hits and actual token usage can lower it",
    }


def _load_credentials(path: Path | None) -> None:
    if path is None:
        return
    from dotenv import dotenv_values

    values = dotenv_values(path)
    for name in ("OPENAI_API_KEY", "EMBEDDING_API_KEY", "RERANK_API_KEY"):
        value = values.get(name)
        if value:
            os.environ[name] = str(value)


def _build_runner_context(cases, args):
    from scripts.ch11_harness import HarnessConfig
    from scripts.eval_ch11 import _build_context

    config = HarnessConfig(
        split=args.split,
        mode="retrieval",
        strategy=RETRIEVAL_STRATEGY,
        max_cost_usd=RUN_COST_LIMIT_USD,
        allow_paid=True,
        temperature=0.0,
    )
    namespace = SimpleNamespace(
        snapshot=args.snapshot,
        cache_dir=args.cache_dir,
        pricing=args.pricing,
        ledger=args.ledger,
        allow_paid=True,
        no_rewrite=True,
        rewrite_cache=None,
        rewrite_cache_mode="read",
    )
    ctx, config, source_contract = _build_context(config, namespace, cases)
    if (ctx.service._candidates, ctx.service._final_top_k) != (50, 10):
        raise RuntimeError("检索上限漂移: 需要 50 candidates / top 10")
    if ctx.service._reranker is None:
        raise RuntimeError("缺少真实 reranker,拒绝把 RRF 序标成 hybrid_rerank")
    ctx.cost.max_cost_usd = RUN_COST_LIMIT_USD
    ctx.cost.ledger.total_limit_usd = RUN_COST_LIMIT_USD
    ctx.cost.attempt_ledger = AttemptLedger(args.attempt_ledger)
    return ctx, config, source_contract


def _structured(cost, journal, model: str, kind: str, schema):
    raw = make_extract_model(Settings())
    audited = AuditedChat(raw, cost, journal, model, kind)
    extractor = make_structured_extractor(
        Settings(), schema, base_model=audited
    )
    return RetryingStructured(extractor, cost, model, kind)


def _variant_trace(
    case: dict,
    variant: VariantQuery,
    *,
    ctx,
    gold: list[frozenset[int]],
    latency_ms: float,
    retrieval_cost_usd: float,
    query_cost_usd: float,
    stage_stats: dict,
) -> dict:
    prepared = PreparedQuery(
        variant.normalized,
        bm25_query_text(QueryRewrite(
            normalized=variant.normalized, synonyms=list(variant.synonyms)
        )),
    )
    result = ctx.service.retrieve(
        variant.text,
        strategy=RETRIEVAL_STRATEGY,
        prepared=prepared,
    )
    ranked = [item.chunk_id for item in result.items]
    rank = coverage_rank(ranked, gold, cap=10)
    rows = {row.id: row for row in ctx.all_rows.values() if row.id in ranked}
    return {
        "case_id": case["id"],
        "split": case["split"],
        "bucket": case["bucket"],
        "gap_type": case.get("gap_type", ""),
        "template_id": case.get("template_id", ""),
        "variant": variant.name,
        "query": case["query"],
        "conversation_context": case.get("conversation_context") or [],
        "constructed_query": {
            "text": variant.text,
            "resolved": variant.resolved,
            "normalized": variant.normalized,
            "synonyms": list(variant.synonyms),
        },
        "query_guard": {
            "violations": list(variant.guard_violations),
            "dropped_synonyms": list(variant.dropped_synonyms),
            "errors": list(variant.errors),
        },
        "metrics": {
            **query_intent_metrics(case, variant.text),
            "coverage_rank_at_10": rank,
            "complete_at_5": rank is not None and rank <= 5,
            "complete_at_10": rank is not None,
            "mrr_at_10": mrr_at(ranked, gold, k=10) or 0.0,
            "evidence_group_recall_at_5":
                evidence_group_recall(ranked, gold, cap=5),
        },
        "retrieved": [
            {
                "rank": index,
                "chunk_id": item.chunk_id,
                "score": item.score,
                "section_path": getattr(rows.get(item.chunk_id), "section_path", ""),
            }
            for index, item in enumerate(result.items, start=1)
        ],
        "latency_ms": round(latency_ms, 3),
        "retrieval_latency_ms": round(result.selection_latency_ms, 3),
        "cost_usd": round(query_cost_usd + retrieval_cost_usd, 8),
        "stage_stats": stage_stats,
        "gate": {
            "low_confidence": result.low_confidence,
            "reason": result.reason,
            "degraded": result.degraded,
        },
    }


def run(args) -> int:
    cases = load_cases(args.cases, args.split)
    estimate = estimate_run(cases)
    if not estimate["within_limit"]:
        print(json.dumps(estimate, ensure_ascii=False), file=sys.stderr)
        return 3
    if args.estimate_only:
        print(json.dumps(estimate, ensure_ascii=False))
        output = args.out / "budget-estimate.json"
        _write_json(output, estimate)
        return 0
    if not args.allow_paid:
        raise BudgetExceeded("真实模型运行需要 --allow-paid")
    if not args.frozen_lock and args.split.startswith("frozen"):
        raise RuntimeError("冻结集运行必须提供 --frozen-lock")
    if args.frozen_lock:
        lock = json.loads(Path(args.frozen_lock).read_text(encoding="utf-8"))
        actual = split_sha256(cases)
        if lock.get("split_sha256") != actual:
            raise RuntimeError(
                f"冻结集哈希不匹配:{actual} != {lock.get('split_sha256')}"
            )
    run_contract = args.out / "run-contract.json"
    if run_contract.exists() and not args.resume:
        raise RuntimeError(f"结果目录已存在,拒绝覆盖:{args.out}")

    _load_credentials(args.credential_env_file)
    os.environ["LLM_MAX_RETRIES"] = "0"
    os.environ["OPENAI_MODEL"] = "deepseek-flash"
    os.environ["HYBRID_CANDIDATES"] = "50"
    os.environ["RERANK_TOP_K"] = "10"
    os.environ["DATABASE_URL"] = f"sqlite:///{args.sqlite_db.resolve()}"
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("缺少 OPENAI_API_KEY(未打印凭据)")
    ctx, harness_config, source_contract = _build_runner_context(cases, args)
    model = Settings().openai_model
    journal = CallResultJournal(args.call_cache)
    legacy_resolve = _structured(
        ctx.cost, journal, model, "query_resolve_legacy", ResolvedQuestion
    )
    legacy_rewrite = _structured(
        ctx.cost, journal, model, "query_rewrite_legacy", QueryRewrite
    )
    safe_resolve = _structured(
        ctx.cost, journal, model, "query_resolve_safe", ResolvedQuestion
    )
    safe_rewrite = _structured(
        ctx.cost, journal, model, "query_rewrite_safe", QueryRewrite
    )
    gold_by_case = {
        case["id"]: resolve_gold_chunks(case, ctx.all_rows) for case in cases
    }
    contract = {
        "schema_version": 1,
        "split": args.split,
        "dataset_sha256": dataset_sha256(args.cases),
        "split_sha256": split_sha256(cases),
        "snapshot_sha256": tree_sha256(args.snapshot),
        "corpus_sha256": corpus_sha256(DOCS_DIR, load_manifest()),
        "strategy": RETRIEVAL_STRATEGY,
        "candidates": ctx.service._candidates,
        "final_top_k": ctx.service._final_top_k,
        "answer_generation_calls": 0,
        "judge_calls": 0,
        "query_model": model,
        "query_max_attempts": 2,
        "budget_limit_usd": RUN_COST_LIMIT_USD,
        "pricing_sha256": _file_sha(args.pricing),
        "git": _git_state(),
        "estimate": estimate,
        "source_contract": source_contract,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    _write_json(run_contract, contract)
    _write_json(args.out / "budget-estimate.json", estimate)

    traces: list[dict] = []
    for case in cases:
        stage_before = {
            "legacy_resolve": legacy_resolve.snapshot(),
            "legacy_rewrite": legacy_rewrite.snapshot(),
            "safe_resolve": safe_resolve.snapshot(),
            "safe_rewrite": safe_rewrite.snapshot(),
        }
        variants = build_query_variants(
            case,
            legacy_resolve=legacy_resolve.invoke,
            legacy_rewrite=legacy_rewrite.invoke,
            safe_resolve=safe_resolve.invoke,
            safe_rewrite=safe_rewrite.invoke,
        )
        stage_after = {
            "legacy_resolve": legacy_resolve.snapshot(),
            "legacy_rewrite": legacy_rewrite.snapshot(),
            "safe_resolve": safe_resolve.snapshot(),
            "safe_rewrite": safe_rewrite.snapshot(),
        }

        def delta(name: str) -> dict:
            before = stage_before[name]
            after = stage_after[name]
            return {key: round(after[key] - before[key], 8)
                    if isinstance(after[key], float)
                    else after[key] - before[key]
                    for key in after}

        deltas = {name: delta(name) for name in stage_after}
        for name in ("raw", "legacy", "safe"):
            stage_stats = {} if name == "raw" else {
                key: value for key, value in deltas.items()
                if key.startswith(name)
            }
            query_cost = sum(
                float(value.get("cost_usd") or 0.0)
                for value in stage_stats.values()
            )
            query_latency = sum(
                float(value.get("latency_ms") or 0.0)
                for value in stage_stats.values()
            )
            query_calls = sum(
                int(value.get("calls") or 0)
                for value in stage_stats.values()
            )
            retrieval_before = float(ctx.cost.spent_usd)
            retrieval_started = time.perf_counter()
            trace = _variant_trace(
                case,
                variants[name],
                ctx=ctx,
                gold=gold_by_case[case["id"]],
                latency_ms=0.0,
                retrieval_cost_usd=0.0,
                query_cost_usd=query_cost,
                stage_stats=stage_stats,
            )
            retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
            trace["retrieval_cost_usd"] = round(
                max(0.0, float(ctx.cost.spent_usd) - retrieval_before), 8
            )
            trace["latency_ms"] = round(
                query_latency + retrieval_ms, 3,
            )
            trace["cost_usd"] = round(
                trace["cost_usd"] + trace["retrieval_cost_usd"], 8
            )
            trace["query_model_calls"] = query_calls
            traces.append(trace)

    summary = summarize_variant_traces(traces)
    summary["budget"] = ctx.cost.snapshot()
    summary["query_stage_stats"] = {
        "legacy_resolve": legacy_resolve.snapshot(),
        "legacy_rewrite": legacy_rewrite.snapshot(),
        "safe_resolve": safe_resolve.snapshot(),
        "safe_rewrite": safe_rewrite.snapshot(),
    }
    summary["estimate"] = estimate
    summary["answer_generation_calls"] = 0
    summary["judge_calls"] = 0
    _write_jsonl(args.out / "responses.jsonl", traces)
    _write_json(args.out / "summary.json", summary)
    print(json.dumps({
        "out": str(args.out),
        "cases": len(cases),
        "traces": len(traces),
        "variants": summary["variants"],
        "budget": summary["budget"],
    }, ensure_ascii=False, indent=2))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="query construction retrieval ablation")
    parser.add_argument("--split", required=True)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT_PATH)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--sqlite-db", type=Path, default=SQLITE_PATH)
    parser.add_argument("--pricing", type=Path, default=DEFAULT_PRICING)
    parser.add_argument("--ledger", type=Path, default=RUN_DATA_DIR / "budget.json")
    parser.add_argument("--attempt-ledger", type=Path,
                        default=RUN_DATA_DIR / "attempts.jsonl")
    parser.add_argument("--call-cache", type=Path,
                        default=RUN_DATA_DIR / "call-results.json")
    parser.add_argument("--credential-env-file", type=Path)
    parser.add_argument("--frozen-lock", type=Path)
    parser.add_argument("--allow-paid", action="store_true")
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
