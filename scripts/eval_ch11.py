"""ch11 分层评测入口。

mode:
- offline: 只校验数据/泄漏,不构造 Settings/DB/模型。
- retrieval: L1,需要 MySQL+Milvus;嵌入/重排可走缓存。
- evidence: L1+L2,启用证据判定(需付费时显式 --allow-paid)。
- full: L1-L5,评测回答 prompt + 确定性检查 + 可选裁判。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from scripts.ch11_harness import (
    HarnessConfig,
    HarnessContext,
    dump_traces,
    run_split,
)
from scripts.eval_checks import check_case_deterministic
from scripts.eval_contract import build_contract, write_contract
from scripts.eval_dataset import (
    bucket_counts,
    dataset_sha256,
    load_split,
    split_sha256,
    validate_cases,
)
from scripts.eval_metrics import resolve_gold_chunks
from scripts.eval_prompts import EVAL_ANSWER_PROMPT, EVAL_JUDGE_PROMPT, prompt_sha256
from scripts.eval_report import aggregate_traces, write_report

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "reports" / "ch11"


def _file_sha(path: Path) -> str:
    import hashlib

    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return ""


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return ""


def _offline(split: str, out: Path, limit: int) -> int:
    from scripts.corpus_facts import load_factbook
    from scripts.corpus_manifest import DOCS_DIR
    from scripts.leakage_scan import scan_split

    cases = load_split(split)
    if limit:
        cases = cases[:limit]
    validate_cases(
        cases,
        split=split,
        docs_dir=DOCS_DIR,
        factbook=load_factbook(),
        bucket_counts=None,
    )
    findings = scan_split(split)
    blocking = [item for item in findings if item.severity == "blocking"]
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "split": split,
        "n": len(cases),
        "sha256": split_sha256(cases),
        "buckets": bucket_counts(cases),
        "leakage_blocking": len(blocking),
        "leakage_warning": sum(1 for item in findings if item.severity == "warning"),
    }
    (out / "offline_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False))
    return 1 if blocking else 0


def _prepare_queries(cases, args, settings, rewriter, split: str):
    from app.retrieval import PreparedQuery
    from scripts.rewrite_cache import RewriteCacheError, load_rewrites

    if args.no_rewrite or rewriter is None:
        return {case["id"]: PreparedQuery(case["query"], case["query"]) for case in cases}
    cache_path = args.rewrite_cache or (
        ROOT / "tests" / "eval" / "caches" / "rewrite" / f"{split}.json"
    )
    if not Path(cache_path).exists():
        print(f"[warn] 改写缓存不存在,使用恒等改写: {cache_path}", file=sys.stderr)
        return {case["id"]: PreparedQuery(case["query"], case["query"]) for case in cases}
    try:
        rewrites = load_rewrites(
            cases, cache_path, None, settings.openai_model, rewriter,
            mode="read", eval_sha256=split_sha256(cases),
        )
    except RewriteCacheError as exc:
        print(f"[warn] 改写缓存不可用,使用恒等改写: {exc}", file=sys.stderr)
        return {case["id"]: PreparedQuery(case["query"], case["query"]) for case in cases}
    from app.retrieval import bm25_query_text
    from app.rewrite import QueryRewrite

    prepared = {}
    for case in cases:
        qr = rewrites[case["id"]]
        rewritten = (qr.normalized or "").strip() or case["query"]
        prepared[case["id"]] = PreparedQuery(
            rewritten, bm25_query_text(QueryRewrite(normalized=rewritten, synonyms=qr.synonyms))
        )
    return prepared


def _build_context(config, args, cases):
    from app.config import Settings
    from app.db import make_engine, make_session_factory
    from app.evidence import make_evidence_judge
    from app.graph.builder import _load_evidence_calibration
    from app.kb import KnowledgeBaseStore
    from app.llm import make_chat_model, make_extract_model
    from app.prompts import FaithfulnessVerdict
    from app.rerank import make_reranker
    from app.retrieval import RetrievalService
    from app.rewrite import make_rewriter
    from app.vector_store import KnowledgeVectorStore
    from scripts.corpus_manifest import DOCS_DIR, corpus_sha256, load_manifest
    from scripts.eval_cache import EmbeddingCache, ResponseCache, RerankCache
    from scripts.eval_contract import contract_sha256
    from scripts.eval_cost import (
        DEFAULT_PRICING_PATH,
        BudgetLedger,
        CostTracker,
        PricingTable,
    )
    from scripts.eval_snapshot import compute_build_id

    settings = Settings()
    factory = make_session_factory(make_engine(settings))
    kb = KnowledgeBaseStore(factory)
    db_path = args.snapshot or settings.milvus_db_path
    vectors = KnowledgeVectorStore(str(db_path), dim=settings.embedding_dim)
    all_rows = {row.id: row for row in kb.get_chunks(kb.doc_chunk_ids_all())}
    gold_by_case = {
        case["id"]: resolve_gold_chunks(case, all_rows)
        for case in cases if case.get("required_evidence")
    }

    cache_dir = Path(args.cache_dir or ROOT / "data" / "eval-caches")
    embedder = EmbeddingCache(
        __import__("app.embedding", fromlist=["make_embedder"]).make_embedder(settings),
        cache_dir / "embedding.json",
        settings.embedding_model,
    )
    reranker = make_reranker(settings)
    if reranker is not None:
        reranker = RerankCache(reranker, cache_dir / "rerank.json", settings.rerank_model)
    rewriter = make_rewriter(settings) if settings.query_rewrite_enabled else None
    prepared = _prepare_queries(cases, args, settings, rewriter, config.split)

    evidence_judge = None
    if config.mode in {"evidence", "full"} and config.allow_paid:
        evidence_judge = make_evidence_judge(settings)
    service = RetrievalService(
        embedder, vectors, kb, rewriter=rewriter, reranker=reranker,
        candidates=settings.hybrid_candidates,
        final_top_k=settings.rerank_top_k,
        rerank_score_floor=settings.rerank_score_floor,
        evidence_calibration=_load_evidence_calibration(settings),
        evidence_judge=evidence_judge,
        evidence_top_n=settings.evidence_entailment_top_n,
    )

    pricing = PricingTable.load(Path(args.pricing) if args.pricing else DEFAULT_PRICING_PATH)
    cost = CostTracker(
        pricing, BudgetLedger(), config.max_cost_usd, allow_paid=config.allow_paid
    )
    response_cache = ResponseCache(cache_dir / "responses.json", "responses")
    chat = make_chat_model(settings) if config.mode == "full" else None
    judge = None
    if config.mode == "full" and config.use_judge:
        judge = make_extract_model(settings).with_structured_output(
            FaithfulnessVerdict, method="function_calling"
        )

    manifest = load_manifest()
    facts_sha = __import__("scripts.corpus_facts", fromlist=["load_factbook"]).load_factbook().sha256()
    split_path = ROOT / "tests" / "eval" / "splits" / f"{config.split}.jsonl"
    file_sha = dataset_sha256(split_path) if split_path.exists() else ""
    rewrite_cache_path = args.rewrite_cache or (
        ROOT / "tests" / "eval" / "caches" / "rewrite" / f"{config.split}.json"
    )
    contract = build_contract(
        dataset={
            "split": config.split,
            "sha256": split_sha256(cases),
            "file_sha256": file_sha,
            "label_version": "v2",
            "n": len(cases),
            "buckets": bucket_counts(cases),
        },
        corpus={
            "manifest_sha256": _file_sha(ROOT / "corpus" / "manifest.yaml"),
            "docs_sha256": corpus_sha256(DOCS_DIR, manifest),
            "facts_sha256": facts_sha,
            "doc_count": len(manifest.docs),
            "chunk_count": len(all_rows),
            "chunk_max_chars": settings.chunk_max_chars,
            "overlap_chars": settings.chunk_overlap_chars,
        },
        index={
            "backend": "milvus-lite",
            "build_id": compute_build_id(
                list(all_rows.values()),
                corpus_sha256=corpus_sha256(DOCS_DIR, manifest),
                embedding_model=settings.embedding_model,
                dim=settings.embedding_dim,
                max_chars=settings.chunk_max_chars,
                overlap_chars=settings.chunk_overlap_chars,
            ),
            "collection": "knowledge",
            "embedding_model": settings.embedding_model,
            "dim": settings.embedding_dim,
            "snapshot_path": str(db_path),
        },
        retrieval={
            "strategy": config.strategy,
            "candidates": settings.hybrid_candidates,
            "top_k": settings.rerank_top_k,
            "rerank_score_floor": settings.rerank_score_floor,
            "calibration_sha256": _file_sha(Path(settings.evidence_calibration_path)),
            "calibration_status": "stale",
        },
        rewrite={
            "enabled": rewriter is not None,
            "model": settings.openai_model,
            "mode": "identity" if args.no_rewrite else "frozen",
            "cache_sha256": _file_sha(Path(rewrite_cache_path)),
        },
        llm={"model": settings.openai_model, "temperature": config.temperature,
             "prompt_sha256": prompt_sha256(EVAL_ANSWER_PROMPT)},
        judge={"model": settings.openai_model, "temperature": 0,
               "prompt_sha256": prompt_sha256(EVAL_JUDGE_PROMPT)},
        pricing={
            "version": pricing.version,
            "currency": pricing.currency,
            "verified": pricing.verified,
            "sha256": _file_sha(Path(args.pricing) if args.pricing else DEFAULT_PRICING_PATH),
        },
        run={"git_commit": _git_commit(), "git_dirty": False, "seed": config.seed,
             "python": sys.version.split()[0], "host": ""},
        harness={"version": "ch11-harness-1", "mode": config.mode},
    )
    contract["_sha"] = contract_sha256(contract)
    config = HarnessConfig(
        **{**config.__dict__, "contract_sha": contract["_sha"], "model": settings.openai_model}
    )
    ctx = HarnessContext(
        service=service,
        all_rows=all_rows,
        gold_by_case=gold_by_case,
        prepared=prepared,
        chat=chat,
        judge=judge,
        cache=response_cache,
        cost=cost,
    )
    return ctx, config, contract


def _summarize(traces: list[dict]) -> dict:
    l1_cases = [trace for trace in traces if trace["l1"]["n"]]
    by_bucket: dict[str, list[dict]] = {}
    for trace in traces:
        by_bucket.setdefault(trace["bucket"], []).append(trace)

    def avg(values):
        values = [value for value in values if value is not None]
        return sum(values) / len(values) if values else None

    l1 = {
        "n": len(l1_cases),
        "coverage_rank_avg": avg([t["l1"]["coverage_rank"] for t in l1_cases]),
        "group_recall": avg([t["l1"]["group_recall"] for t in l1_cases]),
        "mrr": avg([t["l1"]["mrr"] for t in l1_cases]),
        "single_chunk_rank_avg": avg([t["l1"]["single_chunk_rank"] for t in l1_cases]),
        "evidence_precision": avg([t["l1"]["evidence_precision"] for t in l1_cases]),
        "hard_negative_hit_rate": avg([t["l1"]["hard_negative_hit_rate"] for t in traces]),
        "by_bucket": {
            bucket: {
                "n": sum(1 for t in rows if t["l1"]["n"]),
                "coverage_rank_avg": avg([t["l1"]["coverage_rank"] for t in rows]),
                "group_recall": avg([t["l1"]["group_recall"] for t in rows]),
            }
            for bucket, rows in sorted(by_bucket.items())
        },
    }
    l2 = {
        "gate_false_accept": sum(
            1 for t in traces
            if t["expected_behavior"] == "refuse" and t["gate"]["passed"]
        ),
        "gate_false_block": sum(
            1 for t in traces
            if t["expected_behavior"] != "refuse" and not t["gate"]["passed"]
        ),
        "false_accept_ids": [
            t["case_id"] for t in traces
            if t["expected_behavior"] == "refuse" and t["gate"]["passed"]
        ],
        "false_block_ids": [
            t["case_id"] for t in traces
            if t["expected_behavior"] != "refuse" and not t["gate"]["passed"]
        ],
    }
    l3 = {
        "failures": {},
        "gold_facts_checked": sum(
            t.get("deterministic_checks", {}).get("gold_facts_checked", 0) for t in traces
        ),
        "condition_coverage": avg([
            t["deterministic_checks"]["condition_coverage"]["score"]
            for t in traces if t.get("deterministic_checks")
        ]),
        "unsupported_claim_rate": avg([
            t["deterministic_checks"]["unsupported_claim_rate"]
            for t in traces if t.get("deterministic_checks")
        ]),
    }
    for trace in traces:
        for label in trace.get("failure_labels") or []:
            l3["failures"][label] = l3["failures"].get(label, 0) + 1
    l5 = {
        "llm_calls": sum(t.get("llm_calls", 0) for t in traces),
        "tokens_in": sum(t.get("tokens_in", 0) for t in traces),
        "tokens_out": sum(t.get("tokens_out", 0) for t in traces),
        "cost_estimate": sum(t.get("cost_estimate", 0.0) for t in traces),
        "latency_ms_avg": avg([t.get("latency_ms") for t in traces]),
    }
    return {"n": len(traces), "l1": l1, "l2": l2, "l3": l3, "l5": l5}


def _write_report(out: Path, summary: dict, contract: dict, split: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# ch11 基线评测：{split}",
        "",
        "## 契约",
        f"- contract_sha256: `{contract.get('_sha', '')}`",
        f"- dataset sha256: `{contract['dataset']['sha256']}`",
        f"- build_id: `{contract['index']['build_id']}`",
        f"- calibration: `{contract['retrieval']['calibration_status']}`",
        "",
        "## L1 检索",
        f"- n: {summary['l1']['n']}",
        f"- MRR@10: {summary['l1']['mrr']}",
        f"- group_recall: {summary['l1']['group_recall']}",
        f"- evidence_precision: {summary['l1']['evidence_precision']}",
        f"- hard_negative_hit_rate: {summary['l1']['hard_negative_hit_rate']}",
        "",
        "## L2 证据",
        f"- gate_false_accept: {summary['l2']['gate_false_accept']}",
        f"- gate_false_block: {summary['l2']['gate_false_block']}",
        "",
        "## L3/L4 回答行为",
        f"- gold_facts_checked: {summary['l3']['gold_facts_checked']}",
        f"- condition_coverage: {summary['l3']['condition_coverage']}",
        f"- unsupported_claim_rate: {summary['l3']['unsupported_claim_rate']}",
        f"- failure_labels: {summary['l3']['failures']}",
        "",
        "## L5 过程",
        f"- llm_calls: {summary['l5']['llm_calls']}",
        f"- tokens_in/out: {summary['l5']['tokens_in']}/{summary['l5']['tokens_out']}",
        f"- cost_estimate: ${summary['l5']['cost_estimate']:.4f}",
        f"- latency_ms_avg: {summary['l5']['latency_ms_avg']}",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ch11 分层评测")
    parser.add_argument("--split", default="frozen-v1", help="逗号分隔")
    parser.add_argument("--mode", choices=("offline", "retrieval", "evidence", "full"), default="offline")
    parser.add_argument("--strategy", default="hybrid_rerank")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-cost-usd", type=float, default=0.0)
    parser.add_argument("--allow-paid", action="store_true")
    parser.add_argument("--judge", dest="use_judge", action="store_true")
    parser.add_argument("--no-judge", dest="use_judge", action="store_false")
    parser.set_defaults(use_judge=False)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--pricing", type=Path)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--no-rewrite", action="store_true")
    parser.add_argument("--rewrite-cache", type=Path)
    parser.add_argument("--rewrite-cache-mode", choices=("read", "write", "refresh"), default="read")
    parser.add_argument("--seed", type=int, default=20261024)
    args = parser.parse_args(argv)

    splits = [split.strip() for split in args.split.split(",") if split.strip()]
    if args.mode == "offline":
        exit_code = 0
        for split in splits:
            exit_code |= _offline(split, args.out / f"{split}-offline", args.limit)
        return exit_code

    all_traces = []
    for split in splits:
        cases = load_split(split)
        if args.limit:
            cases = cases[: args.limit]
        config = HarnessConfig(
            split=split,
            mode=args.mode,
            strategy=args.strategy,
            limit=args.limit,
            max_cost_usd=args.max_cost_usd,
            allow_paid=args.allow_paid,
            use_judge=args.use_judge,
            seed=args.seed,
        )
        ctx, config, contract = _build_context(config, args, cases)
        traces = run_split(cases, ctx, config)
        split_out = args.out / split
        split_out.mkdir(parents=True, exist_ok=True)
        dump_traces(traces, split_out / "traces.jsonl")
        write_contract(split_out / "contract.json", contract)
        summary = aggregate_traces(traces)
        write_report(split_out, summary, contract, traces)
        all_traces.extend(traces)
    print(f"split={','.join(splits)} mode={args.mode} traces={len(all_traces)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
