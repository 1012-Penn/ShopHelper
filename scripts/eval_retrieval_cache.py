"""Real-API cold/warm replay for the in-process retrieval static cache.

This harness builds private SQLite + Milvus Lite indexes from the repository
corpus, then drives the same RetrievalService used by the application. It
measures embedding + rerank calls only; answer generation, evidence judging,
and query rewriting are deliberately outside the measured scope.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.embedding import SiliconFlowEmbedder
from app.kb import KnowledgeBaseStore
from app.models import Base
from app.rerank import SiliconFlowReranker
from app.retrieval import RetrievalService
from app.retrieval_cache import CachedEmbedder, CachedReranker
from app.vector_store import KnowledgeVectorStore
from scripts.build_kb import build
from scripts.corpus_manifest import corpus_sha256, load_manifest
from scripts.eval_cost import BudgetLedger, CostTracker, PricingTable

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "tests" / "eval" / "retrieval_cache" / "runs"
DEFAULT_PRICING = ROOT / "config" / "pricing.yaml"
QUERIES = {
    "repeated": ["SH-E300 无线降噪耳机怎么样?"],
    "non_repeated": [
        "退货申请提交后多久审核?",
        "现货商品多久发货?",
        "无线耳机保修期多久?",
        "怎么申请保修?",
    ],
    "updated": ["SH-E300 现在售价多少?"],
}


class _CountingHttpClient:
    def __init__(self, inner, counter: _HttpCounter, kind: str) -> None:
        self.inner = inner
        self.counter = counter
        self.kind = kind

    def post(self, url, *args, **kwargs):
        with self.counter.lock:
            setattr(self.counter, self.kind, getattr(self.counter, self.kind) + 1)
        return self.inner.post(url, *args, **kwargs)


class _HttpCounter:
    def __init__(self) -> None:
        import threading

        self.lock = threading.Lock()
        self.embedding = 0
        self.rerank = 0

    def snapshot(self) -> dict[str, int]:
        with self.lock:
            return {
                "embedding_http_requests": self.embedding,
                "rerank_http_requests": self.rerank,
                "total_http_requests": self.embedding + self.rerank,
            }


def _estimate_tokens(texts: list[str]) -> int:
    return max(1, sum(max(1, len(text) // 2) for text in texts))


class _MeteredRetryEmbedder:
    def __init__(self, inner, cost: CostTracker, model: str, attempts: int) -> None:
        self.inner = inner
        self.cost = cost
        self.model = model
        self.attempts = attempts

    def embed(self, texts: list[str]) -> list[list[float]]:
        tokens = _estimate_tokens(texts)
        for attempt in range(self.attempts):
            self.cost.guard(tokens, self.model, kind="embedding")
            try:
                vectors = self.inner.embed(texts)
            except Exception as exc:
                self.cost.fail_attempt(exc)
                if attempt + 1 >= self.attempts:
                    raise
                time.sleep(1.0)
                continue
            self.cost.add(self.model, tokens, 0, kind="embedding")
            return vectors
        raise AssertionError("unreachable")


class _MeteredRetryReranker:
    def __init__(self, inner, cost: CostTracker, model: str, attempts: int) -> None:
        self.inner = inner
        self.cost = cost
        self.model = model
        self.attempts = attempts

    def rerank(self, query: str, documents: list[str], top_n: int | None = None):
        tokens = _estimate_tokens([query, *documents])
        for attempt in range(self.attempts):
            self.cost.guard(tokens, self.model, kind="rerank")
            try:
                results = self.inner.rerank(query, documents, top_n=top_n)
            except Exception as exc:
                self.cost.fail_attempt(exc)
                if attempt + 1 >= self.attempts:
                    raise
                time.sleep(1.0)
                continue
            self.cost.add(self.model, tokens, 0, kind="rerank")
            return results
        raise AssertionError("unreachable")


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return round(ordered[index], 3)


def _digest(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _config_version(settings: Settings) -> str:
    payload = {
        "schema": "retrieval-cache-eval-v1",
        "strategy": "hybrid_rerank",
        "candidates": 50,
        "final_top_k": 10,
        "rerank_score_floor": settings.rerank_score_floor,
        "embedding_dim": settings.embedding_dim,
        "chunk_max_chars": settings.chunk_max_chars,
        "chunk_overlap_chars": settings.chunk_overlap_chars,
    }
    return _digest(payload)


def _make_index(
    root: Path,
    docs_dir: Path,
    settings: Settings,
    embedder,
    *,
    label: str,
) -> dict:
    index_dir = root / label
    index_dir.mkdir(parents=True, exist_ok=True)
    db_path = index_dir / "knowledge.sqlite3"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    kb = KnowledgeBaseStore(factory)
    vectors = KnowledgeVectorStore(str(index_dir / "milvus.db"), dim=settings.embedding_dim)
    manifest_out = index_dir / "build_manifest.json"
    stats = build(
        docs_dir,
        kb,
        vectors,
        embedder,
        max_chars=settings.chunk_max_chars,
        overlap_chars=settings.chunk_overlap_chars,
        rebuild_docs=True,
        manifest=load_manifest(),
        embedding_model=settings.embedding_model,
        embedding_dim=settings.embedding_dim,
        manifest_out=manifest_out,
    )
    return {
        "kb": kb,
        "vectors": vectors,
        "stats": stats,
        "manifest": json.loads(manifest_out.read_text(encoding="utf-8")),
    }


def _load_index(root: Path, label: str, settings: Settings) -> dict:
    index_dir = root / label
    db_path = index_dir / "knowledge.sqlite3"
    manifest_path = index_dir / "build_manifest.json"
    if not db_path.exists() or not manifest_path.exists():
        raise RuntimeError(f"复用索引缺少文件:{index_dir}")
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    return {
        "kb": KnowledgeBaseStore(factory),
        "vectors": KnowledgeVectorStore(str(index_dir / "milvus.db"), dim=settings.embedding_dim),
        "stats": {
            "chunks": json.loads(manifest_path.read_text(encoding="utf-8"))["chunks"]
        },
        "manifest": json.loads(manifest_path.read_text(encoding="utf-8")),
    }


def _prepare_updated_corpus(root: Path) -> Path:
    target = root / "corpus-v2"
    shutil.copytree(ROOT / "knowledge", target)
    for filename in ("商品FAQ.md", "商品目录.md"):
        path = target / filename
        text = path.read_text(encoding="utf-8")
        if "299 元" not in text:
            raise RuntimeError(f"更新夹具未找到 299 元:{filename}")
        path.write_text(text.replace("299 元", "329 元"), encoding="utf-8")
    return target


def _service(kb, vectors, settings: Settings, embedder, reranker) -> RetrievalService:
    return RetrievalService(
        embedder,
        vectors,
        kb,
        rewriter=None,
        reranker=reranker,
        candidates=50,
        final_top_k=10,
        rerank_score_floor=settings.rerank_score_floor,
    )


def _response_snapshot(result, kb) -> dict:
    rows = kb.get_chunks([item.chunk_id for item in result.items])
    by_id = {row.id: row for row in rows}
    evidence = []
    for rank, item in enumerate(result.items[:3], start=1):
        row = by_id.get(item.chunk_id)
        evidence.append({
            "rank": rank,
            "chunk_id": item.chunk_id,
            "score": item.score,
            "question": (row.questions.splitlines() or [""])[0] if row else "",
            "answer": row.answer if row else "",
            "category": row.category if row else "",
        })
    return {
        "chunk_ids": [item.chunk_id for item in result.items],
        "scores": [item.score for item in result.items],
        "low_confidence": result.low_confidence,
        "reason": result.reason,
        "evidence_entailment": result.evidence_entailment,
        "degraded": result.degraded,
        "evidence_confidence": (
            result.evidence_confidence.signals
            if result.evidence_confidence is not None
            else None
        ),
        "top_evidence": evidence,
    }


def _measure(
    *,
    service: RetrievalService,
    kb,
    embedded,
    reranked,
    query: str,
    request_class: str,
    cache_state: str,
    iteration: int,
    http_counter: _HttpCounter,
) -> dict:
    emb_before = embedded.stats()
    rer_before = reranked.stats()
    http_before = http_counter.snapshot()
    started = time.perf_counter()
    result = service.retrieve(query, strategy="hybrid_rerank")
    duration_ms = (time.perf_counter() - started) * 1000
    emb_after = embedded.stats()
    rer_after = reranked.stats()
    http_after = http_counter.snapshot()
    response = _response_snapshot(result, kb)
    return {
        "request_class": request_class,
        "cache_state": cache_state,
        "iteration": iteration,
        "query": query,
        "latency_ms": round(duration_ms, 3),
        "embedding": {
            "hits": emb_after["hits"] - emb_before["hits"],
            "misses": emb_after["misses"] - emb_before["misses"],
            "coalesced": emb_after["coalesced"] - emb_before["coalesced"],
        },
        "rerank": {
            "hits": rer_after["hits"] - rer_before["hits"],
            "misses": rer_after["misses"] - rer_before["misses"],
            "coalesced": rer_after["coalesced"] - rer_before["coalesced"],
        },
        "http": {
            key: http_after[key] - http_before[key]
            for key in http_before
        },
        "response": response,
        "response_sha256": _digest(response),
    }


def _run_class(
    *,
    label: str,
    queries: list[str],
    kb,
    vectors,
    settings: Settings,
    embedder,
    reranker,
    corpus_version: str,
    config_version: str,
    iterations: int,
    http_counter: _HttpCounter,
) -> tuple[list[dict], dict]:
    rows = []
    for iteration in range(iterations):
        cold_embedder = CachedEmbedder(
            embedder,
            model=settings.embedding_model,
            corpus_version=corpus_version,
            config_version=config_version,
            max_entries=512,
            ttl_seconds=3600,
        )
        cold_reranker = CachedReranker(
            reranker,
            model=settings.rerank_model,
            corpus_version=corpus_version,
            config_version=config_version,
            max_entries=256,
            ttl_seconds=3600,
        )
        service = _service(kb, vectors, settings, cold_embedder, cold_reranker)
        for query in queries:
            row = _measure(
                service=service,
                kb=kb,
                embedded=cold_embedder,
                reranked=cold_reranker,
                query=query,
                request_class=label,
                cache_state="cold",
                iteration=iteration,
                http_counter=http_counter,
            )
            rows.append(row)

    warm_embedder = CachedEmbedder(
        embedder,
        model=settings.embedding_model,
        corpus_version=corpus_version,
        config_version=config_version,
        max_entries=512,
        ttl_seconds=3600,
    )
    warm_reranker = CachedReranker(
        reranker,
        model=settings.rerank_model,
        corpus_version=corpus_version,
        config_version=config_version,
        max_entries=256,
        ttl_seconds=3600,
    )
    service = _service(kb, vectors, settings, warm_embedder, warm_reranker)
    replay_reference: dict[str, str] = {}
    for query in queries:
        row = _measure(
            service=service,
            kb=kb,
            embedded=warm_embedder,
            reranked=warm_reranker,
            query=query,
            request_class=label,
            cache_state="replay_reference",
            iteration=0,
            http_counter=http_counter,
        )
        replay_reference[query] = row["response_sha256"]
        rows.append(row)
    replay_replayed: dict[str, str] = {}
    for iteration in range(iterations):
        for query in queries:
            row = _measure(
                service=service,
                kb=kb,
                embedded=warm_embedder,
                reranked=warm_reranker,
                query=query,
                request_class=label,
                cache_state="warm",
                iteration=iteration,
                http_counter=http_counter,
            )
            replay_replayed.setdefault(query, row["response_sha256"])
            rows.append(row)

    by_state = {}
    for state in ("cold", "warm"):
        state_rows = [row for row in rows if row["cache_state"] == state]
        latencies = [row["latency_ms"] for row in state_rows]
        by_state[state] = {
            "requests": len(state_rows),
            "p50_ms": _percentile(latencies, 0.50),
            "p95_ms": _percentile(latencies, 0.95),
            "embedding_hits": sum(row["embedding"]["hits"] for row in state_rows),
            "embedding_misses": sum(row["embedding"]["misses"] for row in state_rows),
            "rerank_hits": sum(row["rerank"]["hits"] for row in state_rows),
            "rerank_misses": sum(row["rerank"]["misses"] for row in state_rows),
            "http_requests": sum(row["http"]["total_http_requests"] for row in state_rows),
        }
        lookups = by_state[state]["embedding_hits"] + by_state[state]["embedding_misses"]
        by_state[state]["embedding_hit_rate"] = (
            round(by_state[state]["embedding_hits"] / lookups, 6) if lookups else 0.0
        )
    summary = {
        "request_class": label,
        "queries": queries,
        "iterations": iterations,
        "cold": by_state["cold"],
        "warm": by_state["warm"],
        "within_cache_replay_identical": replay_reference == replay_replayed,
        "cache": {
            "embedding_capacity": warm_embedder.stats()["capacity"],
            "rerank_capacity": warm_reranker.stats()["capacity"],
            "embedding_entries": warm_embedder.stats()["entries"],
            "rerank_entries": warm_reranker.stats()["entries"],
            "ttl_seconds": warm_embedder.stats()["ttl_seconds"],
        },
    }
    return rows, summary


def run(args) -> dict:
    if not args.allow_paid:
        raise RuntimeError("真实 API 对照需显式传入 --allow-paid")
    settings = Settings(_env_file=args.env_file) if args.env_file else Settings()
    if not settings.embedding_api_key:
        raise RuntimeError("EMBEDDING_API_KEY 未配置")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_dir = Path(args.run_dir) if args.run_dir else DEFAULT_OUTPUT / f"live-{stamp}"
    if args.resume:
        if args.run_dir is None or not run_dir.is_dir():
            raise RuntimeError("--resume 需要存在且已建索引的 --run-dir")
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
    ledger = BudgetLedger(
        run_dir / "cost-ledger.json",
        total_limit_usd=args.max_cost_usd,
    )
    cost = CostTracker(
        PricingTable.load(DEFAULT_PRICING),
        ledger,
        max_cost_usd=args.max_cost_usd,
        allow_paid=True,
    )
    http_counter = _HttpCounter()

    raw_embedder = SiliconFlowEmbedder(
        settings.embedding_api_base,
        settings.embedding_api_key,
        settings.embedding_model,
        batch_size=settings.embedding_batch_size,
    )
    raw_embedder._client = _CountingHttpClient(raw_embedder._client, http_counter, "embedding")
    rerank_key = settings.rerank_api_key or settings.embedding_api_key
    raw_reranker = SiliconFlowReranker(
        settings.rerank_api_base,
        rerank_key,
        settings.rerank_model,
    )
    raw_reranker._client = _CountingHttpClient(raw_reranker._client, http_counter, "rerank")
    metered_embedder = _MeteredRetryEmbedder(
        raw_embedder, cost, settings.embedding_model, args.max_attempts
    )
    metered_reranker = _MeteredRetryReranker(
        raw_reranker, cost, settings.rerank_model, args.max_attempts
    )

    started = time.perf_counter()
    updated_docs = run_dir / "corpus-v2"
    if args.resume:
        v1 = _load_index(run_dir, "index-v1", settings)
        v2 = _load_index(run_dir, "index-v2", settings)
    else:
        v1 = _make_index(
            run_dir, ROOT / "knowledge", settings, metered_embedder, label="index-v1"
        )
        updated_docs = _prepare_updated_corpus(run_dir)
        v2 = _make_index(run_dir, updated_docs, settings, metered_embedder, label="index-v2")
    config_version = _config_version(settings)
    v1_version = corpus_sha256(ROOT / "knowledge")
    v2_version = corpus_sha256(updated_docs)

    request_rows: list[dict] = []
    class_summaries: list[dict] = []
    for label in ("repeated", "non_repeated"):
        rows, summary = _run_class(
            label=label,
            queries=QUERIES[label],
            kb=v1["kb"],
            vectors=v1["vectors"],
            settings=settings,
            embedder=metered_embedder,
            reranker=metered_reranker,
            corpus_version=v1_version,
            config_version=config_version,
            iterations=args.iterations,
            http_counter=http_counter,
        )
        request_rows.extend(rows)
        class_summaries.append(summary)

    update_rows, update_summary = _run_class(
        label="updated",
        queries=QUERIES["updated"],
        kb=v2["kb"],
        vectors=v2["vectors"],
        settings=settings,
        embedder=metered_embedder,
        reranker=metered_reranker,
        corpus_version=v2_version,
        config_version=config_version,
        iterations=args.iterations,
        http_counter=http_counter,
    )
    request_rows.extend(update_rows)
    class_summaries.append(update_summary)

    # The update comparison above runs v2. Run a single uncached v1 reference
    # through a properly initialized service so its response remains auditable.
    v1_ref_embedder = CachedEmbedder(
        metered_embedder,
        model=settings.embedding_model,
        corpus_version=v1_version,
        config_version=config_version,
    )
    v1_ref_reranker = CachedReranker(
        metered_reranker,
        model=settings.rerank_model,
        corpus_version=v1_version,
        config_version=config_version,
    )
    v1_ref = _measure(
        service=_service(
            v1["kb"], v1["vectors"], settings, v1_ref_embedder, v1_ref_reranker
        ),
        kb=v1["kb"],
        embedded=v1_ref_embedder,
        reranked=v1_ref_reranker,
        query=QUERIES["updated"][0],
        request_class="update_reference_v1",
        cache_state="cold",
        iteration=0,
        http_counter=http_counter,
    )
    request_rows.append(v1_ref)

    v2_top_answer = update_rows[0]["response"]["top_evidence"][0]["answer"]
    v1_top_answer = v1_ref["response"]["top_evidence"][0]["answer"]
    update_summary["reference"] = {
        "v1_top_answer": v1_top_answer,
        "v2_top_answer": v2_top_answer,
        "corpus_changed": v1_version != v2_version,
        "v1_contains_299": "299" in v1_top_answer,
        "v2_contains_329": "329" in v2_top_answer,
        "first_v2_request_was_miss": (
            update_rows[0]["embedding"]["misses"] == 1
            and update_rows[0]["rerank"]["misses"] == 1
        ),
        "first_v2_warm_request_was_hit": any(
            row["cache_state"] == "warm" and row["iteration"] == 0
            and row["embedding"]["hits"] == 1
            and row["rerank"]["hits"] == 1
            for row in update_rows
        ),
    }

    summary = {
        "experiment": "retrieval-static-cache-cold-warm",
        "generated_at": datetime.now().astimezone().isoformat(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "scope": {
            "chain": "RetrievalService.retrieve(strategy=hybrid_rerank)",
            "query_rewrite": False,
            "answer_generation": False,
            "evidence_judge_calls": False,
            "evidence_confidence_gate": True,
            "provider": "SiliconFlow BGE-M3 embedding + BGE reranker",
            "private_indexes_only": True,
            "retry_policy": f"max_attempts={args.max_attempts}",
        },
        "settings": {
            "embedding_model": settings.embedding_model,
            "rerank_model": settings.rerank_model,
            "config_version": config_version,
            "max_cost_usd": args.max_cost_usd,
            "iterations": args.iterations,
        },
        "corpus": {
            "v1": {
                "sha256": v1_version,
                "chunks": v1["stats"]["chunks"],
                "build_id": v1["manifest"]["build_id"],
            },
            "v2": {
                "sha256": v2_version,
                "chunks": v2["stats"]["chunks"],
                "build_id": v2["manifest"]["build_id"],
            },
        },
        "classes": class_summaries,
        "update_invalidation": update_summary["reference"],
        "provider_http": http_counter.snapshot(),
        "cost": cost.snapshot(),
        "price_boundary": (
            "BAAI 标准版在价格页标注免费;账本仍按 $0.01/M 保守占位计费。"
            "因此不把缓存命中折算成金额节省。"
        ),
    }

    with (run_dir / "responses.jsonl").open("w", encoding="utf-8") as fh:
        for row in request_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "run_dir": str(run_dir),
        "elapsed_seconds": summary["elapsed_seconds"],
        "classes": class_summaries,
        "update_invalidation": summary["update_invalidation"],
        "provider_http": summary["provider_http"],
        "cost": summary["cost"],
    }, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="真实 API 检索静态缓存冷/热对照")
    parser.add_argument("--allow-paid", action="store_true", help="显式允许真实提供方调用")
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="复用 --run-dir 中已建好的私有索引与累计费用账本",
    )
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--max-cost-usd", type=float, default=2.0)
    args = parser.parse_args()
    if args.iterations < 1:
        raise SystemExit("--iterations 必须 >= 1")
    if not 1 <= args.max_attempts <= 3:
        raise SystemExit("--max-attempts 必须在 1..3")
    if args.max_cost_usd <= 0 or args.max_cost_usd > 2.0:
        raise SystemExit("--max-cost-usd 必须在 (0, 2.0]")
    run(args)


if __name__ == "__main__":
    main()
