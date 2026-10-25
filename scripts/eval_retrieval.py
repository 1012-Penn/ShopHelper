"""ch04 检索质量评估（严格证据版）。

判定真值来自评估集的 required_evidence：每组 evidence 的 quotes 必须同时落在
同一个 chunk 内；多组 evidence 之间是 AND，coverage_rank 取各组首次命中名次的最大值。
不再使用「文档名 + 任一关键词」的宽松判定。

用法：
  .venv/bin/python scripts/eval_retrieval.py                      # 冻结改写 + 四策略
  .venv/bin/python scripts/eval_retrieval.py --no-rewrite         # 恒等改写基线
  .venv/bin/python scripts/eval_retrieval.py --with-baseline      # 主模式 + 恒等改写对照
  .venv/bin/python scripts/eval_retrieval.py --rewrite-cache-mode write --strategy hybrid_rerank
  .venv/bin/python scripts/eval_retrieval.py --rewrite-live       # 诊断用，不更新 latest 报告
退出码：0 全命中；1 有 R@10 未命中；2 库空；3 索引/DB 不一致；
        4 证据无法映射；5 改写缓存不可用。
"""
import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings
from app.db import make_engine, make_session_factory
from app.embedding import make_embedder
from app.graph.builder import _load_evidence_calibration
from app.kb import KnowledgeBaseStore
from app.rerank import make_reranker
from app.retrieval import (
    RETRIEVAL_SCORE_FLOOR,
    STRATEGIES,
    PreparedQuery,
    RetrievalService,
)
from app.rewrite import QueryRewrite, bm25_query_text, make_rewriter
from app.vector_store import KnowledgeVectorStore
from scripts.eval_metrics import (
    EvalDataError,
    chunk_covers,
    coverage_rank,
    evidence_group_recall,
    needs_from_case,
    resolve_gold_chunks,
    single_chunk_rank,
)
from scripts.rewrite_cache import (
    DEFAULT_CACHE_PATH,
    RewriteCacheError,
    cache_sha256,
    load_rewrites,
)

ROOT = Path(__file__).resolve().parent.parent
LATEST_REPORT = ROOT / "reports" / "ch04-retrieval-report.md"


# ---- 兼容旧诊断脚本 ---------------------------------------------------------

def chunk_is_relevant(row, case: dict) -> bool:
    """兼容壳：新评估请用 eval_metrics；旧数据仍按 expect_doc + 关键词判定。"""
    if "required_evidence" in case:
        return any(chunk_covers(row, need) for need in needs_from_case(case))
    docs = case["expect_doc"] if isinstance(case["expect_doc"], list) else [case["expect_doc"]]
    if not row.section_path or not any(d in row.section_path for d in docs):
        return False
    if case.get("expect_section") and case["expect_section"] not in row.section_path:
        return False
    return any(kw in row.answer or kw in row.questions for kw in case.get("expect_keywords", []))


def recall_mrr(ranked: list[tuple[int, float]], rows_by_id: dict, case: dict,
               k_list: tuple[int, ...] = (3, 5, 10), mrr_k: int = 10):
    """兼容壳：旧指标（首次相关命中即止）。新评估用 coverage_rank。"""
    hits = {k: False for k in k_list}
    rr = 0.0
    for pos, (cid, _score) in enumerate(ranked):
        row = rows_by_id.get(cid)
        if row is None or not chunk_is_relevant(row, case):
            continue
        for k in k_list:
            hits[k] = pos < k
        if pos < mrr_k:
            rr = 1.0 / (pos + 1)
        break
    return hits, rr


# ---- 元数据 -----------------------------------------------------------------

def _git(*args: str) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                             text=True, check=False)
        return out.stdout.strip()
    except Exception:
        return ""


def _file_sha(path: str | Path) -> str:
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else ""


def _cache_meta(path: str | Path) -> dict:
    p = Path(path)
    meta = {"path": str(p), "sha256": _file_sha(p), "created_at": ""}
    if p.exists():
        try:
            meta["created_at"] = json.loads(p.read_text(encoding="utf-8")).get("created_at", "")
        except Exception:
            pass
    return meta


def _doc_counts(all_rows: dict) -> str:
    counts = Counter(row.section_path.split(" >", 1)[0] for row in all_rows.values())
    return ", ".join(f"{doc}={n}" for doc, n in sorted(counts.items()))


# ---- 改写准备 ---------------------------------------------------------------

def _prepared_from_rewrite(query: str, qr: QueryRewrite) -> PreparedQuery:
    rewritten = (qr.normalized or "").strip() or query
    search_text = bm25_query_text(QueryRewrite(normalized=rewritten, synonyms=qr.synonyms))
    return PreparedQuery(rewritten, search_text)


def _identity(cases: list[dict]) -> dict[str, PreparedQuery]:
    return {c["id"]: PreparedQuery(c["query"], c["query"]) for c in cases}


def _prepare(args, cases: list[dict], service: RetrievalService, settings: Settings,
             eval_path: Path, rewriter) -> tuple[dict[str, PreparedQuery], dict]:
    if args.no_rewrite or args.rewrite_mode == "identity":
        return _identity(cases), {"mode": "identity"}
    if args.rewrite_live:
        return {c["id"]: service.prepare_query(c["query"]) for c in cases}, {"mode": "live"}
    if rewriter is None:
        return _identity(cases), {"mode": "identity(query_rewrite_disabled)"}
    rewrites = load_rewrites(cases, args.rewrite_cache, eval_path, settings.openai_model,
                             rewriter, mode=args.rewrite_cache_mode)
    return ({c["id"]: _prepared_from_rewrite(c["query"], rewrites[c["id"]]) for c in cases},
            {"mode": "frozen", "cache_mode": args.rewrite_cache_mode})


# ---- 评估 -------------------------------------------------------------------

def _empty_bucket() -> dict:
    return {"n": 0, "r3": 0, "r5": 0, "r10": 0, "mrr": 0.0,
            "single1": 0, "single3": 0, "group_recall": 0.0,
            "degraded": 0, "low_conf": 0, "filter_fallback": 0}


def _run(cases: list[dict], service: RetrievalService, all_rows: dict,
         gold_by_case: dict, prepared: dict[str, PreparedQuery],
         strategies: list[str], *, print_detail: bool = True):
    stats = {s: defaultdict(_empty_bucket) for s in strategies}
    misses: list[tuple[str, str, str]] = []
    for strategy in strategies:
        for case in cases:
            result = service.retrieve(case["query"], strategy=strategy,
                                      prepared=prepared[case["id"]])
            ranked = [item.chunk_id for item in result.items]
            gold = gold_by_case[case["id"]]
            cov = coverage_rank(ranked, gold, cap=10)
            single = single_chunk_rank(ranked, case, all_rows, cap=10)
            group_recall = evidence_group_recall(ranked, gold, cap=10)
            b = stats[strategy][case["bucket"]]
            b["n"] += 1
            b["r3"] += cov is not None and cov <= 3
            b["r5"] += cov is not None and cov <= 5
            b["r10"] += cov is not None and cov <= 10
            b["mrr"] += 1.0 / cov if cov is not None and cov <= 10 else 0.0
            b["single1"] += single == 1
            b["single3"] += single is not None and single <= 3
            b["group_recall"] += group_recall
            b["degraded"] += result.degraded
            b["low_conf"] += result.low_confidence
            b["filter_fallback"] += result.filter_fallback
            if cov is None:
                misses.append((strategy, case["id"], case["query"]))
            if print_detail:
                mark = "✅" if cov is not None and cov <= 10 else "❌"
                print(f"{mark} [{strategy}] {case['id']} cov={cov} "
                      f"single={single} gr={group_recall:.2f} {case['query']}")
    return stats, misses


def _total(stats_by_bucket: dict) -> dict:
    total = _empty_bucket()
    for b in stats_by_bucket.values():
        for key in total:
            total[key] += b[key]
    return total


def _metrics(bucket: dict) -> dict:
    n = bucket["n"] or 1
    return {
        "r3": bucket["r3"] / n, "r5": bucket["r5"] / n, "r10": bucket["r10"] / n,
        "mrr": bucket["mrr"] / n, "single1": bucket["single1"] / n,
        "single3": bucket["single3"] / n, "group_recall": bucket["group_recall"] / n,
        "degraded": bucket["degraded"], "low_conf": bucket["low_conf"],
        "filter_fallback": bucket["filter_fallback"], "n": bucket["n"],
    }


def _summary_row(name: str, bucket: dict) -> str:
    m = _metrics(bucket)
    return (f"| {name} | {m['r3']:.3f} | {m['r5']:.3f} | {m['r10']:.3f} | "
            f"{m['mrr']:.3f} | {m['single1']:.3f} | {m['single3']:.3f} | "
            f"{m['group_recall']:.3f} | {m['degraded']} | {m['low_conf']} | {m['n']} |")


def _bucket_table(name: str, stats_by_bucket: dict, buckets: list[str]) -> list[str]:
    cells = []
    for bucket in buckets:
        m = _metrics(stats_by_bucket.get(bucket, _empty_bucket()))
        cells.append(f"{m['r3']:.2f}/{m['mrr']:.2f}")
    return [f"| {name} | " + " | ".join(cells) + " |"]


# ---- 报告 -------------------------------------------------------------------

def _write_report(meta: list[tuple[str, str]], primary: dict, baseline: dict | None,
                  misses: list[tuple[str, str, str]], strategies: list[str],
                  buckets: list[str], args) -> Path:
    now = datetime.now(timezone.utc)
    lines = ["# ch04 检索评估（严格证据版）", "", "## 元数据", ""]
    lines += [f"- {key}: {value}" for key, value in meta]
    lines += ["", "## 汇总（47 道正样本，覆盖全部 required_evidence）", "",
              "| strategy | R@3 | R@5 | R@10 | MRR@10 | single@1 | single@3 | "
              "group_recall | degraded | low_conf | n |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for strategy in strategies:
        lines.append(_summary_row(strategy, _total(primary[strategy])))
    lines += ["", "指标口径：R@K = 全部 evidence 组都进 Top-K；single@K = 单个 chunk 同时覆盖全部 evidence；"
                  "group_recall = 命中的 evidence 组比例。", ""]

    lines += ["## 分桶（R@3 / MRR@10）", "",
              "| strategy | " + " | ".join(buckets) + " |",
              "|---|" + "---|" * len(buckets)]
    for strategy in strategies:
        lines += _bucket_table(strategy, primary[strategy], buckets)
    lines.append("")

    if baseline is not None:
        lines += ["## 恒等改写基线对照", "",
                  "| strategy | 主 R@3 | 基线 R@3 | ΔR@3 | 主 MRR | 基线 MRR | ΔMRR |",
                  "|---|---|---|---|---|---|---|"]
        for strategy in strategies:
            main = _metrics(_total(primary[strategy]))
            base = _metrics(_total(baseline[strategy]))
            lines.append(
                f"| {strategy} | {main['r3']:.3f} | {base['r3']:.3f} | "
                f"{main['r3'] - base['r3']:+.3f} | {main['mrr']:.3f} | "
                f"{base['mrr']:.3f} | {main['mrr'] - base['mrr']:+.3f} |")
        lines.append("")

    lines += ["## R@10 未命中", ""]
    if misses:
        lines += [f"- [{s}] {cid} {query}" for s, cid, query in misses]
    else:
        lines.append("- 无")
    lines.append("")

    text = "\n".join(lines)
    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"ch04-retrieval-{now.strftime('%Y%m%dT%H%M%SZ')}.md"
    path.write_text(text, encoding="utf-8")
    if not args.rewrite_live:
        LATEST_REPORT.write_text(text, encoding="utf-8")
    return path


# ---- 一致性守卫 -------------------------------------------------------------

def _load_and_check(kb, vectors) -> tuple[int, str, dict]:
    """→ (退出码, 错误信息, all_rows)。0 表示通过。"""
    mysql_ids = set(kb.doc_chunk_ids_all())
    if not mysql_ids:
        return 2, "MySQL 知识库为空,请先运行:.venv/bin/python -m scripts.build_kb", {}
    all_rows = {r.id: r for r in kb.get_chunks(list(mysql_ids))}
    pending = kb.pending_chunks()
    milvus_ids = set(vectors.all_ids())
    if pending or mysql_ids != milvus_ids:
        missing = sorted(mysql_ids - milvus_ids)[:20]
        extra = sorted(milvus_ids - mysql_ids)[:20]
        return 3, f"索引/DB 不一致:pending={len(pending)} missing={missing} extra={extra}", {}
    return 0, "", all_rows


# ---- 入口 -------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="all", choices=[*STRATEGIES, "all"])
    parser.add_argument("--set", dest="eval_set", default="tests/eval/ch04_eval_set.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-rewrite", action="store_true")
    parser.add_argument("--rewrite-mode", choices=["frozen", "identity"], default="frozen")
    parser.add_argument("--rewrite-cache", default=str(DEFAULT_CACHE_PATH))
    parser.add_argument("--rewrite-cache-mode", choices=["read", "write", "refresh"],
                        default="read")
    parser.add_argument("--rewrite-live", action="store_true",
                        help="诊断用：直接调 LLM，不更新 latest 报告")
    parser.add_argument("--with-baseline", action="store_true",
                        help="同进程再跑一轮恒等改写，报告给出 delta")
    args = parser.parse_args()
    if args.no_rewrite:
        args.rewrite_mode = "identity"

    settings = Settings()
    kb = KnowledgeBaseStore(make_session_factory(make_engine(settings)))
    vectors = KnowledgeVectorStore(settings.milvus_db_path, dim=settings.embedding_dim)
    if vectors.count() == 0:
        print("向量库为空,请先运行:.venv/bin/python -m scripts.build_kb")
        return 2

    code, message, all_rows = _load_and_check(kb, vectors)
    if code:
        print(message)
        return code
    pending = kb.pending_chunks()
    mysql_ids = set(kb.doc_chunk_ids_all())
    milvus_ids = set(vectors.all_ids())

    eval_path = ROOT / args.eval_set
    cases = [json.loads(line) for line in eval_path.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    if args.limit:
        cases = cases[:args.limit]
    if any("required_evidence" not in case for case in cases):
        print("评估集包含没有 required_evidence 的样本;检索评估只接受正样本集")
        return 4
    try:
        gold_by_case = {case["id"]: resolve_gold_chunks(case, all_rows) for case in cases}
    except EvalDataError as exc:
        print(f"评估集证据无法映射:{exc}")
        return 4

    rewriter = make_rewriter(settings) if settings.query_rewrite_enabled else None
    service = RetrievalService(
        make_embedder(settings), vectors, kb, rewriter=rewriter,
        reranker=make_reranker(settings), candidates=settings.hybrid_candidates,
        final_top_k=settings.rerank_top_k, rerank_score_floor=settings.rerank_score_floor,
        evidence_calibration=_load_evidence_calibration(settings),
    )
    try:
        prepared, mode_meta = _prepare(args, cases, service, settings, eval_path, rewriter)
    except RewriteCacheError as exc:
        print(f"改写缓存不可用:{exc}")
        return 5

    strategies = list(STRATEGIES) if args.strategy == "all" else [args.strategy]
    primary, misses = _run(cases, service, all_rows, gold_by_case, prepared, strategies)

    baseline = None
    if args.with_baseline and mode_meta["mode"] != "identity":
        baseline, _ = _run(cases, service, all_rows, gold_by_case, _identity(cases),
                           strategies, print_detail=False)

    cache_meta = _cache_meta(args.rewrite_cache)
    meta = [
        ("generated_at", datetime.now(timezone.utc).isoformat()),
        ("git_commit", _git("rev-parse", "--short", "HEAD") or "unknown"),
        ("git_branch", _git("rev-parse", "--abbrev-ref", "HEAD") or "unknown"),
        ("git_dirty", "yes" if _git("status", "--porcelain") else "no"),
        ("eval_set", str(eval_path.relative_to(ROOT))),
        ("eval_set_sha256", _file_sha(eval_path)),
        ("case_count", str(len(cases))),
        ("buckets", ", ".join(f"{b}={n}" for b, n in sorted(Counter(c["bucket"] for c in cases).items()))),
        ("mysql_chunks", str(len(mysql_ids))),
        ("milvus_chunks", str(len(milvus_ids))),
        ("pending_chunks", str(len(pending))),
        ("doc_chunks", _doc_counts(all_rows)),
        ("embedding_model", settings.embedding_model),
        ("rerank_model", settings.rerank_model),
        ("rewrite_model", settings.openai_model if settings.query_rewrite_enabled else "disabled"),
        ("hybrid_candidates", str(settings.hybrid_candidates)),
        ("rerank_top_k", str(settings.rerank_top_k)),
        ("dense_score_floor", str(RETRIEVAL_SCORE_FLOOR)),
        ("rerank_score_floor", str(settings.rerank_score_floor)),
        ("rewrite_mode", mode_meta["mode"] + (
            f" ({mode_meta['cache_mode']})" if "cache_mode" in mode_meta else "")),
        ("rewrite_cache", cache_meta["path"]),
        ("rewrite_cache_sha256", cache_meta["sha256"]),
        ("rewrite_cache_created_at", cache_meta["created_at"]),
        ("baseline", "identity" if baseline is not None else "none"),
        ("evidence_gate", "v2 score fallback (production adds entailment judge)"),
        ("live", "yes" if args.rewrite_live else "no"),
    ]
    path = _write_report(meta, primary, baseline, misses, strategies,
                         sorted({c["bucket"] for c in cases}), args)
    print(f"\n报告已写入 {path}")
    if misses:
        print(f"R@10 未命中 {len(misses)} 例")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
