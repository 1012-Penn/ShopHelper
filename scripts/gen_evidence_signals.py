"""对 ch04 正样本 + ch09 负样本逐题真检索,生成证据闸校准信号并可选校准。

标签口径：
- label=1：有 required_evidence；evidence_retrieved 表示全部 evidence 组是否进 Top-10；
  检索未命中的正样本只进 retrieval_miss，不参与阈值拟合，也不再算成 junk。
- label=0：无 required_evidence；含同域硬负例（H_hard_negative）和域外 junk；
  clarify 题不参与正样本统计。

用法：
  .venv/bin/python scripts/gen_evidence_signals.py
  .venv/bin/python scripts/gen_evidence_signals.py --calibrate
  .venv/bin/python scripts/gen_evidence_signals.py --calibrate \
      --artifact-config config/evidence_calibration.json
前提：MySQL + Milvus 已灌、embedding key 有效；正样本改写缓存需先由
eval_retrieval.py --rewrite-cache-mode write 生成。
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings
from app.db import make_engine, make_session_factory
from app.embedding import make_embedder
from app.evidence import make_evidence_judge
from app.kb import KnowledgeBaseStore, vector_text
from app.rerank import make_reranker
from app.retrieval import (
    PreparedQuery,
    RetrievalService,
    calibrate_evidence_thresholds,
    combined_confidence_score,
)
from app.rewrite import QueryRewrite, bm25_query_text, make_rewriter
from app.vector_store import KnowledgeVectorStore
from scripts.eval_metrics import EvalDataError, coverage_rank, resolve_gold_chunks
from scripts.rewrite_cache import (
    DEFAULT_CACHE_PATH,
    RewriteCacheError,
    load_rewrites,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SETS = ("tests/eval/ch04_eval_set.jsonl", "tests/eval/ch09_junk_set.jsonl")


def load_cases(paths: tuple[str, ...]) -> list[dict]:
    cases: list[dict] = []
    for rel in paths:
        for line in (ROOT / rel).read_text(encoding="utf-8").splitlines():
            if line.strip():
                cases.append(json.loads(line))
    return cases


def signal_for(case: dict, result, effective_floor: float,
               gold_by_case: dict, verdict=None) -> dict:
    scores = [max(0.0, min(1.0, float(item.score))) for item in result.items]
    top1 = scores[0] if scores else 0.0
    top2 = scores[1] if len(scores) > 1 else 0.0
    margin = top1 - top2 if len(scores) > 1 else top1
    effective_count = sum(score >= effective_floor for score in scores)
    # 检索正样本以“有 required_evidence”为准;clarify 不再被当成检索正样本。
    label = 1 if case.get("required_evidence") else 0
    evidence_retrieved = False
    if label == 1:
        ranked = [item.chunk_id for item in result.items]
        evidence_retrieved = coverage_rank(ranked, gold_by_case[case["id"]], cap=10) is not None
    return {
        "id": case["id"],
        "bucket": case.get("bucket", ""),
        "query": case["query"],
        "label": label,
        "evidence_retrieved": evidence_retrieved,
        "n_items": len(result.items),
        "top1_relevance": round(top1, 6),
        "top1_top2_margin": round(margin, 6),
        "effective_count": effective_count,
        "effective_score_floor": effective_floor,
        "combined_score": round(combined_confidence_score(top1, margin, effective_count), 6),
        "entailed": None if verdict is None else bool(verdict.answered),
        "entailment_reason": "" if verdict is None else verdict.reason,
    }


def _entailment_metrics(signals: list[dict]) -> dict:
    judged = [s for s in signals if s.get("entailed") is not None]
    positives = [s for s in judged if s["label"] == 1]
    hard = [s for s in judged if s["label"] == 0 and s["bucket"] == "H_hard_negative"]
    ood = [
        s for s in judged
        if s["label"] == 0 and s["bucket"] not in {"H_hard_negative", "clarify"}
    ]
    return {
        "enabled": bool(judged),
        "positive_total": len(positives),
        "positive_recall": (sum(bool(s["entailed"]) for s in positives) / len(positives)
                            if positives else 0.0),
        "positive_blocked_ids": [s["id"] for s in positives if not s["entailed"]],
        "hard_negative_false_accept_ids": [s["id"] for s in hard if s["entailed"]],
        "ood_false_accept_ids": [s["id"] for s in ood if s["entailed"]],
    }


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _sets_sha(paths: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for rel in paths:
        digest.update((ROOT / rel).read_bytes())
    return digest.hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                              text=True, check=False).stdout.strip()
    except Exception:
        return ""


def _prepare(cases: list[dict], settings: Settings,
             rewriter, args) -> dict[str, PreparedQuery]:
    if args.no_rewrite or rewriter is None:
        return {c["id"]: PreparedQuery(c["query"], c["query"]) for c in cases}
    positives = [c for c in cases if c.get("required_evidence")]
    positive_sets = [
        path for path in args.eval_sets
        if any(case.get("required_evidence") for case in load_cases((path,)))
    ]
    rewrite_set = ROOT / (positive_sets[0] if positive_sets else args.eval_sets[0])
    rewrites = load_rewrites(positives, args.rewrite_cache,
                             rewrite_set, settings.openai_model,
                             rewriter, mode=args.rewrite_cache_mode)
    prepared: dict[str, PreparedQuery] = {}
    for case in cases:
        qr = rewrites.get(case["id"])
        if qr is None:
            prepared[case["id"]] = PreparedQuery(case["query"], case["query"])
            continue
        rewritten = (qr.normalized or "").strip() or case["query"]
        prepared[case["id"]] = PreparedQuery(
            rewritten,
            bm25_query_text(QueryRewrite(normalized=rewritten, synonyms=qr.synonyms)),
        )
    return prepared


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", dest="eval_sets", nargs="*", default=list(DEFAULT_SETS))
    parser.add_argument("--signals", default="reports/ch09-evidence-signals.jsonl")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--output", default="reports/ch09-evidence-calibration.json")
    parser.add_argument("--artifact-config", default=None,
                        help="同时把 v2 产物写到仓库内配置路径")
    parser.add_argument("--holdout-ratio", type=float, default=1 / 3)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--rewrite-cache", default=str(DEFAULT_CACHE_PATH))
    parser.add_argument("--rewrite-cache-mode", choices=["read", "write", "refresh"],
                        default="read")
    parser.add_argument("--no-rewrite", action="store_true")
    parser.add_argument("--entailment", action=argparse.BooleanOptionalAction,
                        default=None, help="默认跟随 settings.evidence_entailment_enabled")
    args = parser.parse_args()

    settings = Settings()
    entailment_enabled = (settings.evidence_entailment_enabled
                          if args.entailment is None else args.entailment)
    factory = make_session_factory(make_engine(settings))
    kb = KnowledgeBaseStore(factory)
    vectors = KnowledgeVectorStore(settings.milvus_db_path, dim=settings.embedding_dim)
    if vectors.count() == 0:
        print("向量库为空,请先运行:.venv/bin/python -m scripts.build_kb")
        return 2
    all_rows = {r.id: r for r in kb.get_chunks(kb.doc_chunk_ids_all())}
    if not all_rows:
        print("MySQL 知识库为空,请先运行:.venv/bin/python -m scripts.build_kb")
        return 2

    cases = load_cases(tuple(args.eval_sets))
    try:
        gold_by_case = {case["id"]: resolve_gold_chunks(case, all_rows)
                        for case in cases if case.get("required_evidence")}
    except EvalDataError as exc:
        print(f"正样本证据无法映射:{exc}")
        return 4

    rewriter = make_rewriter(settings) if settings.query_rewrite_enabled else None
    service = RetrievalService(
        make_embedder(settings), vectors, kb, rewriter=rewriter,
        reranker=make_reranker(settings), candidates=settings.hybrid_candidates,
        final_top_k=settings.rerank_top_k, rerank_score_floor=settings.rerank_score_floor,
    )
    try:
        prepared = _prepare(cases, settings, rewriter, args)
    except RewriteCacheError as exc:
        print(f"改写缓存不可用:{exc}")
        return 5

    judge = make_evidence_judge(settings) if entailment_enabled else None
    signals = []
    for case in cases:
        result = service.retrieve(case["query"], strategy="hybrid_rerank",
                                  prepared=prepared[case["id"]])
        verdict = None
        if judge is not None:
            documents = []
            for item in result.items[:settings.evidence_entailment_top_n]:
                row = all_rows.get(item.chunk_id)
                if row is not None:
                    documents.append(vector_text(row.category, row.questions, row.answer))
            if documents:
                try:
                    verdict = judge.judge(case["query"], documents)
                except Exception as exc:
                    print(f"⚠️ {case['id']} 证据蕴含校验失败:{exc}")
        signals.append(signal_for(case, result, settings.rerank_score_floor,
                                  gold_by_case, verdict=verdict))

    signals_path = ROOT / args.signals
    signals_path.parent.mkdir(exist_ok=True)
    signals_path.write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in signals) + "\n",
        encoding="utf-8")
    print(f"signals 已写入 {signals_path.relative_to(ROOT)}({len(signals)} 行)")

    if not args.calibrate:
        return 0

    calibration, selection, holdout, split = calibrate_evidence_thresholds(
        signals,
        effective_score_floor=settings.rerank_score_floor,
        holdout_ratio=args.holdout_ratio,
        seed=args.seed,
    )
    status = selection["status"]
    payload = {
        "schema_version": 2,
        "status": status,
        "source": calibration.source,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git("rev-parse", "--short", "HEAD") or "unknown",
        "eval_set_sha256": _sets_sha(tuple(args.eval_sets)),
        "split": split,
        "thresholds": {
            "top1_floor": calibration.top1_floor,
            "effective_score_floor": calibration.effective_score_floor,
            "min_effective_count": calibration.min_effective_count,
            "margin_floor": calibration.margin_floor,
            "combined_floor": calibration.combined_floor,
            "combined_diagnostic_only": True,
        },
        "metrics": {"selection": selection, "holdout": holdout,
                    "entailment": _entailment_metrics(signals)},
        "model": {"embedding": settings.embedding_model, "rerank": settings.rerank_model},
    }
    output = ROOT / args.output
    _atomic_write(output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"校准产物已写入 {output.relative_to(ROOT)}")

    if status != "ok":
        print("⚠️ 校准无可行解,未覆盖生产配置;请检查硬负例/正样本分数分布")
        return 2

    if args.artifact_config:
        cfg = ROOT / args.artifact_config
        if cfg.exists():
            cfg.with_suffix(cfg.suffix + ".bak").write_text(
                cfg.read_text(encoding="utf-8"), encoding="utf-8")
        _atomic_write(cfg, payload)
        print(f"配置产物已写入 {args.artifact_config}(旧文件已备份为 .bak)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
