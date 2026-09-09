"""对 ch04 评估集 + junk 集逐题真检索,生成证据闸校准信号(并可一步产出校准产物)。

用法:
  .venv/bin/python scripts/gen_evidence_signals.py                 # 只产 signals JSONL
  .venv/bin/python scripts/gen_evidence_signals.py --calibrate     # 顺带产出校准产物
  .venv/bin/python scripts/gen_evidence_signals.py --calibrate --artifact-config config/evidence_calibration.json
前提:同 eval_retrieval(MySQL + Milvus 已灌 + .env key 有效)。
relevant 判定复用 chunk_is_relevant(expect_doc 命中 + 关键词命中);junk 集无 expect_doc 恒 false。
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings
from app.db import make_engine, make_session_factory
from app.embedding import make_embedder
from app.kb import KnowledgeBaseStore
from app.rerank import make_reranker
from app.retrieval import RetrievalService, combined_confidence_score
from app.rewrite import make_rewriter
from app.vector_store import KnowledgeVectorStore
from scripts.eval_retrieval import chunk_is_relevant

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SETS = ("tests/eval/ch04_eval_set.jsonl", "tests/eval/ch09_junk_set.jsonl")


def load_cases(paths: tuple[str, ...]) -> list[dict]:
    cases: list[dict] = []
    for rel in paths:
        for line in (ROOT / rel).read_text(encoding="utf-8").splitlines():
            if line.strip():
                cases.append(json.loads(line))
    return cases


def signal_for(case: dict, result, effective_floor: float, kb) -> dict:
    scores = [max(0.0, min(1.0, float(item.score))) for item in result.items]
    top1 = scores[0] if scores else 0.0
    top2 = scores[1] if len(scores) > 1 else 0.0
    margin = top1 - top2 if len(scores) > 1 else top1
    effective_count = sum(score >= effective_floor for score in scores)
    rows = {r.id: r for r in kb.get_chunks([item.chunk_id for item in result.items])}
    relevant = bool(case.get("expect_doc")) and any(
        chunk_is_relevant(row, case) for row in rows.values())
    return {
        "id": case["id"], "bucket": case.get("bucket", ""), "query": case["query"],
        "relevant": relevant, "n_items": len(result.items),
        "top1_relevance": round(top1, 6),
        "top1_top2_margin": round(margin, 6),
        "effective_count": effective_count,
        "effective_score_floor": effective_floor,
        "combined_score": round(combined_confidence_score(top1, margin, effective_count), 6),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", dest="eval_sets", nargs="*", default=list(DEFAULT_SETS))
    parser.add_argument("--signals", default="reports/ch09-evidence-signals.jsonl")
    parser.add_argument("--calibrate", action="store_true", help="生成 signals 后立即校准")
    parser.add_argument("--min-recall", type=float, default=1.0)
    parser.add_argument("--output", default="reports/ch09-evidence-calibration.json")
    parser.add_argument("--artifact-config", default=None,
                        help="同时把产物写到仓库内配置路径(如 config/evidence_calibration.json)")
    args = parser.parse_args()

    settings = Settings()
    factory = make_session_factory(make_engine(settings))
    kb = KnowledgeBaseStore(factory)
    vectors = KnowledgeVectorStore(settings.milvus_db_path, dim=settings.embedding_dim)
    if vectors.count() == 0:
        print("向量库为空,请先运行:.venv/bin/python -m scripts.build_kb")
        return 2
    service = RetrievalService(
        make_embedder(settings), vectors, kb,
        rewriter=make_rewriter(settings) if settings.query_rewrite_enabled else None,
        reranker=make_reranker(settings), candidates=settings.hybrid_candidates,
        final_top_k=settings.rerank_top_k, rerank_score_floor=settings.rerank_score_floor,
    )
    cases = load_cases(tuple(args.eval_sets))
    print(f"评估集 {len(cases)} 题,开始真检索…")

    signals = []
    for case in cases:
        result = service.retrieve(case["query"], strategy="hybrid_rerank")
        signals.append(signal_for(case, result, settings.rerank_score_floor, kb))
        mark = "✅" if signals[-1]["relevant"] else "⬛"
        print(f"{mark} {case['id']} top1={signals[-1]['top1_relevance']:.3f} "
              f"count={signals[-1]['effective_count']} combined={signals[-1]['combined_score']:.3f}")

    signals_path = ROOT / args.signals
    signals_path.parent.mkdir(exist_ok=True)
    signals_path.write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in signals) + "\n", encoding="utf-8")
    print(f"signals 已写入 {signals_path.relative_to(ROOT)}({len(signals)} 行)")

    if not args.calibrate:
        return 0
    from app.retrieval import calibrate_evidence_thresholds

    calibration = calibrate_evidence_thresholds(signals, min_recall=args.min_recall)
    positives = [s for s in signals if s["relevant"]]
    blocked_positive = [p["id"] for p in positives if not (
        p["top1_relevance"] >= calibration.top1_floor
        and p["effective_count"] >= calibration.min_effective_count
        and p["top1_top2_margin"] >= calibration.margin_floor
        and p["combined_score"] >= calibration.combined_floor)]
    junk = [s for s in signals if not s["relevant"] and s["n_items"] > 0]
    blocked_junk = [j["id"] for j in junk if not (
        j["top1_relevance"] >= calibration.top1_floor
        and j["effective_count"] >= calibration.min_effective_count
        and j["top1_top2_margin"] >= calibration.margin_floor
        and j["combined_score"] >= calibration.combined_floor)]
    payload = {
        "source": calibration.source, "min_recall": args.min_recall,
        "top1_floor": calibration.top1_floor,
        "effective_score_floor": calibration.effective_score_floor,
        "min_effective_count": calibration.min_effective_count,
        "margin_floor": calibration.margin_floor,
        "combined_floor": calibration.combined_floor,
        "sample_count": len(signals), "positive_count": len(positives),
        "positive_blocked": blocked_positive,
        "junk_count": len(junk), "junk_blocked": blocked_junk,
    }
    output = ROOT / args.output
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"校准产物已写入 {output.relative_to(ROOT)}")
    if args.artifact_config:
        cfg = ROOT / args.artifact_config
        cfg.parent.mkdir(exist_ok=True)
        cfg.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"配置产物已写入 {args.artifact_config}")
    if blocked_positive:
        print(f"⚠️ 正样本被闸拦下:{blocked_positive}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
