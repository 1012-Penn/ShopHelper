"""复用 ch04 评估集跑一轮 Recall@K、MRR、Faithfulness 并落 eval_runs。

手动运行: .venv/bin/python scripts/run_eval.py --triggered-by 手动
定时任务: .venv/bin/python scripts/run_eval.py --triggered-by 定时
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings
from app.db import make_engine, make_session_factory
from app.embedding import make_embedder
from app.evaluation import EvalRunStore
from app.kb import FaithCaseLedger, KnowledgeBaseStore
from app.llm import make_chat_model, make_extract_model
from app.models import Base
from app.prompts import FaithfulnessVerdict
from app.rerank import make_reranker
from app.retrieval import RetrievalService
from app.rewrite import make_rewriter
from app.vector_store import KnowledgeVectorStore
from scripts.eval_faithfulness import run_case
from scripts.eval_retrieval import chunk_is_relevant, recall_mrr

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--triggered-by", choices=["定时", "手动"], default="手动")
    parser.add_argument("--set", dest="eval_set", default="tests/eval/ch04_eval_set.jsonl")
    parser.add_argument("--limit", type=int, default=None)
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
    cases = [json.loads(line) for line in (ROOT / args.eval_set).read_text(encoding="utf-8").splitlines()
             if line.strip()]
    if args.limit:
        cases = cases[:args.limit]

    recall3 = recall5 = recall10 = mrr = 0.0
    for case in cases:
        result = service.retrieve(case["query"], strategy="hybrid_rerank")
        rows = {r.id: r for r in kb.get_chunks([item.chunk_id for item in result.items])}
        hits, rr = recall_mrr([(item.chunk_id, item.score) for item in result.items], rows, case)
        recall3 += hits[3]
        recall5 += hits[5]
        recall10 += hits[10]
        mrr += rr

    chat = make_chat_model(settings)
    judge = make_extract_model(settings).with_structured_output(
        FaithfulnessVerdict, method="function_calling")
    ledger = FaithCaseLedger(factory)
    faithful = 0
    for case in cases:
        fabricated, _ = run_case(case, service, kb, chat, judge, ledger,
                                 judge_model=settings.openai_model)
        faithful += not fabricated
    n = len(cases) or 1
    metrics = {"recall_at_3": recall3 / n, "recall_at_5": recall5 / n,
               "recall_at_10": recall10 / n, "mrr": mrr / n,
               "faithfulness": faithful / n}
    run_id = EvalRunStore(factory).record(args.triggered_by, len(cases), metrics)
    report = ROOT / "reports" / "ch09-eval-report.md"
    report.parent.mkdir(exist_ok=True)
    report.write_text("# ch09 自动评估\n\n" + json.dumps({"run_id": run_id, **metrics},
                                                    ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps({"run_id": run_id, "dataset_size": len(cases), **metrics},
                     ensure_ascii=False, indent=2))
    print(f"报告已写入 {report.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
