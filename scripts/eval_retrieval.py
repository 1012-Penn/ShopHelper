"""ch04 检索质量评估:四策略 × 评估集,Recall@3/5/10 + MRR@10,按桶分列 + 总表。

用法:
  .venv/bin/python scripts/eval_retrieval.py                       # 四策略全跑 ch04 评估集
  .venv/bin/python scripts/eval_retrieval.py --strategy bm25       # 单策略
  .venv/bin/python scripts/eval_retrieval.py --set tests/eval/retrieval_samples.jsonl  # ch03 回归集
  .venv/bin/python scripts/eval_retrieval.py --no-rewrite          # 关改写对照
前提:MySQL 已建库、Milvus v2 集合已灌(python -m scripts.build_kb)、.env key 有效。
退出码:0 全中 / 1 有用例 Recall@10 未命中 / 2 向量库为空。
注:当前已知硬题 C04(「这个东西能便宜点不」→ 95 折)四策略均未命中,全量跑固定退出 1;
以报告数字为准,退出码只作 CI 严格模式参考。
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
from app.kb import KnowledgeBaseStore
from app.rerank import make_reranker
from app.retrieval import STRATEGIES, RetrievalService
from app.rewrite import make_rewriter
from app.vector_store import KnowledgeVectorStore

ROOT = Path(__file__).resolve().parent.parent


def chunk_is_relevant(row, case: dict) -> bool:
    """命中判定:章节路径含期望文档(及期望章节,如有)+ 原文含任一期望关键词。"""
    docs = case["expect_doc"] if isinstance(case["expect_doc"], list) else [case["expect_doc"]]
    if not row.section_path or not any(d in row.section_path for d in docs):
        return False
    if case.get("expect_section") and case["expect_section"] not in row.section_path:
        return False
    return any(kw in row.answer or kw in row.questions for kw in case["expect_keywords"])


def recall_mrr(ranked: list[tuple[int, float]], rows_by_id: dict, case: dict,
               k_list: tuple[int, ...] = (3, 5, 10), mrr_k: int = 10):
    """ranked 按策略名次排序;命中即止(取首个正确 chunk 的倒数排名)。"""
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="all", choices=[*STRATEGIES, "all"])
    parser.add_argument("--set", dest="eval_set", default="tests/eval/ch04_eval_set.jsonl")
    parser.add_argument("--no-rewrite", action="store_true")
    args = parser.parse_args()

    settings = Settings()
    kb = KnowledgeBaseStore(make_session_factory(make_engine(settings)))
    vectors = KnowledgeVectorStore(settings.milvus_db_path, dim=settings.embedding_dim)
    if vectors.count() == 0:
        print("向量库为空,请先运行:.venv/bin/python -m scripts.build_kb")
        return 2
    rewriter = make_rewriter(settings) if (settings.query_rewrite_enabled and not args.no_rewrite) else None
    service = RetrievalService(
        make_embedder(settings), vectors, kb, rewriter=rewriter, reranker=make_reranker(settings),
        candidates=settings.hybrid_candidates, final_top_k=settings.retrieval_final_top_k,
        rerank_score_floor=settings.rerank_score_floor,
    )
    cases = [json.loads(ln) for ln in
             (ROOT / args.eval_set).read_text(encoding="utf-8").splitlines() if ln.strip()]
    strategies = list(STRATEGIES) if args.strategy == "all" else [args.strategy]

    stats = {s: defaultdict(lambda: {"r3": 0, "r5": 0, "r10": 0, "rr": 0.0, "n": 0})
             for s in strategies}
    misses: list[tuple[str, str, str]] = []
    for s in strategies:
        for case in cases:
            result = service.retrieve(case["query"], strategy=s)
            rows_by_id = {r.id: r for r in kb.get_chunks([it.chunk_id for it in result.items])}
            hits, rr = recall_mrr([(it.chunk_id, it.score) for it in result.items], rows_by_id, case)
            b = stats[s][case["bucket"]]
            b["n"] += 1
            b["r3"] += hits[3]
            b["r5"] += hits[5]
            b["r10"] += hits[10]
            b["rr"] += rr
            if not hits[10]:
                misses.append((s, case["id"], case["query"]))
            top = result.items[0].chunk_id if result.items else None
            mark = "✅" if hits[10] else "❌"
            print(f"{mark} [{s}] {case['id']} {case['query']} -> top1={top} rr={rr:.3f}")

    for s in strategies:
        print(f"\n== 策略 {s} ==")
        print(f"{'bucket':<12}{'R@3':>7}{'R@5':>7}{'R@10':>7}{'MRR@10':>9}{'n':>4}")
        for bucket, b in sorted(stats[s].items()):
            n = b["n"] or 1
            print(f"{bucket:<12}{b['r3']/n:>7.2f}{b['r5']/n:>7.2f}{b['r10']/n:>7.2f}"
                  f"{b['rr']/n:>9.3f}{b['n']:>4}")

    report_dir = ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    lines = [
        "# ch04 检索四策略对比报告", "",
        f"评估集:{args.eval_set}({len(cases)} 题);rewrite={'off' if args.no_rewrite else 'on'}", "",
    ]
    for s in strategies:
        lines += [f"## {s}", "", "| bucket | Recall@3 | Recall@5 | Recall@10 | MRR@10 | n |",
                  "|---|---|---|---|---|---|"]
        for bucket, b in sorted(stats[s].items()):
            n = b["n"] or 1
            lines.append(f"| {bucket} | {b['r3']/n:.2f} | {b['r5']/n:.2f} | {b['r10']/n:.2f} "
                         f"| {b['rr']/n:.3f} | {b['n']} |")
        lines.append("")
    (report_dir / "ch04-retrieval-report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n报告已写入 reports/ch04-retrieval-report.md")
    if misses:
        print(f"Recall@10 未命中 {len(misses)} 例:{[(m[0], m[1]) for m in misses]}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
