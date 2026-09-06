"""ch04 忠实度评估:hybrid_rerank 生成带引用答案 → LLM 裁判判编造 → faith_cases 台账 + 分桶报告。

用法:
  .venv/bin/python scripts/eval_faithfulness.py                # 全量 32 题,落台账
  .venv/bin/python scripts/eval_faithfulness.py --limit 4      # 冒烟
  .venv/bin/python scripts/eval_faithfulness.py --no-db        # 只出报告不落库
前提:同 eval_retrieval(建库 + .env)。判出的编造个案按 spec §3.1 upsert 进 faith_cases。
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
from app.kb import FaithCaseLedger, KnowledgeBaseStore
from app.llm import make_chat_model, make_extract_model
from app.prompts import (
    FAITHFULNESS_JUDGE_PROMPT,
    RAG_ANSWER_PROMPT,
    FaithfulnessVerdict,
    build_evidence_block,
)
from app.rerank import make_reranker
from app.retrieval import RetrievalService, lost_in_middle_order
from app.rewrite import make_rewriter
from app.vector_store import KnowledgeVectorStore

ROOT = Path(__file__).resolve().parent.parent
STRATEGY = "hybrid_rerank"


def make_evidence(result, rows) -> list[dict]:
    """n = 精排名次;摆放反漏斗(喂给模型的顺序),citations 快照保留摆放序。"""
    evidence = [
        {"n": i + 1, "chunk_id": r.chunk_id,
         "question": row.questions.splitlines()[0], "answer": row.answer,
         "section_path": row.section_path or "", "category": row.category}
        for i, (r, row) in enumerate(zip(result.items, rows))
    ]
    return lost_in_middle_order(evidence)


def run_case(case, service, kb, chat, judge, ledger, judge_model: str = "unknown") -> tuple[bool, str]:
    """单题:检索 → 生成带引用答案 → 裁判 → 编造个案落台账。返回 (是否编造, 理由)。"""
    result = service.retrieve(case["query"], strategy=STRATEGY)
    rows = kb.get_chunks([r.chunk_id for r in result.items])
    evidence = make_evidence(result, rows)
    answer = chat.invoke(RAG_ANSWER_PROMPT.format(
        evidence=build_evidence_block(evidence), query=case["query"])).content
    verdict = judge.invoke(FAITHFULNESS_JUDGE_PROMPT.format(
        query=case["query"], evidence=build_evidence_block(evidence), answer=answer))
    if isinstance(verdict, dict):
        verdict = FaithfulnessVerdict(**verdict)
    if verdict.fabricated and ledger is not None:
        ledger.upsert_case(eval_id=case["id"], bucket=case["bucket"], query=case["query"],
                           strategy=STRATEGY, answer=answer, reason=verdict.reason,
                           citations=evidence, judge_model=judge_model)
    return bool(verdict.fabricated), str(verdict.reason)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-db", action="store_true")
    parser.add_argument("--set", dest="eval_set", default="tests/eval/ch04_eval_set.jsonl")
    args = parser.parse_args()

    settings = Settings()
    kb = KnowledgeBaseStore(make_session_factory(make_engine(settings)))
    vectors = KnowledgeVectorStore(settings.milvus_db_path, dim=settings.embedding_dim)
    if vectors.count() == 0:
        print("向量库为空,请先运行:.venv/bin/python -m scripts.build_kb")
        return 2
    service = RetrievalService(
        make_embedder(settings), vectors, kb,
        rewriter=make_rewriter(settings) if settings.query_rewrite_enabled else None,
        reranker=make_reranker(settings),
        candidates=settings.hybrid_candidates, final_top_k=settings.retrieval_final_top_k,
        rerank_score_floor=settings.rerank_score_floor,
    )
    chat = make_chat_model(settings)
    judge = make_extract_model(settings).with_structured_output(FaithfulnessVerdict,
                                                                method="function_calling")
    ledger = None if args.no_db else FaithCaseLedger(make_session_factory(make_engine(settings)))
    judge_model = settings.openai_model

    cases = [json.loads(ln) for ln in
             (ROOT / args.eval_set).read_text(encoding="utf-8").splitlines() if ln.strip()]
    if args.limit:
        cases = cases[: args.limit]

    bucket_stats: dict[str, dict] = defaultdict(lambda: {"ok": 0, "n": 0})
    fabricated_cases: list[str] = []
    for case in cases:
        fab, reason = run_case(case, service, kb, chat, judge, ledger, judge_model=judge_model)
        b = bucket_stats[case["bucket"]]
        b["n"] += 1
        if fab:
            fabricated_cases.append(f"{case['id']} {case['query']} —— {reason}")
            print(f"❌ {case['id']} {case['query']} 编造:{reason}")
        else:
            b["ok"] += 1
            print(f"✅ {case['id']} {case['query']}")

    print("\n== Faithfulness(hybrid_rerank)==")
    print(f"{'bucket':<12}{'非编造率':>9}{'n':>4}")
    for bucket, b in sorted(bucket_stats.items()):
        print(f"{bucket:<12}{b['ok']/b['n']:>9.2f}{b['n']:>4}")

    report_dir = ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    lines = ["# ch04 忠实度评估报告(hybrid_rerank)", "",
             f"评估集:{args.eval_set};judge_model:{judge_model}", "",
             "| bucket | Faithfulness | n |", "|---|---|---|"]
    for bucket, b in sorted(bucket_stats.items()):
        lines.append(f"| {bucket} | {b['ok']/b['n']:.2f} | {b['n']} |")
    if fabricated_cases:
        lines += ["", "## 编造个案(已进 faith_cases 台账)"]
        lines += [f"- {c}" for c in fabricated_cases]
    (report_dir / "ch04-faithfulness-report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n报告已写入 reports/ch04-faithfulness-report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
