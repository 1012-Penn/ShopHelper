"""检索质量评估:对 tests/eval/retrieval_samples.jsonl 逐条真跑向量检索,报 hit@top_k。

用法:.venv/bin/python scripts/eval_retrieval.py
前提:MySQL 已建库(knowledge_chunks 有数据)、Milvus 已建库(python -m scripts.build_kb)、
.env 的 EMBEDDING_API_KEY 有效。退出码:0 全中 / 1 有未命中 / 2 向量库为空。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings
from app.db import make_engine, make_session_factory
from app.embedding import make_embedder
from app.kb import KnowledgeBaseStore
from app.vector_store import KnowledgeVectorStore

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "tests" / "eval" / "retrieval_samples.jsonl"


def main(top_k: int = 3) -> int:
    settings = Settings()
    kb = KnowledgeBaseStore(make_session_factory(make_engine(settings)))
    vectors = KnowledgeVectorStore(settings.milvus_db_path, dim=settings.embedding_dim)
    embedder = make_embedder(settings)
    if vectors.count() == 0:
        print("向量库为空,请先运行:.venv/bin/python -m scripts.build_kb")
        return 2

    hits = 0
    misses: list[str] = []
    cases = [json.loads(ln) for ln in SAMPLES.read_text(encoding="utf-8").splitlines() if ln.strip()]
    qvecs = embedder.embed([c["query"] for c in cases])
    for case, qvec in zip(cases, qvecs):
        results = vectors.search(qvec, top_k=top_k)
        rows = kb.get_chunks([r[0] for r in results])
        scores = {r[0]: r[1] for r in results}
        expect_docs = case["expect_doc"] if isinstance(case["expect_doc"], list) else [case["expect_doc"]]
        ok = any(
            row.section_path
            and any(doc in row.section_path for doc in expect_docs)
            and any(kw in row.answer or kw in row.questions for kw in case["expect_keywords"])
            for row in rows
        )
        hits += ok
        if not ok:
            misses.append(case["query"])
        detail = [(row.section_path, round(scores[row.id], 3)) for row in rows]
        print(f"{'✅' if ok else '❌'} {case['query']} -> {detail}")

    print(f"\nhit@{top_k}: {hits}/{len(cases)}")
    if misses:
        print(f"未命中:{misses}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
