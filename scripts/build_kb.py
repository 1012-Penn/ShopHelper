"""离线建库:knowledge/*.md → 结构感知切分 → MySQL(pending)→ embed → Milvus → done。

用法:
  python -m scripts.build_kb                  # 全量建库(文档重建 + 向量化,幂等)
  python -m scripts.build_kb --max-chunks 5   # 演示中断:只向量化 5 块即停
  python -m scripts.build_kb --resume-only    # 中断后续跑:不动文档,只补 pending 块
"""
import argparse
from pathlib import Path

from app.chunking import chunk_markdown
from app.config import Settings
from app.db import make_engine, make_session_factory
from app.embedding import make_embedder
from app.kb import KnowledgeBaseStore, vectorize_pending
from app.vector_store import KnowledgeVectorStore

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "knowledge"
# 文档名 → 内容类型;未登记的文档按 manual 处理
DOC_TYPES = {"退货政策": "policy", "商品FAQ": "faq"}


def build(
    docs_dir: Path,
    kb: KnowledgeBaseStore,
    vectors,
    embedder,
    *,
    max_chunks: int | None = None,
    max_chars: int = 500,
    overlap_chars: int = 80,
    rebuild_docs: bool = True,
) -> dict:
    vectors.ensure_collection()
    total_chunks = len(kb.doc_chunk_ids_all())
    if rebuild_docs:
        docs = sorted(docs_dir.glob("*.md"))
        total_chunks = 0
        for path in docs:
            ctype = DOC_TYPES.get(path.stem, "manual")
            chunks = chunk_markdown(
                path.read_text(encoding="utf-8"),
                path.name,
                ctype,
                max_chars=max_chars,
                overlap_chars=overlap_chars,
            )
            old_ids, _ = kb.replace_doc_chunks(path.name, chunks)
            vectors.delete(old_ids)  # 旧向量清掉,新块统一走 pending 向量化
            total_chunks += len(chunks)
    vectorized = vectorize_pending(kb, vectors, embedder, max_chunks=max_chunks)
    return {"docs": len(list(docs_dir.glob("*.md"))), "chunks": total_chunks, "vectorized": vectorized}


def main() -> None:
    parser = argparse.ArgumentParser(description="知识库离线建库(幂等,可中断重跑)")
    parser.add_argument("--max-chunks", type=int, default=None, help="只向量化 N 块即停(演示中断)")
    parser.add_argument("--resume-only", action="store_true", help="跳过文档重建,只补齐 pending 块")
    args = parser.parse_args()

    settings = Settings()
    kb = KnowledgeBaseStore(make_session_factory(make_engine(settings)))
    vectors = KnowledgeVectorStore(settings.milvus_db_path, dim=settings.embedding_dim)
    embedder = make_embedder(settings)
    stats = build(
        DOCS_DIR, kb, vectors, embedder,
        max_chunks=args.max_chunks,
        max_chars=settings.chunk_max_chars,
        overlap_chars=settings.chunk_overlap_chars,
        rebuild_docs=not args.resume_only,
    )
    print(
        f"建库完成:文档 {stats['docs']} 份,chunk {stats['chunks']} 条,"
        f"本次向量化 {stats['vectorized']} 条,剩 pending {len(kb.pending_chunks())} 条"
    )


if __name__ == "__main__":
    main()
