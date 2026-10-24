"""离线建库:knowledge/*.md → 结构感知切分 → MySQL(pending)→ embed → Milvus → done。

用法:
  python -m scripts.build_kb                       # 全量建库(文档重建 + 向量化,幂等)
  python -m scripts.build_kb --max-chunks 5        # 演示中断:只向量化 5 块即停
  python -m scripts.build_kb --resume-only         # 中断后续跑:不动文档,只补 pending 块
  python -m scripts.build_kb --dry-run             # 只切分并校验规模,不建 Settings/DB/API
  python -m scripts.build_kb --manifest-out [PATH] # 真实构建后写构建清单(默认 data/build_manifest.json)
"""
import argparse
import json
from pathlib import Path

from app.chunking import Chunk, chunk_markdown
from app.kb import KnowledgeBaseStore, vectorize_pending
from scripts.corpus_manifest import (
    DOCS_DIR,
    CorpusManifest,
    CorpusManifestError,
    corpus_sha256,
    doc_types,
    load_manifest,
    resolve_docs,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BUILD_MANIFEST = ROOT / "data" / "build_manifest.json"


def chunk_corpus(
    docs_dir: Path = DOCS_DIR,
    *,
    max_chars: int = 500,
    overlap_chars: int = 80,
    manifest: CorpusManifest | None = None,
) -> dict[str, list[Chunk]]:
    """按 manifest 登记顺序切分语料;未登记文档直接抛 ValueError。"""
    manifest = manifest or load_manifest()
    content_types = doc_types(manifest)
    return {
        path.name: chunk_markdown(
            path.read_text(encoding="utf-8"),
            path.name,
            content_types[path.name],
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        )
        for path in resolve_docs(docs_dir, manifest)
    }


def _validate_scale(docs: int, chunks: int, manifest: CorpusManifest) -> None:
    if manifest.expected_docs is not None and docs != manifest.expected_docs:
        raise CorpusManifestError(f"文档数 {docs} 与 manifest 期望 {manifest.expected_docs} 不符")
    if manifest.expected_chunks_min is not None and chunks < manifest.expected_chunks_min:
        raise CorpusManifestError(
            f"chunk 数 {chunks} 低于 manifest 下限 {manifest.expected_chunks_min}"
        )
    if manifest.expected_chunks_max is not None and chunks > manifest.expected_chunks_max:
        raise CorpusManifestError(
            f"chunk 数 {chunks} 高于 manifest 上限 {manifest.expected_chunks_max}"
        )


def dry_run(
    docs_dir: Path = DOCS_DIR,
    *,
    max_chars: int = 500,
    overlap_chars: int = 80,
    manifest: CorpusManifest | None = None,
) -> dict:
    """只切分并校验规模;不构造 Settings、数据库连接或任何 API 客户端。"""
    manifest = manifest or load_manifest()
    chunks_by_doc = chunk_corpus(
        docs_dir, max_chars=max_chars, overlap_chars=overlap_chars, manifest=manifest
    )
    counts = {name: len(chunks) for name, chunks in chunks_by_doc.items()}
    total = sum(counts.values())
    _validate_scale(len(counts), total, manifest)
    return {
        "docs": len(counts),
        "chunks": total,
        "doc_chunk_counts": counts,
        "corpus_sha256": corpus_sha256(docs_dir, manifest=manifest),
    }


def _write_build_manifest(
    path: Path,
    manifest: CorpusManifest,
    stats: dict,
    embedding_model: str,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "corpus_version": manifest.corpus_version,
        "corpus_sha256": stats["corpus_sha256"],
        "embedding_model": embedding_model,
        "docs": stats["docs"],
        "chunks": stats["chunks"],
        "vectorized": stats["vectorized"],
        "documents": [
            {
                "file": spec.file,
                "content_type": spec.content_type,
                "role": spec.role,
                "chunk_count": stats["doc_chunk_counts"].get(spec.file, 0),
            }
            for spec in manifest.docs
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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
    manifest: CorpusManifest | None = None,
    embedding_model: str = "",
    manifest_out: Path | None = None,
) -> dict:
    manifest = manifest or load_manifest()
    chunks_by_doc = chunk_corpus(
        docs_dir, max_chars=max_chars, overlap_chars=overlap_chars, manifest=manifest
    )
    doc_chunk_counts = {name: len(chunks) for name, chunks in chunks_by_doc.items()}

    vectors.ensure_collection()
    total_chunks = len(kb.doc_chunk_ids_all())
    if rebuild_docs:
        total_chunks = 0
        for doc_name, chunks in chunks_by_doc.items():
            old_ids, _ = kb.replace_doc_chunks(doc_name, chunks)
            vectors.delete(old_ids)  # 旧向量清掉,新块统一走 pending 向量化
            total_chunks += len(chunks)
    vectorized = vectorize_pending(kb, vectors, embedder, max_chunks=max_chunks)

    stats = {
        "docs": len(chunks_by_doc),
        "chunks": total_chunks,
        "vectorized": vectorized,
        "doc_chunk_counts": doc_chunk_counts,
        "corpus_sha256": corpus_sha256(docs_dir, manifest=manifest),
    }
    if manifest_out is not None:
        _write_build_manifest(manifest_out, manifest, stats, embedding_model)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="知识库离线建库(幂等,可中断重跑)")
    parser.add_argument("--max-chunks", type=int, default=None, help="只向量化 N 块即停(演示中断)")
    parser.add_argument("--resume-only", action="store_true", help="跳过文档重建,只补齐 pending 块")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只切分并校验 manifest 规模,不连 MySQL/Milvus、不构造 Settings",
    )
    parser.add_argument(
        "--manifest-out", type=Path, nargs="?", const=DEFAULT_BUILD_MANIFEST,
        default=DEFAULT_BUILD_MANIFEST, metavar="PATH",
        help="真实构建后写构建清单(默认 data/build_manifest.json);dry-run 不写",
    )
    args = parser.parse_args()

    manifest = load_manifest()
    if args.dry_run:
        stats = dry_run(manifest=manifest)
        counts = ",".join(f"{name}={count}" for name, count in stats["doc_chunk_counts"].items())
        print(f"dry-run 通过:文档 {stats['docs']} 份,chunk {stats['chunks']} 条({counts})")
        print(f"corpus_sha256: {stats['corpus_sha256']}")
        print(f"dry-run 不写 manifest(--manifest-out={args.manifest_out} 只对真实构建生效)")
        return

    # 真实构建才导入并构造 Settings/DB/嵌入客户端,dry-run 不碰这些依赖。
    from app.config import Settings
    from app.db import make_engine, make_session_factory
    from app.embedding import make_embedder
    from app.vector_store import KnowledgeVectorStore

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
        manifest=manifest,
        embedding_model=settings.embedding_model,
        manifest_out=args.manifest_out,
    )
    print(
        f"建库完成:文档 {stats['docs']} 份,chunk {stats['chunks']} 条,"
        f"本次向量化 {stats['vectorized']} 条,剩 pending {len(kb.pending_chunks())} 条"
    )
    if args.manifest_out is not None:
        print(f"构建清单已写入 {args.manifest_out}")


if __name__ == "__main__":
    main()
