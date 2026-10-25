"""Milvus Lite 评测快照:停写后复制,重开并做五项一致性校验。

五项:build_id、pending=0、MySQL/Milvus id 集合相等、逐 chunk 文本/类别相等、
快照文件 SHA-256。只比计数或文件哈希不足以证明内容一致。
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

from app.kb import vector_text

ROOT = Path(__file__).resolve().parent.parent
LOCK_PATH = ROOT / "data" / ".eval.lock"


class SnapshotError(RuntimeError):
    """快照创建/校验失败。"""


RUNTIME_SECTION_PREFIXES = ("review_queue:", "mined:")


@dataclass(frozen=True)
class ConsistencyReport:
    ok: bool
    mysql_chunks: int
    milvus_chunks: int
    pending: int
    missing_ids: list[int]
    extra_ids: list[int]
    text_mismatch_ids: list[int]
    category_mismatch_ids: list[int]
    excluded_runtime_ids: list[int]
    build_id: str
    corpus_sha256: str
    chunk_set_sha256: str

    def as_dict(self) -> dict:
        return asdict(self)


def is_runtime_chunk(row) -> bool:
    section = str(getattr(row, "section_path", "") or "")
    return section.startswith(RUNTIME_SECTION_PREFIXES)


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(Path(path).read_bytes())


def tree_sha256(path: Path) -> str:
    """目录树或单文件的稳定 SHA:按相对路径排序后逐文件哈希。"""
    path = Path(path)
    if path.is_file():
        return _sha_file(path)
    if not path.exists():
        return ""
    parts = []
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        if child.name == "LOCK":
            continue
        parts.append(f"{child.relative_to(path)}\0{_sha_file(child)}")
    return _sha_bytes("\n".join(parts).encode("utf-8"))


def chunk_set_sha256(rows) -> str:
    pairs = sorted((int(row.id), vector_text(row.category, row.questions, row.answer))
                   for row in rows)
    payload = "\n".join(f"{chunk_id}\0{_sha_bytes(text.encode('utf-8'))}" for chunk_id, text in pairs)
    return _sha_bytes(payload.encode("utf-8"))


def compute_build_id(
    rows,
    *,
    corpus_sha256: str,
    embedding_model: str,
    dim: int,
    max_chars: int,
    overlap_chars: int,
) -> str:
    seed = "\0".join([
        corpus_sha256,
        embedding_model,
        str(dim),
        str(max_chars),
        str(overlap_chars),
        chunk_set_sha256(rows),
    ])
    return _sha_bytes(seed.encode("utf-8"))


def verify_consistency(
    kb,
    vectors,
    *,
    corpus_sha256: str,
    embedding_model: str,
    dim: int,
    max_chars: int = 500,
    overlap_chars: int = 80,
) -> ConsistencyReport:
    all_rows = kb.get_chunks(kb.doc_chunk_ids_all())
    rows = [row for row in all_rows if not is_runtime_chunk(row)]
    excluded = [int(row.id) for row in all_rows if is_runtime_chunk(row)]
    mysql_rows = {int(row.id): row for row in rows}
    runtime_ids = set(excluded)
    milvus_rows = {
        int(row["id"]): row
        for row in vectors.all_rows()
        if int(row["id"]) not in runtime_ids
    }
    pending_rows = [
        row for row in kb.pending_chunks() if not is_runtime_chunk(row)
    ]
    pending = len(pending_rows)
    mysql_ids, milvus_ids = set(mysql_rows), set(milvus_rows)
    missing = sorted(mysql_ids - milvus_ids)
    extra = sorted(milvus_ids - mysql_ids)
    text_mismatch: list[int] = []
    category_mismatch: list[int] = []
    for chunk_id in sorted(mysql_ids & milvus_ids):
        row = mysql_rows[chunk_id]
        expected = vector_text(row.category, row.questions, row.answer)
        actual = str(milvus_rows[chunk_id].get("text") or "")
        if expected != actual:
            text_mismatch.append(chunk_id)
        if str(row.category) != str(milvus_rows[chunk_id].get("category") or ""):
            category_mismatch.append(chunk_id)
    build_id = compute_build_id(
        rows,
        corpus_sha256=corpus_sha256,
        embedding_model=embedding_model,
        dim=dim,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
    )
    ok = not (pending or missing or extra or text_mismatch or category_mismatch)
    return ConsistencyReport(
        ok=ok,
        mysql_chunks=len(mysql_rows),
        milvus_chunks=len(milvus_rows),
        pending=pending,
        missing_ids=missing,
        extra_ids=extra,
        text_mismatch_ids=text_mismatch,
        category_mismatch_ids=category_mismatch,
        excluded_runtime_ids=excluded,
        build_id=build_id,
        corpus_sha256=corpus_sha256,
        chunk_set_sha256=chunk_set_sha256(rows),
    )


@contextmanager
def acquire_eval_lock(path: Path = LOCK_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SnapshotError(f"已有评测进程持有锁: {path}") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()))
        handle.flush()
        yield path
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def copy_snapshot(
    source_db: Path, dest_db: Path, *, confirmed_writers_stopped: bool
) -> dict:
    if not confirmed_writers_stopped:
        raise SnapshotError(
            "拒绝复制:必须先停止 uvicorn/建库等写入进程,并显式确认 --confirmed-writers-stopped"
        )
    source_db, dest_db = Path(source_db).resolve(), Path(dest_db).resolve()
    if not source_db.exists():
        raise SnapshotError(f"源 Milvus 文件不存在: {source_db}")
    if dest_db == source_db:
        raise SnapshotError("快照目标不能等于源库")
    if dest_db in source_db.parents:
        raise SnapshotError(f"快照目标不能是源库的父目录: {dest_db}")
    if source_db in dest_db.parents:
        raise SnapshotError(f"快照目标不能位于源库目录内: {dest_db}")
    dest_db.parent.mkdir(parents=True, exist_ok=True)
    if source_db.is_dir():
        # Milvus Lite 3.x 使用目录树(databases/collections/wal);不能按单文件复制。
        if dest_db.exists():
            shutil.rmtree(dest_db)
        shutil.copytree(source_db, dest_db, ignore=shutil.ignore_patterns("LOCK"))
        copied = []
        for path in sorted(dest_db.rglob("*")):
            if path.is_file():
                copied.append({
                    "file": str(path.relative_to(dest_db)),
                    "sha256": _sha_file(path),
                    "bytes": path.stat().st_size,
                })
        return {"source": str(source_db), "dest": str(dest_db), "files": copied}

    copied = []
    for path in sorted(source_db.parent.glob(source_db.name + "*")):
        if path.resolve() == dest_db.resolve():
            continue
        if path == source_db:
            target = dest_db
        else:
            suffix = path.name[len(source_db.name):]
            target = dest_db.with_name(dest_db.name + suffix)
        if path.resolve() == target.resolve():
            continue
        shutil.copy2(path, target)
        with target.open("rb+") as fh:
            os.fsync(fh.fileno())
        copied.append({"file": target.name, "sha256": _sha_file(target), "bytes": target.stat().st_size})
    return {"source": str(source_db), "dest": str(dest_db), "files": copied}


def verify_snapshot(
    snapshot_db: Path,
    kb,
    *,
    corpus_sha256: str,
    embedding_model: str,
    dim: int,
) -> ConsistencyReport:
    from app.vector_store import KnowledgeVectorStore

    vectors = KnowledgeVectorStore(str(snapshot_db), dim=dim)
    return verify_consistency(
        kb,
        vectors,
        corpus_sha256=corpus_sha256,
        embedding_model=embedding_model,
        dim=dim,
    )


def main(argv: list[str] | None = None) -> int:
    from app.config import Settings
    from app.db import make_engine, make_session_factory
    from app.kb import KnowledgeBaseStore
    from app.vector_store import KnowledgeVectorStore
    from scripts.corpus_manifest import DOCS_DIR, corpus_sha256, load_manifest

    parser = argparse.ArgumentParser(description="Milvus Lite 评测快照")
    parser.add_argument("--copy", action="store_true")
    parser.add_argument("--source", type=Path, default=Path("data/milvus_knowledge.db"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--confirmed-writers-stopped", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args(argv)

    if args.copy:
        if args.out is None:
            print("--copy 需要 --out", file=sys.stderr)
            return 2
        with acquire_eval_lock():
            report = copy_snapshot(
                args.source, args.out, confirmed_writers_stopped=args.confirmed_writers_stopped
            )
        report["sha256"] = {item["file"]: item["sha256"] for item in report["files"]}
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.verify_only:
        settings = Settings()
        factory = make_session_factory(make_engine(settings))
        kb = KnowledgeBaseStore(factory)
        manifest = load_manifest()
        corpus = corpus_sha256(DOCS_DIR, manifest)
        target = args.snapshot or Path(settings.milvus_db_path)
        report = verify_snapshot(
            target, kb, corpus_sha256=corpus,
            embedding_model=settings.embedding_model, dim=settings.embedding_dim,
        )
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
        return 0 if report.ok else 1

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
