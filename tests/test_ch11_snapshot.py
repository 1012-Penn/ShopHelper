"""快照五项校验:build_id、pending、id 集合、逐 chunk 文本/类别、文件复制。"""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from app.kb import vector_text
from scripts.eval_snapshot import (
    SnapshotError,
    acquire_eval_lock,
    compute_build_id,
    copy_snapshot,
    verify_consistency,
)


class FakeKB:
    def __init__(self, rows, pending=None):
        self.rows = {row.id: row for row in rows}
        self.pending = list(pending or [])

    def doc_chunk_ids_all(self):
        return list(self.rows)

    def get_chunks(self, ids):
        return [self.rows[cid] for cid in ids if cid in self.rows]

    def pending_chunks(self):
        return self.pending


class FakeVectors:
    def __init__(self, rows):
        self.rows = rows

    def all_rows(self):
        return self.rows


def _row(chunk_id=1, category="售后政策", questions="邮费", answer="普通订单邮费 8 元"):
    return SimpleNamespace(id=chunk_id, category=category, questions=questions, answer=answer)


def _vector_row(row, *, text=None, category=None):
    return {
        "id": row.id,
        "text": text if text is not None else vector_text(row.category, row.questions, row.answer),
        "category": category if category is not None else row.category,
    }


def _verify(kb, vectors):
    return verify_consistency(
        kb, vectors, corpus_sha256="corpus", embedding_model="m", dim=8
    )


def test_happy_path_build_id_stable():
    row = _row()
    report = _verify(FakeKB([row]), FakeVectors([_vector_row(row)]))
    assert report.ok
    assert report.pending == 0
    assert report.build_id == compute_build_id(
        [row], corpus_sha256="corpus", embedding_model="m", dim=8,
        max_chars=500, overlap_chars=80,
    )


def test_pending_chunks_fail_consistency():
    row = _row()
    report = _verify(FakeKB([row], pending=[row]), FakeVectors([_vector_row(row)]))
    assert not report.ok
    assert report.pending == 1


def test_missing_and_extra_ids_fail():
    row = _row()
    report = _verify(FakeKB([row]), FakeVectors([_vector_row(_row(2))]))
    assert report.missing_ids == [1]
    assert report.extra_ids == [2]
    assert not report.ok


def test_same_ids_but_text_or_category_mismatch_fails():
    row = _row()
    report = _verify(
        FakeKB([row]), FakeVectors([_vector_row(row, text="错文本", category="错类别")])
    )
    assert report.text_mismatch_ids == [1]
    assert report.category_mismatch_ids == [1]
    assert not report.ok


def test_eval_lock_is_exclusive(tmp_path):
    path = tmp_path / ".eval.lock"
    with acquire_eval_lock(path):
        with pytest.raises(SnapshotError):
            with acquire_eval_lock(path):
                pass


def test_snapshot_requires_writers_stopped(tmp_path):
    source = tmp_path / "milvus.db"
    source.write_bytes(b"db")
    with pytest.raises(SnapshotError):
        copy_snapshot(source, tmp_path / "snap.db", confirmed_writers_stopped=False)


def test_copy_snapshot_copies_main_and_siblings(tmp_path):
    source = tmp_path / "milvus.db"
    source.write_bytes(b"db")
    (tmp_path / "milvus.db.lock").write_bytes(b"lock")
    dest = tmp_path / "snap.db"
    report = copy_snapshot(source, dest, confirmed_writers_stopped=True)
    assert dest.read_bytes() == b"db"
    assert (tmp_path / "snap.db.lock").read_bytes() == b"lock"
    assert any(item["file"] == "snap.db" for item in report["files"])
    assert report["files"][0]["sha256"] == hashlib.sha256(b"db").hexdigest()


def test_runtime_chunks_are_excluded_from_doc_snapshot():
    runtime = _row(99)
    runtime.section_path = "review_queue:4"
    doc = _row(1)
    doc.section_path = "退货政策.md > 邮费与运费"
    report = _verify(
        FakeKB([doc, runtime]),
        FakeVectors([_vector_row(doc)]),
    )
    assert report.ok
    assert report.excluded_runtime_ids == [99]


def test_snapshot_rejects_dest_equal_or_inside_source(tmp_path):
    source = tmp_path / "milvus.db"
    source.mkdir()
    (source / "file").write_bytes(b"x")
    for dest in (source, tmp_path, source / "nested.db"):
        with pytest.raises(SnapshotError):
            copy_snapshot(source, dest, confirmed_writers_stopped=True)
