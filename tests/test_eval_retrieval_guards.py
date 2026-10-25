"""评估脚本的一致性守卫与指标聚合测试（不连真库）。"""
from dataclasses import dataclass

from app.retrieval import PreparedQuery
from app.rewrite import QueryRewrite
from scripts.eval_retrieval import (
    _empty_bucket,
    _identity,
    _load_and_check,
    _metrics,
    _prepared_from_rewrite,
    _total,
)


@dataclass(frozen=True)
class Row:
    id: int
    section_path: str = "a.md > a > s"
    questions: str = "s"
    answer: str = "x"


class StubKB:
    def __init__(self, rows, pending=None):
        self.rows = rows
        self._pending = pending or []

    def doc_chunk_ids_all(self):
        return [r.id for r in self.rows]

    def get_chunks(self, ids):
        return [r for r in self.rows if r.id in ids]

    def pending_chunks(self):
        return self._pending


class StubVectors:
    def __init__(self, ids):
        self.ids = ids

    def all_ids(self):
        return list(self.ids)


def test_guard_passes_when_ids_match():
    kb = StubKB([Row(1), Row(2)])
    code, message, all_rows = _load_and_check(kb, StubVectors([2, 1]))
    assert code == 0 and message == ""
    assert set(all_rows) == {1, 2}


def test_guard_rejects_empty_kb():
    code, message, _ = _load_and_check(StubKB([]), StubVectors([]))
    assert code == 2 and "为空" in message


def test_guard_rejects_pending_chunks():
    kb = StubKB([Row(1)], pending=[Row(1)])
    code, message, _ = _load_and_check(kb, StubVectors([1]))
    assert code == 3 and "pending=1" in message


def test_guard_rejects_missing_and_extra_ids():
    kb = StubKB([Row(1), Row(2)])
    code, message, _ = _load_and_check(kb, StubVectors([2, 3]))
    assert code == 3
    assert "missing=[1]" in message and "extra=[3]" in message


def test_total_and_metrics():
    bucket = _empty_bucket()
    bucket.update({"n": 2, "r3": 1, "r5": 2, "r10": 2, "mrr": 1.5,
                   "single1": 1, "single3": 2, "group_recall": 1.5,
                   "degraded": 1, "low_conf": 0, "filter_fallback": 0})
    total = _total({"A": bucket})
    assert total == bucket
    m = _metrics(total)
    assert m["r3"] == 0.5 and m["r5"] == 1.0 and m["r10"] == 1.0
    assert m["mrr"] == 0.75 and m["single1"] == 0.5 and m["single3"] == 1.0
    assert m["group_recall"] == 0.75 and m["degraded"] == 1 and m["n"] == 2


def test_prepared_from_rewrite_builds_dense_and_bm25_text():
    prepared = _prepared_from_rewrite("原问法", QueryRewrite(normalized="新问法",
                                                             synonyms=["同义词"]))
    assert prepared == PreparedQuery(rewritten="新问法", search_text="新问法 同义词")


def test_identity_prepared_query():
    prepared = _identity([{"id": "A", "query": "原问法"}])
    assert prepared["A"] == PreparedQuery(rewritten="原问法", search_text="原问法")
