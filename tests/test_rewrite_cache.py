"""改写冻结缓存：原子写、哈希/模型校验、增量复用、refresh、恒等模式。"""
import json
from pathlib import Path

import pytest

from app.rewrite import QueryRewrite
from scripts.rewrite_cache import (
    RewriteCacheError,
    identity_rewrites,
    load_rewrites,
)

CASES = [{"id": "A01", "query": "邮费多少"}, {"id": "A02", "query": "几天到"}]


class FakeRewriter:
    def __init__(self, mapping=None):
        self.mapping = mapping or {}
        self.calls = []

    def rewrite(self, query):
        self.calls.append(query)
        normalized, synonyms = self.mapping.get(query, (query + " 改写", ["同义词"]))
        return QueryRewrite(normalized=normalized, synonyms=list(synonyms))


def _eval_file(tmp_path: Path, cases=CASES) -> Path:
    path = tmp_path / "eval.jsonl"
    path.write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n",
        encoding="utf-8",
    )
    return path


def test_write_then_read_roundtrip(tmp_path):
    cache = tmp_path / "cache.json"
    eval_path = _eval_file(tmp_path)
    writer = FakeRewriter()
    rewrites = load_rewrites(CASES, cache, eval_path, "m", writer, mode="write")
    assert writer.calls == ["邮费多少", "几天到"]
    assert rewrites["A01"].normalized == "邮费多少 改写"
    assert rewrites["A01"].synonyms == ["同义词"]

    reader = FakeRewriter()
    loaded = load_rewrites(CASES, cache, eval_path, "m", reader, mode="read")
    assert reader.calls == []
    assert loaded["A02"].normalized == "几天到 改写"


def test_write_reuses_existing_entries(tmp_path):
    cache = tmp_path / "cache.json"
    eval_path = _eval_file(tmp_path)
    load_rewrites(CASES, cache, eval_path, "m", FakeRewriter(), mode="write")
    second = FakeRewriter()
    load_rewrites(CASES, cache, eval_path, "m", second, mode="write")
    assert second.calls == []


def test_write_does_not_touch_unchanged_file(tmp_path):
    cache = tmp_path / "cache.json"
    eval_path = _eval_file(tmp_path)
    load_rewrites(CASES, cache, eval_path, "m", FakeRewriter(), mode="write")
    before = cache.read_bytes()
    load_rewrites(CASES, cache, eval_path, "m", FakeRewriter(), mode="write")
    assert cache.read_bytes() == before


def test_read_rejects_hash_mismatch(tmp_path):
    cache = tmp_path / "cache.json"
    eval_path = _eval_file(tmp_path)
    load_rewrites(CASES, cache, eval_path, "m", FakeRewriter(), mode="write")
    _eval_file(tmp_path, CASES + [{"id": "A03", "query": "新题"}])
    with pytest.raises(RewriteCacheError, match="评估集已变化"):
        load_rewrites(CASES, cache, eval_path, "m", FakeRewriter(), mode="read")


def test_read_rejects_model_mismatch(tmp_path):
    cache = tmp_path / "cache.json"
    eval_path = _eval_file(tmp_path)
    load_rewrites(CASES, cache, eval_path, "m1", FakeRewriter(), mode="write")
    with pytest.raises(RewriteCacheError, match="模型不匹配"):
        load_rewrites(CASES, cache, eval_path, "m2", FakeRewriter(), mode="read")


def test_read_rejects_missing_entry(tmp_path):
    cache = tmp_path / "cache.json"
    eval_path = _eval_file(tmp_path)
    load_rewrites(CASES, cache, eval_path, "m", FakeRewriter(), mode="write")
    payload = json.loads(cache.read_text(encoding="utf-8"))
    del payload["entries"]["A02"]
    cache.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(RewriteCacheError, match="缺少条目"):
        load_rewrites(CASES, cache, eval_path, "m", FakeRewriter(), mode="read")


def test_refresh_regenerates_all(tmp_path):
    cache = tmp_path / "cache.json"
    eval_path = _eval_file(tmp_path)
    load_rewrites(CASES, cache, eval_path, "m", FakeRewriter(), mode="write")
    refresher = FakeRewriter({"邮费多少": ("新归一", [])})
    loaded = load_rewrites(CASES, cache, eval_path, "m", refresher, mode="refresh")
    assert refresher.calls == ["邮费多少", "几天到"]
    assert loaded["A01"].normalized == "新归一"


def test_atomic_write_leaves_no_tmp_file(tmp_path):
    cache = tmp_path / "cache.json"
    eval_path = _eval_file(tmp_path)
    load_rewrites(CASES, cache, eval_path, "m", FakeRewriter(), mode="write")
    assert not (tmp_path / "cache.json.tmp").exists()


def test_identity_rewrites():
    rewrites = identity_rewrites(CASES)
    assert rewrites["A01"].normalized == "邮费多少"
    assert rewrites["A01"].synonyms == []
