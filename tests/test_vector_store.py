import pytest

from app.vector_store import KnowledgeVectorStore


@pytest.fixture
def store(tmp_path):
    s = KnowledgeVectorStore(str(tmp_path / "kb.db"), collection="knowledge", dim=8)
    yield s


V1 = [1.0, 0, 0, 0, 0, 0, 0, 0]
V2 = [0, 1.0, 0, 0, 0, 0, 0, 0]


def test_lazy_constructor_no_io(tmp_path):
    KnowledgeVectorStore(str(tmp_path / "lazy.db"), dim=8)
    assert not (tmp_path / "lazy.db").exists()  # 构造不得建文件(测试环境不触碰真实路径)


def test_ensure_upsert_search_roundtrip(store):
    store.ensure_collection()
    store.upsert([11, 22], [V1, V2])
    assert store.count() == 2
    hits = store.search(V1, top_k=2)
    assert hits[0][0] == 11 and hits[0][1] > 0.99  # 自身相似度≈1,score 语义为相似度
    assert hits[1][0] == 22
    assert hits[0][1] >= hits[1][1]  # 降序


def test_upsert_same_id_overwrites(store):
    store.ensure_collection()
    store.upsert([7], [V1])
    store.upsert([7], [V2])
    assert store.count() == 1  # 主键幂等:覆盖不重复
    assert store.search(V2, top_k=1)[0][0] == 7


def test_search_missing_collection_returns_empty(store):
    assert store.search(V1, top_k=3) == []


def test_delete(store):
    store.ensure_collection()
    store.upsert([1, 2], [V1, V2])
    store.delete([1])
    assert store.count() == 1
    store.delete([999])  # 删不存在的 id 不抛错
    assert store.count() == 1


def test_ensure_collection_idempotent(store):
    store.ensure_collection()
    store.ensure_collection()  # 二次调用不抛错
