"""Milvus v2 集合封装:真 milvus-lite tmp 文件实测(BM25 函数/jieba/hybrid/expr/旧库重建)。"""
import pytest
from pymilvus import DataType, MilvusClient

from app.vector_store import KnowledgeVectorStore

DIM = 8
V_HEAR = [0.9, 0.1, 0, 0, 0, 0, 0, 0]
V_POLICY = [0.1, 0.9, 0, 0, 0, 0, 0, 0]

ROWS = [
    {"id": 1, "vector": V_HEAR, "text": "无线蓝牙降噪耳机 SH-E300 黑色 支持主动降噪",
     "category": "数码配件"},
    {"id": 2, "vector": V_POLICY, "text": "退货政策 七天无理由 邮费八元 满九十九包邮",
     "category": "售后政策"},
    {"id": 3, "vector": V_HEAR, "text": "机械键盘 SH-K870 青轴 87 键",
     "category": "数码配件"},
]


@pytest.fixture
def store(tmp_path):
    s = KnowledgeVectorStore(str(tmp_path / "v2.db"), dim=DIM)
    s.upsert(ROWS)
    return s


def test_lazy_constructor_no_io(tmp_path):
    KnowledgeVectorStore(str(tmp_path / "lazy.db"), dim=DIM)
    assert not (tmp_path / "lazy.db").exists()  # 构造不得建文件(测试环境不触碰真实路径)


def test_upsert_idempotent_and_count(store):
    store.upsert(ROWS)  # 同主键覆盖
    assert store.count() == 3


def test_bm25_hits_model_number(store):
    hits = store.bm25_search("SH-E300", top_k=3)
    assert hits[0][0] == 1


def test_bm25_expr_filter(store):
    hits = store.bm25_search("退货 邮费", top_k=3, filter_expr='category == "售后政策"')
    assert {h[0] for h in hits} == {2}


def test_dense_search_cosine_order(store):
    hits = store.dense_search(V_HEAR, top_k=3)
    assert hits[0][0] in (1, 3)  # 同向量簇在前
    assert hits[0][1] > hits[-1][1]


def test_dense_search_expr_filter(store):
    hits = store.dense_search(V_HEAR, top_k=3, filter_expr='category == "售后政策"')
    assert {h[0] for h in hits} == {2}


def test_hybrid_search_fuses_both(store):
    hits = store.hybrid_search(V_HEAR, "耳机 SH-E300", top_k=3)
    assert len(hits) == 3 and hits[0][0] == 1  # dense、bm25 双路都提名 id=1,RRF 后居首


def test_hybrid_search_expr_filter_both_legs(store):
    hits = store.hybrid_search(V_HEAR, "退货 邮费", top_k=3, filter_expr='category == "售后政策"')
    assert {h[0] for h in hits} <= {2}


def test_legacy_collection_recreated(tmp_path):
    """ch03 旧 schema(id+vector)无 BM25 字段 → ensure_collection 自动 drop 重建。"""
    path = str(tmp_path / "legacy.db")
    client = MilvusClient(uri=path)
    client.create_collection(collection_name="knowledge", dimension=DIM,
                             primary_field_name="id", id_type=DataType.INT64,
                             vector_field_name="vector", metric_type="COSINE", auto_id=False)
    store = KnowledgeVectorStore(path, dim=DIM)
    store.ensure_collection()
    names = {f["name"] for f in store._get_client().describe_collection("knowledge")["fields"]}
    assert {"text", "sparse", "category"} <= names


def test_search_legacy_alias(store):
    assert store.search(V_HEAR, 3)[0][0] in (1, 3)


def test_delete(store):
    store.delete([1])
    assert store.count() == 2
    store.delete([999])  # 删不存在的 id 不抛错
    assert store.count() == 2


def test_count_missing_collection(tmp_path):
    assert KnowledgeVectorStore(str(tmp_path / "none.db"), dim=DIM).count() == 0
