"""Milvus 集合封装 v2(ch04):dense + 原生 BM25(BM25 Function 生成 sparse)+ category 过滤。

Context7 + 真库探针核对(pymilvus 3.0.1 / milvus-lite 3.2.1,2026-09-06):
- 中文分词:enable_analyzer + analyzer_params={"tokenizer": "jieba"}(文档的 {"type": "chinese"}
  在 Lite 不存在,报错明示 supported: standard/jieba;jieba 需 pip install jieba)
- sparse 由 BM25 Function 自动生成,upsert 数据只带 {id, vector, text, category}
- hybrid_search(collection, reqs, ranker=RRFRanker(), limit, output_fields)——关键字是 ranker
- MilvusClient.search 用 search_params 与 filter(ORM Collection 才是 param/expr);
  AnnSearchRequest 字段名是 param 与 expr
- BM25 打分为负值(实测同库 -2.6 附近):只按返回序取名次,阈值判断不得依赖其方向
- ch03 旧 schema(id+vector)缺 BM25 字段,ensure_collection 检测到即 drop 重建
  (数据权威源在 MySQL,重灌零成本)
"""
import os
from pathlib import Path

from pymilvus import AnnSearchRequest, DataType, Function, FunctionType, MilvusClient, RRFRanker

TEXT_MAX_LENGTH = 8192  # VARCHAR 按字节计;500 字 chunk 三格拼文本约 4.5KB,留余量
_V2_FIELDS = {"text", "category", "sparse"}


class KnowledgeVectorStore:
    def __init__(self, db_path: str, collection: str = "knowledge", dim: int = 1024) -> None:
        self._db_path = db_path
        self._collection = collection
        self._dim = dim
        self._client: MilvusClient | None = None  # 惰性:构造不得产生文件 I/O

    def _get_client(self) -> MilvusClient:
        if self._client is None:
            parent = Path(self._db_path).parent
            if str(parent) not in ("", "."):
                os.makedirs(parent, exist_ok=True)  # milvus-lite 要求父目录先存在
            self._client = MilvusClient(uri=self._db_path)
        return self._client

    def _exists(self) -> bool:
        return self._get_client().has_collection(self._collection)

    def _is_v2(self) -> bool:
        desc = self._get_client().describe_collection(self._collection)
        names = {f["name"] for f in desc.get("fields", [])}
        return _V2_FIELDS <= names

    def ensure_collection(self) -> None:
        if self._exists():
            if self._is_v2():
                return
            self._get_client().drop_collection(self._collection)
        schema = self._get_client().create_schema()
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=False)
        schema.add_field("text", DataType.VARCHAR, max_length=TEXT_MAX_LENGTH,
                         enable_analyzer=True, analyzer_params={"tokenizer": "jieba"})
        schema.add_field("category", DataType.VARCHAR, max_length=255)
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self._dim)
        schema.add_function(Function(
            name="text_bm25", input_field_names=["text"],
            output_field_names=["sparse"], function_type=FunctionType.BM25,
        ))
        index_params = self._get_client().prepare_index_params()
        index_params.add_index(field_name="vector", index_type="FLAT", metric_type="COSINE")
        index_params.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
        self._get_client().create_collection(
            collection_name=self._collection, schema=schema, index_params=index_params,
        )

    def upsert(self, rows: list[dict]) -> None:
        """rows: [{id, vector, text, category}];sparse 由 BM25 函数自动生成,禁止手工写。"""
        if not rows:
            return
        self.ensure_collection()
        data = [{"id": r["id"], "vector": r["vector"], "text": r["text"], "category": r["category"]}
                for r in rows]
        self._get_client().upsert(self._collection, data)

    def _loaded_client(self):
        if not self._exists():
            return None
        client = self._get_client()
        # 跨进程重开文件库时集合处于 released 态,search 前需 load(load 幂等)
        client.load_collection(self._collection)
        return client

    def dense_search(self, vector: list[float], top_k: int,
                     filter_expr: str | None = None) -> list[tuple[int, float]]:
        client = self._loaded_client()
        if client is None:
            return []
        results = client.search(self._collection, data=[vector], anns_field="vector",
                                search_params={"metric_type": "COSINE"}, limit=top_k,
                                filter=filter_expr or "", output_fields=["id"])
        return [(int(h["id"]), float(h["distance"])) for h in results[0]]

    def bm25_search(self, query_text: str, top_k: int,
                    filter_expr: str | None = None) -> list[tuple[int, float]]:
        client = self._loaded_client()
        if client is None:
            return []
        results = client.search(self._collection, data=[query_text], anns_field="sparse",
                                search_params={"metric_type": "BM25"}, limit=top_k,
                                filter=filter_expr or "", output_fields=["id"])
        return [(int(h["id"]), float(h["distance"])) for h in results[0]]

    def hybrid_search(self, vector: list[float], query_text: str, top_k: int,
                      filter_expr: str | None = None) -> list[tuple[int, float]]:
        """dense + BM25 各取 top_k,RRF 融合;返回 [(id, rrf 分)] 融合分降序。"""
        client = self._loaded_client()
        if client is None:
            return []
        dense_req = AnnSearchRequest(data=[vector], anns_field="vector",
                                     param={"metric_type": "COSINE"}, limit=top_k, expr=filter_expr)
        sparse_req = AnnSearchRequest(data=[query_text], anns_field="sparse",
                                      param={"metric_type": "BM25"}, limit=top_k, expr=filter_expr)
        results = client.hybrid_search(self._collection, reqs=[dense_req, sparse_req],
                                       ranker=RRFRanker(), limit=top_k, output_fields=["id"])
        return [(int(h["id"]), float(h["distance"])) for h in results[0]]

    def delete(self, ids: list[int]) -> None:
        if not ids or not self._exists():
            return
        self._get_client().delete(self._collection, ids=ids)

    def count(self) -> int:
        if not self._exists():
            return 0
        stats = self._get_client().get_collection_stats(self._collection)
        return int(stats.get("row_count", 0))
