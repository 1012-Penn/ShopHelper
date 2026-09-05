"""Milvus 集合封装(Milvus Lite 文件库):只存 id(= MySQL chunk 主键)+ vector,元数据一律在 MySQL。

Context7 核对(pymilvus 3.0.1):
- create_collection 快捷参数即支持自定主键:primary_field_name="id", id_type="int", auto_id=False
- upsert(collection, data=[{"id":…, "vector":…}]) 按主键覆盖,幂等
- search(collection, data=[vec], limit=k) → 外层每查询一个列表,hit["id"]/hit["distance"]
- COSINE 度量下 hit["distance"] 实测即相似度(同向量 1.0、正交 0.0,越大越像;pymilvus 3.0.1 + milvus-lite 3.2.1 真库校准)
- get_collection_stats(collection)["row_count"] 计数
"""
from pymilvus import DataType, MilvusClient


class KnowledgeVectorStore:
    def __init__(self, db_path: str, collection: str = "knowledge", dim: int = 1024) -> None:
        self._db_path = db_path
        self._collection = collection
        self._dim = dim
        self._client: MilvusClient | None = None  # 惰性:构造不得产生文件 I/O

    def _get_client(self) -> MilvusClient:
        if self._client is None:
            self._client = MilvusClient(uri=self._db_path)
        return self._client

    def _exists(self) -> bool:
        return self._get_client().has_collection(self._collection)

    def ensure_collection(self) -> None:
        if self._exists():
            return
        self._get_client().create_collection(
            collection_name=self._collection,
            dimension=self._dim,
            primary_field_name="id",
            id_type=DataType.INT64,
            vector_field_name="vector",
            metric_type="COSINE",
            auto_id=False,
        )

    def upsert(self, ids: list[int], vectors: list[list[float]]) -> None:
        self.ensure_collection()
        data = [{"id": i, "vector": v} for i, v in zip(ids, vectors)]
        self._get_client().upsert(self._collection, data)

    def search(self, vector: list[float], top_k: int) -> list[tuple[int, float]]:
        """→ [(chunk_id, 相似度)] 相似度降序;集合不存在返回空。"""
        if not self._exists():
            return []
        results = self._get_client().search(
            self._collection, data=[vector], limit=top_k, output_fields=["id"]
        )
        return [(int(hit["id"]), float(hit["distance"])) for hit in results[0]]

    def delete(self, ids: list[int]) -> None:
        if not ids or not self._exists():
            return
        self._get_client().delete(self._collection, ids=ids)

    def count(self) -> int:
        if not self._exists():
            return 0
        stats = self._get_client().get_collection_stats(self._collection)
        return int(stats.get("row_count", 0))
