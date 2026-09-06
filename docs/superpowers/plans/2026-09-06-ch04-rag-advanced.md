# ch04 RAG 进阶(混合检索+重排+评估体系)Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `query_faq` 从 dense 单路升级为 Milvus 原生 BM25 + dense 混合检索(RRF 融合)+ bge-reranker-v2-m3 精排,回答带可点引用编号、显式拒答落低置信度池,配套四策略评估体系(Recall@K / MRR / Faithfulness 分桶报告)与聊天页引用/反馈配套。

**Architecture:** 新建 `app/retrieval.py` 策略模块(dense / bm25 / hybrid / hybrid_rerank 四策略同一接口,线上与评估共用);Milvus 集合升 v2(id + vector + text(jieba BM25 源)+ sparse(BM25 Function 产出)+ category 过滤);`query_faq` v3 出参带引用编号与低置信标志;chat 路由发 citations 帧并落池;评估脚本 `eval_retrieval.py`(四策略指标)与 `eval_faithfulness.py`(生成 + LLM 裁判 + faith_cases 台账)。

**Tech Stack:** Milvus Lite 3.2.1(`pymilvus[milvus-lite]==3.0.1` + `jieba==0.42.1`)+ 硅基流动 `/v1/rerank`(`BAAI/bge-reranker-v2-m3`)+ 现有 FastAPI / SQLAlchemy 2.x / LangChain 1.x / DeepSeek。

**Spec:** `docs/superpowers/specs/2026-09-06-ch04-rag-advanced-design.md`

## Global Constraints

- 真库探针结论优先于文档(spec §2):analyzer 用 `{"tokenizer": "jieba"}`(文档的 `{"type": "chinese"}` 在 milvus-lite 不存在);`MilvusClient.hybrid_search` 关键字 `ranker`;`MilvusClient.search` 用 `search_params` 与 `filter`(AnnSearchRequest 内才是 `param`/`expr`);BM25 分数为负,只按返回序取名次,阈值不得依赖其方向。
- 新依赖:`jieba==0.42.1`(milvus-lite JiebaAnalyzer 运行时 import)。不引入 Langfuse。
- `query_faq` 契约本章演进(用户放行):入参 `keyword: str, category: str | None = None`;出参 `{"items": [{n, chunk_id, question, answer, category, section_path}], "low_confidence": bool, "reason": str, "filter_fallback": bool}`;任何异常收敛为 `{"items": [], "low_confidence": true, …}` 不抛错。
- 引用编号 n = 精排名次([1] 恒为最强证据);组装顺序用 `lost_in_middle_order`(奇数名次从头、偶数名次从尾)。
- `low_confidence_questions` 本章只写 `retrieval_low_conf` / `self_check` 两个入口,`user_feedback` 只留枚举;不去重。
- `faith_cases` 台账语义:一题一行(`uk_eval_id`);重复判出 `seen_count+1` 并刷新答案/理由/角标快照;已解决/无需解决再判出 → 退回「未解决」、清 `resolution`、保留 `resolved_at`(复发标记)。
- 拒答标记常量 `REFUSAL_MARKER = "【无法回答】"` 唯一来源 `app/guard.py`;prompts 与路由检测都引用它。
- query 改写只在检索侧(normalized 喂 dense,normalized+synonyms 喂 BM25);不做指代消解、多轮改写。
- API key 只进 `.env`;`.env.example` 只有占位符。测试不依赖网络/真上游:FakeEmbedding / FakeReranker / FakeRewriter / FakeVectorStore 或 tmp Milvus Lite 文件。
- milvus-lite 库文件单进程锁:真库测试用独立 tmp 路径,勿与 `data/milvus_knowledge.db` 冲突。
- 每任务完成:全量 `.venv/bin/pytest` 绿 → 追记 `dev-notes/ch04.md` 一段(四样:用户关键原话/关键产出/拒绝纠偏/翻车返工)→ git commit。
- 前端 `static/index.html` 是例外(Task 13):Vibe Coding,不套 TDD/code review。

---

### Task 1: 双表 DDL + ORM + config + .env.example

**Files:**
- Modify: `db/init.sql`(末尾追加两张表)
- Modify: `app/models.py`(追加两个 ORM 模型)
- Modify: `app/config.py`(追加 8 个配置项)
- Modify: `.env.example`(追加 RERANK_* 占位)
- Test: `tests/test_models.py`(追加)、`tests/test_config.py`(追加)

**Interfaces:**
- Produces: ORM `LowConfidenceQuestion` / `FaithCase`;`Settings.rerank_api_base / rerank_api_key / rerank_model / rerank_top_n / rerank_score_floor / hybrid_candidates / retrieval_final_top_k / query_rewrite_enabled`。Task 6/9/12 消费。

- [ ] **Step 1: db/init.sql 追加用户 DDL 原文**

在文件末尾追加(用户随需求提供的 DDL 一字不改,含注释;`SET NAMES utf8mb4` 已在文件头不重复):`low_confidence_questions`(id / conversation_id FK→conversations / raw_question / source ENUM('retrieval_low_conf','self_check','user_feedback') / reason / created_at,KEY idx_source、idx_created_at)与 `faith_cases`(eval_id UNIQUE / bucket / query / strategy DEFAULT 'hybrid_rerank' / answer / reason / citations JSON / judge_model / status ENUM('未解决','已解决','无需解决') / seen_count / first_seen_at / last_seen_at / resolution / resolved_at,KEY idx_status、idx_last_seen_at)。文件头注释补一行「ch04 新增:low_confidence_questions / faith_cases」。

- [ ] **Step 2: app/models.py 追加 ORM**

```python
class LowConfidenceQuestion(Base):
    __tablename__ = "low_confidence_questions"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    )
    conversation_id: Mapped[int | None] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), nullable=True
    )
    raw_question: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(Enum(
        "retrieval_low_conf", "self_check", "user_feedback", name="lcq_source"), nullable=False
    )
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class FaithCase(Base):
    __tablename__ = "faith_cases"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    )
    eval_id: Mapped[str] = mapped_column(String(16), nullable=False, unique=True)
    bucket: Mapped[str] = mapped_column(String(24), nullable=False)
    query: Mapped[str] = mapped_column(String(512), nullable=False)
    strategy: Mapped[str] = mapped_column(String(24), nullable=False, default="hybrid_rerank")
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[list | None] = mapped_column(JSON, nullable=True)
    judge_model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(Enum("未解决", "已解决", "无需解决", name="faith_status"), default="未解决")
    seen_count: Mapped[int] = mapped_column(Integer, default=1)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    resolution: Mapped[str | None] = mapped_column(String(300), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
```

- [ ] **Step 3: app/config.py 追加配置**

```python
    # ch04:混合检索 + 重排(rerank 与嵌入同走硅基流动;key 缺省回落 embedding_api_key)
    rerank_api_base: str = "https://api.siliconflow.cn/v1"
    rerank_api_key: str = ""
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_score_floor: float = 0.30   # 实现期(Task 11/14)拿评估集 rerank 分数分布校准
    hybrid_candidates: int = 50        # dense/BM25 各自召回数(hybrid_search limit)
    retrieval_final_top_k: int = 10    # 精排后进入 prompt 的条数
    query_rewrite_enabled: bool = True
```

- [ ] **Step 4: .env.example 追加**

```
# ch04 新增:bge-reranker-v2-m3 重排(硅基流动 /v1/rerank;留空则复用 EMBEDDING_API_KEY)
RERANK_API_BASE=https://api.siliconflow.cn/v1
RERANK_API_KEY=
RERANK_MODEL=BAAI/bge-reranker-v2-m3
```

- [ ] **Step 5: 写失败测试**

`tests/test_models.py` 追加:

```python
from app.models import FaithCase, LowConfidenceQuestion


def test_low_confidence_question_roundtrip(db_session_factory):
    with db_session_factory() as s:
        s.add(LowConfidenceQuestion(conversation_id=None, raw_question="量子力学怎么退货",
                                    source="self_check", reason="模型自评证据不足"))
        s.commit()
        row = s.query(LowConfidenceQuestion).one()
    assert row.source == "self_check"
    assert row.created_at is not None


def test_faith_case_roundtrip(db_session_factory):
    with db_session_factory() as s:
        s.add(FaithCase(eval_id="A01", bucket="A_policy", query="q", answer="a", reason="r",
                        citations=[{"n": 1, "chunk_id": 7}], judge_model="deepseek-chat"))
        s.commit()
        row = s.query(FaithCase).one()
    assert row.status == "未解决" and row.seen_count == 1 and row.strategy == "hybrid_rerank"
```

`tests/test_config.py` 追加:

```python
def test_ch04_defaults():
    s = Settings(openai_api_key="k", _env_file=None)
    assert s.rerank_model == "BAAI/bge-reranker-v2-m3"
    assert s.rerank_score_floor == 0.30
    assert s.hybrid_candidates == 50 and s.retrieval_final_top_k == 10
    assert s.query_rewrite_enabled is True
    assert s.rerank_api_key == ""
```

- [ ] **Step 6: 跑测试验证失败→实现→通过**

Run: `.venv/bin/pytest tests/test_models.py tests/test_config.py -v`
Expected: 先 FAIL(ImportError: cannot import name)→ 加 Step 2/3 后 PASS。

- [ ] **Step 7: 全量回归 + dev-notes 追记 + commit**

Run: `.venv/bin/pytest` → 全绿(基线 101 passed, 1 xfailed)。
Commit: `ch04: 双表 DDL/ORM/config——low_confidence_questions 与 faith_cases 台账`

### Task 2: Milvus 集合 v2 封装(dense + BM25 + hybrid + 过滤)

**Files:**
- Modify: `app/vector_store.py`(整体重写)
- Modify: `app/kb.py:163-181`(`vectorize_pending` 的 upsert 改带 text/category)
- Modify: `tests/helpers.py`(`FakeVectorStore` 升 v2 接口)
- Modify: `tests/test_vector_store.py`(重写)、`tests/test_kb.py:59`(upsert 调用形态)
- Modify: `pyproject.toml`(dependencies 加 `jieba==0.42.1`)
- Test: `tests/test_vector_store.py`

**Interfaces:**
- Consumes: 无(底层)。
- Produces: `KnowledgeVectorStore(db_path, collection="knowledge", dim=1024)`:
  - `ensure_collection()`(v2 schema;检测到旧 schema 自动 drop 重建)
  - `upsert(rows: list[dict])`,row = `{"id": int, "vector": list[float], "text": str, "category": str}`
  - `dense_search(vector, top_k, filter_expr=None) -> list[tuple[int, float]]`
  - `bm25_search(query_text, top_k, filter_expr=None) -> list[tuple[int, float]]`
  - `hybrid_search(vector, query_text, top_k, filter_expr=None) -> list[tuple[int, float]]`
  - `search(vector, top_k)`(ch03 兼容壳 = dense_search,Task 7 移除)、`delete(ids)`、`count()`
- `FakeVectorStore` 同步该接口(测试内 `expr` 只支持 `category == "X"` 形态)。

- [ ] **Step 1: 安装并声明 jieba**

```bash
.venv/bin/pip install jieba==0.42.1
```

`pyproject.toml` dependencies 加一行 `"jieba==0.42.1",`。

- [ ] **Step 2: 写失败测试(重写 tests/test_vector_store.py)**

```python
"""Milvus v2 集合封装:真 milvus-lite tmp 文件实测(BM25 函数/jieba/hybrid/expr/旧库重建)。"""
import tempfile
from pathlib import Path

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


def test_count_missing_collection(tmp_path):
    assert KnowledgeVectorStore(str(tmp_path / "none.db"), dim=DIM).count() == 0
```

- [ ] **Step 3: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_vector_store.py -v`
Expected: FAIL(`KnowledgeVectorStore` 无 `bm25_search` 等属性)。

- [ ] **Step 4: 实现 app/vector_store.py v2**

```python
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

    def dense_search(self, vector: list[float], top_k: int, filter_expr: str | None = None) -> list[tuple[int, float]]:
        client = self._loaded_client()
        if client is None:
            return []
        results = client.search(self._collection, data=[vector], anns_field="vector",
                                search_params={"metric_type": "COSINE"}, limit=top_k,
                                filter=filter_expr or "", output_fields=["id"])
        return [(int(h["id"]), float(h["distance"])) for h in results[0]]

    def bm25_search(self, query_text: str, top_k: int, filter_expr: str | None = None) -> list[tuple[int, float]]:
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

    def search(self, vector: list[float], top_k: int) -> list[tuple[int, float]]:
        """ch03 旧接口兼容壳(= dense 语义);query_faq 换 v3 后移除。"""
        return self.dense_search(vector, top_k)

    def delete(self, ids: list[int]) -> None:
        if not ids or not self._exists():
            return
        self._get_client().delete(self._collection, ids=ids)

    def count(self) -> int:
        if not self._exists():
            return 0
        stats = self._get_client().get_collection_stats(self._collection)
        return int(stats.get("row_count", 0))
```

- [ ] **Step 5: FakeVectorStore 升 v2 + kb.vectorize_pending 适配**

`tests/helpers.py` 的 `FakeVectorStore` 替换为(文件头部补 `import re`):

```python
class FakeVectorStore:
    """内存向量库,接口与 v2 KnowledgeVectorStore 一致;expr 仅支持 category == "X" 形态。"""

    def __init__(self, dim: int = 64):
        self.dim = dim
        self._rows: dict[int, dict] = {}  # id -> {vector, text, category}

    def ensure_collection(self) -> None:
        pass

    def upsert(self, rows):
        for r in rows:
            self._rows[r["id"]] = {"vector": r["vector"], "text": r["text"], "category": r["category"]}

    @staticmethod
    def _category_of(expr):
        if not expr:
            return None
        m = re.match(r'^category == "(.*)"$', expr.strip())
        return m.group(1) if m else None

    def _filtered(self, expr):
        cat = self._category_of(expr)
        if cat is None:
            return self._rows
        return {i: r for i, r in self._rows.items() if r["category"] == cat}

    def dense_search(self, vector, top_k, filter_expr=None):
        scored = [(i, cosine(vector, r["vector"])) for i, r in self._filtered(filter_expr).items()]
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    def bm25_search(self, query_text, top_k, filter_expr=None):
        toks = query_text.split()
        scored = []
        for i, r in self._filtered(filter_expr).items():
            hits = sum(r["text"].count(t) for t in toks)
            if hits:
                scored.append((i, -float(hits)))  # 仿真 BM25 负分:值越小越靠前
        scored.sort(key=lambda x: x[1])
        return scored[:top_k]

    def hybrid_search(self, vector, query_text, top_k, filter_expr=None):
        K = 60.0
        fused: dict[int, float] = {}
        for rank, (i, _s) in enumerate(self.dense_search(vector, top_k, filter_expr)):
            fused[i] = fused.get(i, 0.0) + 1.0 / (K + rank + 1)
        for rank, (i, _s) in enumerate(self.bm25_search(query_text, top_k, filter_expr)):
            fused[i] = fused.get(i, 0.0) + 1.0 / (K + rank + 1)
        return sorted(fused.items(), key=lambda x: -x[1])[:top_k]

    def search(self, vector, top_k):  # ch03 兼容壳,Task 7 随 query_faq 换核移除
        return self.dense_search(vector, top_k)

    def delete(self, ids):
        for i in ids:
            self._rows.pop(i, None)

    def count(self):
        return len(self._rows)
```

`app/kb.py` 的 `vectorize_pending` 内一行:

```python
        rows = kb.pending_chunks()[...][:limit]  # 原逻辑不动
        texts = [vector_text(r.category, r.questions, r.answer) for r in rows]
        vecs = embedder.embed(texts)
        vectors.upsert([
            {"id": r.id, "vector": v, "text": t, "category": r.category}
            for r, v, t in zip(rows, vecs, texts)
        ])
        kb.mark_vectorized([r.id for r in rows])
```

`tests/test_kb.py:59` 的 `vectors.upsert([999], [[0.1] * 64])` 改为
`vectors.upsert([{"id": 999, "vector": [0.1] * 64, "text": "干扰", "category": "x"}])`。

- [ ] **Step 6: 跑测试到绿 + 全量回归**

Run: `.venv/bin/pytest tests/test_vector_store.py tests/test_kb.py tests/test_build_kb.py -v` → PASS;
`.venv/bin/pytest` 全绿(test_store/test_chat_toolflow 等经 FakeVectorStore 兼容壳不破)。

- [ ] **Step 7: dev-notes 追记 + commit**

Commit: `ch04: Milvus 集合 v2——BM25 函数/jieba/hybrid RRF/expr 过滤 + 旧库自动重建`

### Task 3: Reranker 客户端(硅基流动 /v1/rerank)

**Files:**
- Create: `app/rerank.py`
- Modify: `tests/helpers.py`(追加 `FakeReranker` 与 `__all__`)
- Test: `tests/test_rerank.py`(新)

**Interfaces:**
- Produces: `SiliconFlowReranker(api_base, api_key, model, timeout=10.0, transport=None).rerank(query, documents, top_n=None) -> list[tuple[int, float]]`(按 relevance_score 降序,元素为候选下标与分数);`make_reranker(settings) -> SiliconFlowReranker | None`(key 为空返回 None,调用方降级);`FakeReranker()` 同签名。

- [ ] **Step 1: 写失败测试**

`tests/test_rerank.py`:

```python
"""Reranker 客户端:MockTransport 契约 + 分数降序 + 异常收敛 RuntimeError + FakeReranker。"""
import json

import httpx
import pytest

from app.rerank import SiliconFlowReranker
from tests.helpers import FakeReranker


def _client(handler):
    return SiliconFlowReranker("https://api.siliconflow.cn/v1", "sk-test", "BAAI/bge-reranker-v2-m3",
                               transport=httpx.MockTransport(handler))


def test_rerank_sorted_desc_and_indices():
    def handler(request):
        assert json.loads(request.content)["model"] == "BAAI/bge-reranker-v2-m3"
        body = {"results": [
            {"index": 1, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.2},
        ]}
        return httpx.Response(200, json=body)

    out = _client(handler).rerank("q", ["a", "b"], top_n=2)
    assert out == [(1, 0.9), (0, 0.2)]


def test_rerank_non_200_raises():
    def handler(request):
        return httpx.Response(500, text="boom")

    with pytest.raises(RuntimeError, match="500"):
        _client(handler).rerank("q", ["a"])


def test_rerank_bad_payload_raises():
    def handler(request):
        return httpx.Response(200, json={"unexpected": []})

    with pytest.raises(RuntimeError, match="格式"):
        _client(handler).rerank("q", ["a"])


def test_fake_reranker_deterministic_overlap():
    r = FakeReranker()
    out = r.rerank("降噪耳机", ["无线降噪耳机 WH", "退货政策 邮费", "耳机 主动降噪 功能"], top_n=2)
    assert len(out) == 2
    assert out[0][1] >= out[1][1]
    assert out[0][0] in (0, 2)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_rerank.py -v` → FAIL(No module named app.rerank)。

- [ ] **Step 3: 实现 app/rerank.py**

```python
"""bge-reranker-v2-m3 重排客户端:硅基流动 /v1/rerank,httpx 直连(2026-09-06 真接口实测)。

POST /rerank {model, query, documents, top_n, return_documents:false}
→ 200 {"results": [{index, relevance_score}, …]} 已按分数降序;客户端再校验形态并重排序防上游变更。
"""
import httpx

from app.config import Settings


class SiliconFlowReranker:
    def __init__(self, api_base: str, api_key: str, model: str,
                 timeout: float = 10.0, transport: httpx.BaseTransport | None = None) -> None:
        self._client = httpx.Client(
            base_url=api_base.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            transport=transport,
        )
        self._model = model

    def rerank(self, query: str, documents: list[str], top_n: int | None = None) -> list[tuple[int, float]]:
        """→ [(候选下标, relevance_score)] 分数降序;非 200 / 格式异常一律 RuntimeError。"""
        payload: dict = {"model": self._model, "query": query, "documents": documents,
                         "return_documents": False}
        if top_n is not None:
            payload["top_n"] = top_n
        resp = self._client.post("/rerank", json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"重排 API 调用失败:HTTP {resp.status_code} {resp.text[:200]}")
        try:
            results = resp.json()["results"]
            out = [(int(r["index"]), float(r["relevance_score"])) for r in results]
        except (ValueError, KeyError, TypeError) as exc:
            raise RuntimeError(f"重排 API 返回格式异常:{exc}") from exc
        out.sort(key=lambda x: -x[1])
        return out


def make_reranker(settings: Settings) -> SiliconFlowReranker | None:
    """key 缺省回落 embedding_api_key;两者皆空返回 None(调用方降级为不重排)。"""
    key = settings.rerank_api_key or settings.embedding_api_key
    if not key:
        return None
    return SiliconFlowReranker(settings.rerank_api_base, key, settings.rerank_model)
```

- [ ] **Step 4: tests/helpers.py 追加 FakeReranker 并入 __all__**

```python
class FakeReranker:
    """确定性假重排:按 query 字符在文档中的覆盖率打分(|query∩doc| / |query unique|),
    降序返回 (下标, 分)。按 query 口径而非 doc 口径,短问法命中关键词即可得高分。"""

    def rerank(self, query: str, documents: list[str], top_n: int | None = None) -> list[tuple[int, float]]:
        q = set(query)

        def score(doc: str) -> float:
            if not q:
                return 0.0
            return sum(1 for ch in q if ch in doc) / len(q)

        out = sorted(((i, score(doc)) for i, doc in enumerate(documents)), key=lambda x: -x[1])
        return out[:top_n] if top_n else out
```

`__all__` 追加 `"FakeReranker"`。

- [ ] **Step 5: 测试到绿 + 全量回归 + dev-notes + commit**

Run: `.venv/bin/pytest tests/test_rerank.py -v` → PASS;`.venv/bin/pytest` 全绿。
Commit: `ch04: bge-reranker-v2-m3 重排客户端——httpx 直连 /v1/rerank + FakeReranker`

### Task 4: Query 改写器(LLM 归一 + 同义词)

**Files:**
- Create: `app/rewrite.py`
- Modify: `tests/helpers.py`(追加 `FakeRewriter`)
- Test: `tests/test_rewrite.py`(新)

**Interfaces:**
- Produces: `QueryRewrite(BaseModel)`: `normalized: str`, `synonyms: list[str] = []`;`QueryRewriter(llm).rewrite(query) -> QueryRewrite`(任何异常/空归一 → 原句兜底,synonyms 截到 3);`make_rewriter(settings) -> QueryRewriter`;`bm25_query_text(qr) -> str`(normalized + synonyms 空格拼接)。`FakeRewriter(mapping)` 同 rewrite 签名,映射外恒等返回。

- [ ] **Step 1: 写失败测试**

`tests/test_rewrite.py`:

```python
"""Query 改写器:结构化归一 + 同义词;LLM 异常/空归一兜底原句。"""
import pytest

from app.rewrite import QueryRewrite, QueryRewriter, bm25_query_text
from tests.helpers import FakeRewriter


class _OkLLM:
    def invoke(self, prompt):
        assert "退了" in prompt  # prompt 里带原 query
        return {"normalized": "怎么申请退货", "synonyms": ["退货流程", "怎么退", "退换货"]}


class _BoomLLM:
    def invoke(self, prompt):
        raise RuntimeError("上游挂了")


class _EmptyLLM:
    def invoke(self, prompt):
        return QueryRewrite(normalized="  ", synonyms=[])


def test_rewrite_normalizes_and_expands():
    qr = QueryRewriter(_OkLLM()).rewrite("东西退了")
    assert qr.normalized == "怎么申请退货"
    assert bm25_query_text(qr) == "怎么申请退货 退货流程 怎么退 退换货"


def test_rewrite_boom_falls_back():
    qr = QueryRewriter(_BoomLLM()).rewrite("东西退了")
    assert qr.normalized == "东西退了" and qr.synonyms == []


def test_rewrite_blank_falls_back():
    qr = QueryRewriter(_EmptyLLM()).rewrite("邮费多少")
    assert qr.normalized == "邮费多少"


def test_synonyms_capped_at_three():
    llm = _OkLLM()
    llm.invoke = lambda p: {"normalized": "n", "synonyms": ["a", "b", "c", "d", "e"]}
    qr = QueryRewriter(llm).rewrite("q")
    assert qr.synonyms == ["a", "b", "c"]


def test_fake_rewriter_identity_default():
    assert FakeRewriter().rewrite("随便").normalized == "随便"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_rewrite.py -v` → FAIL(No module named app.rewrite)。

- [ ] **Step 3: 实现 app/rewrite.py**

```python
"""Query 理解(ch04):口语模糊问法 LLM 改写归一 + 检索侧同义词扩展。只在检索侧,不动入库侧。"""
from pydantic import BaseModel
from langchain_core.prompts import PromptTemplate

from app.config import Settings
from app.llm import make_extract_model

# 模板无花括号文案,只有 {query} 占位
REWRITE_PROMPT = PromptTemplate.from_template(
    "把用户的客服问法改写成知识库里的标准问法,不改意思,不新增假设;再给最多 3 个同义问法。\n"
    "用户问法:{query}"
)


class QueryRewrite(BaseModel):
    normalized: str
    synonyms: list[str] = []


class QueryRewriter:
    def __init__(self, llm) -> None:
        self._llm = llm  # with_structured_output(QueryRewrite) 的产物,测试可注替身

    def rewrite(self, query: str) -> QueryRewrite:
        """失败兜底原句:改写是增强,不是依赖——上游挂了检索照常走。"""
        try:
            result = self._llm.invoke(REWRITE_PROMPT.format(query=query))
            if isinstance(result, dict):
                result = QueryRewrite(**result)
            normalized = (result.normalized or "").strip()
            if not normalized:
                return QueryRewrite(normalized=query, synonyms=[])
            synonyms = [s.strip() for s in result.synonyms if s and s.strip()][:3]
            return QueryRewrite(normalized=normalized, synonyms=synonyms)
        except Exception:
            return QueryRewrite(normalized=query, synonyms=[])


def bm25_query_text(qr: QueryRewrite) -> str:
    """BM25 词面扩展:归一问法 + 同义词拼接(dense 只吃 normalized)。"""
    return " ".join([qr.normalized, *qr.synonyms])


def make_rewriter(settings: Settings) -> QueryRewriter:
    llm = make_extract_model(settings).with_structured_output(QueryRewrite, method="function_calling")
    return QueryRewriter(llm)
```

`tests/helpers.py` 追加(入 `__all__`):

```python
class FakeRewriter:
    """映射表命中返回预置归一,否则恒等;测试检索改写链路用。"""

    def __init__(self, mapping: dict[str, tuple[str, list[str]]] | None = None):
        self.mapping = mapping or {}

    def rewrite(self, query: str):
        from app.rewrite import QueryRewrite

        if query in self.mapping:
            n, syn = self.mapping[query]
            return QueryRewrite(normalized=n, synonyms=list(syn))
        return QueryRewrite(normalized=query, synonyms=[])
```

- [ ] **Step 4: 测试到绿 + 全量回归 + dev-notes + commit**

Run: `.venv/bin/pytest tests/test_rewrite.py -v` → PASS;`.venv/bin/pytest` 全绿。
Commit: `ch04: Query 改写器——LLM 归一 + 检索侧同义词扩展 + 失败兜底`

### Task 5: 检索策略模块(反漏斗 + 四策略)

**Files:**
- Create: `app/retrieval.py`
- Test: `tests/test_retrieval.py`(新)

**Interfaces:**
- Consumes: Task 2 `KnowledgeVectorStore` v2 接口、Task 3 rerank 签名、Task 4 `QueryRewrite`/`bm25_query_text`、`app.kb.KnowledgeBaseStore.get_chunks` 与 `vector_text`。
- Produces:
  - `STRATEGIES = ("dense", "bm25", "hybrid", "hybrid_rerank")`
  - `RETRIEVAL_SCORE_FLOOR = 0.5`(自 definitions.py 迁来,dense 相似度下限)
  - `@dataclass Retrieved(chunk_id: int, score: float)`
  - `@dataclass RetrievalResult(items, low_confidence=False, reason="", filter_fallback=False, rewritten="", search_text="")`
  - `lost_in_middle_order(seq) -> list`(奇数位次从头、偶数位次从尾;`[1..10]→[1,3,5,7,9,10,8,6,4,2]`)
  - `RetrievalService(embedder, vectors, kb, rewriter=None, reranker=None, *, candidates=50, final_top_k=10, dense_score_floor=0.5, rerank_score_floor=0.30).retrieve(query, strategy="hybrid_rerank", category=None, use_rewrite=True) -> RetrievalResult`

- [ ] **Step 1: 写失败测试**

`tests/test_retrieval.py`(真 milvus-lite tmp + FakeEmbedding;FakeEmbedding 词库无型号词 → 型号文本 dense 零向量,BM25 按词面命中——正是 B 桶场景的替身仿真):

```python
"""检索策略模块:四策略同一真库(milvus-lite tmp),反漏斗纯函数,低置信与过滤回退。"""
import tempfile

import pytest

from app.chunking import Chunk
from app.kb import KnowledgeBaseStore, vectorize_pending
from app.retrieval import RETRIEVAL_SCORE_FLOOR, RetrievalService, lost_in_middle_order
from tests.helpers import FakeEmbedding, FakeReranker, FakeRewriter, FakeVectorStore
from app.models import Base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

DOCS = [
    # (id, category, question, answer)
    (1, "售后政策", "退货政策是什么", "支持七天无理由退货,商品需保持完好。"),
    (2, "数码配件", "SH-E300 无线降噪耳机", "SH-E300 头戴式无线降噪耳机,售价 299 元,支持主动降噪。"),
    (3, "物流", "多久能发货", "现货商品 48 小时内发出。"),
]


@pytest.fixture
def service():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    kb = KnowledgeBaseStore(factory)
    old, new = kb.replace_doc_chunks("t.md", [
        Chunk(category=cat, questions=q, answer=a, section_path=f"t.md > {q}",
              content_type="faq", is_key_clause=False)
        for cid, cat, q, a in DOCS
    ])
    vectors = FakeVectorStore()
    vectorize_pending(kb, vectors, FakeEmbedding())
    return RetrievalService(FakeEmbedding(), vectors, kb,
                            rewriter=FakeRewriter({"东西坏了想退": ("质量问题怎么退货", ["质量问题退货"])}),
                            reranker=FakeReranker())


def test_lost_in_middle_order():
    assert lost_in_middle_order(list(range(1, 11))) == [1, 3, 5, 7, 9, 10, 8, 6, 4, 2]
    assert lost_in_middle_order([]) == [] and lost_in_middle_order([7]) == [7]


def test_dense_hits_semantic_but_misses_model(service):
    r = service.retrieve("退货的规定是什么", strategy="dense")
    assert r.items[0].chunk_id == 1 and not r.low_confidence
    # 型号文本对 FakeEmbedding 是零词向量 → 相似度低于 floor 全被滤掉
    r2 = service.retrieve("SH-E300 怎么样", strategy="dense")
    assert all(it.chunk_id != 2 for it in r2.items)


def test_bm25_hits_model_number(service):
    r = service.retrieve("SH-E300", strategy="bm25")
    assert r.items[0].chunk_id == 2 and not r.low_confidence


def test_hybrid_fuses(service):
    r = service.retrieve("SH-E300 降噪", strategy="hybrid")
    assert r.items[0].chunk_id == 2


def test_hybrid_rerank_ranks_model_first(service):
    r = service.retrieve("SH-E300 降噪耳机", strategy="hybrid_rerank")
    assert r.items[0].chunk_id == 2
    assert len(r.items) <= 10


def test_rewrite_applies(service):
    r = service.retrieve("东西坏了想退", strategy="bm25",
                         use_rewrite=True)
    assert r.rewritten == "质量问题怎么退货"
    assert "质量问题退货" in r.search_text


def test_empty_store_low_confidence(service):
    r = service.retrieve("量子力学", strategy="hybrid_rerank")
    assert r.low_confidence and r.reason


def test_category_filter_fallback(service):
    r = service.retrieve("退货的规定是什么", strategy="hybrid_rerank", category="不存在的品类")
    assert r.filter_fallback is True and r.items  # 过滤空 → 回退无过滤


def test_category_filter_honored(service):
    r = service.retrieve("退货的规定是什么", strategy="hybrid_rerank", category="售后政策")
    assert r.filter_fallback is False and all(it.chunk_id == 1 for it in r.items)


def test_unknown_strategy_raises(service):
    with pytest.raises(ValueError):
        service.retrieve("q", strategy="magic")


def test_dense_floor_constant():
    assert RETRIEVAL_SCORE_FLOOR == 0.5
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_retrieval.py -v` → FAIL(No module named app.retrieval)。

- [ ] **Step 3: 实现 app/retrieval.py**

```python
"""检索策略模块(ch04):dense / bm25 / hybrid / hybrid_rerank 四策略同一接口。

线上一律 hybrid_rerank;评估脚本逐策略调同一实现,四策略对比才是同一套代码。
依赖(embedder/vectors/kb/rewriter/reranker)全部构造注入,测试可替换身。
低置信判定:候选空(知识库无相关内容)或精排 Top-1 分低于 rerank_score_floor(证据置信度低)。
"""
import logging
from dataclasses import dataclass

from app.kb import KnowledgeBaseStore, vector_text
from app.rewrite import QueryRewrite, bm25_query_text

logger = logging.getLogger(__name__)

STRATEGIES = ("dense", "bm25", "hybrid", "hybrid_rerank")
RETRIEVAL_SCORE_FLOOR = 0.5  # dense 相似度下限(ch03 语义迁入)


@dataclass
class Retrieved:
    chunk_id: int
    score: float


@dataclass
class RetrievalResult:
    items: list[Retrieved]
    low_confidence: bool = False
    reason: str = ""
    filter_fallback: bool = False
    rewritten: str = ""
    search_text: str = ""


def lost_in_middle_order(seq: list) -> list:
    """反漏斗:位次 1,3,5… 从头排,2,4,6… 从尾排;最强证据在首、次强在尾。"""
    return seq[0::2] + seq[1::2][::-1]


class RetrievalService:
    def __init__(self, embedder, vectors, kb: KnowledgeBaseStore, rewriter=None, reranker=None, *,
                 candidates: int = 50, final_top_k: int = 10,
                 dense_score_floor: float = RETRIEVAL_SCORE_FLOOR,
                 rerank_score_floor: float = 0.30) -> None:
        self._embedder = embedder
        self._vectors = vectors
        self._kb = kb
        self._rewriter = rewriter    # None = 不改写
        self._reranker = reranker    # None = hybrid_rerank 降级 RRF 序
        self._candidates = candidates
        self._final_top_k = final_top_k
        self._dense_floor = dense_score_floor
        self._rerank_floor = rerank_score_floor

    def retrieve(self, query: str, strategy: str = "hybrid_rerank",
                 category: str | None = None, use_rewrite: bool = True) -> RetrievalResult:
        if strategy not in STRATEGIES:
            raise ValueError(f"未知检索策略:{strategy}")

        rewritten, search_text = query, query
        if use_rewrite and self._rewriter is not None:
            qr = self._rewriter.rewrite(query)
            rewritten = qr.normalized.strip() or query
            search_text = bm25_query_text(QueryRewrite(normalized=rewritten, synonyms=qr.synonyms))
        expr = f'category == "{category}"' if category else None

        if strategy == "dense":
            items = self._dense(rewritten, expr)
            fallback = not items and bool(expr)
            if fallback:
                items = self._dense(rewritten, None)
            return RetrievalResult(items, not items, "知识库无相关内容" if not items else "",
                                   fallback, rewritten, search_text)

        if strategy == "bm25":
            items = self._bm25(search_text, expr)
            fallback = not items and bool(expr)
            if fallback:
                items = self._bm25(search_text, None)
            return RetrievalResult(items, not items, "知识库无相关内容" if not items else "",
                                   fallback, rewritten, search_text)

        cand = self._hybrid(rewritten, search_text, expr)
        fallback = not cand and bool(expr)
        if fallback:
            cand = self._hybrid(rewritten, search_text, None)
        if not cand:
            return RetrievalResult([], True, "知识库无相关内容", fallback, rewritten, search_text)
        if strategy == "hybrid":
            return RetrievalResult(cand, False, "", fallback, rewritten, search_text)

        # hybrid_rerank:精排 Top-N;reranker 缺失降级 RRF 序(质量掉档,记 warning)
        if self._reranker is None:
            logger.warning("reranker 未配置,hybrid_rerank 降级为 RRF 序")
            return RetrievalResult(cand[: self._final_top_k], False, "", fallback, rewritten, search_text)
        texts = self._chunk_texts([r.chunk_id for r in cand])
        ranked = self._reranker.rerank(rewritten, [texts.get(r.chunk_id, "") for r in cand],
                                       top_n=self._final_top_k)
        items = [Retrieved(chunk_id=cand[idx].chunk_id, score=score) for idx, score in ranked]
        if not items:
            return RetrievalResult([], True, "知识库无相关内容", fallback, rewritten, search_text)
        low = items[0].score < self._rerank_floor
        reason = f"证据置信度低(最高 {items[0].score:.2f})" if low else ""
        return RetrievalResult(items, low, reason, fallback, rewritten, search_text)

    # ---- 单路封装 ----

    def _dense(self, query: str, expr: str | None) -> list[Retrieved]:
        qvec = self._embedder.embed([query])[0]
        return [Retrieved(cid, s)
                for cid, s in self._vectors.dense_search(qvec, self._candidates, filter_expr=expr)
                if s >= self._dense_floor]

    def _bm25(self, search_text: str, expr: str | None) -> list[Retrieved]:
        return [Retrieved(cid, s) for cid, s in
                self._vectors.bm25_search(search_text, self._candidates, filter_expr=expr)]

    def _hybrid(self, dense_query: str, search_text: str, expr: str | None) -> list[Retrieved]:
        qvec = self._embedder.embed([dense_query])[0]
        return [Retrieved(cid, s) for cid, s in
                self._vectors.hybrid_search(qvec, search_text, self._candidates, filter_expr=expr)]

    def _chunk_texts(self, ids: list[int]) -> dict[int, str]:
        return {r.id: vector_text(r.category, r.questions, r.answer)
                for r in self._kb.get_chunks(ids)}
```

- [ ] **Step 4: 测试到绿 + 全量回归 + dev-notes + commit**

Run: `.venv/bin/pytest tests/test_retrieval.py -v` → PASS;`.venv/bin/pytest` 全绿。
Commit: `ch04: 检索策略模块——四策略同一接口 + 反漏斗排列 + 低置信判定 + 过滤回退`

### Task 6: 质量台账持久层(低置信池 + faith_cases)

**Files:**
- Create: `app/guard.py`
- Modify: `app/kb.py`(追加 `FaithCaseLedger`)
- Test: `tests/test_guard.py`(新)、`tests/test_faith_ledger.py`(新)

**Interfaces:**
- Consumes: Task 1 的 `LowConfidenceQuestion` / `FaithCase` ORM。
- Produces:
  - `app.guard.REFUSAL_MARKER = "【无法回答】"`;`is_refusal(text) -> bool`;`LowConfidencePool(session_factory).insert(source, conversation_id, question, reason="") -> None`
  - `app.kb.FaithCaseLedger(session_factory).upsert_case(*, eval_id, bucket, query, strategy, answer, reason, citations, judge_model) -> None`(Task 12 消费;语义见 Global Constraints)

- [ ] **Step 1: 写失败测试**

`tests/test_guard.py`:

```python
"""低置信度池:REFUSAL_MARKER 检测 + 两入口落池(conversation_id 可空)。"""
from app.guard import REFUSAL_MARKER, LowConfidencePool, is_refusal
from app.models import LowConfidenceQuestion


def test_refusal_marker():
    assert is_refusal("【无法回答】这个问题我查不到")
    assert is_refusal("  【无法回答】xx")
    assert not is_refusal("普通回答")


def test_marker_is_single_source():
    from app.prompts import SERVICE_PROMPT_TEMPLATE
    assert REFUSAL_MARKER in SERVICE_PROMPT_TEMPLATE.template


def test_insert_two_sources(db_session_factory):
    pool = LowConfidencePool(db_session_factory)
    pool.insert("retrieval_low_conf", 1, "邮费多少", "知识库无相关内容")
    pool.insert("self_check", None, "会飞的手机怎么买", "模型自评证据不足:……")
    with db_session_factory() as s:
        rows = s.query(LowConfidenceQuestion).order_by(LowConfidenceQuestion.id).all()
    assert [(r.source, r.conversation_id, r.raw_question) for r in rows] == [
        ("retrieval_low_conf", 1, "邮费多少"),
        ("self_check", None, "会飞的手机怎么买"),
    ]
```

`tests/test_faith_ledger.py`:

```python
"""faith_cases 台账:一题一行 / seen_count 跨轮累加 / 复发退回未解决。"""
from app.kb import FaithCaseLedger
from app.models import FaithCase


def _seed(factory):
    return FaithCaseLedger(factory)


def test_first_insert_then_accumulate(db_session_factory):
    led = _seed(db_session_factory)
    led.upsert_case(eval_id="A01", bucket="A_policy", query="q1", strategy="hybrid_rerank",
                    answer="a1", reason="编了一句", citations=[{"n": 1}], judge_model="m1")
    led.upsert_case(eval_id="A01", bucket="A_policy", query="q1", strategy="hybrid_rerank",
                    answer="a2", reason="又编了", citations=[{"n": 2}], judge_model="m2")
    with db_session_factory() as s:
        rows = s.query(FaithCase).all()
    assert len(rows) == 1
    row = rows[0]
    assert row.seen_count == 2 and row.answer == "a2" and row.reason == "又编了"
    assert row.citations == [{"n": 2}] and row.judge_model == "m2"
    assert row.status == "未解决"


def test_resolved_case_relapses(db_session_factory):
    led = _seed(db_session_factory)
    led.upsert_case(eval_id="B03", bucket="B_model", query="q", strategy="hybrid_rerank",
                    answer="a", reason="r", citations=None, judge_model="m")
    with db_session_factory() as s:
        row = s.query(FaithCase).one()
        row.status = "已解决"
        row.resolution = "补了知识"
        row.resolved_at = row.last_seen_at
        s.commit()
    led.upsert_case(eval_id="B03", bucket="B_model", query="q", strategy="hybrid_rerank",
                    answer="a2", reason="复发", citations=None, judge_model="m")
    with db_session_factory() as s:
        row = s.query(FaithCase).one()
    assert row.status == "未解决"
    assert row.resolution is None      # 复发清空处置交代
    assert row.resolved_at is not None  # 保留作复发标记
    assert row.seen_count == 2 and row.reason == "复发"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_guard.py tests/test_faith_ledger.py -v` → FAIL(No module named app.guard / No attribute FaithCaseLedger)。

- [ ] **Step 3: 实现 app/guard.py**

```python
"""生成质量护栏(ch04):拒答标记唯一来源 + 低置信度问题池落库。"""
from sqlalchemy.orm import sessionmaker

from app.models import LowConfidenceQuestion

REFUSAL_MARKER = "【无法回答】"  # prompts 与 chat 路由都从这里取,禁止另写字面量


def is_refusal(text: str | None) -> bool:
    return bool(text) and text.lstrip().startswith(REFUSAL_MARKER)


class LowConfidencePool:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._factory = session_factory

    def insert(self, source: str, conversation_id: int | None, question: str, reason: str = "") -> None:
        """source ∈ retrieval_low_conf / self_check / user_feedback(本章只写前两个,不去重)。"""
        with self._factory() as session:
            session.add(LowConfidenceQuestion(
                conversation_id=conversation_id, raw_question=question, source=source, reason=reason,
            ))
            session.commit()
```

`app/kb.py` 追加(文件头 `from app.models import` 行加 `FaithCase`,补 `from sqlalchemy import func`):

```python
class FaithCaseLedger:
    """忠实度编造个案台账(spec §3.1):一题一行,跨轮累加,复发退回未解决。"""

    def __init__(self, session_factory: sessionmaker) -> None:
        self._factory = session_factory

    def upsert_case(self, *, eval_id: str, bucket: str, query: str, strategy: str,
                    answer: str, reason: str, citations: list | None, judge_model: str | None) -> None:
        with self._factory() as session:
            row = session.scalars(select(FaithCase).where(FaithCase.eval_id == eval_id)).first()
            if row is None:
                session.add(FaithCase(eval_id=eval_id, bucket=bucket, query=query[:512],
                                      strategy=strategy, answer=answer, reason=reason,
                                      citations=citations, judge_model=judge_model, seen_count=1))
            else:
                row.bucket, row.query, row.strategy = bucket, query[:512], strategy
                row.answer, row.reason = answer, reason
                row.citations, row.judge_model = citations, judge_model
                row.seen_count = (row.seen_count or 0) + 1
                row.last_seen_at = func.now()
                if row.status != "未解决":  # 复发:退回未解决、清处置;resolved_at 保留作复发标记
                    row.status = "未解决"
                    row.resolution = None
            session.commit()
```

- [ ] **Step 4: prompts 先落自评行(test_guard 的 single-source 断言需要)**

`app/prompts.py` 文件头加 `from app.guard import REFUSAL_MARKER`;`SERVICE_PROMPT_TEMPLATE` 改为拼接表达式(原六行文案不动,插入自评一行;Task 8 再补齐引用规范与负面知识两行):

```python
SERVICE_PROMPT_TEMPLATE = PromptTemplate.from_template(
    "你是「小帮」,一家电商店铺的智能客服。\n"
    "你只处理三类话题:售前咨询、售后问题、订单相关。\n"
    "医疗建议、法律意见、投资理财等超出店铺服务范围的请求,礼貌说明无法提供,并建议用户寻求专业渠道。\n"
    "不确定或不知道时,坦白告知,并建议用户转人工客服。\n"
    "工具查询结果为空或与用户问题对不上时,如实告知用户暂时查不到相关信息,并建议转人工客服;严禁编造政策、价格、时效或任何承诺。\n"
    "回答前先自评:召回的知识不足以回答用户问题、或知识库里没有依据时,回答第一行以"
    + REFUSAL_MARKER + "开头,简要说明原因并建议用户转人工客服。\n"
    "语气友好、称呼亲切,回答简洁,尽量不超过 3 句。"
)
```

(`REFUSAL_MARKER` 无花括号,拼接后进 `from_template` 安全;guard 不 import prompts,无循环。)

- [ ] **Step 5: 测试到绿 + 全量回归 + dev-notes + commit**

Run: `.venv/bin/pytest tests/test_guard.py tests/test_faith_ledger.py -v` → PASS;`.venv/bin/pytest` 全绿(test_schemas_prompts 若断言旧模板文案,同步更新)。
Commit: `ch04: 质量台账持久层——REFUSAL_MARKER/低置信池 + faith_cases 复发语义`

### Task 7: query_faq v3(引用编号 + 反漏斗 + 低置信)

**Files:**
- Modify: `app/tools/definitions.py`
- Modify: `app/vector_store.py`(删 `search` 兼容壳)、`tests/helpers.py`(删 FakeVectorStore.search)
- Modify: `tests/conftest.py`(注入 rewriter/reranker)
- Test: `tests/test_tools.py`(追加/改写)

**Interfaces:**
- Consumes: Task 3/4/5 全部产物。
- Produces: `build_tools(session_factory, embedder=None, vectors=None, top_k=None, rewriter=None, reranker=None)`;`query_faq(keyword, category=None)` 出参契约见 Global Constraints。conftest 的 app.state.registry 用替身四件套构建。

- [ ] **Step 1: 写失败测试(tests/test_tools.py 追加)**

```python
def _seed_kb(db_session_factory):
    from app.chunking import Chunk
    from app.kb import KnowledgeBaseStore, vectorize_pending
    kb = KnowledgeBaseStore(db_session_factory)
    kb.replace_doc_chunks("d.md", [
        Chunk(category="售后政策", questions="退货政策是什么", answer="支持七天无理由退货,需保持完好。",
              section_path="d.md > 退货政策是什么", content_type="faq", is_key_clause=False),
        Chunk(category="数码配件", questions="SH-E300 无线降噪耳机", answer="SH-E300 售价 299 元,支持主动降噪。",
              section_path="d.md > SH-E300", content_type="faq", is_key_clause=False),
    ])
    vectors = FakeVectorStore()
    vectorize_pending(kb, vectors, FakeEmbedding())
    return kb, vectors


async def test_query_faq_v3_contract(db_session_factory):
    kb, vectors = _seed_kb(db_session_factory)
    tools = build_tools(db_session_factory, embedder=FakeEmbedding(), vectors=vectors, top_k=3,
                        rewriter=FakeRewriter(), reranker=FakeReranker())
    query_faq = {t.name: t for t in tools}["query_faq"]
    out = json.loads(query_faq.invoke({"keyword": "SH-E300 降噪"}))
    assert out["low_confidence"] is False
    assert [it["n"] for it in out["items"]] == sorted(it["n"] for it in out["items"])  # n 连续从 1
    first = out["items"][0]
    assert first["chunk_id"] == 2 and first["section_path"] == "d.md > SH-E300"
    assert {"question", "answer", "category"} <= set(first)


async def test_query_faq_v3_lost_in_middle_arrangement(db_session_factory):
    """精排名次 n 与摆放解耦:名次 1 在首位,名次 2 在末位(≥3 条时)。"""
    kb, vectors = _seed_kb(db_session_factory)
    # 再灌 3 条含共同词「退货」的块,让 hybrid 候选 ≥4
    kb.replace_doc_chunks("d.md", [
        Chunk(category="售后政策", questions=f"退货问题{i}", answer=f"退货说明{i},七天无理由。",
              section_path=f"d.md > 退货问题{i}", content_type="faq", is_key_clause=False)
        for i in range(1, 4)
    ] + [Chunk(category="售后政策", questions="退货政策是什么", answer="支持七天无理由退货,需保持完好。",
               section_path="d.md > 退货政策是什么", content_type="faq", is_key_clause=False)])
    from app.kb import vectorize_pending as _vp
    _vp(kb, vectors, FakeEmbedding())
    tools = build_tools(db_session_factory, embedder=FakeEmbedding(), vectors=vectors, top_k=4,
                        rewriter=FakeRewriter(), reranker=FakeReranker())
    query_faq = {t.name: t for t in tools}["query_faq"]
    out = json.loads(query_faq.invoke({"keyword": "退货"}))
    ns = [it["n"] for it in out["items"]]
    assert ns[0] == 1 and ns[-1] == 2  # 摆放反漏斗:n=1 首位、n=2 末位


async def test_query_faq_v3_low_confidence_on_empty(db_session_factory):
    vectors = FakeVectorStore()
    tools = build_tools(db_session_factory, embedder=FakeEmbedding(), vectors=vectors, top_k=3,
                        rewriter=FakeRewriter(), reranker=FakeReranker())
    query_faq = {t.name: t for t in tools}["query_faq"]
    out = json.loads(query_faq.invoke({"keyword": "量子力学"}))
    assert out == {"items": [], "low_confidence": True, "reason": out["reason"],
                   "filter_fallback": False}
    assert out["reason"]


async def test_query_faq_v3_exception_converges(db_session_factory):
    """内部异常(如 milvus 库锁)收敛为空结果 + low_confidence,不抛错。"""
    class Broken:
        def ensure_collection(self): raise RuntimeError("db lock")
        def __getattr__(self, name): raise RuntimeError("db lock")

    tools = build_tools(db_session_factory, embedder=FakeEmbedding(), vectors=Broken(),
                        top_k=3, rewriter=FakeRewriter(), reranker=FakeReranker())
    out = json.loads({t.name: t for t in tools}["query_faq"].invoke({"keyword": "邮费"}))
    assert out["items"] == [] and out["low_confidence"] is True and "检索失败" in out["reason"]
```

注:`Broken.__getattr__` 让任何方法调用都抛错,覆盖 `retrieve` 内部的 dense/bm25/hybrid 路径。

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_tools.py -v` → FAIL(契约形状不变:缺 n/chunk_id/section_path)。

- [ ] **Step 3: 改造 app/tools/definitions.py**

文件头 import 调整:

```python
from app.kb import KnowledgeBaseStore
from app.models import Faq, Ticket
from app.rerank import make_reranker
from app.retrieval import RetrievalService, RETRIEVAL_SCORE_FLOOR, lost_in_middle_order
from app.rewrite import make_rewriter
```

删除常量 `RETRIEVAL_SCORE_FLOOR = 0.5`(迁至 app.retrieval)。`build_tools` 替换为:

```python
def build_tools(session_factory, embedder=None, vectors=None, top_k=None,
                rewriter=None, reranker=None) -> list:
    """注入面(ch04 扩展):只有生产路径(embedder/vectors 未注入)才按 Settings 构造真实现,
    含真 rewriter/reranker;测试部分注入(embedder/vectors 给替身、rewriter/reranker 缺省)时
    保持 None → 无改写、hybrid_rerank 降级 RRF 序,测试不出网。top_k 显式传入优先。
    query_faq 自 ch04 起走 retrieval.RetrievalService(hybrid_rerank)。"""
    production = embedder is None or vectors is None
    settings = None
    if production:
        from app.config import Settings
        from app.embedding import make_embedder
        from app.vector_store import KnowledgeVectorStore

        settings = Settings()
        embedder = embedder or make_embedder(settings)
        vectors = vectors or KnowledgeVectorStore(settings.milvus_db_path, dim=settings.embedding_dim)
    top_k = top_k or (settings.retrieval_final_top_k if production else 3)
    if production:
        if settings.query_rewrite_enabled:
            rewriter = rewriter or make_rewriter(settings)
        reranker = reranker if reranker is not None else make_reranker(settings)
    kb = KnowledgeBaseStore(session_factory)
    service = RetrievalService(
        embedder, vectors, kb, rewriter=rewriter, reranker=reranker,
        candidates=settings.hybrid_candidates if production else 50,
        final_top_k=top_k,
        rerank_score_floor=settings.rerank_score_floor if production else 0.30,
    )
```

`query_faq` 替换为:

```python
    @tool
    def query_faq(keyword: str, category: str | None = None) -> str:
        """语义检索常见问题知识库并精排。用户问退货政策、发货时间、邮费运费、付款、发票、会员等常见问题,
        或问具体商品型号(如 SH-E300)的参数价格时使用;能从用户话里明确判断品类时传 category,判断不了不要传。"""
        try:
            result = service.retrieve(keyword, strategy="hybrid_rerank", category=category)
            rows = kb.get_chunks([r.chunk_id for r in result.items])
            evidence = [
                {
                    "n": i + 1,  # n = 精排名次,[1] 恒为最强证据
                    "chunk_id": r.chunk_id,
                    "question": row.questions.splitlines()[0],
                    "answer": row.answer,
                    "category": row.category,
                    "section_path": row.section_path or "",
                }
                for i, (r, row) in enumerate(zip(result.items, rows))
            ]
            return json.dumps({
                "items": lost_in_middle_order(evidence),
                "low_confidence": result.low_confidence,
                "reason": result.reason,
                "filter_fallback": result.filter_fallback,
            }, ensure_ascii=False)
        except Exception as exc:
            return json.dumps({
                "items": [], "low_confidence": True, "reason": f"检索失败:{exc}",
                "filter_fallback": False,
            }, ensure_ascii=False)
```

同时删除 `app/vector_store.py` 与 `tests/helpers.py` 里的 `search` 兼容壳(以及 test_vector_store.py 的 `test_search_legacy_alias` 用例);`conftest.py` 的 `build_tools(...)` 调用补 `rewriter=FakeRewriter(), reranker=FakeReranker()`(import 同步)。test_tools.py / test_chat_toolflow.py 里旧 query_faq 契约断言(如「items 键只有 question/answer/category」)逐条改成 v3 契约。

- [ ] **Step 4: 测试到绿 + 全量回归 + dev-notes + commit**

Run: `.venv/bin/pytest tests/test_tools.py tests/test_vector_store.py -v` → PASS;`.venv/bin/pytest` 全绿。
Commit: `ch04: query_faq v3——hybrid_rerank 内核 + 引用编号反漏斗 + 低置信标志`

### Task 8: prompts 三件套(引用规范 / 自评拒答 / 负面知识)

**Files:**
- Modify: `app/prompts.py`(SERVICE_PROMPT_TEMPLATE 补齐;追加 RAG_ANSWER_PROMPT、FAITHFULNESS_JUDGE_PROMPT、FaithfulnessVerdict、build_evidence_block)
- Test: `tests/test_schemas_prompts.py`(追加)

**Interfaces:**
- Produces: Task 9 消费 SERVICE_PROMPT_TEMPLATE(citations/自评/负面知识文本);Task 12 消费 `RAG_ANSWER_PROMPT`(`{evidence}`/`{query}` 槽)、`FAITHFULNESS_JUDGE_PROMPT`(`{query}`/`{evidence}`/`{answer}` 槽)、`FaithfulnessVerdict(fabricated: bool, reason: str, fabricated_claims: list[str])`、`build_evidence_block(items: list[dict]) -> str`。

- [ ] **Step 1: 写失败测试(tests/test_schemas_prompts.py 追加)**

```python
def test_service_prompt_has_citation_rule():
    t = SERVICE_PROMPT_TEMPLATE.template
    assert "[n]" in t and "严禁编造" in t and "给定的编号" in t


def test_service_prompt_has_refusal_self_check():
    t = SERVICE_PROMPT_TEMPLATE.template
    assert REFUSAL_MARKER in t and "自评" in t


def test_service_prompt_has_negative_knowledge():
    t = SERVICE_PROMPT_TEMPLATE.template
    assert "退款到账" in t and "送达日期" in t and "赔付" in t and "型号参数" in t


def test_rag_answer_prompt_slots():
    block = build_evidence_block([
        {"n": 1, "section_path": "d > a", "question": "q1", "answer": "a1"},
        {"n": 2, "section_path": "d > b", "question": "q2", "answer": "a2"},
    ])
    assert "[1] (d > a) 问:q1 答:a1" in block
    out = RAG_ANSWER_PROMPT.format(evidence=block, query="邮费多少")
    assert "邮费多少" in out and "[1]" in out


def test_faithfulness_verdict_shape():
    v = FaithfulnessVerdict(fabricated=True, reason="r", fabricated_claims=["x"])
    assert v.fabricated and v.fabricated_claims == ["x"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_schemas_prompts.py -v` → FAIL。

- [ ] **Step 3: 实现 app/prompts.py**

```python
# app/prompts.py
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel

from app.guard import REFUSAL_MARKER

# 客服 System Prompt(spec §6)。文案不含花括号(PromptTemplate 占位符敏感);
# 拒答标记从 app.guard 引入,禁止另写字面量。
SERVICE_PROMPT_TEMPLATE = PromptTemplate.from_template(
    "你是「小帮」,一家电商店铺的智能客服。\n"
    "你只处理三类话题:售前咨询、售后问题、订单相关。\n"
    "医疗建议、法律意见、投资理财等超出店铺服务范围的请求,礼貌说明无法提供,并建议用户寻求专业渠道。\n"
    "不确定或不知道时,坦白告知,并建议用户转人工客服。\n"
    "工具查询结果为空或与用户问题对不上时,如实告知用户暂时查不到相关信息,并建议转人工客服;严禁编造政策、价格、时效或任何承诺。\n"
    "引用知识库作答时,必须在对应句子末尾标注角标 [n];n 只能使用工具结果里给定的编号,严禁编造编号或引用不存在的编号。\n"
    "回答前先自评:召回的知识不足以回答用户问题、或知识库里没有依据时,回答第一行以"
    + REFUSAL_MARKER + "开头,简要说明原因并建议用户转人工客服。\n"
    "负面知识清单(知识库没有依据时一律禁止承诺):不承诺退款到账的具体时间与银行处理时长;"
    "不承诺具体送达日期,库内时效区间只能以「一般」口径引用;不承诺任何赔付、补偿或额外优惠;"
    "不编造价格、库存与型号参数。\n"
    "语气友好、称呼亲切,回答简洁,尽量不超过 3 句。"
)

# ch04 忠实度评估:离线生成带引用答案的 prompt(与线上 system prompt 同一套规则,收敛成可复用文本)
RAG_ANSWER_PROMPT = PromptTemplate.from_template(
    "你是「小帮」,电商店铺智能客服。仅依据下面给出的编号证据回答用户问题。\n"
    "规则:\n"
    "1. 引用证据作答时在对应句子末尾标注角标 [n],只能使用给定编号,严禁编造编号。\n"
    "2. 回答前先自评:证据不足以回答时,第一行以" + REFUSAL_MARKER + "开头,简要说明并建议转人工。\n"
    "3. 负面知识:严禁承诺退款到账具体时间、具体送达日期、任何赔付补偿;严禁编造价格、库存、型号参数。\n"
    "4. 语气友好,回答简洁,不超过 3 句。\n\n"
    "【证据】\n{evidence}\n\n【用户问题】{query}"
)

FAITHFULNESS_JUDGE_PROMPT = PromptTemplate.from_template(
    "你是忠实度裁判。判断「答案」里的事实性断言是否全部被「编号证据」支撑,以及答案的角标 [n] "
    "所指向的证据是否真的包含该断言。\n"
    "只判编造:断言超出证据、与证据矛盾、或角标指向的证据并不包含该断言,判 fabricated=true;"
    "同义转述、语言组织差异不算编造。证据不足以判断时不算编造。\n"
    "【用户问题】{query}\n【编号证据】\n{evidence}\n【答案】{answer}"
)


class FaithfulnessVerdict(BaseModel):
    fabricated: bool
    reason: str
    fabricated_claims: list[str] = []


def build_evidence_block(items: list[dict]) -> str:
    """citations 列表 → 评审/生成共用的证据文本块。"""
    return "\n".join(
        f"[{it['n']}] ({it['section_path']}) 问:{it['question']} 答:{it['answer']}"
        for it in items
    )
```

注意:`from_template` 的入参是拼接后的普通字符串;`REFUSAL_MARKER` 不含花括号,安全。

- [ ] **Step 4: 测试到绿 + 全量回归 + dev-notes + commit**

Run: `.venv/bin/pytest tests/test_schemas_prompts.py -v` → PASS;`.venv/bin/pytest` 全绿。
Commit: `ch04: prompts 三件套——引用规范/自评拒答/负面知识 + 评估用生成与裁判模板`

### Task 9: chat 路由 citations 帧 + 落池挂钩

**Files:**
- Modify: `app/routers/chat.py`
- Modify: `app/main.py`(装配 pool/rewriter/reranker)
- Modify: `tests/conftest.py`(app.state.pool)
- Test: `tests/test_chat_toolflow.py`(追加)

**Interfaces:**
- Consumes: `app.guard.is_refusal` / `LowConfidencePool`;query_faq v3 出参。
- Produces: SSE 帧 `{"type": "citations", "items": [{n, chunk_id, question, answer, category, section_path}]}`(工具执行后、最终答案流之前,一次);`app.state.pool: LowConfidencePool`。

- [ ] **Step 1: 写失败测试(tests/test_chat_toolflow.py 追加)**

前置:conftest 的 `_make` 把替身向量库暴露为 `app.state.vectors`(见 Step 3),测试直接往里灌数据。文件底部补公共 helper:

```python
def _pool_rows(factory):
    from app.models import LowConfidenceQuestion
    with factory() as s:
        return [{"source": r.source, "raw_question": r.raw_question, "reason": r.reason or ""}
                for r in s.query(LowConfidenceQuestion).all()]


def _seed_shipping_kb(app):
    """灌一条运费知识(questions 与 query 同形,保证 FakeReranker 高分不误触低置信)。"""
    from app.chunking import Chunk
    from app.kb import KnowledgeBaseStore, vectorize_pending

    kb = KnowledgeBaseStore(app.state.session_factory)
    kb.replace_doc_chunks("d.md", [Chunk(category="售后政策", questions="邮费是多少",
                                         answer="普通订单邮费 8 元,满 99 元包邮。",
                                         section_path="d.md > 邮费与运费",
                                         content_type="faq", is_key_clause=False)])
    vectorize_pending(kb, app.state.vectors, FakeEmbedding())
```

(文件头补 `from app.kb import KnowledgeBaseStore, vectorize_pending`、`from app.chunking import Chunk`、`from tests.helpers import FakeEmbedding`。)

```python
async def test_citations_frame_before_tokens(make_client):
    model = FakeToolChatModel("邮费一般8元|满99包邮")
    client, app = await make_client(model)
    _seed_shipping_kb(app)

    events = await post_chat_sse(client, {"message": "邮费是多少"})
    cites = [e for e in events if e["type"] == "citations"]
    assert len(cites) == 1
    items = cites[0]["items"]
    assert items and items[0]["n"] == 1 and "section_path" in items[0]
    types = [e["type"] for e in events]
    assert types.index("citations") < types.index("token")  # 帧在答案流之前
    assert _pool_rows(app.state.session_factory) == []      # 正常轮不落池


async def test_low_confidence_tool_result_pools(make_client):
    model = FakeToolChatModel("这个我查不到|建议转人工")
    client, app = await make_client(model)  # 空向量库 → query_faq low_confidence=true
    events = await post_chat_sse(client, {"message": "量子力学怎么退货"})
    assert [e for e in events if e["type"] == "citations"] == []
    rows = _pool_rows(app.state.session_factory)
    assert len(rows) == 1
    assert rows[0]["source"] == "retrieval_low_conf"
    assert rows[0]["raw_question"] == "量子力学怎么退货"


async def test_refusal_answer_pools_self_check(make_client):
    """检索有结果、但模型自评答不了 → 只落 self_check 一条。"""
    model = FakeToolChatModel("【无法回答】知识库中的依据不足以回答|建议转人工")
    client, app = await make_client(model)
    _seed_shipping_kb(app)
    await post_chat_sse(client, {"message": "退货政策是什么"})
    rows = _pool_rows(app.state.session_factory)
    assert [r["source"] for r in rows] == ["self_check"]
    assert "模型自评证据不足" in rows[0]["reason"]
```

(`test_refusal_answer_pools_self_check` 必须先灌知识:否则检索低置信与拒答两条都落,断言就不是单一入口了;query「退货政策是什么」对灌入 chunk 的 FakeReranker 词面重合率高于 0.30,不触发 retrieval_low_conf。)

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_chat_toolflow.py -v` → 新增用例 FAIL(无 citations 帧 / 无落池)。

- [ ] **Step 3: 实现**

`app/routers/chat.py`:

1. import 增加 `from app.guard import is_refusal` 与 `from app.models import ...`(不需要,见下);
2. 模块级加 `_safe_json`:

```python
def _safe_json(raw: str) -> dict | None:
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    return obj if isinstance(obj, dict) else None
```

3. `event_stream` 内 `pending` 初始化后加 `citations: list[dict] = []`;
4. 工具循环里 `result = await registry.execute(...)` 之后、`pending.append` 之前:

```python
                    parsed = _safe_json(result)
                    if isinstance(parsed, dict):
                        if parsed.get("low_confidence"):
                            request.app.state.pool.insert(
                                "retrieval_low_conf", session_id, body.message,
                                str(parsed.get("reason") or ""),
                            )
                        for it in parsed.get("items") or []:
                            if isinstance(it, dict) and "n" in it:
                                citations.append(it)
```

5. 工具循环结束后、第二段流之前:

```python
                if citations:
                    yield sse_frame({"type": "citations", "items": citations})
```

6. 第二段收敛后:

```python
                final_text = "".join(final_parts)
                if is_refusal(final_text):
                    request.app.state.pool.insert(
                        "self_check", session_id, body.message,
                        "模型自评证据不足:" + final_text[:200],
                    )
                pending.append({"role": "assistant", "content": final_text})
```

(注意同时把原来的 `"".join(final_parts)` 引用替换为 `final_text`。)

`app/main.py` 装配(rewriter/reranker 由 build_tools 的生产路径内部构造,main 只补 pool):

```python
from app.guard import LowConfidencePool

    # create_app 内:删除原 embedder/vectors 局部构造与 build_tools 的显式注入,改为
    app.state.pool = LowConfidencePool(session_factory)
    app.state.registry = ToolRegistry(build_tools(
        session_factory, top_k=settings.retrieval_final_top_k,
    ))
    # (build_tools 生产路径自行 make_embedder / KnowledgeVectorStore / make_rewriter / make_reranker)
```

`tests/conftest.py` 的 `_make`:

```python
        vectors_stub = FakeVectorStore()
        app.state.vectors = vectors_stub
        app.state.pool = LowConfidencePool(factory)
        app.state.registry = ToolRegistry(build_tools(
            factory, embedder=FakeEmbedding(), vectors=vectors_stub, top_k=3,
            rewriter=FakeRewriter(), reranker=FakeReranker(),
        ))
```

(imports 补 `LowConfidencePool, FakeRewriter, FakeReranker`。)

- [ ] **Step 4: 测试到绿 + 全量回归 + dev-notes + commit**

Run: `.venv/bin/pytest tests/test_chat_toolflow.py -v` → PASS;`.venv/bin/pytest` 全绿。
Commit: `ch04: chat 路由 citations 帧 + 拒答/低置信双入口落池`

### Task 10: 知识文档型号扩充 + 评估集 32 题(数据任务)

**Files:**
- Modify: `knowledge/商品FAQ.md`(追加「## 商品型号」分组,6 个自有 SKU)
- Create: `tests/eval/ch04_eval_set.jsonl`(32 行)
- Test: `tests/test_ch04_eval_set.py`(新;数据任务的「标注样例验证」替代 TDD)

**Interfaces:**
- Produces: 评估集行格式 `{id, bucket, query, expect_doc, expect_section?, expect_keywords, notes}`;Task 11/12 消费。B 桶靶点 = 新增型号 chunk。

- [ ] **Step 1: 商品FAQ.md 追加型号分组**

在文件末尾追加(风格与现有 H2/H3/「其他问法」一致):

```markdown
## 商品型号

### SH-E300 无线降噪耳机怎么样?

SH-E300 头戴式无线降噪耳机,售价 299 元,支持主动降噪,蓝牙 5.3,整机续航 30 小时,整机一年质保。

- 其他问法:SH-E300 参数 / E300 耳机多少钱

### SH-E120 半入耳耳机怎么样?

SH-E120 半入耳无线耳机,售价 99 元,蓝牙 5.0,续航 24 小时,不支持主动降噪,整机一年质保。

### SH-K870 机械键盘是什么配置?

SH-K870 87 键机械键盘,青轴,白光背光,售价 199 元,支持全键无冲,一年质保。

### SH-K104 机械键盘是什么配置?

SH-K104 104 键机械键盘,红轴,RGB 背光,售价 219 元,支持全键无冲,一年质保。

### SH-B5 智能手环有什么功能?

SH-B5 智能手环,售价 129 元,支持心率监测、睡眠监测、50 米防水,续航 14 天,一年质保。

### SH-W20 智能手表有什么功能?

SH-W20 智能手表,售价 399 元,支持血氧监测、GPS 定位、NFC 刷卡,续航 7 天,一年质保。
```

- [ ] **Step 2: 写评估集 tests/eval/ch04_eval_set.jsonl(32 行)**

```jsonl
{"id": "A01", "bucket": "A_policy", "query": "普通订单的邮费是多少", "expect_doc": "退货政策.md", "expect_keywords": ["邮费", "8 元"], "notes": "运费说明"}
{"id": "A02", "bucket": "A_policy", "query": "满多少包邮", "expect_doc": "退货政策.md", "expect_keywords": ["99", "包邮"], "notes": "包邮门槛"}
{"id": "A03", "bucket": "A_policy", "query": "质量问题的退货运费谁出", "expect_doc": ["退货政策.md", "商品FAQ.md"], "expect_keywords": ["商家"], "notes": "质量退货运费"}
{"id": "A04", "bucket": "A_policy", "query": "哪些商品不支持七天无理由退货", "expect_doc": "退货政策.md", "expect_keywords": ["不予退货", "定制"], "notes": "负面条款"}
{"id": "A05", "bucket": "A_policy", "query": "退款多久能原路退回", "expect_doc": "退货政策.md", "expect_keywords": ["退款", "工作日"], "notes": "退款时效"}
{"id": "A06", "bucket": "A_policy", "query": "寄到偏远地区要加钱吗", "expect_doc": ["退货政策.md", "售后手册.md"], "expect_keywords": ["偏远"], "notes": "偏远加收"}
{"id": "A07", "bucket": "A_policy", "query": "退货申请审核要多久", "expect_doc": "商品FAQ.md", "expect_keywords": ["24 小时", "审核"], "notes": "审核时限"}
{"id": "A08", "bucket": "A_policy", "query": "发票多久能开出来", "expect_doc": "商品FAQ.md", "expect_keywords": ["电子发票", "24 小时"], "notes": "开票时限"}
{"id": "B01", "bucket": "B_model", "query": "SH-E300 支持主动降噪吗", "expect_doc": "商品FAQ.md", "expect_keywords": ["SH-E300", "主动降噪"], "notes": "型号+功能"}
{"id": "B02", "bucket": "B_model", "query": "SH-E300 多少钱", "expect_doc": "商品FAQ.md", "expect_keywords": ["SH-E300", "299"], "notes": "型号+价格"}
{"id": "B03", "bucket": "B_model", "query": "SH-E120 有降噪功能吗", "expect_doc": "商品FAQ.md", "expect_keywords": ["SH-E120", "主动降噪"], "notes": "型号+否定参数"}
{"id": "B04", "bucket": "B_model", "query": "SH-K870 是什么轴的", "expect_doc": "商品FAQ.md", "expect_keywords": ["SH-K870", "青轴"], "notes": "型号+参数"}
{"id": "B05", "bucket": "B_model", "query": "SH-K104 卖多少", "expect_doc": "商品FAQ.md", "expect_keywords": ["SH-K104", "219"], "notes": "型号+价格"}
{"id": "B06", "bucket": "B_model", "query": "SH-B5 能监测睡眠吗", "expect_doc": "商品FAQ.md", "expect_keywords": ["SH-B5", "睡眠"], "notes": "型号+功能"}
{"id": "B07", "bucket": "B_model", "query": "SH-W20 续航怎么样", "expect_doc": "商品FAQ.md", "expect_keywords": ["SH-W20", "7 天"], "notes": "型号+续航"}
{"id": "B08", "bucket": "B_model", "query": "SH-K870 是 87 键还是 104 键", "expect_doc": "商品FAQ.md", "expect_keywords": ["SH-K870", "87 键"], "notes": "型号辨析"}
{"id": "C01", "bucket": "C_colloquial", "query": "耳机戴了两天耳朵疼想退了", "expect_doc": "商品FAQ.md", "expect_keywords": ["七天无理由"], "notes": "口语改写靶"}
{"id": "C02", "bucket": "C_colloquial", "query": "钱啥时候能回来啊", "expect_doc": "退货政策.md", "expect_keywords": ["工作日"], "notes": "口语改写靶"}
{"id": "C03", "bucket": "C_colloquial", "query": "发啥快递啊一般几天到", "expect_doc": "商品FAQ.md", "expect_keywords": ["3-5 天"], "notes": "口语改写靶"}
{"id": "C04", "bucket": "C_colloquial", "query": "这个东西能便宜点不", "expect_doc": "商品FAQ.md", "expect_keywords": ["95 折"], "notes": "口语→会员权益"}
{"id": "C05", "bucket": "C_colloquial", "query": "开胶了算不算质量问题啊", "expect_doc": "商品FAQ.md", "expect_keywords": ["质量"], "notes": "口语改写靶"}
{"id": "C06", "bucket": "C_colloquial", "query": "用微信给钱行不行", "expect_doc": "商品FAQ.md", "expect_keywords": ["微信"], "notes": "口语改写靶"}
{"id": "C07", "bucket": "C_colloquial", "query": "买东西能开票不", "expect_doc": "商品FAQ.md", "expect_keywords": ["电子发票"], "notes": "口语改写靶"}
{"id": "C08", "bucket": "C_colloquial", "query": "住得太远寄过来是不是要加钱", "expect_doc": ["退货政策.md", "售后手册.md"], "expect_keywords": ["偏远"], "notes": "口语改写靶"}
{"id": "E01", "bucket": "E_multi", "query": "98 块的订单不包邮,那我再凑一件够 99 行吗", "expect_doc": "退货政策.md", "expect_keywords": ["99", "包邮"], "notes": "门槛+动作"}
{"id": "E02", "bucket": "E_multi", "query": "质量问题退货的话运费谁出、退款多久到", "expect_doc": ["退货政策.md", "商品FAQ.md"], "expect_keywords": ["商家", "工作日"], "notes": "双约束"}
{"id": "E03", "bucket": "E_multi", "query": "定制商品过了七天还能退吗", "expect_doc": "退货政策.md", "expect_keywords": ["定制", "不予退货"], "notes": "条款叠加"}
{"id": "E04", "bucket": "E_multi", "query": "SH-E300 用了六天降噪坏了能退吗", "expect_doc": ["商品FAQ.md", "退货政策.md"], "expect_keywords": ["七天无理由", "SH-E300"], "notes": "型号+政策跨文档"}
{"id": "E05", "bucket": "E_multi", "query": "预售的商品也是 48 小时内发货吗", "expect_doc": "商品FAQ.md", "expect_keywords": ["48 小时", "预售"], "notes": "条件辨析"}
{"id": "E06", "bucket": "E_multi", "query": "会员买 SH-B5 手环是多少钱", "expect_doc": ["商品FAQ.md"], "expect_keywords": ["95 折", "SH-B5"], "notes": "会员+型号"}
{"id": "E07", "bucket": "E_multi", "query": "质量问题退货运费商家出,那七天无理由的呢", "expect_doc": ["退货政策.md", "商品FAQ.md"], "expect_keywords": ["商家", "运费"], "notes": "对照条款"}
{"id": "E08", "bucket": "E_multi", "query": "满 99 包邮对偏远地区也适用吗", "expect_doc": ["退货政策.md", "售后手册.md"], "expect_keywords": ["偏远", "包邮"], "notes": "条款叠加"}
```

- [ ] **Step 3: 写数据验证测试(替代 TDD 红绿的那一步)**

`tests/test_ch04_eval_set.py`:

```python
"""ch04 评估集数据验证(工作要求 1:数据类任务以标注样例验证替代 TDD)。

四桶各 8 题;expect_doc 真实存在;每个 expect_keyword 都能在期望文档原文里找到
(ground truth 必须真实可命中,防「评估集飘在知识库外」);型号关键词有对应 chunk。
"""
import json
from pathlib import Path

from app.chunking import chunk_markdown

ROOT = Path(__file__).resolve().parent.parent
CASES = [json.loads(ln) for ln in
         (ROOT / "tests" / "eval" / "ch04_eval_set.jsonl").read_text(encoding="utf-8").splitlines()
         if ln.strip()]
BUCKETS = {"A_policy", "B_model", "C_colloquial", "E_multi"}


def test_32_cases_four_buckets():
    assert len(CASES) == 32
    assert {c["bucket"] for c in CASES} == BUCKETS
    for b in BUCKETS:
        assert sum(1 for c in CASES if c["bucket"] == b) == 8
    assert len({c["id"] for c in CASES}) == 32


def test_expect_docs_exist():
    for c in CASES:
        docs = c["expect_doc"] if isinstance(c["expect_doc"], list) else [c["expect_doc"]]
        for d in docs:
            assert (ROOT / "knowledge" / d).exists(), f"{c['id']}: {d} 不存在"


def test_keywords_exist_in_expected_docs():
    for c in CASES:
        docs = c["expect_doc"] if isinstance(c["expect_doc"], list) else [c["expect_doc"]]
        corpus = "".join((ROOT / "knowledge" / d).read_text(encoding="utf-8") for d in docs)
        for kw in c["expect_keywords"]:
            assert kw in corpus, f"{c['id']}: 关键词「{kw}」不在期望文档原文里"


def test_ground_truth_reachable_by_chunker():
    """切分后存在这样的 chunk:路径含期望文档且命中任一关键词——保证评估集可被检索命中。"""
    for c in CASES:
        docs = c["expect_doc"] if isinstance(c["expect_doc"], list) else [c["expect_doc"]]
        found = False
        for d in docs:
            text = (ROOT / "knowledge" / d).read_text(encoding="utf-8")
            for chunk in chunk_markdown(text, d, "faq"):
                blob = f"{chunk.questions}\n{chunk.answer}"
                if any(kw in blob for kw in c["expect_keywords"]):
                    found = True
        assert found, f"{c['id']}: 没有任何 chunk 能命中关键词"
```

- [ ] **Step 4: 跑验证测试并修正数据(允许的循环:改评估集行或补文档,不改断言)**

Run: `.venv/bin/pytest tests/test_ch04_eval_set.py -v` → 全 PASS(关键词落空时优先修评估集行;B 桶关键词应以新增型号 chunk 为准)。

- [ ] **Step 5: 全量回归 + dev-notes + commit**

Run: `.venv/bin/pytest` 全绿(test_knowledge_docs 若断言文档结构需同步)。
Commit: `ch04: 知识文档型号 SKU + 四桶评估集 32 题 + 数据验证`

### Task 11: 建库收口 + 检索四策略评估(Recall@K / MRR)

**Files:**
- Modify: `scripts/eval_retrieval.py`(整体重写)
- Modify: `.gitignore`(加 `reports/`)
- Test: `tests/test_eval_retrieval_metrics.py`(新,指标纯函数)

**Interfaces:**
- Consumes: Task 5 `RetrievalService` / `STRATEGIES`;Task 3 `make_reranker`;Task 4 `make_rewriter`;Task 10 评估集。
- Produces: `scripts/eval_retrieval.py` CLI(`--strategy all|dense|bm25|hybrid|hybrid_rerank`、`--set PATH`、`--no-rewrite`);`chunk_is_relevant(row, case) -> bool`;`recall_mrr(ranked, rows_by_id, case) -> (hits: dict[int, bool], rr: float)`;报告 `reports/ch04-retrieval-report.md`。

- [ ] **Step 1: 写指标纯函数的失败测试**

`tests/test_eval_retrieval_metrics.py`:

```python
"""检索评估指标:命中判定与 Recall@K / MRR 纯函数(不发网络、不碰库)。"""
from types import SimpleNamespace

from scripts.eval_retrieval import chunk_is_relevant, recall_mrr

ROW = SimpleNamespace(section_path="退货政策.md > 邮费与运费",
                      questions="邮费与运费", answer="普通订单邮费 8 元,满 99 元包邮。")
ROW_OTHER = SimpleNamespace(section_path="商品FAQ.md > 物流", questions="多久能发货", answer="48 小时内。")
CASE = {"expect_doc": "退货政策.md", "expect_keywords": ["邮费"]}


def test_chunk_is_relevant():
    assert chunk_is_relevant(ROW, CASE)
    assert not chunk_is_relevant(ROW_OTHER, CASE)


def test_chunk_is_relevant_section_pin():
    case = {**CASE, "expect_section": "邮费与运费"}
    assert chunk_is_relevant(ROW, case)
    row_other_section = SimpleNamespace(section_path="退货政策.md > 退款说明",
                                        questions="退款", answer="邮费说明")
    assert not chunk_is_relevant(row_other_section, case)


def test_recall_and_mrr():
    # 直接构造:第 3 位才命中
    rows_by_id = {1: ROW_OTHER, 2: ROW_OTHER, 3: ROW}
    hits, rr = recall_mrr([(1, 0.9), (2, 0.8), (3, 0.7), (4, 0.6)], rows_by_id, CASE)
    assert hits == {3: False, 5: True, 10: True}
    assert rr == 1.0 / 3


def test_recall_miss_all():
    rows_by_id = {1: ROW_OTHER}
    hits, rr = recall_mrr([(1, 0.9)], rows_by_id, CASE)
    assert hits == {3: False, 5: False, 10: False} and rr == 0.0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_eval_retrieval_metrics.py -v` → FAIL(无 scripts.eval_retrieval 新接口)。

- [ ] **Step 3: 重写 scripts/eval_retrieval.py**

```python
"""ch04 检索质量评估:四策略 × 评估集,Recall@3/5/10 + MRR@10,按桶分列 + 总表。

用法:
  .venv/bin/python scripts/eval_retrieval.py                       # 四策略全跑 ch04 评估集
  .venv/bin/python scripts/eval_retrieval.py --strategy bm25       # 单策略
  .venv/bin/python scripts/eval_retrieval.py --set tests/eval/retrieval_samples.jsonl  # ch03 回归集
  .venv/bin/python scripts/eval_retrieval.py --no-rewrite          # 关改写对照
前提:MySQL 已建库、Milvus v2 集合已灌(python -m scripts.build_kb)、.env key 有效。
退出码:0 全中 / 1 有用例 Recall@10 未命中 / 2 向量库为空。
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings
from app.db import make_engine, make_session_factory
from app.embedding import make_embedder
from app.kb import KnowledgeBaseStore
from app.rerank import make_reranker
from app.retrieval import STRATEGIES, RetrievalService
from app.rewrite import make_rewriter
from app.vector_store import KnowledgeVectorStore

ROOT = Path(__file__).resolve().parent.parent


def chunk_is_relevant(row, case: dict) -> bool:
    docs = case["expect_doc"] if isinstance(case["expect_doc"], list) else [case["expect_doc"]]
    if not row.section_path or not any(d in row.section_path for d in docs):
        return False
    if case.get("expect_section") and case["expect_section"] not in row.section_path:
        return False
    return any(kw in row.answer or kw in row.questions for kw in case["expect_keywords"])


def recall_mrr(ranked: list[tuple[int, float]], rows_by_id: dict, case: dict,
               k_list: tuple[int, ...] = (3, 5, 10), mrr_k: int = 10):
    """ranked 按策略名次排序;命中即止(取首个正确 chunk 的倒数排名)。"""
    hits = {k: False for k in k_list}
    rr = 0.0
    for pos, (cid, _score) in enumerate(ranked):
        row = rows_by_id.get(cid)
        if row is None or not chunk_is_relevant(row, case):
            continue
        for k in k_list:
            hits[k] = pos < k
        if pos < mrr_k:
            rr = 1.0 / (pos + 1)
        break
    return hits, rr


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="all", choices=[*STRATEGIES, "all"])
    parser.add_argument("--set", dest="eval_set", default="tests/eval/ch04_eval_set.jsonl")
    parser.add_argument("--no-rewrite", action="store_true")
    args = parser.parse_args()

    settings = Settings()
    kb = KnowledgeBaseStore(make_session_factory(make_engine(settings)))
    vectors = KnowledgeVectorStore(settings.milvus_db_path, dim=settings.embedding_dim)
    if vectors.count() == 0:
        print("向量库为空,请先运行:.venv/bin/python -m scripts.build_kb")
        return 2
    rewriter = make_rewriter(settings) if (settings.query_rewrite_enabled and not args.no_rewrite) else None
    service = RetrievalService(
        make_embedder(settings), vectors, kb, rewriter=rewriter, reranker=make_reranker(settings),
        candidates=settings.hybrid_candidates, final_top_k=settings.retrieval_final_top_k,
        rerank_score_floor=settings.rerank_score_floor,
    )
    cases = [json.loads(ln) for ln in
             (ROOT / args.eval_set).read_text(encoding="utf-8").splitlines() if ln.strip()]
    strategies = list(STRATEGIES) if args.strategy == "all" else [args.strategy]

    stats = {s: defaultdict(lambda: {"r3": 0, "r5": 0, "r10": 0, "rr": 0.0, "n": 0})
             for s in strategies}
    misses: list[tuple[str, str, str]] = []
    for s in strategies:
        for case in cases:
            result = service.retrieve(case["query"], strategy=s)
            rows_by_id = {r.id: r for r in kb.get_chunks([it.chunk_id for it in result.items])}
            hits, rr = recall_mrr([(it.chunk_id, it.score) for it in result.items], rows_by_id, case)
            b = stats[s][case["bucket"]]
            b["n"] += 1
            b["r3"] += hits[3]; b["r5"] += hits[5]; b["r10"] += hits[10]; b["rr"] += rr
            if not hits[10]:
                misses.append((s, case["id"], case["query"]))
            mark = "✅" if hits[10] else "❌"
            top = result.items[0].chunk_id if result.items else None
            print(f"{mark} [{s}] {case['id']} {case['query']} -> top1={top} rr={rr:.3f}")

    for s in strategies:
        print(f"\n== 策略 {s} ==")
        print(f"{'bucket':<12}{'R@3':>7}{'R@5':>7}{'R@10':>7}{'MRR@10':>9}{'n':>4}")
        for bucket, b in sorted(stats[s].items()):
            n = b["n"] or 1
            print(f"{bucket:<12}{b['r3']/n:>7.2f}{b['r5']/n:>7.2f}{b['r10']/n:>7.2f}"
                  f"{b['rr']/n:>9.3f}{b['n']:>4}")

    report_dir = ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    lines = [
        "# ch04 检索四策略对比报告", "",
        f"评估集:{args.eval_set}({len(cases)} 题);rewrite={'off' if args.no_rewrite else 'on'}", "",
    ]
    for s in strategies:
        lines += [f"## {s}", "", "| bucket | Recall@3 | Recall@5 | Recall@10 | MRR@10 | n |",
                  "|---|---|---|---|---|---|"]
        for bucket, b in sorted(stats[s].items()):
            n = b["n"] or 1
            lines.append(f"| {bucket} | {b['r3']/n:.2f} | {b['r5']/n:.2f} | {b['r10']/n:.2f} "
                         f"| {b['rr']/n:.3f} | {b['n']} |")
        lines.append("")
    (report_dir / "ch04-retrieval-report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n报告已写入 reports/ch04-retrieval-report.md")
    if misses:
        print(f"Recall@10 未命中 {len(misses)} 例:{[(m[0], m[1]) for m in misses]}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 指标测试到绿 + 全量回归**

Run: `.venv/bin/pytest tests/test_eval_retrieval_metrics.py -v` → PASS;`.venv/bin/pytest` 全绿。(端到端真跑在 Task 14 统一执行。)

- [ ] **Step 5: .gitignore 加 `reports/` + dev-notes + commit**

Commit: `ch04: 检索四策略评估——Recall@K/MRR 分桶报告 + 指标纯函数`

### Task 12: 忠实度评估(生成 + LLM 裁判 + faith_cases 台账)

**Files:**
- Create: `scripts/eval_faithfulness.py`
- Test: `tests/test_eval_faithfulness.py`(新)

**Interfaces:**
- Consumes: Task 5 service、Task 6 `FaithCaseLedger`、Task 8 `RAG_ANSWER_PROMPT` / `FAITHFULNESS_JUDGE_PROMPT` / `FaithfulnessVerdict` / `build_evidence_block`。
- Produces: `make_evidence(result, rows) -> list[dict]`(n=精排名次,反漏斗摆放);`run_case(case, service, kb, chat, judge, ledger) -> tuple[bool, str]`;报告 `reports/ch04-faithfulness-report.md`。

- [ ] **Step 1: 写失败测试**

`tests/test_eval_faithfulness.py`:

```python
"""忠实度评估:证据快照组装 + 裁判落台账(替身全链,SQLite),复发累加。"""
import pytest

from app.chunking import Chunk
from app.kb import KnowledgeBaseStore
from app.prompts import FaithfulnessVerdict
from app.retrieval import RetrievalResult, Retrieved
from scripts.eval_faithfulness import make_evidence, run_case
from tests.helpers import FakeEmbedding, FakeReranker, FakeRewriter, FakeVectorStore


class _JudgeFab:
    def invoke(self, prompt):
        return FaithfulnessVerdict(fabricated=True, reason="到账时间无证据", fabricated_claims=["3 天到账"])


class _JudgeOk:
    def invoke(self, prompt):
        return FaithfulnessVerdict(fabricated=False, reason="", fabricated_claims=[])


class _Chat:
    def invoke(self, prompt):
        return type("Msg", (), {"content": "退款一般 3 天到账 [1]"})()


@pytest.fixture
def env(db_session_factory):
    kb = KnowledgeBaseStore(db_session_factory)
    kb.replace_doc_chunks("d.md", [Chunk(category="售后政策", questions="退款说明",
                                         answer="退款原路返回,3-5 个工作日。",
                                         section_path="d.md > 退款说明",
                                         content_type="policy", is_key_clause=False)])
    vectors = FakeVectorStore()
    from app.kb import vectorize_pending
    vectorize_pending(kb, vectors, FakeEmbedding())
    service = RetrievalService(FakeEmbedding(), vectors, kb,
                               rewriter=FakeRewriter(), reranker=FakeReranker())
    return kb, service, db_session_factory


def test_make_evidence_arrangement():
    result = RetrievalResult(items=[Retrieved(11, 0.5), Retrieved(12, 0.4), Retrieved(13, 0.3)])
    rows = {11: SimpleRow(), 12: SimpleRow(), 13: SimpleRow()}
    evidence = make_evidence(result, rows)
    assert [e["n"] for e in evidence] == [1, 3, 2]  # 反漏斗摆放,n 不变


class SimpleRow:
    questions = "q"
    answer = "a"
    section_path = "d > s"
    category = "c"


def test_run_case_fabricated_writes_ledger(env):
    kb, service, factory = env
    case = {"id": "A01", "bucket": "A_policy", "query": "退款多久到账"}
    fab, reason = run_case(case, service, kb, _Chat(), _JudgeFab(), FaithCaseLedger(factory),
                           judge_model="stub")
    assert fab and reason
    from app.models import FaithCase
    with factory() as s:
        row = s.query(FaithCase).one()
    assert row.eval_id == "A01" and row.seen_count == 1 and row.judge_model == "stub"
    assert row.citations and row.citations[0]["n"] == 1
    # 复跑一轮:同题累加
    run_case(case, service, kb, _Chat(), _JudgeFab(), FaithCaseLedger(factory), judge_model="stub")
    with factory() as s:
        assert s.query(FaithCase).one().seen_count == 2


def test_run_case_ok_no_ledger(env):
    kb, service, factory = env
    case = {"id": "A02", "bucket": "A_policy", "query": "退款多久到账"}
    fab, _ = run_case(case, service, kb, _Chat(), _JudgeOk(), FaithCaseLedger(factory),
                      judge_model="stub")
    assert not fab
    from app.models import FaithCase
    with factory() as s:
        assert s.query(FaithCase).count() == 0
```

(文件头补 `from app.kb import FaithCaseLedger`;`SimpleRow` 类定义挪到 `test_make_evidence_arrangement` 之前更清晰。`judge_model` 由 `run_case` 调用方传入,测试显式传 `"stub"`;台账行的 bucket/query 字段取自 `case`。)

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_eval_faithfulness.py -v` → FAIL(No module named scripts.eval_faithfulness)。

- [ ] **Step 3: 实现 scripts/eval_faithfulness.py**

```python
"""ch04 忠实度评估:hybrid_rerank 生成带引用答案 → LLM 裁判判编造 → faith_cases 台账 + 分桶报告。

用法:
  .venv/bin/python scripts/eval_faithfulness.py                # 全量 32 题,落台账
  .venv/bin/python scripts/eval_faithfulness.py --limit 4      # 冒烟
  .venv/bin/python scripts/eval_faithfulness.py --no-db        # 只出报告不落库
前提:同 eval_retrieval(建库 + .env)。判出的编造个案按 spec §3.1 upsert 进 faith_cases。
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings
from app.db import make_engine, make_session_factory
from app.embedding import make_embedder
from app.kb import FaithCaseLedger, KnowledgeBaseStore
from app.llm import make_chat_model, make_extract_model
from app.prompts import (FAITHFULNESS_JUDGE_PROMPT, RAG_ANSWER_PROMPT,
                         FaithfulnessVerdict, build_evidence_block)
from app.rerank import make_reranker
from app.retrieval import RetrievalService, lost_in_middle_order
from app.rewrite import make_rewriter
from app.vector_store import KnowledgeVectorStore

ROOT = Path(__file__).resolve().parent.parent
STRATEGY = "hybrid_rerank"


def make_evidence(result, rows) -> list[dict]:
    """n = 精排名次;摆放反漏斗(喂给模型的顺序),citations 快照保留摆放序。"""
    evidence = [
        {"n": i + 1, "chunk_id": r.chunk_id,
         "question": row.questions.splitlines()[0], "answer": row.answer,
         "section_path": row.section_path or "", "category": row.category}
        for i, (r, row) in enumerate(zip(result.items, rows))
    ]
    return lost_in_middle_order(evidence)


def run_case(case, service, kb, chat, judge, ledger, judge_model: str = "unknown") -> tuple[bool, str]:
    result = service.retrieve(case["query"], strategy=STRATEGY)
    rows = kb.get_chunks([r.chunk_id for r in result.items])
    evidence = make_evidence(result, rows)
    answer = chat.invoke(RAG_ANSWER_PROMPT.format(
        evidence=build_evidence_block(evidence), query=case["query"])).content
    verdict = judge.invoke(FAITHFULNESS_JUDGE_PROMPT.format(
        query=case["query"], evidence=build_evidence_block(evidence), answer=answer))
    if isinstance(verdict, dict):
        verdict = FaithfulnessVerdict(**verdict)
    if verdict.fabricated and ledger is not None:
        ledger.upsert_case(eval_id=case["id"], bucket=case["bucket"], query=case["query"],
                           strategy=STRATEGY, answer=answer, reason=verdict.reason,
                           citations=evidence, judge_model=judge_model)
    return bool(verdict.fabricated), str(verdict.reason)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-db", action="store_true")
    parser.add_argument("--set", dest="eval_set", default="tests/eval/ch04_eval_set.jsonl")
    args = parser.parse_args()

    settings = Settings()
    kb = KnowledgeBaseStore(make_session_factory(make_engine(settings)))
    vectors = KnowledgeVectorStore(settings.milvus_db_path, dim=settings.embedding_dim)
    if vectors.count() == 0:
        print("向量库为空,请先运行:.venv/bin/python -m scripts.build_kb")
        return 2
    service = RetrievalService(
        make_embedder(settings), vectors, kb,
        rewriter=make_rewriter(settings) if settings.query_rewrite_enabled else None,
        reranker=make_reranker(settings),
        candidates=settings.hybrid_candidates, final_top_k=settings.retrieval_final_top_k,
        rerank_score_floor=settings.rerank_score_floor,
    )
    chat = make_chat_model(settings)
    judge = make_extract_model(settings).with_structured_output(FaithfulnessVerdict,
                                                                method="function_calling")
    ledger = None if args.no_db else FaithCaseLedger(make_session_factory(make_engine(settings)))
    judge_model = settings.openai_model

    cases = [json.loads(ln) for ln in
             (ROOT / args.eval_set).read_text(encoding="utf-8").splitlines() if ln.strip()]
    if args.limit:
        cases = cases[: args.limit]

    bucket_stats: dict[str, dict] = defaultdict(lambda: {"ok": 0, "n": 0})
    fabricated_cases: list[str] = []
    for case in cases:
        fab, reason = run_case(case, service, kb, chat, judge, ledger, judge_model=judge_model)
        b = bucket_stats[case["bucket"]]
        b["n"] += 1
        if fab:
            fabricated_cases.append(f"{case['id']} {case['query']} —— {reason}")
            print(f"❌ {case['id']} {case['query']} 编造:{reason}")
        else:
            b["ok"] += 1
            print(f"✅ {case['id']} {case['query']}")

    print("\n== Faithfulness(hybrid_rerank)==")
    print(f"{'bucket':<12}{'非编造率':>9}{'n':>4}")
    for bucket, b in sorted(bucket_stats.items()):
        print(f"{bucket:<12}{b['ok']/b['n']:>9.2f}{b['n']:>4}")

    report_dir = ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    lines = ["# ch04 忠实度评估报告(hybrid_rerank)", "",
             f"评估集:{args.eval_set};judge_model:{judge_model}", "",
             "| bucket | Faithfulness | n |", "|---|---|---|"]
    for bucket, b in sorted(bucket_stats.items()):
        lines.append(f"| {bucket} | {b['ok']/b['n']:.2f} | {b['n']} |")
    if fabricated_cases:
        lines += ["", "## 编造个案(已进 faith_cases 台账)"]
        lines += [f"- {c}" for c in fabricated_cases]
    (report_dir / "ch04-faithfulness-report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n报告已写入 reports/ch04-faithfulness-report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 测试到绿 + 全量回归 + dev-notes + commit**

Run: `.venv/bin/pytest tests/test_eval_faithfulness.py -v` → PASS;`.venv/bin/pytest` 全绿。
Commit: `ch04: 忠实度评估——生成+LLM 裁判+faith_cases 台账落库 + 分桶报告`

### Task 13: 前端配套(引用可点 + 满意度反馈)【Vibe Coding,例外流程】

**Files:**
- Modify: `static/index.html`(唯一文件)

**Interfaces:**
- Consumes: SSE `citations` 帧(`items: [{n, chunk_id, question, answer, category, section_path}]`)。
- 不写测试、不 code review;效果以用户描述为准,现场迭代。

- [ ] **Step 1: citations 帧暂存**

`send()` 里新增 `citations = []` 局部状态;事件分支加:

```js
        } else if (evt.type === "citations") {
          citations = evt.items || [];
        }
```

并把 `citations` 传给收尾逻辑(与 `stream` 一起)。

- [ ] **Step 2: 答案收尾时把 [n] 渲染成可点角标 + 来源卡**

`startStreamText` 的 `finish()` 之后,新增:

```js
function renderCitations(bubble, citations) {
  if (!citations || !citations.length) return;
  const content = bubble.querySelector(".content");
  const text = content.textContent;
  const byN = new Map(citations.map((c) => [c.n, c]));
  content.textContent = "";
  const re = /\[(\d+)\]/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    const n = parseInt(m[1], 10);
    content.appendChild(document.createTextNode(text.slice(last, m.index)));
    if (byN.has(n)) {
      const sup = document.createElement("sup");
      sup.className = "cite";
      sup.textContent = "[" + n + "]";
      sup.onclick = () => toggleSource(bubble, n, citations);
      content.appendChild(sup);
    } else {
      content.appendChild(document.createTextNode(m[0]));
    }
    last = m.index + m[0].length;
  }
  content.appendChild(document.createTextNode(text.slice(last)));
}

function toggleSource(bubble, n, citations) {
  const existing = bubble.querySelector(".source-card");
  if (existing && existing.dataset.n == n) { existing.remove(); return; }
  if (existing) existing.remove();
  const c = citations.find((x) => x.n === n);
  if (!c) return;
  const card = document.createElement("div");
  card.className = "source-card";
  card.dataset.n = n;
  card.innerHTML = '<div class="src-path"></div><div class="src-q"></div><div class="src-a"></div>';
  card.querySelector(".src-path").textContent = "来源[" + n + "] " + (c.section_path || "(无章节路径)");
  card.querySelector(".src-q").textContent = c.question;
  card.querySelector(".src-a").textContent = c.answer;
  bubble.appendChild(card);
  scrollDown();
}
```

CSS 追加:

```css
  .cite { color: #4f7cff; cursor: pointer; margin: 0 1px; font-weight: 600; }
  .cite:hover { text-decoration: underline; }
  .source-card {
    margin-top: 8px; padding: 8px 10px; border-radius: 8px;
    background: #f6f8fc; border: 1px solid #e0e5ee; font-size: 13px;
  }
  .source-card .src-path { color: #5a6270; font-size: 12px; margin-bottom: 4px; }
  .source-card .src-q { font-weight: 600; margin-bottom: 2px; }
  .source-card .src-a { color: #333; white-space: pre-wrap; }
```

- [ ] **Step 3: 满意度反馈(左下角,一次性锁定)**

CSS:

```css
  .feedback { display: flex; align-items: center; gap: 6px; margin-top: 6px; font-size: 12px; }
  .feedback button {
    border: 1px solid #d6dae2; background: #fff; border-radius: 6px;
    padding: 1px 8px; cursor: pointer; font-size: 13px;
  }
  .feedback button.picked { border-color: #4f7cff; background: #eef3ff; }
  .feedback button:disabled { cursor: default; opacity: .75; }
  .feedback .fb-status { color: #8a919e; }
```

收尾时(`renderCitations` 后)给每个机器人气泡挂:

```js
function addFeedback(bubble) {
  const bar = document.createElement("div");
  bar.className = "feedback";
  const up = document.createElement("button"); up.textContent = "👍";
  const down = document.createElement("button"); down.textContent = "👎";
  const status = document.createElement("span"); status.className = "fb-status";
  function pick(which) {
    (which === "up" ? up : down).classList.add("picked");
    status.textContent = "已反馈";
    up.disabled = down.disabled = true;
  }
  up.onclick = () => pick("up");
  down.onclick = () => pick("down");
  bar.append(up, down, status);
  bubble.appendChild(bar);
}
```

`send()` 的 `finally` 里,气泡非 error 时依次调 `renderCitations(bubble, citations)` 与 `addFeedback(bubble)`(只挂一次;可给气泡加 `data-feedback` 标记防重复)。

- [ ] **Step 4: 人工过一遍**

起服务(`.venv/bin/uvicorn app.main:app --port 8000`),页面问「邮费是多少」:角标出现、点击展开来源卡、再点收起;👍 点击点亮 + 「已反馈」+ 锁死;刷新页面历史不回放(与现状一致)。效果不对就地改,直到用户认可。

- [ ] **Step 5: dev-notes 追记 + commit**

Commit: `ch04: 聊天页引用角标可点 + 满意度反馈(纯前端采集)`

### Task 14: README/pyproject 收口 + 真机终验收

**Files:**
- Modify: `README.md`(ch04 定位、配置、评估、验收、已知边界)
- Modify: `pyproject.toml`(description 更新,依赖已在 Task 2 加 jieba)
- Modify: `dev-notes/ch04.md`(finish 段)

- [ ] **Step 1: 全链路真跑(演示命令定稿进 README)**

```bash
docker compose down -v && docker compose up -d          # 等 ~30s,八表由 init.sql 创建
.venv/bin/python -m db.seed && .venv/bin/python -m db.seed_conversations
.venv/bin/python -m scripts.build_kb                    # 首次建集合 v2 + 全量向量化
.venv/bin/python -m scripts.mine_qa                     # 可选:挖知识增量
.venv/bin/python scripts/eval_retrieval.py              # 四策略对比报告 → reports/
.venv/bin/python scripts/eval_faithfulness.py           # 忠实度 + faith_cases 台账
.venv/bin/uvicorn app.main:app --port 8000
```

- [ ] **Step 2: 四条验收逐条打勾(真库真模型)**

0. **rerank_score_floor 校准**(spec §5 承诺的实现期定稿):跑 `eval_retrieval.py --strategy hybrid_rerank`,从输出/临时日志看 32 题 rerank Top-1 分数分布,确认 0.30 能把「库外题」与「库内题」分开;偏了就调 `config.rerank_score_floor` 默认值并在 dev-notes 记录定稿依据。
1. `eval_retrieval.py --strategy all` 出四策略 × 四桶数字(Recall@3/5/10 + MRR@10),报告落 `reports/`;
2. `--strategy bm25` 下 B 桶(如 B01「SH-E300 支持主动降噪吗」)命中;聊天页问「SH-E300 多少钱」答 299;
3. 聊天页「邮费是多少」:角标可点、来源卡显示 `退货政策.md > 邮费与运费` 与原文;
4. 问「会飞的手机怎么买」:回答 `【无法回答】` 开头,`docker exec <mysql> mysql -ushophelper -pshophelper shophelper -e "select source, raw_question from low_confidence_questions"` 可见该题;
   另验 `self_check` 入口:构造证据在但答不出的场景(如「量子力学怎么退货」)看 source 区分。

- [ ] **Step 3: Code Review(requesting-code-review 技能,subagent 全量评审)+ 修复波**

- [ ] **Step 4: README/pyproject 更新 + dev-notes finish 段(演示命令/测试结果/交付路径)+ commit**

Commit: `ch04: README/pyproject 收口 + 真机终验收 + finish 留痕`
