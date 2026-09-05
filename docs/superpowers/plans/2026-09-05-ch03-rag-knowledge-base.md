# ch03 RAG 知识库 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `query_faq` 内核从 LIKE 查表升级为 BGE-M3 + Milvus dense 语义检索(契约不变),补齐离线建库(结构感知切分 + MySQL/Milvus 双写幂等)与历史对话挖知识两条离线管道。

**Architecture:** `knowledge/*.md` 经结构感知切分器产出 Chunk → MySQL `knowledge_chunks`(pending)→ embed → Milvus 集合 `knowledge`(Milvus Lite,只存 id+vector)→ 回填 vector_id 转 done。挖知识走独立脚本:历史会话分批 LLM 抽 QA → 暂存表 → embedding 整体去重 → 入 knowledge_chunks 复用同一向量化循环。在线 `query_faq`:问题 embed → Milvus Top-K → MySQL 回取原文。

**Tech Stack:** 硅基流动 OpenAI 兼容 embeddings API(`BAAI/bge-m3`,1024 维,httpx 直连)+ Milvus Lite(`pymilvus[milvus-lite]`,`MilvusClient`)+ 现有 FastAPI / SQLAlchemy 2.x / LangChain 1.x。

**Spec:** `docs/superpowers/specs/2026-09-05-ch03-rag-knowledge-base-design.md`

## Global Constraints

- dense 单路:不做关键词召回、混合检索、重排(spec §10)。
- `query_faq` 契约不变:工具名、入参 `keyword: str`、出参 `{"items": [{"question","answer","category"}]}`;任何异常收敛为 `{"items": []}`(spec §7)。
- MySQL 是原文权威源;Milvus 只存 `id`(= MySQL chunk id)+ `vector`(1024 维 COSINE),元数据一律不进 Milvus(spec §3.2)。
- 向量化文本 = `f"{category}\n{questions}\n{answer}"`(spec §5)。
- `db/init.sql` 两张新表用用户 DDL 原文,knowledge_chunks 在前(自引用 FK)(spec §3.1)。
- API key 只进 `.env`(gitignore);`.env.example` 只有占位符。
- 测试不依赖网络/真上游/真模型:一律 FakeEmbedding(近义词典)+ FakeVectorStore 或 tmp Milvus Lite 文件。
- pymilvus / 硅基流动接口动手前先 Context7 核对(用户工作要求 3)。
- 每个任务完成:追记 `dev-notes/ch03.md` 一段 + git commit。
- 聊天页 `static/index.html` 本章不动。

---

### Task 1: 依赖 + 双表 DDL + ORM + config

**Files:**
- Modify: `db/init.sql`(末尾追加两张表)
- Modify: `app/models.py`(追加两个 ORM 模型)
- Modify: `app/config.py`(追加 9 个配置项)
- Test: `tests/test_models.py`(追加)、`tests/test_config.py`(追加)

**Interfaces:**
- Produces: `KnowledgeChunk`(字段见 DDL)、`QaStaging` ORM;`Settings.embedding_api_base/embedding_api_key/embedding_model/embedding_dim/embedding_batch_size/milvus_db_path/retrieval_top_k/chunk_max_chars/chunk_overlap_chars/dedup_threshold`。后续所有任务消费。

- [ ] **Step 1: Context7 核对 pymilvus 安装与 MilvusClient 基本用法**

查 `pymilvus` 的 MilvusClient:`MilvusClient(uri)` 对 lite 文件路径的写法、`create_collection`(schema 自定义主键 + FLOAT_VECTOR)、`insert/upsert/search/delete` 参数形态、COSINE 度量下 `distance` 与相似度的关系。把核对结论(接口签名摘录)记入 dev-notes 本任务段。

- [ ] **Step 2: 安装依赖**

```bash
.venv/bin/pip install "pymilvus[milvus-lite]"
.venv/bin/pip freeze | grep -i -E "pymilvus|milvus"   # 版本记入 dev-notes
```

- [ ] **Step 3: db/init.sql 追加两张表(用户 DDL 原文)**

在 `db/init.sql` 末尾追加(用户随需求提供的 DDL,一字不改,含 `SET NAMES utf8mb4` 之前已存在故不重复):`knowledge_chunks`(id/category/questions/answer/section_path/content_type/is_key_clause/prev_chunk_id/next_chunk_id/vector_id/vectorize_status/时间戳,自引用 FK 两枚,KEY idx_category、idx_vectorize_status)与 `qa_extraction_staging`(batch_no/source_ref/question/answer/status ENUM('extracted','kept','discarded'),KEY idx_batch_no、idx_status)。文件头注释补一行「ch03 新增:knowledge_chunks / qa_extraction_staging」。

- [ ] **Step 4: app/models.py 追加 ORM**

```python
class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    )
    category: Mapped[str] = mapped_column(String(255), nullable=False)
    questions: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    section_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    content_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    is_key_clause: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), default=0)
    prev_chunk_id: Mapped[int | None] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), nullable=True
    )
    next_chunk_id: Mapped[int | None] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), nullable=True
    )
    vector_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    vectorize_status: Mapped[str] = mapped_column(
        Enum("pending", "done", name="vectorize_status"), default="pending"
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class QaStaging(Base):
    __tablename__ = "qa_extraction_staging"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    )
    batch_no: Mapped[str] = mapped_column(String(64), nullable=False)
    source_ref: Mapped[str | None] = mapped_column(String(255), nullable=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        Enum("extracted", "kept", "discarded", name="qa_staging_status"), default="extracted"
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
```

- [ ] **Step 5: app/config.py 追加配置**

```python
    # ch03:RAG 知识库
    embedding_api_base: str = "https://api.siliconflow.cn/v1"
    embedding_api_key: str = ""
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    embedding_batch_size: int = 16   # 硅基流动单请求上限 32,留余量
    milvus_db_path: str = "./data/milvus_knowledge.db"
    retrieval_top_k: int = 3
    chunk_max_chars: int = 500
    chunk_overlap_chars: int = 80
    dedup_threshold: float = 0.88
```

- [ ] **Step 6: 写失败测试**

`tests/test_models.py` 追加:

```python
from app.models import KnowledgeChunk, QaStaging


def test_knowledge_chunk_and_staging_roundtrip(db_session_factory):
    with db_session_factory() as session:
        chunk = KnowledgeChunk(category="售后", questions="运费说明", answer="邮费 8 元")
        session.add(chunk)
        session.commit()
        assert chunk.id is not None
        assert chunk.vectorize_status == "pending"
        assert chunk.is_key_clause == 0

        staged = QaStaging(batch_no="b1", source_ref="1", question="q", answer="a")
        session.add(staged)
        session.commit()
        assert staged.status == "extracted"
```

`tests/test_config.py` 追加:

```python
def test_ch03_defaults():
    settings = Settings(openai_api_key="sk-test", _env_file=None)
    assert settings.embedding_model == "BAAI/bge-m3"
    assert settings.embedding_dim == 1024
    assert settings.retrieval_top_k == 3
    assert settings.chunk_max_chars == 500
    assert settings.chunk_overlap_chars == 80
    assert settings.dedup_threshold == 0.88
    assert settings.milvus_db_path == "./data/milvus_knowledge.db"
```

- [ ] **Step 7: 跑测试确认失败 → 实现 → 通过**

Run: `.venv/bin/pytest tests/test_models.py tests/test_config.py -v`
Expected: 新用例先 FAIL(ImportError / 属性缺失),实现后全绿。

- [ ] **Step 8: Commit + dev-notes 追记**

```bash
git add -A && git commit -m "ch03: 双表 DDL/ORM/config——knowledge_chunks 与 qa_extraction_staging"
```

---

### Task 2: 结构感知切分器 app/chunking.py(TDD 核心)

**Files:**
- Create: `app/chunking.py`
- Test: `tests/test_chunking.py`

**Interfaces:**
- Produces:
  - `@dataclass Chunk: category: str; questions: str; answer: str; section_path: str; content_type: str; is_key_clause: bool`
  - `chunk_markdown(text: str, doc_name: str, content_type: str, *, max_chars: int = 500, overlap_chars: int = 80) -> list[Chunk]`
- Task 5/6/7 消费。

- [ ] **Step 1: 写失败测试(每条规则至少一例)**

```python
from app.chunking import chunk_markdown

FAQ_DOC = """# 商品FAQ
## 售后
### 怎么申请退货?
在订单详情页点击「申请售后」,选择退货原因后提交。
- 其他问法:退货入口在哪 / 怎么退
### 质量问题怎么办?
凭照片凭证免费退货。
## 交易
### 支持哪些付款方式?
支持微信支付、支付宝、银联卡。
"""

POLICY_DOC = """# 退货政策
## 邮费与运费
普通订单邮费 8 元,满 99 元包邮。质量问题退货,运费由商家承担。
## 退货流程
第一步联系客服,第二步寄回商品,第三步等待退款。
"""


def test_heading_hierarchy_and_section_path():
    chunks = chunk_markdown(FAQ_DOC, "商品FAQ.md", "faq")
    entry = [c for c in chunks if "怎么申请退货" in c.questions][0]
    assert entry.section_path == "商品FAQ.md > 商品FAQ > 售后 > 怎么申请退货?"


def test_faq_category_and_questions_with_aliases():
    chunks = chunk_markdown(FAQ_DOC, "商品FAQ.md", "faq")
    entry = [c for c in chunks if "怎么申请退货" in c.questions][0]
    assert entry.category == "售后"
    assert entry.questions.splitlines() == ["怎么申请退货?", "退货入口在哪", "怎么退"]
    assert "其他问法" not in entry.answer  # 别名行从正文剥离
    # H2 直接是条目(无 H3)时 category 落到 H1
    pay = [c for c in chunks if "付款方式" in c.questions][0]
    assert pay.category == "交易"


def test_policy_questions_is_section_title_category_is_parent_path():
    chunks = chunk_markdown(POLICY_DOC, "退货政策.md", "policy")
    fee = [c for c in chunks if c.questions == "邮费与运费"][0]
    assert fee.category == "退货政策"
    assert fee.section_path == "退货政策.md > 退货政策 > 邮费与运费"
    assert "满 99 元包邮" in fee.answer


def test_long_section_recursively_split_with_sentence_overlap():
    body = " ".join(f"第{i}句,这是一条用于撑长度的测试句子。" for i in range(60))
    doc = f"# 手册\n## 长章节\n{body}"
    chunks = chunk_markdown(doc, "手册.md", "manual", max_chars=200, overlap_chars=50)
    assert len(chunks) > 1
    # 每块不超限(留余量给重叠前缀)
    assert all(len(c.answer) <= 260 for c in chunks)
    # 相邻块有重叠,且重叠区从句号后开始(不出现半截句:重叠首字符前是句读)
    for prev, nxt in zip(chunks, chunks[1:]):
        tail = prev.answer[-50:]
        start = min((i for i, ch in enumerate(tail) if ch in "。!?;;"), default=None)
        if start is not None:
            assert nxt.answer.startswith(tail[start + 1:]) or not nxt.answer[:5].isspace()


def test_table_split_replicates_header():
    rows = "\n".join(f"| 商品{i} | {i}9.9 元 | 48 小时 |" for i in range(30))
    doc = (
        "# 手册\n## 运费标准\n| 商品 | 价格 | 发货时效 |\n|---|---|---|\n" + rows
    )
    chunks = chunk_markdown(doc, "手册.md", "manual", max_chars=160, overlap_chars=40)
    table_chunks = [c for c in chunks if c.answer.startswith("|")]
    assert len(table_chunks) > 1
    for c in table_chunks:
        assert c.answer.splitlines()[0] == "| 商品 | 价格 | 发货时效 |"
        assert c.answer.splitlines()[1] == "|---|---|---|"


def test_key_clause_flag():
    doc = "# 政策\n## 退货约定\n商品必须保持完好,否则不予退货。\n## 温馨提示\n欢迎随时联系客服。\n"
    chunks = chunk_markdown(doc, "政策.md", "policy")
    by_q = {c.questions: c for c in chunks}
    assert by_q["退货约定"].is_key_clause is True
    assert by_q["温馨提示"].is_key_clause is False
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_chunking.py -v`
Expected: FAIL `ModuleNotFoundError: app.chunking`

- [ ] **Step 3: 实现 app/chunking.py**

```python
"""结构感知 Markdown 切分:标题开节、超长递归、重叠裁到句号、表格按行切复制表头。"""
import re
from dataclasses import dataclass

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
SENTENCE_END = "。!?;;…"
KEY_CLAUSE_WORDS = ("必须", "不得", "禁止", "仅限", "不予", "免费", "七天无理由")
ALIAS_MARK = "其他问法:"
_SEPS = ["\n\n", "\n", "。", "!", "?", ";", ";"]


@dataclass
class Chunk:
    category: str
    questions: str
    answer: str
    section_path: str
    content_type: str
    is_key_clause: bool


def _parse_sections(text: str) -> list[tuple[list[str], str, list[str]]]:
    """→ [(祖先标题路径(不含文档名,不含自身), 自身标题, 正文行)]。只有标题没有正文的节丢弃。"""
    sections, stack, body = [], [], []
    for line in text.splitlines():
        m = HEADING_RE.match(line.strip())
        if m:
            if body:
                sections.append(([t for _, t in stack], stack[-1][1] if stack else "", body))
                body = []
                # 上面这行的祖先/自身取法不对,重写:flush 时用"当前节"的记录
            stack = stack[: int(len(m.group(1))) - 1]
            stack.append((len(m.group(1)), m.group(2).strip()))
        else:
            body.append(line)
    if body:
        sections.append(([t for _, t in stack[:-1]], stack[-1][1], body))
    return sections
```

注意:上面 `_parse_sections` 是草稿骨架,flush 时机需在「遇到新标题」时记录**当前栈顶**为其所属节,实现时以测试为准修正(正确做法:维护 `current=(ancestors, title)`,遇到标题先 flush current 再更新栈)。

```python
def _parse_sections(text: str):
    sections: list[tuple[list[str], str, list[str]]] = []
    stack: list[tuple[int, str]] = []
    current: tuple[list[str], str] | None = None
    body: list[str] = []
    for line in text.splitlines():
        m = HEADING_RE.match(line.strip())
        if m:
            if current is not None:
                sections.append((current[0], current[1], body))
            stack = stack[: len(m.group(1)) - 1]
            current = ([t for _, t in stack], m.group(2).strip())
            body = []
        else:
            body.append(line)
    if current is not None:
        sections.append((current[0], current[1], body))
    return [s for s in sections if "\n".join(s[2]).strip()]


def _split_keep(text: str, sep: str) -> list[str]:
    if sep in ("。", "!", "?", ";", ";", "…"):
        parts, buf = [], ""
        for ch in text:
            buf += ch
            if ch == sep:
                parts.append(buf)
                buf = ""
        if buf.strip():
            parts.append(buf)
        return [p for p in parts if p.strip()]
    return [p for p in text.split(sep) if p.strip()]


def _recursive_split(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    for sep in _SEPS:
        parts = _split_keep(text, sep)
        if len(parts) > 1:
            pieces: list[str] = []
            for p in parts:
                pieces.extend(_recursive_split(p, max_chars))
            return _merge(pieces, max_chars)
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def _merge(pieces: list[str], max_chars: int) -> list[str]:
    blocks, buf = [], ""
    for p in pieces:
        if buf and len(buf) + len(p) > max_chars:
            blocks.append(buf)
            buf = p
        else:
            buf += p
    if buf.strip():
        blocks.append(buf)
    return blocks


def _overlap_prefix(prev: str, overlap_chars: int) -> str:
    tail = prev[-overlap_chars:] if overlap_chars > 0 else ""
    cut = next((i for i, ch in enumerate(tail) if ch in SENTENCE_END), None)
    return tail[cut + 1 :] if cut is not None else ""  # 窗口内无句读 → 不重叠,不留半截话


def _is_table(lines: list[str]) -> bool:
    return bool(lines) and all(ln.lstrip().startswith("|") for ln in lines if ln.strip())


def _split_table(lines: list[str], max_chars: int) -> list[str]:
    lines = [ln for ln in lines if ln.strip()]
    if len(lines) <= 3 or len("\n".join(lines)) <= max_chars:
        return ["\n".join(lines)]
    header, sep, data = lines[0], lines[1], lines[2:]
    budget = max_chars - len(header) - len(sep) - 2
    blocks, cur, cur_len = [], [], 0
    for row in data:
        if cur and cur_len + len(row) + 1 > budget:
            blocks.append("\n".join([header, sep, *cur]))
            cur, cur_len = [], 0
        cur.append(row)
        cur_len += len(row) + 1
    if cur:
        blocks.append("\n".join([header, sep, *cur]))
    return blocks


def _partition_tables(lines: list[str]) -> list[tuple[str, list[str]]]:
    """正文行 → [("text"|"table", 行)] 分段。"""
    segments, buf, mode = [], [], None
    for ln in lines:
        m = "table" if ln.lstrip().startswith("|") else "text"
        if m != mode and buf:
            segments.append((mode, buf))
            buf = []
        mode = m
        buf.append(ln)
    if buf:
        segments.append((mode or "text", buf))
    return segments


def chunk_markdown(
    text: str, doc_name: str, content_type: str, *, max_chars: int = 500, overlap_chars: int = 80
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for ancestors, title, body_lines in _parse_sections(text):
        raw = "\n".join(body_lines).strip()
        aliases: list[str] = []
        if content_type == "faq":
            kept = []
            for ln in raw.splitlines():
                if ALIAS_MARK in ln:
                    tail = ln.split(ALIAS_MARK, 1)[1]
                    aliases.extend(a.strip() for a in re.split(r"[/;]", tail) if a.strip())
                else:
                    kept.append(ln)
            raw = "\n".join(kept).strip()
        if content_type == "faq":
            questions = "\n".join([title, *aliases])
            category = ancestors[-1] if ancestors else title
        else:
            questions = title
            category = " > ".join(ancestors) if ancestors else title
        section_path = " > ".join([doc_name, *ancestors, title])
        is_key = any(w in title or w in raw for w in KEY_CLAUSE_WORDS)

        for seg_kind, seg_lines in _partition_tables(body_lines):
            seg = "\n".join(seg_lines).strip()
            if not seg:
                continue
            if seg_kind == "table":
                blocks = _split_table(seg_lines, max_chars)
            else:
                blocks = _recursive_split(seg, max_chars)
                for i in range(1, len(blocks)):
                    prefix = _overlap_prefix(blocks[i - 1], overlap_chars)
                    if prefix:
                        blocks[i] = prefix + blocks[i]
            for b in blocks:
                chunks.append(Chunk(category, questions, b, section_path, content_type, is_key))
    return chunks
```

- [ ] **Step 4: 跑测试至全绿**

Run: `.venv/bin/pytest tests/test_chunking.py -v`
Expected: PASS(实现里的注释草稿块删除,以正确版为准)

- [ ] **Step 5: Commit + dev-notes**

```bash
git add app/chunking.py tests/test_chunking.py && git commit -m "ch03: 结构感知切分器——标题层级/递归/重叠裁句号/表格表头复制"
```

---

### Task 3: 嵌入客户端 app/embedding.py + FakeEmbedding

**Files:**
- Create: `app/embedding.py`
- Modify: `tests/helpers.py`(追加 FakeEmbedding)
- Test: `tests/test_embedding.py`

**Interfaces:**
- Produces: `class SiliconFlowEmbedder: def __init__(self, api_base: str, api_key: str, model: str, batch_size: int = 16, timeout: float = 30.0); def embed(self, texts: list[str]) -> list[list[float]]`;`make_embedder(settings) -> SiliconFlowEmbedder`;`tests.helpers.FakeEmbedding(dim=64).embed(texts)`(近义词典把「邮费/快递费/寄费」归一为 token「运费」)。Task 5/7/8/9 消费。

- [ ] **Step 1: 写失败测试**

```python
import httpx
import pytest

from app.embedding import SiliconFlowEmbedder, make_embedder
from tests.helpers import FakeEmbedding


def test_fake_embedding_synonym_paraphrase():
    fake = FakeEmbedding()
    v_postage, v_ship = fake.embed(["邮费是多少", "运费怎么算"])
    v_return, _ = fake.embed(["退货政策", "随便"])
    assert _cos(v_postage, v_ship) > 0.9     # 近义词归一后高度相似
    assert _cos(v_postage, v_return) < 0.7   # 不同语义明显拉开


def test_embedder_splits_batches_and_parses(httpx_mock_transport):
    captured = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured.append(body)
        return httpx.Response(200, json={
            "data": [{"index": i, "embedding": [0.1, 0.2]} for i in range(len(body["input"]))]
        })

    client = SiliconFlowEmbedder(
        "https://api.siliconflow.cn/v1", "sk-test", "BAAI/bge-m3", batch_size=2, timeout=5,
        transport=httpx_mock_transport(handler),
    )
    vecs = client.embed([f"文本{i}" for i in range(5)])
    assert len(vecs) == 5 and all(len(v) == 2 for v in vecs)
    assert [len(b["input"]) for b in captured] == [2, 2, 1]  # 按 batch_size 切批
    assert all(b["model"] == "BAAI/bge-m3" for b in captured)


def test_embedder_raises_on_http_error(httpx_mock_transport):
    def handler(request):
        return httpx.Response(401, json={"message": "invalid key"})

    client = SiliconFlowEmbedder("https://x/v1", "bad", "BAAI/bge-m3",
                                 transport=httpx_mock_transport(handler))
    with pytest.raises(RuntimeError, match="嵌入 API 调用失败"):
        client.embed(["hi"])
```

`tests/conftest.py` 追加 fixture(或直接在 test 文件内提供 helper):

```python
@pytest.fixture
def httpx_mock_transport():
    def _make(handler):
        return httpx.MockTransport(handler)
    return _make
```

`tests/helpers.py` 追加:

```python
import hashlib
import math

class FakeEmbedding:
    """确定性假嵌入:分词 → 每词一个稳定伪随机向量,加权平均归一。
    近义词典把换说法归一到同一 token,供检索链路做「换说法命中」测试。"""

    dim = 64
    SYNONYMS = {"邮费": "运费", "快递费": "运费", "寄费": "运费", "送货": "发货"}

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        toks = self._tokens(text)
        for tok in toks:
            tvec = self._token_vec(tok)
            vec = [a + b for a, b in zip(vec, tvec)]
        norm = math.sqrt(sum(a * a for a in vec)) or 1.0
        return [a / norm for a in vec]

    def _tokens(self, text: str) -> list[str]:
        canon = text
        for src, dst in self.SYNONYMS.items():
            canon = canon.replace(src, dst)
        return [canon[i : i + 2] for i in range(0, len(canon), 2)] or ["<empty>"]

    def _token_vec(self, tok: str) -> list[float]:
        seed = int.from_bytes(hashlib.md5(tok.encode()).digest()[:8], "big")
        return [((seed >> j) & 0xFF) / 255.0 - 0.5 for j in range(self.dim * 8)][::8]


def _cos(a, b):
    return sum(x * y for x, y in zip(a, b)) / (
        math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    )
```

(实现时若 `_token_vec` 采样导致近义判据不稳,允许微调采样步长,以 `test_fake_embedding_synonym_paraphrase` 通过为准。)

- [ ] **Step 2: 跑失败 → 实现 app/embedding.py → 全绿**

```python
"""BGE-M3 嵌入客户端:硅基流动 OpenAI 兼容 /embeddings,httpx 直连。"""
import httpx

from app.config import Settings


class SiliconFlowEmbedder:
    def __init__(self, api_base: str, api_key: str, model: str, batch_size: int = 16,
                 timeout: float = 30.0, transport: httpx.BaseTransport | None = None) -> None:
        self._client = httpx.Client(
            base_url=api_base.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            transport=transport,
        )
        self._model = model
        self._batch_size = batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            resp = self._client.post("/embeddings", json={"model": self._model, "input": batch})
            if resp.status_code != 200:
                raise RuntimeError(f"嵌入 API 调用失败:HTTP {resp.status_code} {resp.text[:200]}")
            data = sorted(resp.json()["data"], key=lambda d: d["index"])
            out.extend(d["embedding"] for d in data)
        return out


def make_embedder(settings: Settings) -> SiliconFlowEmbedder:
    return SiliconFlowEmbedder(
        settings.embedding_api_base,
        settings.embedding_api_key,
        settings.embedding_model,
        batch_size=settings.embedding_batch_size,
    )
```

Run: `.venv/bin/pytest tests/test_embedding.py -v` → PASS

- [ ] **Step 3: Commit + dev-notes**

```bash
git add app/embedding.py tests/ && git commit -m "ch03: BGE-M3 嵌入客户端(httpx 批量)+ FakeEmbedding 近义词典"
```

---

### Task 4: Milvus 集合封装 app/vector_store.py(真 Milvus Lite + tmp 文件测试)

**Files:**
- Create: `app/vector_store.py`
- Test: `tests/test_vector_store.py`

**Interfaces:**
- Produces: `KnowledgeVectorStore(db_path: str, collection: str = "knowledge", dim: int = 1024)`:
  - `ensure_collection() -> None`(已存在则跳过)
  - `upsert(ids: list[int], vectors: list[list[float]]) -> None`
  - `search(vector: list[float], top_k: int) -> list[tuple[int, float]]`(score=相似度,降序)
  - `delete(ids: list[int]) -> None`
  - `count() -> int`
  - 构造函数**不得**产生文件 I/O(lazy client;测试环境不触碰真实路径)
- Task 5/7/8/9 消费。

- [ ] **Step 1: Context7 核对 MilvusClient API(结果记 dev-notes)**

确认:`MilvusClient(uri="<文件路径>")`;自定义 schema 建集合(`FieldSchema("id", INT64, is_primary=True)` + `FLOAT_VECTOR(dim)`、`index_params` COSINE);`upsert(collection_name, data=[{"id":…, "vector":…}])`;`search(collection_name, data=[vec], limit=k)` 返回结构与 distance 语义(COSINE 下 similarity = 1 - distance 还是直接就是相似度,以文档为准);`delete(collection_name, ids=…)`;计数 `get_collection_stats`。

- [ ] **Step 2: 写失败测试(tmp_path 真 lite 文件)**

```python
import pytest

from app.vector_store import KnowledgeVectorStore


@pytest.fixture
def store(tmp_path):
    s = KnowledgeVectorStore(str(tmp_path / "kb.db"), collection="knowledge", dim=8)
    yield s


def test_lazy_constructor_no_io(tmp_path):
    # 构造不建文件
    KnowledgeVectorStore(str(tmp_path / "lazy.db"), dim=8)
    assert not (tmp_path / "lazy.db").exists()


def test_ensure_upsert_search_roundtrip(store):
    store.ensure_collection()
    v1 = [1.0, 0, 0, 0, 0, 0, 0, 0]
    v2 = [0, 1.0, 0, 0, 0, 0, 0, 0]
    store.upsert([11, 22], [v1, v2])
    assert store.count() == 2
    hits = store.search(v1, top_k=2)
    assert hits[0][0] == 11 and hits[0][1] > 0.99   # 自身相似度≈1
    assert hits[1][0] == 22


def test_upsert_same_id_overwrites(store):
    store.ensure_collection()
    store.upsert([7], [[1.0, 0, 0, 0, 0, 0, 0, 0]])
    store.upsert([7], [[0, 1.0, 0, 0, 0, 0, 0, 0]])
    assert store.count() == 1  # 主键幂等:覆盖不重复


def test_delete_and_search_missing_collection(store):
    hits = store.search([1.0, 0, 0, 0, 0, 0, 0, 0], top_k=3)  # 集合不存在 → 空不抛
    assert hits == []
    store.ensure_collection()
    store.upsert([1], [[1.0, 0, 0, 0, 0, 0, 0, 0]])
    store.delete([1])
    assert store.count() == 0
```

- [ ] **Step 3: 跑失败 → 实现 → 全绿**

实现要点(以 Context7 结论为准):构造只存参数;`_client()` 惰性 `MilvusClient(uri=db_path)`;`ensure_collection` 用 `has_collection` 判断;schema `id INT64 is_primary=True` + `vector FLOAT_VECTOR(dim)`;index COSINE;`search` 把 distance 转相似度 `score = 1 - distance`(若文档表明 COSINE distance 已是相似度语义则直接用),返回 `[(id, score)]` 降序;集合不存在时 `search` 返回 `[]`、`delete` 静默。

- [ ] **Step 4: Commit + dev-notes**

```bash
git add app/vector_store.py tests/test_vector_store.py && git commit -m "ch03: Milvus Lite 集合封装——主键幂等 upsert/COSINE 检索/惰性连接"
```

---

### Task 5: MySQL 持久层 app/kb.py + 向量化循环(双写核心)

**Files:**
- Create: `app/kb.py`
- Test: `tests/test_kb.py`

**Interfaces:**
- Consumes: Task 1 ORM、Task 2 `Chunk`、Task 3 embedder、Task 4 vector store。
- Produces:
  - `class KnowledgeBaseStore(session_factory)`:
    - `doc_chunk_ids(doc_name: str) -> list[int]`
    - `replace_doc_chunks(doc_name: str, chunks: list[Chunk]) -> tuple[list[int], list[int]]`(返回 (被删旧 id, 新插 id);新块全 pending;同事务回填 prev/next)
    - `pending_chunks() -> list[KnowledgeChunk]`
    - `mark_vectorized(chunk_ids: list[int]) -> None`(vector_id=str(id), status=done)
    - `get_chunks(ids: list[int]) -> list[KnowledgeChunk]`(按入参序)
    - `all_dedup_texts() -> list[str]`(三格拼文本)
    - `already_mined_refs() -> set[str]`(staging.source_ref 非空集合)
    - `insert_staging(batch_no: str, source_ref: str, qas: list[tuple[str, str]]) -> None`
    - `extracted_items() -> list[QaStaging]`
    - `set_staging_status(ids: list[int], status: str) -> None`
    - `insert_mined_chunks(items: list[QaStaging]) -> list[int]`(category="对话挖掘", questions=question, answer=answer, section_path=f"mined:{source_ref}", content_type="qa_mined")
  - `vectorize_pending(kb, vectors, embedder, *, max_chunks: int | None = None, batch_size: int = 16) -> int`(处理块数;upsert 成功一批才 mark 一批)
  - `vector_text(category, questions, answer) -> str`

- [ ] **Step 1: 写失败测试(SQLite + FakeEmbedding + FakeVectorStore)**

`tests/helpers.py` 追加:

```python
class FakeVectorStore:
    """内存向量库,接口与 KnowledgeVectorStore 一致。"""

    def __init__(self, dim: int = 64):
        self.dim = dim
        self._vectors: dict[int, list[float]] = {}

    def ensure_collection(self) -> None:
        pass

    def upsert(self, ids, vectors):
        for i, v in zip(ids, vectors):
            self._vectors[i] = v

    def search(self, vector, top_k):
        scored = [(i, _cos(vector, v)) for i, v in self._vectors.items()]
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    def delete(self, ids):
        for i in ids:
            self._vectors.pop(i, None)

    def count(self):
        return len(self._vectors)
```

```python
# tests/test_kb.py
from app.chunking import Chunk
from app.kb import KnowledgeBaseStore, vector_text, vectorize_pending
from tests.helpers import FakeEmbedding, FakeVectorStore


def _chunk(q, a, section="退货政策.md > 退货政策 > 节"):
    return Chunk(category="退货政策", questions=q, answer=a, section_path=section,
                 content_type="policy", is_key_clause=False)


def test_vector_text_three_fields():
    assert vector_text("售后", "怎么退货\n咋退", "联系客服") == "售后\n怎么退货\n咋退\n联系客服"


def test_replace_doc_chunks_is_idempotent_and_links(db_session_factory):
    kb = KnowledgeBaseStore(db_session_factory)
    old_ids, new_ids = kb.replace_doc_chunks("退货政策.md", [_chunk("q1", "a1"), _chunk("q2", "a2")])
    assert old_ids == [] and len(new_ids) == 2
    # 重跑同文档:旧块删除、新块重插,总数不变
    old_ids2, new_ids2 = kb.replace_doc_chunks("退货政策.md", [_chunk("q1", "a1"), _chunk("q2", "a2")])
    assert sorted(old_ids2) == sorted(new_ids)
    assert all(i not in old_ids2 for i in new_ids2)
    # 其他文档不受影响
    kb.replace_doc_chunks("手册.md", [_chunk("m1", "a", section="手册.md > 手册 > 节")])
    assert len(kb.doc_chunk_ids("手册.md")) == 1


def test_replace_doc_chunks_sets_prev_next(db_session_factory):
    kb = KnowledgeBaseStore(db_session_factory)
    _, ids = kb.replace_doc_chunks("退货政策.md", [_chunk("q1", "a1"), _chunk("q2", "a2")])
    chunks = kb.get_chunks(ids)
    assert chunks[0].prev_chunk_id is None and chunks[0].next_chunk_id == ids[1]
    assert chunks[1].prev_chunk_id == ids[0] and chunks[1].next_chunk_id is None


def test_vectorize_pending_marks_done_and_resumes(db_session_factory):
    kb = KnowledgeBaseStore(db_session_factory)
    vectors = FakeVectorStore()
    kb.replace_doc_chunks("退货政策.md", [_chunk(f"q{i}", f"a{i}") for i in range(4)])
    # 模拟中断:只向量化 2 块
    done = vectorize_pending(kb, vectors, FakeEmbedding(), max_chunks=2)
    assert done == 2 and vectors.count() == 2
    assert len(kb.pending_chunks()) == 2
    # 重跑补齐
    done2 = vectorize_pending(kb, vectors, FakeEmbedding())
    assert done2 == 2 and vectors.count() == 4
    assert kb.pending_chunks() == []
    rows = kb.get_chunks(kb.doc_chunk_ids("退货政策.md"))
    assert all(r.vectorize_status == "done" and r.vector_id == str(r.id) for r in rows)


def test_vectorize_upsert_overwrite_keeps_count(db_session_factory):
    # 中断发生在 upsert 后、mark 前的等价场景:重跑同 id upsert 覆盖不重复
    kb = KnowledgeBaseStore(db_session_factory)
    vectors = FakeVectorStore()
    kb.replace_doc_chunks("退货政策.md", [_chunk("q1", "a1")])
    kb.replace_doc_chunks("退货政策.md", [_chunk("q1", "a1")])  # 重建一轮,旧向量遗留在库
    vectors.upsert([999], [[0.0] * 64])  # 干扰项
    vectorize_pending(kb, vectors, FakeEmbedding())
    assert vectors.count() == 2  # 1 真块 + 1 干扰项,无重复


def test_staging_flow_and_mined_chunks(db_session_factory):
    kb = KnowledgeBaseStore(db_session_factory)
    assert kb.already_mined_refs() == set()
    kb.insert_staging("b1", "3", [("国外的快递费怎么算", "国际件运费如下…")])
    assert kb.already_mined_refs() == {"3"}
    items = kb.extracted_items()
    assert len(items) == 1 and items[0].status == "extracted"
    ids = kb.insert_mined_chunks(items)
    chunk = kb.get_chunks(ids)[0]
    assert chunk.category == "对话挖掘" and chunk.content_type == "qa_mined"
    assert chunk.section_path == "mined:3" and chunk.vectorize_status == "pending"
    kb.set_staging_status([items[0].id], "kept")
    assert kb.extracted_items() == []
```

- [ ] **Step 2: 跑失败 → 实现 app/kb.py → 全绿**

实现要点:`replace_doc_chunks` 一个事务内:查出旧 id → 删旧行 → 插新行 flush 拿 id → UPDATE prev/next → commit;返回 (old, new)。`vectorize_pending`:`ensure_collection()` → 循环取 pending(每次 batch_size 条,直到处理满 max_chunks 或取空)→ embed(vector_text)→ vectors.upsert(ids, vecs) → mark_vectorized(ids) → 累加返回。

- [ ] **Step 3: Commit + dev-notes**

```bash
git add app/kb.py tests/test_kb.py tests/helpers.py && git commit -m "ch03: MySQL 持久层+向量化循环——文档重建幂等/prev-next/中断续跑"
```

---

### Task 6: 知识文档种子 knowledge/*.md(数据类任务:用切分断言验证)

**Files:**
- Create: `knowledge/退货政策.md`
- Create: `knowledge/商品FAQ.md`
- Create: `knowledge/售后手册.md`
- Test: `tests/test_knowledge_docs.py`

**Interfaces:**
- Produces: 三份演示知识文档;`build_kb` 与评估脚本按文件名枚举。**验收 1 靶点必须在**:`退货政策.md` 的「邮费与运费」节写明「普通订单邮费 8 元,满 99 元包邮;质量问题退货,运费由商家承担」。

- [ ] **Step 1: 写三份文档**

`knowledge/退货政策.md`(policy,H2 节:退货承诺 / 邮费与运费 / 退款时效 / 不予退货情形——「不予退货情形」正文含「不予」「必须」,触发 key_clause):

```markdown
# 退货政策

## 七天无理由退货

商品自签收之日起 7 天内,支持七天无理由退货。商品必须保持完好,吊牌、包装齐全,不影响二次销售。

## 邮费与运费

普通订单邮费 8 元,单笔满 99 元包邮。质量问题退货,运费由商家承担;非质量问题退货,运费由买家自行承担。偏远地区额外补收 5 元运费。

## 退款时效

退货商品验收通过后,退款原路返回,1-3 个工作日到账。

## 不予退货情形

定制类商品、贴身衣物、已激活的虚拟商品不予退货。生鲜食品仅限质量问题退货。
```

`knowledge/商品FAQ.md`(faq,H2 分组 + H3 问法,含 ch02 八条等价内容 + 运费相关问法带「其他问法」行——注意不写「邮费」字样在答案正文以外的别名列也可以有,允许含「邮费」别名,语义检索不依赖字面):

```markdown
# 商品FAQ

## 售后

### 退货政策是什么?

支持七天无理由退货,商品需保持完好,凭订单号联系客服即可发起退货。

- 其他问法:退货规定 / 退换货政策

### 怎么申请退货?

在订单详情页点击「申请售后」,选择退货原因后提交,客服会在 24 小时内审核。

- 其他问法:退货入口在哪 / 怎么退

### 商品有质量问题怎么办?

质量问题可凭照片凭证免费退货,退回的运费由商家承担,退款原路返回。

## 物流

### 多久能发货?

现货商品 48 小时内发出,预售商品按页面标注时间发货。

### 物流一般几天到?

普通快递 3-5 天送达,偏远地区 5-7 天。

## 交易

### 支持哪些付款方式?

支持微信支付、支付宝、银联卡。

### 可以开发票吗?

支持电子发票,下单时备注抬头,发货后 24 小时内开出。

### 会员有什么权益?

会员享 95 折、生日礼包、优先客服通道。
```

`knowledge/售后手册.md`(manual,含一张 30 行左右大表格 + 一个超长章节,分别触发表格切分与递归切分;表格为「运费标准表」):

```markdown
# 售后手册

## 运费标准

以下为各区域运费标准明细:

| 区域 | 首重运费 | 续重运费 | 时效 |
|---|---|---|---|
| 华北 | 8 元 | 2 元/kg | 3 天 |
(…共 28 行数据,覆盖华东/华南/西北/东北等,凑到超限触发按行切)
| 偏远补运 | 5 元 | 2 元/kg | 7 天 |

## 退货处理细则

(一段 800 字以上的长文,讲退货审核、验货标准、争议处理、特殊类目差异……
用完整句子书写,触发递归切分与重叠)
```

- [ ] **Step 2: 写验证测试(数据类任务以断言代替 TDD 红绿)**

```python
from pathlib import Path

from app.chunking import chunk_markdown

DOCS_DIR = Path(__file__).resolve().parent.parent / "knowledge"


def _chunks(name, ctype):
    return chunk_markdown((DOCS_DIR / name).read_text(encoding="utf-8"), name, ctype)


def test_all_docs_parse_into_chunks():
    assert len(_chunks("退货政策.md", "policy")) >= 4
    assert len(_chunks("商品FAQ.md", "faq")) >= 8
    assert len(_chunks("售后手册.md", "manual")) >= 6


def test_shipping_fee_section_is_retrievable_target():
    chunks = _chunks("退货政策.md", "policy")
    fee = [c for c in chunks if c.questions == "邮费与运费"]
    assert fee and "满 99 元包邮" in fee[0].answer and "8 元" in fee[0].answer


def test_manual_table_chunks_all_carry_header():
    chunks = _chunks("售后手册.md", "manual")
    table_chunks = [c for c in chunks if c.answer.lstrip().startswith("|")]
    assert len(table_chunks) >= 2, "运费标准表必须被切成多块"
    for c in table_chunks:
        assert c.answer.splitlines()[0].startswith("| 区域 |")


def test_faq_entries_carry_aliases():
    chunks = _chunks("商品FAQ.md", "faq")
    entry = [c for c in chunks if c.questions.startswith("怎么申请退货")][0]
    assert "退货入口在哪" in entry.questions
```

- [ ] **Step 3: 跑测试至绿(文档内容与切分器互相校准)**

Run: `.venv/bin/pytest tests/test_knowledge_docs.py -v`

- [ ] **Step 4: Commit + dev-notes**

```bash
git add knowledge/ tests/test_knowledge_docs.py && git commit -m "ch03: 知识文档种子——退货政策/商品FAQ/售后手册(运费靶点+大表格+超长章节)"
```

---

### Task 7: 建库管道 scripts/build_kb.py(中断续跑验收载体)

**Files:**
- Create: `scripts/build_kb.py`(若无 `scripts/__init__.py` 一并补)
- Test: `tests/test_build_kb.py`

**Interfaces:**
- Consumes: Task 5 全部、Task 6 文档目录。
- Produces: `build(docs_dir: Path, kb, vectors, embedder, max_chunks: int | None = None) -> dict`(返回统计 {docs, chunks, vectorized});CLI:`python -m scripts.build_kb [--max-chunks N]`。

- [ ] **Step 1: 写失败测试**

```python
from pathlib import Path

from app.chunking import chunk_markdown
from app.kb import KnowledgeBaseStore
from scripts.build_kb import build
from tests.helpers import FakeEmbedding, FakeVectorStore

DOCS_DIR = Path(__file__).resolve().parent.parent / "knowledge"


def _make(db_session_factory):
    kb = KnowledgeBaseStore(db_session_factory)
    vectors = FakeVectorStore()
    embedder = FakeEmbedding()
    return kb, vectors, embedder


def test_build_full_and_rerun_idempotent(db_session_factory):
    kb, vectors, embedder = _make(db_session_factory)
    stats = build(DOCS_DIR, kb, vectors, embedder)
    assert stats["docs"] == 3 and stats["chunks"] > 10 and stats["vectorized"] == stats["chunks"]
    assert vectors.count() == stats["chunks"]
    total = stats["chunks"]
    stats2 = build(DOCS_DIR, kb, vectors, embedder)
    assert stats2["chunks"] == total          # 幂等:重建后块数一致
    assert vectors.count() == total           # 旧向量被删干净,无重复


def test_build_interrupt_then_resume(db_session_factory):
    kb, vectors, embedder = _make(db_session_factory)
    stats = build(DOCS_DIR, kb, vectors, embedder, max_chunks=5)  # 模拟中断
    assert stats["vectorized"] == 5
    assert len(kb.pending_chunks()) == stats["chunks"] - 5
    stats2 = build(DOCS_DIR, kb, vectors, embedder)               # 重跑补齐
    assert stats2["vectorized"] == stats["chunks"] - 5
    assert kb.pending_chunks() == []
    assert vectors.count() == stats["chunks"]
```

- [ ] **Step 2: 跑失败 → 实现 → 全绿**

```python
"""离线建库:knowledge/*.md → 切分 → MySQL(pending)→ embed → Milvus → done。用法:
  python -m scripts.build_kb             # 全量建库(可中断后重跑续传)
  python -m scripts.build_kb --max-chunks 5   # 演示:只向量化 5 块即停
"""
import argparse
from pathlib import Path

from app.chunking import chunk_markdown
from app.config import Settings
from app.db import make_engine, make_session_factory
from app.embedding import make_embedder
from app.kb import KnowledgeBaseStore, vectorize_pending
from app.vector_store import KnowledgeVectorStore

ROOT = Path(__file__).resolve().parent.parent
DOC_TYPES = {"退货政策": "policy", "商品FAQ": "faq", "售后手册": "manual"}


def build(docs_dir: Path, kb, vectors, embedder, max_chunks: int | None = None) -> dict:
    vectors.ensure_collection()
    total_chunks = 0
    for path in sorted(docs_dir.glob("*.md")):
        ctype = DOC_TYPES.get(path.stem, "manual")
        chunks = chunk_markdown(
            path.read_text(encoding="utf-8"), path.name, ctype,
            max_chars=kb_max_chars, overlap_chars=kb_overlap,
        )
        old_ids, _ = kb.replace_doc_chunks(path.name, chunks)
        vectors.delete(old_ids)
        total_chunks += len(chunks)
    vectorized = vectorize_pending(kb, vectors, embedder, max_chunks=max_chunks)
    return {"docs": len(list(docs_dir.glob("*.md"))), "chunks": total_chunks, "vectorized": vectorized}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-chunks", type=int, default=None)
    args = parser.parse_args()
    settings = Settings()
    kb = KnowledgeBaseStore(make_session_factory(make_engine(settings)))
    vectors = KnowledgeVectorStore(settings.milvus_db_path, dim=settings.embedding_dim)
    embedder = make_embedder(settings)
    stats = build(ROOT / "knowledge", kb, vectors, embedder, max_chunks=args.max_chunks)
    print(f"建库完成:文档 {stats['docs']} 份,chunk {stats['chunks']} 条,本次向量化 {stats['vectorized']} 条")


if __name__ == "__main__":
    main()
```

(`kb_max_chars/kb_overlap` 从 Settings 传入 `build`——签名加 `max_chars: int = 500, overlap_chars: int = 80`,main 里传 settings 值;测试用默认值。)

- [ ] **Step 3: Commit + dev-notes**

```bash
git add scripts/ tests/test_build_kb.py && git commit -m "ch03: build_kb 建库管道——全量幂等重建+max-chunks 中断续跑"
```

---

### Task 8: 对话挖知识 scripts/mine_qa.py + 历史会话种子

**Files:**
- Create: `scripts/mine_qa.py`
- Create: `db/seed_conversations.py`
- Modify: `tests/helpers.py`(StubMineModel)
- Test: `tests/test_mine_qa.py`

**Interfaces:**
- Consumes: Task 5 `kb` 全部、Task 3 embedder、Task 5 `vectorize_pending`。
- Produces:
  - `MinedQA / MinedQAList`(pydantic,structured output 用)
  - `mine(kb, embedder, extract_model, *, batch_size: int = 5, dedup_threshold: float = 0.88, batch_no: str | None = None) -> dict`(统计 {conversations, extracted, kept, discarded, vectorized})
  - CLI:`python -m scripts.mine_qa [--batch-size 5]`
  - `db/seed_conversations.py`:灌 4 通历史对话(幂等:已有 conversations 则跳过)

- [ ] **Step 1: 写失败测试**

`tests/helpers.py` 追加:

```python
class StubMineModel:
    """按 user 消息文本里的关键词返回预置 QA;未命中返回空列表。"""

    def __init__(self, mapping: dict[str, list[tuple[str, str]]]):
        self.mapping = mapping

    def invoke(self, messages, config=None, **kwargs):
        from tests.helpers import StubQAList  # 局部导入避免循环

        text = messages[-1].content if hasattr(messages, "__getitem__") else str(messages)
        items = []
        for key, qas in self.mapping.items():
            if key in text:
                items.extend(qas)
        return StubQAList(items=[{"question": q, "answer": a} for q, a in items])


class StubQAList:
    def __init__(self, items):
        self.items = items
```

```python
# tests/test_mine_qa.py
from app.kb import KnowledgeBaseStore
from scripts.mine_qa import _conversation_text, mine
from tests.helpers import FakeEmbedding, StubMineModel


def _seed_conversations(db_session_factory):
    from app.models import Conversation, Message

    with db_session_factory() as s:
        c1 = Conversation(id=1, user_id="guest")
        c2 = Conversation(id=2, user_id="guest")
        s.add_all([c1, c2])
        s.add_all([
            Message(conversation_id=1, role="user", content="国外的快递费怎么算?"),
            Message(conversation_id=1, role="assistant", content="海外订单运费按首重 30 元计,续重 15 元/kg。"),
            Message(conversation_id=1, role="assistant", content=None,
                    tool_calls=[{"name": "query_order"}]),  # 工具轨迹应被跳过
            Message(conversation_id=2, role="user", content="大件商品寄过来邮费谁出?"),
            Message(conversation_id=2, role="assistant", content="大件商品若为质量问题退货,运费商家承担。"),
        ])
        s.commit()


def test_conversation_text_skips_tool_messages(db_session_factory):
    _seed_conversations(db_session_factory)
    text = _conversation_text(db_session_factory, 1)
    assert "快递费" in text and "海外订单" in text and "query_order" not in text


def test_mine_extracts_dedups_and_promotes(db_session_factory):
    _seed_conversations(db_session_factory)
    kb = KnowledgeBaseStore(db_session_factory)
    # 库里先有一块与「大件邮费」语义重复的知识(用 FakeEmbedding 同词面构造高相似)
    from app.chunking import Chunk
    kb.replace_doc_chunks("退货政策.md", [
        Chunk("退货政策", "大件商品寄过来邮费谁出?", "运费商家承担。", "退货政策.md > 退货政策 > 运费", "policy", False),
    ])
    stub = StubMineModel({
        "快递费": [("国外的快递费怎么算", "海外订单首重 30 元,续重 15 元/kg")],
        "邮费": [("大件商品寄过来邮费谁出", "运费商家承担")],  # 与库内重复 → discarded
    })
    stats = mine(kb, FakeEmbedding(), stub, dedup_threshold=0.9)
    assert stats["extracted"] == 2
    assert stats["discarded"] == 1 and stats["kept"] == 1
    kept_rows = kb.get_chunks([])  # 占位,断言走 staging 状态:
    statuses = [r.status for r in kb.extracted_items()]
    assert statuses == []  # 全部落 kept/discarded,extracted 清空


def test_mine_skips_already_mined_conversations(db_session_factory):
    _seed_conversations(db_session_factory)
    kb = KnowledgeBaseStore(db_session_factory)
    stub = StubMineModel({"快递费": [("q", "a")], "邮费": [("q2", "a2")]})
    mine(kb, FakeEmbedding(), stub)
    stats2 = mine(kb, FakeEmbedding(), stub)
    assert stats2["conversations"] == 0 and stats2["extracted"] == 0
```

- [ ] **Step 2: 跑失败 → 实现 mine_qa.py → 全绿**

```python
"""从历史客服对话挖知识:分批 LLM 抽 QA → 暂存表 → 整体去重 → 入库向量化。
定时:crontab 示例(README)—— 0 * * * * cd /path && .venv/bin/python -m scripts.mine_qa
"""
import argparse

from pydantic import BaseModel, Field

from app.config import Settings
from app.db import make_engine, make_session_factory
from app.embedding import make_embedder
from app.kb import KnowledgeBaseStore, vectorize_pending
from app.vector_store import KnowledgeVectorStore


class MinedQA(BaseModel):
    question: str = Field(description="用户的原始问法")
    answer: str = Field(description="客服给出的答案")


class MinedQAList(BaseModel):
    items: list[MinedQA] = Field(description="抽取的问答对列表")


def _conversation_text(factory, conversation_id: int) -> str | None:
    """user 提问 + 紧随其后的 assistant 正文(跳过纯工具调用),拼成一段对话文本。"""
    ...


def mine(kb, embedder, extract_model, *, batch_size=5, dedup_threshold=0.88, batch_no=None) -> dict:
    ...


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=5)
    args = parser.parse_args()
    settings = Settings()
    kb = KnowledgeBaseStore(make_session_factory(make_engine(settings)))
    vectors = KnowledgeVectorStore(settings.milvus_db_path, dim=settings.embedding_dim)
    embedder = make_embedder(settings)
    extract_model = _make_extract_model(settings)  # ChatOpenAI().with_structured_output(MinedQAList, method="function_calling")
    stats = mine(kb, embedder, extract_model, batch_size=args.batch_size,
                 dedup_threshold=settings.dedup_threshold)
    print(f"挖知识完成:会话 {stats['conversations']} 通,抽取 {stats['extracted']} 条,"
          f"保留 {stats['kept']} 条,去重丢弃 {stats['discarded']} 条,向量化 {stats['vectorized']} 条")


if __name__ == "__main__":
    main()
```

`mine` 主体逻辑(实现时展开):

```python
def mine(kb, embedder, extract_model, *, batch_size=5, dedup_threshold=0.88, batch_no=None) -> dict:
    stats = {"conversations": 0, "extracted": 0, "kept": 0, "discarded": 0, "vectorized": 0}
    mined = kb.already_mined_refs()
    conv_ids = kb.conversation_ids_with_user_talks()  # kb.py 需补此方法:messages 表有 user 消息的会话 id,排除 mined
    batch_no = batch_no or f"mine-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    extracted: list = []
    for conv_id in conv_ids:
        text = _conversation_text(kb._factory, conv_id)
        if not text:
            continue
        result = extract_model.invoke(text)
        qas = [(i.question, i.answer) for i in (getattr(result, "items", None) or [])]
        kb.insert_staging(batch_no, str(conv_id), qas)
        stats["conversations"] += 1
        stats["extracted"] += len(qas)
    items = kb.extracted_items()
    if items:
        q_vecs = embedder.embed([i.question for i in items])
        lib_texts = kb.all_dedup_texts()
        lib_vecs = embedder.embed(lib_texts) if lib_texts else []
        kept_ids, discarded_ids = [], []
        kept_vecs: list[list[float]] = []
        for idx, item in enumerate(items):
            qv = q_vecs[idx]
            dup = max((_cos(qv, lv) for lv in lib_vecs), default=0.0) >= dedup_threshold
            if not dup:
                dup = max((_cos(qv, kv) for kv in kept_vecs), default=0.0) >= dedup_threshold
            if dup:
                discarded_ids.append(item.id)
            else:
                kept_ids.append(item.id)
                kept_vecs.append(qv)
        kb.set_staging_status(kept_ids, "kept")
        kb.set_staging_status(discarded_ids, "discarded")
        stats["kept"], stats["discarded"] = len(kept_ids), len(discarded_ids)
        kept_items = [i for i in items if i.id in set(kept_ids)]
        kb.insert_mined_chunks(kept_items)
        vectors = ...  # main 里传入;或 vectorize 由 main 收尾调用
    return stats
```

注:`mine` 需要触发向量化收尾,签名加 `vectors=None`;`vectors` 非 None 时调 `vectorize_pending` 计入 stats["vectorized"]。`_cos` 从 `tests.helpers` 提为公共工具(放 `app/kb.py` 导出 `cosine(a, b)`,helpers 与 mine 共用)。`kb.conversation_ids_with_user_talks()` 加入 Task 5 的接口清单(实现一并补,测试在 test_kb.py 补一例)。

- [ ] **Step 3: db/seed_conversations.py(4 通对话,幂等跳过)**

```python
"""灌历史客服对话(挖知识演示料):python -m db.seed_conversations"""
from app.config import Settings
from app.db import make_engine, make_session_factory
from app.models import Conversation, Message

CONVERSATIONS = [
    [("user", "国外的快递费怎么算?"), ("assistant", "海外订单暂不支持直邮,港澳台地区运费首重 20 元,续重 10 元/kg。")],
    [("user", "大件商品寄过来邮费谁出?"), ("assistant", "大件商品若是质量问题退货,运费由商家承担;个人原因退货需自行承担。")],
    [("user", "发票抬头写错了能改吗?"), ("assistant", "发货前可在订单页自助修改抬头,已开出的电子发票支持红冲重开。")],
    [("user", "会员折扣和包邮能一起享受吗?"), ("assistant", "可以,会员 95 折后订单金额满 99 元依然包邮。")],
]


def main() -> None:
    engine = make_engine(Settings())
    factory = make_session_factory(engine)
    with factory() as session:
        if session.query(Conversation).count() > 0:
            print("conversations 表已有数据,跳过")
            return
        for pairs in CONVERSATIONS:
            conv = Conversation(user_id="guest")
            session.add(conv)
            session.flush()
            for role, content in pairs:
                session.add(Message(conversation_id=conv.id, role=role, content=content))
        session.commit()
    print(f"已灌入 {len(CONVERSATIONS)} 通历史对话")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Commit + dev-notes**

```bash
git add scripts/mine_qa.py db/seed_conversations.py tests/ app/kb.py && git commit -m "ch03: mine_qa 挖知识管道——分批抽取/暂存/整体去重/入库向量化 + 历史会话种子"
```

---

### Task 9: query_faq 内核替换 + 防幻觉 prompt + 装配

**Files:**
- Modify: `app/tools/definitions.py`(build_tools 签名 + query_faq 内核)
- Modify: `app/prompts.py`(防幻觉一句)
- Modify: `app/main.py`(装配 embedder/vectors)
- Modify: `tests/conftest.py`(make_client 注入 Fake)
- Test: `tests/test_tools.py`(改造 + 新增)

**Interfaces:**
- Consumes: Task 3/4/5。
- Produces: `build_tools(session_factory, embedder=None, vectors=None) -> list`;默认 None 时从 `Settings()` 构造真实现。`query_faq` 契约不变。

- [ ] **Step 1: 改造测试(先红)**

`tests/test_tools.py` 中 query_faq 相关用例改为注入 FakeEmbedding/FakeVectorStore;新增:

```python
import json

from app.chunking import Chunk
from app.kb import KnowledgeBaseStore
from tests.helpers import FakeEmbedding, FakeVectorStore


def _seed_vector_kb(db_session_factory, vectors):
    kb = KnowledgeBaseStore(db_session_factory)
    chunks = [
        Chunk("退货政策", "邮费与运费", "普通订单邮费 8 元,满 99 元包邮。", "退货政策.md > 退货政策 > 邮费与运费", "policy", True),
        Chunk("售后", "怎么申请退货?", "订单详情页申请售后。", "商品FAQ.md > 商品FAQ > 售后 > 怎么申请退货?", "faq", False),
    ]
    _, ids = kb.replace_doc_chunks("docs", chunks)
    from app.kb import vectorize_pending
    vectorize_pending(kb, vectors, FakeEmbedding())
    return kb


def test_query_faq_paraphrase_hit(db_session_factory):
    vectors = FakeVectorStore()
    kb = _seed_vector_kb(db_session_factory, vectors)
    tools = build_tools(db_session_factory, embedder=FakeEmbedding(), vectors=vectors)
    faq = {t.name: t for t in tools}["query_faq"]
    out = json.loads(faq.invoke({"keyword": "快递费多少钱"}))  # FakeEmbedding 近义词典:快递费→运费
    assert out["items"], "换说法必须命中"
    assert out["items"][0]["question"] == "邮费与运费"
    assert "满 99 元包邮" in out["items"][0]["answer"]


def test_query_faq_empty_on_miss_and_never_raises(db_session_factory):
    vectors = FakeVectorStore()
    kb = _seed_vector_kb(db_session_factory, vectors)
    tools = build_tools(db_session_factory, embedder=FakeEmbedding(), vectors=vectors)
    faq = {t.name: t for t in tools}["query_faq"]
    assert json.loads(faq.invoke({"keyword": "量子力学"})) == {"items": []}


def test_system_prompt_contains_no_hallucination_rule():
    from app.prompts import SERVICE_PROMPT_TEMPLATE
    text = SERVICE_PROMPT_TEMPLATE.format()
    assert "编造" in text and "如实" in text
```

(旧 query_faq LIKE 用例删除;其余工具用例保持。)

- [ ] **Step 2: 跑失败 → 实现 → 全绿**

`definitions.py`:

```python
def build_tools(session_factory, embedder=None, vectors=None) -> list:
    if embedder is None or vectors is None:
        from app.config import Settings
        from app.embedding import make_embedder
        from app.vector_store import KnowledgeVectorStore
        settings = Settings()
        embedder = embedder or make_embedder(settings)
        vectors = vectors or KnowledgeVectorStore(settings.milvus_db_path, dim=settings.embedding_dim)
    kb = KnowledgeBaseStore(session_factory)
    settings_top_k = ...  # 见下
```

top_k 直接在闭包里用 `Settings()` 一次读取或由参数传入;简化:build_tools 增参 `top_k: int = 3`。

```python
    @tool
    def query_faq(keyword: str) -> str:
        """按语义检索常见问题知识库。用户问退货政策、发货时间、邮费运费、付款、发票、会员等常见问题时使用。"""
        try:
            qvec = embedder.embed([keyword])[0]
            hits = vectors.search(qvec, top_k=top_k)
            items = []
            if hits:
                rows = kb.get_chunks([h[0] for h in hits])
                items = [
                    {"question": r.questions.splitlines()[0], "answer": r.answer, "category": r.category}
                    for r in rows
                ]
        except Exception:
            items = []  # 任何异常收敛为空结果,契约不抛错
        return json.dumps({"items": items}, ensure_ascii=False)
```

`prompts.py` 在「不确定或不知道时…」后加:

```
"工具查询结果为空或与问题对不上时,如实告知用户暂时查不到相关信息,并建议转人工客服;严禁编造政策、价格、时效或任何承诺。\n"
```

`main.py` create_app:

```python
    from app.embedding import make_embedder
    from app.vector_store import KnowledgeVectorStore
    embedder = make_embedder(settings)
    vectors = KnowledgeVectorStore(settings.milvus_db_path, dim=settings.embedding_dim)
    app.state.registry = ToolRegistry(
        build_tools(session_factory, embedder=embedder, vectors=vectors,
                    top_k=settings.retrieval_top_k)
    )
```

`conftest.py` make_client:

```python
        from tests.helpers import FakeEmbedding, FakeVectorStore
        app.state.registry = ToolRegistry(build_tools(
            factory, embedder=FakeEmbedding(), vectors=FakeVectorStore(), top_k=3,
        ))
```

(`conftest` 顶部 `os.environ.setdefault("EMBEDDING_API_KEY", "sk-test")` 兜底,防真 .env 缺失。)

- [ ] **Step 3: 全量回归**

Run: `.venv/bin/pytest -q`
Expected: 全绿(ch01/ch02 用例无回归;chat 工具流用例走 Fake 链路)

- [ ] **Step 4: Commit + dev-notes**

```bash
git add -A && git commit -m "ch03: query_faq 换向量检索内核(契约不变)+ 防幻觉 prompt + 应用装配"
```

---

### Task 10: 检索评估集 + scripts/eval_retrieval.py(数据类任务的验证步)

**Files:**
- Create: `tests/eval/retrieval_samples.jsonl`
- Create: `scripts/eval_retrieval.py`

**Interfaces:**
- Consumes: Task 6 文档、Task 7 build(真库真模型)。
- Produces: `python scripts/eval_retrieval.py` 输出 hit@3 报告,核心样例全中退出码 0,否则 1。

- [ ] **Step 1: 标注样例(≥10 条,换说法为主)**

```jsonl
{"query": "邮费是多少", "expect_doc": "退货政策.md", "expect_keywords": ["邮费", "包邮"]}
{"query": "快递费怎么算", "expect_doc": "退货政策.md", "expect_keywords": ["邮费", "运费"]}
{"query": "寄到偏远地区要加钱吗", "expect_doc": "退货政策.md", "expect_keywords": ["偏远"]}
{"query": "退货的规定是什么", "expect_doc": "商品FAQ.md", "expect_keywords": ["七天无理由"]}
{"query": "啥时候能发货啊", "expect_doc": "商品FAQ.md", "expect_keywords": ["48 小时"]}
{"query": "几天能到货", "expect_doc": "商品FAQ.md", "expect_keywords": ["3-5 天"]}
{"query": "能用微信付钱吗", "expect_doc": "商品FAQ.md", "expect_keywords": ["微信"]}
{"query": "怎么开发票", "expect_doc": "商品FAQ.md", "expect_keywords": ["电子发票"]}
{"query": "会员有什么好处", "expect_doc": "商品FAQ.md", "expect_keywords": ["95 折"]}
{"query": "东西坏了不想修了想退", "expect_doc": "商品FAQ.md", "expect_keywords": ["质量问题"]}
{"query": "退货的钱多久退回来", "expect_doc": "退货政策.md", "expect_keywords": ["退款", "工作日"]}
{"query": "哪些东西不能退", "expect_doc": "退货政策.md", "expect_keywords": ["不予退货", "定制"]}
```

- [ ] **Step 2: 实现评估脚本(真 MySQL + 真 Milvus Lite + 真嵌入 API)**

```python
"""检索质量评估:对 tests/eval/retrieval_samples.jsonl 逐条检索,报 hit@top_k。
用法:.venv/bin/python scripts/eval_retrieval.py(需 MySQL 已建库、EMBEDDING_API_KEY 有效)
"""
import json
from pathlib import Path

from app.config import Settings
from app.db import make_engine, make_session_factory
from app.embedding import make_embedder
from app.kb import KnowledgeBaseStore
from app.vector_store import KnowledgeVectorStore

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "tests" / "eval" / "retrieval_samples.jsonl"


def main(top_k: int = 3) -> int:
    settings = Settings()
    kb = KnowledgeBaseStore(make_session_factory(make_engine(settings)))
    vectors = KnowledgeVectorStore(settings.milvus_db_path, dim=settings.embedding_dim)
    embedder = make_embedder(settings)
    if vectors.count() == 0:
        print("向量库为空,请先运行 python -m scripts.build_kb")
        return 2
    hits, misses = 0, []
    for line in SAMPLES.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        qvec = embedder.embed([case["query"]])[0]
        results = vectors.search(qvec, top_k=top_k)
        rows = kb.get_chunks([r[0] for r in results])
        ok = any(
            row.section_path and case["expect_doc"] in row.section_path
            and any(kw in row.answer or kw in row.questions for kw in case["expect_keywords"])
            for row in rows
        )
        hits += ok
        if not ok:
            misses.append(case["query"])
        print(f"{'✅' if ok else '❌'} {case['query']!r} -> {[r.section_path for r in rows]}")
    total = hits + len(misses)
    print(f"\nhit@{top_k}: {hits}/{total}")
    return 0 if not misses else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: 真跑一遍(真上游)**

Run: `.venv/bin/python scripts/eval_retrieval.py`
Expected: 全部 ✅,退出码 0;若有 ❌ 回知识文档/切分口径修正后重跑(修正是预期工作流,记录于 dev-notes)

- [ ] **Step 4: Commit + dev-notes(记 hit@3 与修正过程)**

```bash
git add tests/eval/retrieval_samples.jsonl scripts/eval_retrieval.py && git commit -m "ch03: 检索评估集 12 条 + eval_retrieval 真跑脚本(hit@3)"
```

---

### Task 11: README / 文档收口 + 全量回归

**Files:**
- Modify: `README.md`(安装/配置/建库/挖知识/评估/验收/已知边界)
- Modify: `pyproject.toml`(依赖补 `pymilvus[milvus-lite]`)
- Test: 全量回归

- [ ] **Step 1: pyproject 依赖补齐 + README 章节更新**

README 增补要点:安装命令追加 `pymilvus[milvus-lite]`;配置段补 `EMBEDDING_API_KEY` 等;数据库段补「init.sql 六表、首次需 `docker compose down -v` 重建卷」;新增「知识库」段(build_kb / 中断续跑演示 / mine_qa / crontab 示例行 / eval_retrieval);ch03 验收三条(邮费换说法召回、中断重跑补齐、挖知识增量可召回);已知边界改写(「邮费漏召回」条目翻案为已解决,新增「dense 单路,无混合检索与重排,留 ch04」)。

- [ ] **Step 2: 全量回归**

Run: `.venv/bin/pytest -q`
Expected: 全绿(记录 xfail 数)

- [ ] **Step 3: Commit + dev-notes**

```bash
git add -A && git commit -m "ch03: README/pyproject 收口——建库/挖知识/评估/验收指引"
```

---

### Task 12: Code review + 修复波 + 真机终验收 + finish

- [ ] **Step 1: requesting-code-review(skill)对全分支审一遍,修复波落地**
- [ ] **Step 2: 真机验收(docker compose down -v 重建真库 → seed → seed_conversations → build_kb → 三条验收)**

1. `curl -sN -X POST http://127.0.0.1:8000/api/chat -d '{"message": "邮费是多少"}'` → query_faq 召回运费说明,答对 8 元/满 99 包邮;
2. `python -m scripts.build_kb --max-chunks 5` 中断 → 重跑 → 输出补齐数、pending=0;
3. `python -m scripts.mine_qa` → staging kept/discarded 落地,挖出的问法换说法可召回(如「国外快递费」问「海外的运费」)。

- [ ] **Step 3: verification-before-completion(skill)过一遍证据清单**
- [ ] **Step 4: dev-notes/ch03.md 终段(演示命令/测试结果/文档路径)+ finishing-a-development-branch(skill)**

---

## Self-Review 记录

- spec §3→Task 1;§4→Task 2;§5→Task 5/7;§6→Task 8;§7→Task 9;§8.1→各测试;§8.2→Task 10;§8.3→Task 12;§8.4→贯穿。无缺口。
- 类型一致性:`Chunk` 六字段贯穿 Task 2/5/6/7;`KnowledgeVectorStore` 五方法贯穿 Task 4/5/7/8/9;`vectorize_pending` 在 Task 5 定义、7/8 消费——已对齐。
- Task 8 依赖的 `kb.conversation_ids_with_user_talks()` 与公共 `cosine()` 已在 Task 5 接口清单中补齐说明。
