"""测试替身与 SSE 解析助手——与实现代码无关,只面向 HTTP 契约。"""
import hashlib
import json
import math
import re

from app.embedding import cosine  # 公共余弦工具,生产/测试同源
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

__all__ = ["cosine", "FakeChatWithTools", "fake_chat", "StubExtractModel", "parse_sse",
           "post_chat_sse", "FakeEmbedding", "FakeVectorStore", "StubMineModel", "StubQAList", "FakeReranker", "FakeRewriter"]


class FakeChatWithTools(GenericFakeChatModel):
    """GenericFakeChatModel 不实现 bind_tools(基类 NotImplementedError);替身直接返回自身。"""

    def bind_tools(self, tools, **kwargs):
        return self


def fake_chat(*reply_texts: str) -> GenericFakeChatModel:
    """每个回复一条 AIMessage;调用顺序消费。**每个测试新建**,迭代器一次性。"""
    return FakeChatWithTools(messages=iter(AIMessage(content=t) for t in reply_texts))


class StubExtractModel:
    """模拟 with_structured_output 产物:result 与 error 二选一。"""

    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error

    async def ainvoke(self, text):
        if self.error is not None:
            raise self.error
        return self.result


def parse_sse(raw: str) -> list[dict]:
    return [
        json.loads(line[len("data: "):])
        for line in raw.split("\n\n")
        if line.startswith("data: ")
    ]


async def post_chat_sse(client, payload: dict) -> list[dict]:
    async with client.stream("POST", "/api/chat", json=payload) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        raw = ""
        async for chunk in resp.aiter_text():
            raw += chunk
    return parse_sse(raw)


class FakeEmbedding:
    """确定性假嵌入:近义词归一 → 只保留领域词(词库锚定,长词优先)→ 每词一个稳定伪随机向量求和归一。
    「邮费/快递费/寄费」都归一到「运费」,换说法 cos=1.0;异话题仅靠随机向量近似正交自然拉开;
    多话题文本按词叠加,得到可控的梯度相似度。无领域词的文本返回零向量(与任何向量 cos=0)。"""

    dim = 64
    SYNONYMS = {"邮费": "运费", "快递费": "运费", "寄费": "运费", "送货": "发货"}
    LEXICON = sorted(
        ["七天无理由", "运费", "退货", "退款", "发货", "物流", "发票", "会员", "付款", "支付", "质量"],
        key=len,
        reverse=True,
    )

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in self._tokens(text):
            for j, v in enumerate(self._token_vec(tok)):
                vec[j] += v
        norm = math.sqrt(sum(a * a for a in vec))
        if norm == 0:
            return vec
        return [a / norm for a in vec]

    def _tokens(self, text: str) -> list[str]:
        canon = text
        for src, dst in self.SYNONYMS.items():
            canon = canon.replace(src, dst)
        toks: list[str] = []
        i = 0
        while i < len(canon):
            for word in self.LEXICON:
                if canon.startswith(word, i):
                    toks.append(word)
                    i += len(word)
                    break
            else:
                i += 1
        return toks

    def _token_vec(self, tok: str) -> list[float]:
        digest = hashlib.sha512(tok.encode()).digest()  # 64 字节 = 64 维,字节独立近似正交
        return [b / 255.0 - 0.5 for b in digest]


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


class StubQAList:
    """with_structured_output 产物的列表壳:mine 只消费 .items。"""

    def __init__(self, items):
        self.items = items


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


class StubMineModel:
    """按对话文本里的关键词返回预置 QA(dict 形式,同 structured output 的 dict 形态);未命中返回空。"""

    def __init__(self, mapping: dict[str, list[tuple[str, str]]]):
        self.mapping = mapping

    def invoke(self, text, config=None, **kwargs):
        items: list[dict] = []
        for key, qas in self.mapping.items():
            if key in text:
                items.extend({"question": q, "answer": a} for q, a in qas)
        return StubQAList(items=items)
