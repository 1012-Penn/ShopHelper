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

        # hybrid / hybrid_rerank 共用候选召回
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
