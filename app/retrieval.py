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
    degraded: bool = False  # reranker 缺失/失败降级 RRF 序时为 True(未精排,证据质量降档)


def sanitize_category(category: str | None) -> str | None:
    """品类过滤值进 Milvus expr,LLM 工具参数可能带引号/反斜杠——剥掉防注入炸表达式。"""
    if not category:
        return None
    cleaned = category.replace('"', "").replace("\\", "").strip()
    return cleaned or None


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
        category = sanitize_category(category)
        expr = f'category == "{category}"' if category else None

        def search(search_fn, *args) -> list[Retrieved]:
            """带过滤的检索异常时视为空结果 → 走无过滤回退,不让坏 expr 炸穿拒答。"""
            try:
                return search_fn(*args, expr)
            except Exception as exc:
                if expr:
                    logger.warning("带过滤检索失败,回退无过滤:%s", exc)
                    return []
                raise  # 无过滤也炸:真故障,交上层收敛为低置信

        if strategy == "dense":
            items = search(self._dense, rewritten)
            fallback = not items and bool(expr)
            if fallback:
                items = self._dense(rewritten, None)
            return RetrievalResult(items, not items, "知识库无相关内容" if not items else "",
                                   fallback, rewritten, search_text)

        if strategy == "bm25":
            items = search(self._bm25, search_text)
            fallback = not items and bool(expr)
            if fallback:
                items = self._bm25(search_text, None)
            return RetrievalResult(items, not items, "知识库无相关内容" if not items else "",
                                   fallback, rewritten, search_text)

        # hybrid / hybrid_rerank 共用候选召回
        cand = search(self._hybrid, rewritten, search_text)
        fallback = not cand and bool(expr)
        if fallback:
            cand = self._hybrid(rewritten, search_text, None)
        if not cand:
            return RetrievalResult([], True, "知识库无相关内容", fallback, rewritten, search_text)
        if strategy == "hybrid":
            return RetrievalResult(cand, False, "", fallback, rewritten, search_text)

        # hybrid_rerank:精排 Top-N;reranker 缺失或调用失败降级 RRF 序(质量掉档,记 warning 并标记)
        if self._reranker is None:
            logger.warning("reranker 未配置,hybrid_rerank 降级为 RRF 序")
            return RetrievalResult(cand[: self._final_top_k], False, "", fallback, rewritten,
                                   search_text, degraded=True)
        texts = self._chunk_texts([r.chunk_id for r in cand])
        try:
            ranked = self._reranker.rerank(rewritten, [texts.get(r.chunk_id, "") for r in cand],
                                           top_n=self._final_top_k)
        except Exception as exc:
            logger.warning("重排失败,降级 RRF 序:%s", exc)
            return RetrievalResult(cand[: self._final_top_k], False, "", fallback, rewritten,
                                   search_text, degraded=True)
        items = [Retrieved(chunk_id=cand[idx].chunk_id, score=score) for idx, score in ranked]
        if not items:
            return RetrievalResult([], True, "知识库无相关内容", fallback, rewritten, search_text)
        low = items[0].score < self._rerank_floor
        if low:
            # 低置信不出证据:别把零分候选递给生成层当编造素材(spec §5 拒答语义)
            reason = f"证据置信度低(最高 {items[0].score:.2f})"
            return RetrievalResult([], True, reason, fallback, rewritten, search_text)
        return RetrievalResult(items, False, "", fallback, rewritten, search_text)

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
