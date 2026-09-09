"""检索策略模块(ch04):dense / bm25 / hybrid / hybrid_rerank 四策略同一接口。

线上一律 hybrid_rerank;评估脚本逐策略调同一实现,四策略对比才是同一套代码。
依赖(embedder/vectors/kb/rewriter/reranker)全部构造注入,测试可替换身。
低置信判定:候选空(知识库无相关内容)或精排结果未通过证据置信度闸。
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


@dataclass(frozen=True)
class EvidenceCalibration:
    """证据闸阈值,由 ch04 评估集校准并可由脚本产物覆盖。"""

    top1_floor: float = 0.03
    effective_score_floor: float = 0.03
    min_effective_count: int = 1
    margin_floor: float = 0.0
    combined_floor: float = 0.15
    source: str = "ch04_eval_set"


@dataclass(frozen=True)
class EvidenceConfidence:
    top1_relevance: float
    effective_count: int
    top1_top2_margin: float
    score: float
    passed: bool
    signals: dict
    calibration_source: str


@dataclass
class RetrievalResult:
    items: list[Retrieved]
    low_confidence: bool = False
    reason: str = ""
    filter_fallback: bool = False
    rewritten: str = ""
    search_text: str = ""
    degraded: bool = False  # reranker 缺失/失败降级 RRF 序时为 True(未精排,证据质量降档)
    evidence_confidence: EvidenceConfidence | None = None

    def snapshot(self, top_k: int = 3) -> list[dict]:
        """保留入池时可供人工审核的 Top-K 原始 chunk id 与精排分数。"""
        return [
            {"rank": rank, "chunk_id": item.chunk_id, "score": item.score,
             "rerank_score": item.score}
            for rank, item in enumerate(self.items[:max(0, top_k)], start=1)
        ]


def evidence_confidence(items: list[Retrieved], calibration: EvidenceCalibration) -> EvidenceConfidence:
    """用 Top1、有效证据数、Top1/Top2 分差合成可解释的证据置信度。"""
    scores = [max(0.0, min(1.0, float(item.score))) for item in items]
    top1 = scores[0] if scores else 0.0
    top2 = scores[1] if len(scores) > 1 else 0.0
    margin = top1 - top2 if len(scores) > 1 else top1
    effective_count = sum(score >= calibration.effective_score_floor for score in scores)
    normalized_margin = min(1.0, max(0.0, margin / 0.5))
    normalized_count = min(1.0, effective_count / 3.0)
    combined = 0.60 * top1 + 0.25 * normalized_margin + 0.15 * normalized_count
    passed = bool(scores) and top1 >= calibration.top1_floor \
        and effective_count >= calibration.min_effective_count \
        and margin >= calibration.margin_floor \
        and combined >= calibration.combined_floor
    signals = {
        "top1_relevance": round(top1, 6),
        "effective_evidence_count": effective_count,
        "top1_top2_margin": round(margin, 6),
        "combined_score": round(combined, 6),
    }
    return EvidenceConfidence(top1, effective_count, round(margin, 6), combined, passed, signals,
                              calibration.source)


def calibrate_evidence_thresholds(samples: list[dict], min_recall: float = 1.0) -> EvidenceCalibration:
    """在带标注的 ch04 样本上选出仍满足目标 Recall 的最高 Top1 门槛。"""
    positives = [s for s in samples if bool(s.get("relevant"))]
    if not positives:
        raise ValueError("校准集至少需要一条 relevant=true 样本")
    candidates = sorted({float(s["top1_relevance"]) for s in positives}, reverse=True)
    target = max(0.0, min(1.0, float(min_recall)))
    top1_floor = min(candidates)
    for candidate in candidates:
        recall = sum(float(s["top1_relevance"]) >= candidate for s in positives) / len(positives)
        if recall >= target:
            top1_floor = candidate
            break
    return EvidenceCalibration(
        top1_floor=top1_floor,
        effective_score_floor=min(float(s.get("effective_score_floor", s["top1_relevance"]))
                                  for s in positives),
        min_effective_count=min(int(s.get("effective_count", 1)) for s in positives),
        margin_floor=min(float(s.get("top1_top2_margin", 0.0)) for s in positives),
        source="ch04_eval_set",
    )


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
                 rerank_score_floor: float = 0.30,
                 evidence_calibration: EvidenceCalibration | None = None) -> None:
        self._embedder = embedder
        self._vectors = vectors
        self._kb = kb
        self._rewriter = rewriter    # None = 不改写
        self._reranker = reranker    # None = hybrid_rerank 降级 RRF 序
        self._candidates = candidates
        self._final_top_k = final_top_k
        self._dense_floor = dense_score_floor
        self._rerank_floor = rerank_score_floor
        self._evidence_calibration = evidence_calibration or EvidenceCalibration(
            top1_floor=rerank_score_floor)

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
        confidence = evidence_confidence(items, self._evidence_calibration)
        if not confidence.passed:
            # 低置信不进 Agent,但保留候选快照给问题池和人工审核。
            reason = (f"证据置信度低(top1={confidence.top1_relevance:.2f}, "
                      f"有效证据={confidence.effective_count}, "
                      f"Top1/Top2差={confidence.top1_top2_margin:.2f})")
            return RetrievalResult(items, True, reason, fallback, rewritten, search_text,
                                   evidence_confidence=confidence)
        return RetrievalResult(items, False, "", fallback, rewritten, search_text,
                               evidence_confidence=confidence)

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
