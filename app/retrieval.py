"""检索策略模块(ch04):dense / bm25 / hybrid / hybrid_rerank 四策略同一接口。

线上一律 hybrid_rerank;评估脚本逐策略调同一实现,四策略对比才是同一套代码。
依赖(embedder/vectors/kb/rewriter/reranker)全部构造注入,测试可替换身。
低置信判定:候选空(知识库无相关内容)或精排结果未通过证据置信度闸。
"""
import hashlib
import logging
from collections import defaultdict
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
class PreparedQuery:
    """一次改写结果;评估时四策略共用,避免每条策略各改一次。"""
    rewritten: str
    search_text: str


@dataclass(frozen=True)
class EvidenceCalibration:
    """证据闸阈值,由 ch04 评估集校准并可由脚本产物覆盖。"""

    top1_floor: float = 0.03
    effective_score_floor: float = 0.03
    min_effective_count: int = 1
    margin_floor: float = 0.0
    combined_floor: float = 0.15
    source: str = "ch04_eval_set"

    def __post_init__(self) -> None:
        if int(self.min_effective_count) < 1:
            raise ValueError("min_effective_count 必须 >= 1")
        for name in ("top1_floor", "effective_score_floor", "margin_floor", "combined_floor"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} 必须在 [0,1] 内:{value}")


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
    evidence_entailment: bool | None = None  # v2:LLM 判断证据是否真的回答问题;None=未启用/失败

    def snapshot(self, top_k: int = 3) -> list[dict]:
        """保留入池时可供人工审核的 Top-K 原始 chunk id 与精排分数。"""
        return [
            {"rank": rank, "chunk_id": item.chunk_id, "score": item.score,
             "rerank_score": item.score}
            for rank, item in enumerate(self.items[:max(0, top_k)], start=1)
        ]


# 证据置信度合成口径(生成侧与校准侧共用,改权重必须重新校准):
# combined = 0.60*top1 + 0.25*min(margin/0.5, 1) + 0.15*min(有效证据数/3, 1)
EVIDENCE_WEIGHT_TOP1 = 0.60
EVIDENCE_WEIGHT_MARGIN = 0.25
EVIDENCE_WEIGHT_COUNT = 0.15
EVIDENCE_MARGIN_NORM = 0.5
EVIDENCE_COUNT_NORM = 3.0


def combined_confidence_score(top1: float, margin: float, effective_count: int) -> float:
    """三信号合成分;校准器按同一公式给样本定 combined_floor。"""
    normalized_margin = min(1.0, max(0.0, margin / EVIDENCE_MARGIN_NORM))
    normalized_count = min(1.0, effective_count / EVIDENCE_COUNT_NORM)
    return (EVIDENCE_WEIGHT_TOP1 * top1 + EVIDENCE_WEIGHT_MARGIN * normalized_margin
            + EVIDENCE_WEIGHT_COUNT * normalized_count)


def evidence_confidence(items: list[Retrieved], calibration: EvidenceCalibration) -> EvidenceConfidence:
    """用 Top1、有效证据数、Top1/Top2 分差合成可解释的证据置信度。"""
    scores = [max(0.0, min(1.0, float(item.score))) for item in items]
    top1 = scores[0] if scores else 0.0
    top2 = scores[1] if len(scores) > 1 else 0.0
    margin = top1 - top2 if len(scores) > 1 else top1
    effective_count = sum(score >= calibration.effective_score_floor for score in scores)
    combined = combined_confidence_score(top1, margin, effective_count)
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


CALIBRATION_SEED = 20261023
CALIBRATION_HOLDOUT_RATIO = 1 / 3


def _sample_label(sample: dict) -> int:
    if "label" in sample:
        return int(sample["label"])
    return int(bool(sample.get("relevant")))


def _split_samples(samples: list[dict], holdout_ratio: float,
                   seed: int) -> tuple[list[dict], list[dict]]:
    """按 (label, bucket) 分层；同组内按 sha256(id+seed) 排序取前 1/3 做留出。"""
    groups: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for sample in samples:
        groups[(_sample_label(sample), str(sample.get("bucket", "")))].append(sample)
    calibration: list[dict] = []
    holdout: list[dict] = []
    for key in sorted(groups):
        items = sorted(groups[key], key=lambda s: hashlib.sha256(
            f"{s['id']}:{seed}".encode()).hexdigest())
        n_hold = int(round(len(items) * holdout_ratio))
        if holdout_ratio <= 0:
            n_hold = 0
        elif len(items) > 1:
            n_hold = min(max(n_hold, 1), len(items) - 1)
        else:
            n_hold = 0
        holdout.extend(items[:n_hold])
        calibration.extend(items[n_hold:])
    return calibration, holdout


def _passes(sample: dict, top1_floor: float, margin_floor: float,
            min_count: int) -> bool:
    return (int(sample.get("effective_count", 0)) >= min_count
            and float(sample.get("top1_relevance", 0.0)) >= top1_floor
            and float(sample.get("top1_top2_margin", 0.0)) >= margin_floor)


def _evaluate_thresholds(samples: list[dict], top1_floor: float,
                         margin_floor: float, min_count: int) -> dict:
    positives = [s for s in samples
                 if _sample_label(s) == 1 and s.get("evidence_retrieved", True)]
    retrieval_miss = [s["id"] for s in samples
                      if _sample_label(s) == 1 and not s.get("evidence_retrieved", True)]
    negatives = [s for s in samples if _sample_label(s) == 0]
    passed_positive = [s for s in positives
                       if _passes(s, top1_floor, margin_floor, min_count)]
    blocked_positive = [s["id"] for s in positives
                        if not _passes(s, top1_floor, margin_floor, min_count)]
    hard_false = [s["id"] for s in negatives
                  if s.get("bucket") == "H_hard_negative"
                  and _passes(s, top1_floor, margin_floor, min_count)]
    ood_false = [s["id"] for s in negatives
                 if s.get("bucket") != "H_hard_negative"
                 and _passes(s, top1_floor, margin_floor, min_count)]
    recall = len(passed_positive) / len(positives) if positives else 0.0
    denominator = len(passed_positive) + len(hard_false) + len(ood_false)
    precision = len(passed_positive) / denominator if denominator else 0.0
    f2 = (5 * precision * recall / (4 * precision + recall)
          if (4 * precision + recall) > 0 else 0.0)
    return {
        "positives": len(positives),
        "positive_recall": round(recall, 6),
        "positive_blocked_ids": blocked_positive,
        "retrieval_miss_ids": retrieval_miss,
        "hard_negative_false_accept_ids": hard_false,
        "ood_false_accept_ids": ood_false,
        "precision": round(precision, 6),
        "f2": round(f2, 6),
        "top1_floor": top1_floor,
        "margin_floor": margin_floor,
        "min_effective_count": min_count,
    }


def calibrate_evidence_thresholds(
    samples: list[dict],
    *,
    effective_score_floor: float = 0.03,
    holdout_ratio: float = CALIBRATION_HOLDOUT_RATIO,
    seed: int = CALIBRATION_SEED,
) -> tuple[EvidenceCalibration, dict, dict, dict]:
    """在正样本/同域硬负例/OOD 上网格搜索阈值。

    目标（字典序）：硬负例零误放 → 正样本召回最大 → OOD 误放最少 → F2 最大
    → 更保守的阈值。无可行解时 status=infeasible，调用方不得覆盖生产配置。
    combined_floor 固定 0，combined_score 仅作诊断。
    """
    if not samples:
        raise ValueError("校准集为空")
    normalized = [
        {**sample, "label": _sample_label(sample),
         "evidence_retrieved": bool(sample.get("evidence_retrieved", True))}
        for sample in samples
    ]
    calibration, holdout = _split_samples(normalized, holdout_ratio, seed)
    split_ids = {
        "seed": seed,
        "holdout_ratio": holdout_ratio,
        "calibration_ids": [s["id"] for s in calibration],
        "holdout_ids": [s["id"] for s in holdout],
    }
    eligible = [s for s in calibration
                if s["label"] == 1 and s["evidence_retrieved"]]
    if not eligible:
        metrics = {
            "status": "infeasible",
            "reason": "校准集没有检索到证据的正样本",
            "positives": 0,
            "positive_recall": 0.0,
            "positive_blocked_ids": [],
            "retrieval_miss_ids": [s["id"] for s in calibration if s["label"] == 1],
            "hard_negative_false_accept_ids": [],
            "ood_false_accept_ids": [],
            "precision": 0.0,
            "f2": 0.0,
            "top1_floor": 0.03,
            "margin_floor": 0.0,
            "min_effective_count": 1,
        }
        calibration_obj = EvidenceCalibration(
            top1_floor=0.03, effective_score_floor=effective_score_floor,
            min_effective_count=1, margin_floor=0.0, combined_floor=0.0,
            source="ch04_eval_set+v2")
        return calibration_obj, metrics, {**metrics, "status": "infeasible"}, split_ids

    top1_values = sorted({0.0} | {round(float(s.get("top1_relevance", 0.0)), 6)
                                  for s in calibration})
    margin_values = sorted({0.0} | {round(float(s.get("top1_top2_margin", 0.0)), 6)
                                     for s in calibration})
    candidates = [
        _evaluate_thresholds(calibration, top1_floor, margin_floor, min_count)
        for top1_floor in top1_values
        for margin_floor in margin_values
        for min_count in (1, 2, 3)
    ]
    feasible = [c for c in candidates if not c["hard_negative_false_accept_ids"]]
    if not feasible:
        fallback = max(candidates, key=lambda c: (
            c["positive_recall"], -len(c["ood_false_accept_ids"]), c["f2"]))
        fallback = {**fallback, "status": "infeasible"}
        calibration_obj = EvidenceCalibration(
            top1_floor=fallback["top1_floor"],
            effective_score_floor=effective_score_floor,
            min_effective_count=max(1, fallback["min_effective_count"]),
            margin_floor=fallback["margin_floor"],
            combined_floor=0.0, source="ch04_eval_set+v2")
        holdout_metrics = _evaluate_thresholds(
            holdout, fallback["top1_floor"], fallback["margin_floor"],
            fallback["min_effective_count"])
        holdout_metrics["status"] = "infeasible"
        return calibration_obj, fallback, holdout_metrics, split_ids

    best = min(feasible, key=lambda c: (
        -c["positive_recall"],
        len(c["ood_false_accept_ids"]),
        -c["f2"],
        -c["top1_floor"],
        -c["margin_floor"],
        -c["min_effective_count"],
    ))
    best = {**best, "status": "ok"}
    calibration_obj = EvidenceCalibration(
        top1_floor=best["top1_floor"],
        effective_score_floor=effective_score_floor,
        min_effective_count=best["min_effective_count"],
        margin_floor=best["margin_floor"],
        combined_floor=0.0,
        source="ch04_eval_set+v2",
    )
    holdout_metrics = _evaluate_thresholds(
        holdout, best["top1_floor"], best["margin_floor"],
        best["min_effective_count"])
    holdout_metrics["status"] = "ok"
    return calibration_obj, best, holdout_metrics, split_ids


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
                 evidence_calibration: EvidenceCalibration | None = None,
                 evidence_judge=None, evidence_top_n: int = 3) -> None:
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
        self._evidence_judge = evidence_judge
        self._evidence_top_n = max(1, int(evidence_top_n))

    def prepare_query(self, query: str, use_rewrite: bool = True) -> PreparedQuery:
        """一次性改写;评估脚本可先缓存再让四策略共用。"""
        if not use_rewrite or self._rewriter is None:
            return PreparedQuery(query, query)
        qr = self._rewriter.rewrite(query)
        rewritten = qr.normalized.strip() or query
        search_text = bm25_query_text(QueryRewrite(normalized=rewritten, synonyms=qr.synonyms))
        return PreparedQuery(rewritten, search_text)

    def retrieve(self, query: str, strategy: str = "hybrid_rerank",
                 category: str | None = None, use_rewrite: bool = True,
                 prepared: PreparedQuery | None = None) -> RetrievalResult:
        if strategy not in STRATEGIES:
            raise ValueError(f"未知检索策略:{strategy}")

        if prepared is None:
            prepared = self.prepare_query(query, use_rewrite=use_rewrite)
        rewritten, search_text = prepared.rewritten, prepared.search_text
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
            logger.warning("reranker 未配置,hybrid_rerank 降级为 RRF 序,证据闸无法判定")
            return RetrievalResult(cand[: self._final_top_k], True,
                                   "重排不可用,证据闸无法判定", fallback, rewritten,
                                   search_text, degraded=True)
        texts = self._chunk_texts([r.chunk_id for r in cand])
        try:
            ranked = self._reranker.rerank(rewritten, [texts.get(r.chunk_id, "") for r in cand],
                                           top_n=self._final_top_k)
        except Exception as exc:
            if exc.__class__.__name__ == "BudgetExceeded":
                raise
            logger.warning("重排失败,降级 RRF 序,证据闸无法判定:%s", exc)
            return RetrievalResult(cand[: self._final_top_k], True,
                                   "重排不可用,证据闸无法判定", fallback, rewritten,
                                   search_text, degraded=True)
        items = [Retrieved(chunk_id=cand[idx].chunk_id, score=score) for idx, score in ranked]
        if not items:
            return RetrievalResult([], True, "知识库无相关内容", fallback, rewritten, search_text)
        confidence = evidence_confidence(items, self._evidence_calibration)

        # v2:语义层优先。判断器说“证据真的回答了”就放行(即使分数低);
        # 说“没回答”就拒答(即使分数高);判断器不可用时回退分数闸。
        verdict = None
        if self._evidence_judge is not None:
            documents = [texts.get(item.chunk_id, "")
                         for item in items[: self._evidence_top_n]]
            documents = [doc for doc in documents if doc]
            if documents:
                try:
                    verdict = self._evidence_judge.judge(query, documents)
                except Exception as exc:
                    if exc.__class__.__name__ == "BudgetExceeded":
                        raise
                    logger.warning("证据蕴含校验失败,回退分数闸:%s", exc)
        if verdict is not None:
            if not verdict.answered:
                reason = f"证据未直接回答问题:{verdict.reason or '判断器未给出理由'}"
                return RetrievalResult(items, True, reason, fallback, rewritten, search_text,
                                       evidence_confidence=confidence,
                                       evidence_entailment=False)
            return RetrievalResult(items, False, "", fallback, rewritten, search_text,
                                   evidence_confidence=confidence,
                                   evidence_entailment=True)

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
