"""ch12 受控 Agentic 检索循环核心:线上图与评测 harness 共用同一实现。

流程与界限见设计稿 docs/superpowers/specs/2026-10-25-ch12-controlled-agentic-rag-design.md:
- 首轮检索(原查询,hybrid_rerank,与基线同一调用)通过证据闸 → 零额外 LLM 调用直接放行,
  单跳行为与基线一致;
- 未通过才进入「蕴含再判 → 缺口分析 → 子查询去重 → 补检索 → max-score 合并」的有界循环;
  全部轮次结束后对最终合并证据做一次终判(decompose_once = max_rounds=1 特例)。
放行判定与基线同源:同一分数闸(evidence_confidence+同一 calibration)+ 同一蕴含判断器
(同一 EvidenceJudge/prompt/温度);缺口分析器的 sufficient/action=answer 仅供参考,不作为放行依据。
故障语义 = 按已知证据谨慎收敛(不放宽、宁可拒答),停止原因如实记录;
BudgetExceeded 一律向上抛(评测预算闸必须能中止运行)。
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Callable

from app.history import estimate_tokens
from app.prompts import GAP_ANALYZER_PROMPT, GapReport
from app.retrieval import (
    EvidenceCalibration,
    EvidenceConfidence,
    Retrieved,
    evidence_confidence,
)

logger = logging.getLogger(__name__)

STOP_FIRST_PASS = "first_pass_sufficient"
STOP_SUFFICIENT = "sufficient"
STOP_ROUNDS = "rounds_exhausted"
STOP_TIME = "time_budget"
STOP_TOKENS = "token_budget"
STOP_NO_NEW = "no_new_queries"
STOP_CLARIFY = "clarify_needed"
STOP_REFUSE = "refuse_no_evidence"
STOP_ESCALATE = "escalate_conflict"
STOP_RETRIEVAL_ERROR = "retrieval_error"
STOP_ANALYZER_ERROR = "analyzer_error"

# 预算/故障熔断的停止原因:谨慎收敛(不再发起任何 LLM 调用,不放行)
CONVERGE_STOPS = {STOP_ROUNDS, STOP_TIME, STOP_TOKENS, STOP_NO_NEW,
                  STOP_RETRIEVAL_ERROR, STOP_ANALYZER_ERROR}


def normalize_query(query: str) -> str:
    """查询去重键:折叠全部空白+小写。同键查询绝不重复检索(设计稿 3.2)。"""
    return "".join(str(query or "").split()).lower()


@dataclass(frozen=True)
class RagLoopLimits:
    max_rounds: int = 3          # 最大补检索轮数(不含首轮;decompose_once 传 1)
    token_budget: int = 3000     # 缺口分析输出侧 token 预算(estimate_tokens 口径)
    time_budget_s: float = 45.0  # 循环墙钟预算(秒)
    max_sub_queries: int = 3     # 每轮最多子查询数
    final_top_k: int = 10        # 合并后进入证据的条数(= rerank_top_k)
    evidence_top_n: int = 10     # 送蕴含判断器的证据条数(= evidence_entailment_top_n)


@dataclass
class RagLoopResult:
    items: list[Retrieved] = field(default_factory=list)  # 合并后 top-k(最终排序)
    gate_passed: bool = False
    stop_reason: str = ""
    reason: str = ""
    clarify_question: str = ""
    conflicts: list[str] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)
    rounds: int = 0
    latency_ms: float = 0.0
    evidence_confidence: EvidenceConfidence | None = None
    evidence_entailment: bool | None = None
    snapshots: list[dict] = field(default_factory=list)  # retrieved_chunks 诊断快照(含低置信轮)
    analyzer_calls: int = 0
    judge_calls: int = 0
    degraded: bool = False


class GapAnalyzer:
    """生产缺口分析器:结构化输出 GapReport,温度 0 口径由注入的 extract 模型保证。"""

    def __init__(self, llm) -> None:
        self._llm = llm  # with_structured_output(GapReport) 的产物,测试可注替身

    def analyze(self, query: str, evidence: str) -> GapReport:
        result = self._llm.invoke(GAP_ANALYZER_PROMPT.format(query=query, evidence=evidence))
        if isinstance(result, dict):
            result = GapReport(**result)
        return result


def make_gap_analyzer(settings):
    from app.llm import make_extract_model

    return GapAnalyzer(make_extract_model(settings).with_structured_output(
        GapReport, method="function_calling"))


def _step_record(round_index: int, phase: str, query: str) -> dict:
    return {
        "round_index": round_index,
        "phase": phase,
        "query": query,
        "queries": [],
        "sub_queries": [],
        "evidence_ids": [],
        "new_evidence_ids": [],
        "merged_evidence_ids": [],
        "judge_answered": None,
        "analyzer_action": "",
        "conflicts": [],
        "missing_aspects": [],
        "latency_ms": 0.0,
        "stop_reason": "",
    }


class ControlledRagLoop:
    """有界受控检索循环。依赖全部注入(retrieve/chunk_texts/judge/analyzer),可离线替身单测。

    retrieve(query, *, use_rewrite, prepared) -> RetrievalResult:首轮流经调用方改写,
      子查询 use_rewrite=False(分析器输出已是规范问法,同 refund/policy_retrieve 惯例);
    chunk_texts(ids) -> dict[int, str]:证据文本(蕴含判断与缺口分析用);
    judge(query, documents) -> EvidenceVerdict:与基线同一蕴含判断器(评测侧为缓存+守卫包装);
    analyzer(query, evidence_text) -> GapReport:仅决定拆解/澄清/拒答/转人工,不决定放行;
    on_step(step):每轮结束回调(逐轮审计落库),异常仅 warning 绝不阻断循环。
    """

    def __init__(self, retrieve: Callable, chunk_texts: Callable, judge, analyzer,
                 calibration: EvidenceCalibration, limits: RagLoopLimits, *,
                 on_step: Callable[[dict], None] | None = None,
                 snapshot_top_k: int = 3) -> None:
        self._retrieve = retrieve
        self._chunk_texts = chunk_texts
        self._judge = judge
        self._analyzer = analyzer
        self._calibration = calibration
        self._limits = limits
        self._on_step = on_step
        self._snapshot_top_k = snapshot_top_k

    # ---- 内部工具 ----

    @staticmethod
    def _budget_exceeded(exc: BaseException) -> bool:
        return exc.__class__.__name__ == "BudgetExceeded"

    def _retrieve_one(self, query: str, *, use_rewrite: bool, prepared=None):
        """单次检索;非预算异常吞掉返回 None(该查询按空结果处理,谨慎收敛)。"""
        try:
            return self._retrieve(query, use_rewrite=use_rewrite, prepared=prepared)
        except Exception as exc:
            if self._budget_exceeded(exc):
                raise
            logger.warning("受控循环检索失败,该查询按空结果处理:%s", exc, exc_info=True)
            return None

    def _snapshots_of(self, retrieval_result) -> list[dict]:
        """候选快照;替身检索服务缺 snapshot 方法时按 items 兜底(refund.py 同惯例)。"""
        snapshot_fn = getattr(retrieval_result, "snapshot", None)
        if callable(snapshot_fn):
            try:
                return snapshot_fn(self._snapshot_top_k) or []
            except Exception:
                logger.warning("候选快照构建失败,回退 items", exc_info=True)
        return [{"rank": i + 1, "chunk_id": item.chunk_id, "score": item.score,
                 "rerank_score": item.score}
                for i, item in enumerate((getattr(retrieval_result, "items", None) or [])
                                         [: self._snapshot_top_k])]

    def _judge_merged(self, query: str, merged_top: list[Retrieved], texts: dict[int, str]):
        """对合并证据重跑蕴含判断(与基线同一判断器/同一 prompt);异常回退分数闸。"""
        documents = [texts[item.chunk_id]
                     for item in merged_top[: self._limits.evidence_top_n]
                     if item.chunk_id in texts]
        documents = [doc for doc in documents if doc]
        if self._judge is None or not documents:
            return None
        try:
            return self._judge.judge(query, documents)
        except Exception as exc:
            if self._budget_exceeded(exc):
                raise
            logger.warning("合并证据蕴含判断失败,回退分数闸:%s", exc, exc_info=True)
            return None

    def _emit_step(self, step: dict) -> None:
        if self._on_step is None:
            return
        try:
            self._on_step(step)
        except Exception:
            logger.warning("rag 步骤审计回调失败(不阻断循环)", exc_info=True)

    def _evidence_text(self, merged_top: list[Retrieved], texts: dict[int, str]) -> str:
        return "\n".join(
            f"[{i + 1}] {texts[item.chunk_id]}"
            for i, item in enumerate(merged_top[: self._limits.evidence_top_n])
            if item.chunk_id in texts
        )

    # ---- 主流程 ----

    def run(self, query: str, prepared=None) -> RagLoopResult:
        limits = self._limits
        t0 = time.monotonic()
        executed: set[str] = {normalize_query(query)}
        merged: dict[int, float] = {}
        result = RagLoopResult()
        spent_tokens = 0
        stop = ""
        reason = ""
        last_entailment: bool | None = None

        def merged_top() -> list[Retrieved]:
            ranked = sorted(merged.items(), key=lambda kv: -kv[1])
            return [Retrieved(chunk_id=cid, score=score)
                    for cid, score in ranked[: limits.final_top_k]]

        def merge_items(items: list[Retrieved]) -> int:
            new_ids = 0
            for item in items or []:
                if item.chunk_id not in merged:
                    new_ids += 1
                if item.chunk_id not in merged or item.score > merged[item.chunk_id]:
                    merged[item.chunk_id] = item.score
            return new_ids

        def time_left() -> float:
            return limits.time_budget_s - (time.monotonic() - t0)

        def current_texts() -> dict[int, str]:
            return self._chunk_texts([item.chunk_id for item in merged_top()])

        def admit():
            """放行判定 = 分数闸(同一 calibration)+ 蕴含判断器(同一 prompt);
            判断器不可用/失败时回退分数闸(与基线 retrieve 的回退语义一致,未放宽)。"""
            verdict = self._judge_merged(query, merged_top(), current_texts())
            confidence = evidence_confidence(merged_top(), self._calibration)
            if verdict is not None:
                return confidence.passed and bool(verdict.answered), verdict, confidence
            return confidence.passed, None, confidence

        # ---- 首轮:原查询,与基线完全同一调用口径 ----
        round0 = self._retrieve_one(query, use_rewrite=True, prepared=prepared)
        round0_items = list(round0.items) if round0 is not None else []
        result.degraded = bool(round0.degraded) if round0 is not None else False
        step0 = _step_record(0, "retrieve", query)
        step0["queries"] = [query]
        step0["evidence_ids"] = [item.chunk_id for item in round0_items]
        step0["latency_ms"] = round((time.monotonic() - t0) * 1000, 3)
        if round0 is None:
            step0["stop_reason"] = STOP_RETRIEVAL_ERROR
            result.steps.append(step0)
            self._emit_step(step0)
            return self._finish(result, t0, STOP_RETRIEVAL_ERROR, "首轮检索服务不可用",
                                merged_top())
        for snap in self._snapshots_of(round0):
            result.snapshots.append(snap)
        if round0.low_confidence:
            # 低置信候选只进诊断快照,不进证据合并(同 refund/policy_retrieve 语义)
            for item in round0_items:
                result.snapshots.append({"chunk_id": item.chunk_id, "score": item.score,
                                         "rerank_score": item.score})
        else:
            merge_items(round0_items)
            step0["new_evidence_ids"] = [item.chunk_id for item in round0_items]
            step0["merged_evidence_ids"] = [item.chunk_id for item in merged_top()]
        last_entailment = round0.evidence_entailment
        if round0.evidence_entailment is not None:
            result.judge_calls += 1
        if not round0.low_confidence:
            step0["judge_answered"] = round0.evidence_entailment
            result.steps.append(step0)
            self._emit_step(step0)
            return self._finish(result, t0, STOP_FIRST_PASS, "", merged_top(),
                                confidence=getattr(round0, "evidence_confidence", None),
                                entailment=round0.evidence_entailment, passed=True)
        result.steps.append(step0)
        self._emit_step(step0)

        # ---- 有界循环 r=1..max_rounds:预算检查 → 蕴含再判 → 缺口分析 → 子查询补检索合并 ----
        for r in range(1, limits.max_rounds + 1):
            result.rounds = r
            round_t0 = time.monotonic()
            step = _step_record(r, "retrieve", query)
            if time_left() <= 0:
                stop, reason = STOP_TIME, f"时间预算耗尽({limits.time_budget_s}s)"
                break
            if spent_tokens >= limits.token_budget:
                stop, reason = STOP_TOKENS, f"token 预算耗尽({limits.token_budget})"
                break

            # 蕴含再判(判的是含前几轮合并的证据)
            passed, verdict, confidence = admit()
            if verdict is not None:
                result.judge_calls += 1
                last_entailment = bool(verdict.answered)
                step["judge_answered"] = bool(verdict.answered)
            if passed:
                result.evidence_confidence = confidence
                step["merged_evidence_ids"] = [item.chunk_id for item in merged_top()]
                step["latency_ms"] = round((time.monotonic() - round_t0) * 1000, 3)
                result.steps.append(step)
                self._emit_step(step)
                return self._finish(result, t0, STOP_SUFFICIENT, "", merged_top(),
                                    confidence=confidence,
                                    entailment=(verdict.answered if verdict is not None else None),
                                    passed=True)

            # 缺口分析(先查 token/时间预算)
            if self._analyzer is None:
                stop, reason = STOP_ANALYZER_ERROR, "缺口分析器不可用"
                break
            texts = current_texts()
            estimate = (estimate_tokens(query)
                        + estimate_tokens(self._evidence_text(merged_top(), texts)))
            if spent_tokens + estimate > limits.token_budget:
                stop, reason = STOP_TOKENS, f"token 预算耗尽({limits.token_budget})"
                break
            if time_left() <= 0:
                stop, reason = STOP_TIME, f"时间预算耗尽({limits.time_budget_s}s)"
                break
            try:
                report = self._analyzer.analyze(query, self._evidence_text(merged_top(), texts))
            except Exception as exc:
                if self._budget_exceeded(exc):
                    raise
                logger.warning("缺口分析失败,谨慎收敛:%s", exc, exc_info=True)
                stop, reason = STOP_ANALYZER_ERROR, f"缺口分析服务不可用:{exc}"[:200]
                break
            result.analyzer_calls += 1
            spent_tokens += estimate_tokens(json.dumps(
                {k: getattr(report, k) for k in
                 ("sufficient", "action", "missing_aspects", "conflicts", "sub_queries",
                  "clarify_question", "reason")}, ensure_ascii=False, default=str))
            step["analyzer_action"] = str(report.action)
            step["conflicts"] = [str(c) for c in report.conflicts][:5]
            step["missing_aspects"] = [str(m) for m in report.missing_aspects][:5]
            result.conflicts = step["conflicts"]
            if report.action == "clarify" and report.clarify_question:
                stop, reason = STOP_CLARIFY, report.clarify_question
                break
            if report.action == "refuse":
                stop, reason = STOP_REFUSE, report.reason or "证据缺口无法通过检索补齐"
                break
            if report.action == "escalate":
                stop, reason = STOP_ESCALATE, report.reason or "证据存在无法自行裁决的冲突"
                break

            # 子查询去重后补检索(同键查询绝不重复)
            subs: list[str] = []
            for raw in (report.sub_queries or [])[: limits.max_sub_queries]:
                text = str(raw or "").strip()
                if not text or normalize_query(text) in executed:
                    continue
                executed.add(normalize_query(text))
                subs.append(text)
            step["sub_queries"] = subs
            if not subs:
                stop, reason = STOP_NO_NEW, "缺口分析未给出新的可执行子查询"
                break
            round_new_ids = 0
            for sub in subs:
                sub_result = self._retrieve_one(sub, use_rewrite=False)
                step["queries"].append(sub)
                if sub_result is None:
                    continue
                result.degraded = result.degraded or bool(sub_result.degraded)
                for snap in self._snapshots_of(sub_result):
                    result.snapshots.append(snap)
                if sub_result.low_confidence:
                    for item in sub_result.items:
                        result.snapshots.append({"chunk_id": item.chunk_id,
                                                 "score": item.score,
                                                 "rerank_score": item.score})
                    continue
                for item in sub_result.items:
                    if item.chunk_id not in step["evidence_ids"]:
                        step["evidence_ids"].append(item.chunk_id)
                round_new_ids += merge_items(sub_result.items)
            step["merged_evidence_ids"] = [item.chunk_id for item in merged_top()]
            step["latency_ms"] = round((time.monotonic() - round_t0) * 1000, 3)
            result.steps.append(step)
            self._emit_step(step)
            if round_new_ids == 0:
                stop, reason = STOP_NO_NEW, "本轮补检索未产生新证据"
                break

        # ---- 全部轮次结束后的终判:对最终合并证据再做一次蕴含再判(decompose_once 即此特例) ----
        if not stop:
            final_t0 = time.monotonic()
            final_step = _step_record(result.rounds, "final", query)
            if time_left() <= 0:
                stop, reason = STOP_TIME, f"时间预算耗尽({limits.time_budget_s}s)"
            elif spent_tokens >= limits.token_budget:
                stop, reason = STOP_TOKENS, f"token 预算耗尽({limits.token_budget})"
            else:
                passed, verdict, confidence = admit()
                if verdict is not None:
                    result.judge_calls += 1
                    last_entailment = bool(verdict.answered)
                    final_step["judge_answered"] = bool(verdict.answered)
                final_step["merged_evidence_ids"] = [item.chunk_id for item in merged_top()]
                final_step["latency_ms"] = round((time.monotonic() - final_t0) * 1000, 3)
                result.steps.append(final_step)
                self._emit_step(final_step)
                if passed:
                    return self._finish(
                        result, t0, STOP_SUFFICIENT, "", merged_top(),
                        confidence=confidence,
                        entailment=(verdict.answered if verdict is not None else None),
                        passed=True)
                stop, reason = STOP_ROUNDS, f"补检索轮数耗尽({limits.max_rounds})"
                result.evidence_confidence = confidence

        # ---- 终态收口:未放行即谨慎收敛(拒答/澄清/转人工由节点层映射) ----
        result.reason = reason
        result.clarify_question = reason if stop == STOP_CLARIFY else ""
        return self._finish(result, t0, stop or STOP_ROUNDS, reason, merged_top(),
                            entailment=last_entailment)

    def _finish(self, result: RagLoopResult, t0: float, stop: str, reason: str,
                items: list[Retrieved], *,
                confidence: EvidenceConfidence | None = None,
                entailment: bool | None = None, passed: bool = False) -> RagLoopResult:
        result.items = items
        result.stop_reason = stop
        result.reason = reason
        result.gate_passed = passed
        result.latency_ms = round((time.monotonic() - t0) * 1000, 3)
        if confidence is None and items and stop in CONVERGE_STOPS:
            # 收敛路径没有再判蕴含:只补分数闸信号供观测,放行仍为 False
            result.evidence_confidence = evidence_confidence(items, self._calibration)
        else:
            result.evidence_confidence = result.evidence_confidence or confidence
        result.evidence_entailment = entailment
        if result.steps:
            result.steps[-1]["stop_reason"] = stop
        return result
