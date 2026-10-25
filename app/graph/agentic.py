"""ch12 受控 Agentic 检索节点:包 ControlledRagLoop,整个循环在 to_thread 内完成。

出口(设计稿 3.3):证据充分 → agent(证据重新编号 1..n,citations 全量单发,覆盖语义);
不充分 → 本节点收口,不使用 interrupt()(py3.10 下被 langgraph 守卫禁用):
- clarify:普通澄清回复结束本回合,下回合重走图;
- refuse/预算与故障收敛:REFUSAL_MARKER + 落池(retrieval_low_conf)+ 转人工动作;
- escalate:说明冲突 + 转人工动作。
逐轮步骤经 on_step 落 rag_loop_steps(仅可审计,ADR-1);审计写入失败绝不阻断回合。
本路径纯读,不产生任何业务写。
"""
import asyncio
import logging

from langchain_core.messages import AIMessage

from app.guard import REFUSAL_MARKER
from app.kb import vector_text
from app.rag_loop import STOP_CLARIFY, STOP_ESCALATE, ControlledRagLoop, RagLoopLimits
from app.retrieval import EvidenceCalibration

logger = logging.getLogger(__name__)

CLARIFY_PREFIX = "想帮您查得更准:"
REFUSAL_CONFLICT = f"{REFUSAL_MARKER}现有资料存在口径冲突,需要人工确认适用版本,请选择下方方式继续。"


def _emit(writer, frame: dict) -> None:
    if writer is not None:
        writer(frame)


def make_agentic_retrieve_node(service, kb, pool, settings, *, max_rounds: int,
                               analyzer=None, steps_store=None, snapshot_top_k: int = 3):
    """analyzer/steps_store 缺省时分别自建/不落库;测试全部可注入替身。

    放行判定与 knowledge 链同源:同一 RetrievalService(分数闸+蕴含判断器+同一校准产物)。
    """
    limits = RagLoopLimits(
        max_rounds=max_rounds,
        token_budget=settings.rag_token_budget,
        time_budget_s=settings.rag_time_budget_s,
        max_sub_queries=settings.rag_max_sub_queries,
        final_top_k=settings.rerank_top_k,
        evidence_top_n=settings.evidence_entailment_top_n)
    calibration = getattr(service, "_evidence_calibration", None) or EvidenceCalibration()
    judge = getattr(service, "_evidence_judge", None)

    def _retrieve(query, *, use_rewrite, prepared=None):
        return service.retrieve(query, strategy="hybrid_rerank",
                                use_rewrite=use_rewrite, prepared=prepared)

    def _chunk_texts(ids):
        return {row.id: vector_text(row.category, row.questions, row.answer)
                for row in kb.get_chunks(ids)}

    async def agentic_retrieve_node(state, writer=None, config=None) -> dict:
        session_id = state["session_id"]
        turn_id = state.get("turn_user_msg_id") or ""
        query = state["resolved_message"]
        _emit(writer, {"type": "tool_status", "name": "query_faq", "label": "知识检索(受控)"})

        def on_step(step):
            if steps_store is not None:
                steps_store.insert_step(session_id=session_id,
                                        turn_user_msg_id=turn_id, step=step)

        loop = ControlledRagLoop(_retrieve, _chunk_texts, judge, analyzer, calibration,
                                 limits, on_step=on_step, snapshot_top_k=snapshot_top_k)
        rag = await asyncio.to_thread(loop.run, query)
        if steps_store is not None:
            steps_store.finalize_stop(session_id, turn_id, rag.stop_reason)

        rows = await asyncio.to_thread(kb.get_chunks, [r.chunk_id for r in rag.items])
        evidence = [
            {"n": i + 1, "chunk_id": r.chunk_id,
             "question": (row.questions.splitlines() or [""])[0], "answer": row.answer,
             "category": row.category, "section_path": row.section_path or "",
             "score": r.score, "rerank_score": r.score}
            for i, (r, row) in enumerate(zip(rag.items, rows))
        ]
        confidence = rag.evidence_confidence
        confidence_payload = None
        if confidence is not None:
            confidence_payload = {**confidence.signals, "passed": confidence.passed,
                                  "calibration_source": confidence.calibration_source}
        base = {
            "retrieved_chunks": rag.snapshots,
            "evidence_confidence": confidence_payload,
            "rag_rounds": rag.rounds,
            "rag_stop_reason": rag.stop_reason,
            "rag_steps": rag.steps,
            "trace": [*state["trace"],
                      f"node=agentic_retrieve route={settings.retrieval_route} "
                      f"rounds={rag.rounds} stop={rag.stop_reason} n={len(evidence)}",
                      f"node=agentic_retrieve analyzer={rag.analyzer_calls} "
                      f"judge={rag.judge_calls} latency={rag.latency_ms:.0f}ms"],
        }
        if rag.gate_passed:
            # citations 帧为覆盖语义:循环结束一次性全量发送最终合并证据
            if evidence:
                _emit(writer, {"type": "citations", "items": list(evidence)})
            return {**base, "evidence": evidence, "low_confidence": False,
                    "low_reason": "", "gate_passed": True}

        # ---- 未放行:本节点收口(澄清/拒答/转人工) ----
        if rag.stop_reason == STOP_CLARIFY and rag.clarify_question:
            reply = f"{CLARIFY_PREFIX}{rag.clarify_question}"
            actions: list[str] = []
            low_confidence = False
            low_reason = ""
        else:
            low_confidence = True
            low_reason = rag.reason or rag.stop_reason
            if rag.stop_reason == STOP_ESCALATE and rag.conflicts:
                reply = REFUSAL_CONFLICT
            else:
                detail = rag.reason or "证据不足"
                reply = (f"{REFUSAL_MARKER}这个问题我暂时无法给出可靠答案({detail[:80]}),"
                         "您可以换个问法,或选择下方方式继续。")
            actions = ["transfer_human", "create_ticket"]
            try:
                await asyncio.to_thread(
                    pool.insert, "retrieval_low_conf", session_id,
                    state["user_message"], f"agentic:{rag.stop_reason}:{low_reason[:200]}",
                    rag.snapshots or None)
            except Exception:
                logger.warning("受控检索低置信落池失败", exc_info=True)
        _emit(writer, {"type": "token", "content": reply})
        return {**base, "evidence": [], "low_confidence": low_confidence,
                "low_reason": low_reason, "gate_passed": False,
                "final_reply": reply, "messages": [AIMessage(content=reply)],
                "suggested_actions": actions}

    return agentic_retrieve_node
