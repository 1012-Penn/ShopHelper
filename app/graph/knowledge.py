"""知识路径:强制 RAG 检索 → 置信度闸 → 弱证据兜底。service/kb/pool 构造注入,测试可替身。"""
import asyncio

from langchain_core.messages import AIMessage

from app.guard import REFUSAL_MARKER

FALLBACK_REPLY = f"{REFUSAL_MARKER}这个问题我暂时无法给出可靠答案,您可以换个问法,或选择下方方式继续。"


def _emit(writer, frame: dict) -> None:
    if writer is not None:
        writer(frame)


def make_knowledge_nodes(service, kb, pool, snapshot_top_k: int = 3):
    async def retrieve_node(state, writer=None) -> dict:
        result = service.retrieve(state["resolved_message"], strategy="hybrid_rerank")
        rows = kb.get_chunks([r.chunk_id for r in result.items])
        evidence = [
            {"n": i + 1, "chunk_id": r.chunk_id,
             "question": (row.questions.splitlines() or [""])[0], "answer": row.answer,
             "category": row.category, "section_path": row.section_path or "",
             "score": r.score, "rerank_score": r.score}
            for i, (r, row) in enumerate(zip(result.items, rows))
        ]
        if hasattr(result, "snapshot"):
            snapshot = result.snapshot(snapshot_top_k)
        else:
            snapshot = [
                {"rank": i + 1, "chunk_id": r.chunk_id, "score": r.score,
                 "rerank_score": r.score}
                for i, r in enumerate(result.items[:snapshot_top_k])
            ]
        row_by_id = {r.chunk_id: row for r, row in zip(result.items, rows)}
        for item in snapshot:
            row = row_by_id.get(item["chunk_id"])
            if row is not None:
                item.update({"question": (row.questions.splitlines() or [""])[0],
                             "answer": row.answer, "category": row.category,
                             "section_path": row.section_path or "",
                             "text": f"{row.questions}\n{row.answer}"})
        top1 = f"{result.items[0].score:.2f}" if result.items else "0.00"
        _emit(writer, {"type": "tool_status", "name": "query_faq", "label": "知识检索"})
        confidence = getattr(result, "evidence_confidence", None)
        confidence_payload = None
        if confidence is not None:
            confidence_payload = {**confidence.signals,
                                  "passed": confidence.passed,
                                  "calibration_source": confidence.calibration_source}
        return {"evidence": evidence, "low_confidence": result.low_confidence,
                "low_reason": result.reason,
                "retrieved_chunks": snapshot,
                "evidence_confidence": confidence_payload,
                "trace": [*state["trace"], f"node=retrieve strategy=hybrid_rerank top1={top1} "
                                           f"low_confidence={result.low_confidence} n={len(evidence)}"]}

    async def gate_node(state, writer=None) -> dict:
        if state["low_confidence"]:
            # 落池可能内联标准化/查重(LLM),放线程池避免阻塞事件循环
            await asyncio.to_thread(
                pool.insert, "retrieval_low_conf", state["session_id"],
                state["user_message"], str(state.get("low_reason") or ""),
                state.get("retrieved_chunks") or None)
            return {"gate_passed": False,
                    "retrieved_chunks": state.get("retrieved_chunks", []),
                    "evidence_confidence": state.get("evidence_confidence"),
                    "trace": [*state["trace"], "node=gate blocked=low_confidence"]}
        return {"gate_passed": True, "trace": [*state["trace"], "node=gate passed"]}

    async def fallback_node(state, writer=None) -> dict:
        _emit(writer, {"type": "token", "content": FALLBACK_REPLY})
        return {"final_reply": FALLBACK_REPLY,
                "messages": [AIMessage(content=FALLBACK_REPLY)],  # user 由 initial_state 带入
                "trace": [*state["trace"], "node=fallback"]}

    return retrieve_node, gate_node, fallback_node
