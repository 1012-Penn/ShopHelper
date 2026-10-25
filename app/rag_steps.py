"""ch12 rag_loop_steps 审计存储:逐轮插入 + 末轮回填停止原因。

仅可审计(ADR-1):写入失败只 warning 绝不阻断检索回合(审计不拦执行,同 tool_audit 惯例)。
"""
import logging

from sqlalchemy import select, update
from sqlalchemy.orm import sessionmaker

from app.models import RagLoopStep

logger = logging.getLogger(__name__)


class RagLoopStepStore:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._factory = session_factory

    def insert_step(self, *, session_id: int, turn_user_msg_id: str, step: dict) -> None:
        """一轮一行;stop_reason 由 finalize_stop 在循环结束后回填到末行。"""
        try:
            with self._factory() as session:
                session.add(RagLoopStep(
                    conversation_id=session_id,
                    turn_user_msg_id=turn_user_msg_id[:64],
                    round_index=int(step.get("round_index", 0)),
                    phase=str(step.get("phase", ""))[:16],
                    query=str(step.get("query", "")),
                    queries=list(step.get("queries") or []),
                    sub_queries=list(step.get("sub_queries") or []),
                    evidence_ids=list(step.get("evidence_ids") or []),
                    new_evidence_ids=list(step.get("new_evidence_ids") or []),
                    merged_evidence_ids=list(step.get("merged_evidence_ids") or []),
                    judge_answered=step.get("judge_answered"),
                    analyzer_action=(str(step["analyzer_action"])[:24]
                                     if step.get("analyzer_action") else None),
                    conflicts=list(step.get("conflicts") or []),
                    missing_aspects=list(step.get("missing_aspects") or []),
                    stop_reason=None,
                    latency_ms=int(float(step.get("latency_ms") or 0)),
                ))
                session.commit()
        except Exception:
            logger.warning("rag_loop_steps 写入失败(不阻断检索回合)", exc_info=True)

    def finalize_stop(self, session_id: int, turn_user_msg_id: str, stop_reason: str) -> None:
        """把循环终止原因回填到本轮最后一行(按 id 取最大)。"""
        try:
            with self._factory() as session:
                last_id = session.scalar(
                    select(RagLoopStep.id)
                    .where(RagLoopStep.conversation_id == session_id,
                           RagLoopStep.turn_user_msg_id == turn_user_msg_id)
                    .order_by(RagLoopStep.id.desc()).limit(1))
                if last_id is None:
                    return
                session.execute(
                    update(RagLoopStep).where(RagLoopStep.id == last_id)
                    .values(stop_reason=str(stop_reason or "")[:32]))
                session.commit()
        except Exception:
            logger.warning("rag_loop_steps 停止原因回填失败(不阻断检索回合)", exc_info=True)
