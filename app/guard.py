"""生成质量护栏(ch04):拒答标记唯一来源 + 低置信度问题池落库。"""
from sqlalchemy.orm import sessionmaker

from app.models import LowConfidenceQuestion

REFUSAL_MARKER = "【无法回答】"  # prompts 与 chat 路由都从这里取,禁止另写字面量


def is_refusal(text: str | None) -> bool:
    return bool(text) and text.lstrip().startswith(REFUSAL_MARKER)


class LowConfidencePool:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._factory = session_factory

    def insert(self, source: str, conversation_id: int | None, question: str, reason: str = "",
               retrieved_chunks: list | None = None, matched_review_id: int | None = None) -> int:
        """写入问题池,并可保存落池当轮的召回片段快照。"""
        with self._factory() as session:
            row = LowConfidenceQuestion(
                conversation_id=conversation_id, raw_question=question, source=source, reason=reason,
                retrieved_chunks=retrieved_chunks, matched_review_id=matched_review_id,
            )
            session.add(row)
            session.commit()
            return row.id
