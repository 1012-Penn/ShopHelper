"""生成质量护栏(ch04):拒答标记唯一来源 + 低置信度问题池落库。"""
from sqlalchemy.orm import sessionmaker

from app.models import LowConfidenceQuestion

REFUSAL_MARKER = "【无法回答】"  # prompts 与 chat 路由都从这里取,禁止另写字面量


def is_refusal(text: str | None) -> bool:
    return bool(text) and text.lstrip().startswith(REFUSAL_MARKER)


class LowConfidencePool:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._factory = session_factory

    def insert(self, source: str, conversation_id: int | None, question: str, reason: str = "") -> None:
        """source ∈ retrieval_low_conf / self_check / user_feedback(本章只写前两个,不去重)。"""
        with self._factory() as session:
            session.add(LowConfidenceQuestion(
                conversation_id=conversation_id, raw_question=question, source=source, reason=reason,
            ))
            session.commit()
