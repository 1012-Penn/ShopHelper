"""MySQL 会话存储:conversations/messages 表承载多轮上下文与工具轨迹。"""
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from app.models import Conversation, Message

class ConversationStore:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._factory = session_factory

    async def resolve(self, session_id: int | None) -> int:
        """缺省则新建会话壳;客户端给的未知 id 一并注册(ch01 语义延续)。"""
        with self._factory() as session:
            if session_id is not None and session.get(Conversation, session_id) is not None:
                return session_id
            conv = Conversation(id=session_id, user_id="guest")  # id=None → 自增
            session.add(conv)
            session.commit()
            return conv.id

    async def exists(self, conversation_id: int) -> bool:
        with self._factory() as session:
            return session.get(Conversation, conversation_id) is not None

    async def get_history(self, conversation_id: int) -> list[dict]:
        with self._factory() as session:
            rows = session.scalars(
                select(Message).where(Message.conversation_id == conversation_id).order_by(Message.id)
            ).all()
            return [
                {
                    "role": r.role,
                    "content": r.content,
                    "tool_calls": r.tool_calls,
                    "tool_call_id": r.tool_call_id,
                }
                for r in rows
            ]

    async def append(self, conversation_id: int, msgs: list[dict]) -> None:
        with self._factory() as session:
            for m in msgs:
                session.add(Message(
                    conversation_id=conversation_id,
                    role=m["role"],
                    content=m.get("content"),
                    tool_calls=m.get("tool_calls"),
                    tool_call_id=m.get("tool_call_id"),
                ))
            session.commit()
