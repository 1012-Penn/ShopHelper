"""MySQL 会话存储:conversations/messages 表承载多轮上下文与工具轨迹。"""
from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker

from app.models import Conversation, ConversationSummary, Message

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

    # ---- ch07:分层上下文——行带 id、锚点读写、分段摘要 ----

    async def get_rows(self, conversation_id: int,
                       upto_id: int | None = None) -> list[dict]:
        """全量消息行(带自增 id),id 升序;upto_id 限定 id ≤ upto_id(摘要批上界)。"""
        stmt = select(Message).where(Message.conversation_id == conversation_id)
        if upto_id is not None:
            stmt = stmt.where(Message.id <= upto_id)
        with self._factory() as session:
            rows = session.scalars(stmt.order_by(Message.id)).all()
            return [
                {
                    "id": r.id,
                    "role": r.role,
                    "content": r.content,
                    "tool_calls": r.tool_calls,
                    "tool_call_id": r.tool_call_id,
                }
                for r in rows
            ]

    async def get_anchors(self, conversation_id: int) -> dict:
        """三层边界 + 摘要投影:summary_upto/layer1_from 为 NULL 表示从未降级/未摘要。"""
        with self._factory() as session:
            conv = session.get(Conversation, conversation_id)
            if conv is None:
                return {"summary_upto": None, "layer1_from": None,
                        "summary_text": None, "summary_seqs": 0}
            seqs = session.scalar(
                select(func.count()).select_from(ConversationSummary)
                .where(ConversationSummary.conversation_id == conversation_id))
            return {"summary_upto": conv.summary_upto_msg_id,
                    "layer1_from": conv.layer1_from_msg_id,
                    "summary_text": conv.summary,
                    "summary_seqs": int(seqs or 0)}

    async def set_layer1_from(self, conversation_id: int, msg_id: int | None) -> None:
        """级联降级只挪锚点,不搬数据。"""
        with self._factory() as session:
            conv = session.get(Conversation, conversation_id)
            conv.layer1_from_msg_id = msg_id
            session.commit()

    async def next_seq(self, conversation_id: int) -> int:
        with self._factory() as session:
            seq = session.scalar(
                select(func.max(ConversationSummary.seq))
                .where(ConversationSummary.conversation_id == conversation_id))
            return int(seq or 0) + 1

    async def append_summary(self, conversation_id: int, seq: int, from_id: int,
                             upto_id: int, content: str) -> None:
        """单事务:插一段摘要 + 推进 summary_upto 锚点 + 重拼 summary 投影列。"""
        with self._factory() as session:
            session.add(ConversationSummary(
                conversation_id=conversation_id, seq=seq,
                from_msg_id=from_id, upto_msg_id=upto_id, content=content))
            conv = session.get(Conversation, conversation_id)
            conv.summary_upto_msg_id = upto_id
            segs = session.scalars(
                select(ConversationSummary.content)
                .where(ConversationSummary.conversation_id == conversation_id)
                .order_by(ConversationSummary.seq)).all()
            conv.summary = "\n".join(segs)
            session.commit()

    async def list_conversations(self, user_id: str = "guest") -> list[dict]:
        """多会话侧栏:新在前(updated_at DESC)+ 首问预览 + 已摘要标记。"""
        with self._factory() as session:
            convs = session.scalars(
                select(Conversation).where(Conversation.user_id == user_id)
                .order_by(Conversation.updated_at.desc(), Conversation.id.desc())).all()
            items = []
            for c in convs:
                first = session.scalar(
                    select(Message.content).where(Message.conversation_id == c.id,
                                                  Message.role == "user")
                    .order_by(Message.id).limit(1))
                items.append({
                    "id": c.id,
                    "preview": (first or "")[:100],
                    "summarized": c.summary is not None,
                    "updated_at": c.updated_at.isoformat() if c.updated_at else "",
                })
            return items
