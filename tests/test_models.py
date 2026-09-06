from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.models import (
    Base,
    Conversation,
    Faq,
    FaithCase,
    KnowledgeChunk,
    LowConfidenceQuestion,
    Message,
    QaStaging,
    Ticket,
)


def create_memory_engine():
    from sqlalchemy.pool import StaticPool

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,  # 全线程共享一条连接,工具在工作线程执行时也能看到数据
    )
    Base.metadata.create_all(engine)
    return engine


def _seed(session: Session) -> int:
    conv = Conversation(user_id="guest")
    session.add(conv)
    session.flush()
    session.add(Faq(question="退货政策是什么", answer="七天无理由退货", category="售后"))
    session.add(Message(conversation_id=conv.id, role="user", content="退货政策是什么"))
    session.add(
        Message(
            conversation_id=conv.id,
            role="assistant",
            content=None,
            tool_calls=[{"name": "query_faq", "args": {"keyword": "退货"}, "id": "call_1"}],
        )
    )
    session.add(Message(conversation_id=conv.id, role="tool", content='{"items":[]}', tool_call_id="call_1"))
    session.add(Ticket(ticket_no="T20260904001", conversation_id=conv.id, description="想退货", ticket_type="售后"))
    session.commit()
    return conv.id


def test_models_roundtrip():
    engine = create_memory_engine()
    with Session(engine) as session:
        conv_id = _seed(session)
        conv = session.get(Conversation, conv_id)
        assert conv.status == "进行中"
        assert session.scalar(select(Faq).limit(1)).answer == "七天无理由退货"
        msgs = session.scalars(select(Message).order_by(Message.id)).all()
        assert [m.role for m in msgs] == ["user", "assistant", "tool"]
        assert msgs[2].tool_call_id == "call_1"
        assert session.get(Ticket, "T20260904001").status == "待处理"


def test_knowledge_chunk_and_staging_roundtrip():
    engine = create_memory_engine()
    with Session(engine) as session:
        chunk = KnowledgeChunk(category="售后", questions="运费说明", answer="邮费 8 元")
        session.add(chunk)
        session.commit()
        assert chunk.id is not None
        assert chunk.vectorize_status == "pending"
        assert chunk.is_key_clause == 0

        staged = QaStaging(batch_no="b1", source_ref="1", question="q", answer="a")
        session.add(staged)
        session.commit()
        assert staged.status == "extracted"


def test_low_confidence_question_roundtrip():
    engine = create_memory_engine()
    with Session(engine) as session:
        session.add(LowConfidenceQuestion(conversation_id=None, raw_question="量子力学怎么退货",
                                          source="self_check", reason="模型自评证据不足"))
        session.commit()
        row = session.query(LowConfidenceQuestion).one()
    assert row.source == "self_check"
    assert row.created_at is not None


def test_faith_case_roundtrip():
    engine = create_memory_engine()
    with Session(engine) as session:
        session.add(FaithCase(eval_id="A01", bucket="A_policy", query="q", answer="a", reason="r",
                              citations=[{"n": 1, "chunk_id": 7}], judge_model="deepseek-chat"))
        session.commit()
        row = session.query(FaithCase).one()
    assert row.status == "未解决" and row.seen_count == 1 and row.strategy == "hybrid_rerank"
