"""低置信度池:REFUSAL_MARKER 检测 + 两入口落池(conversation_id 可空)。"""
from app.guard import REFUSAL_MARKER, LowConfidencePool, is_refusal
from app.models import LowConfidenceQuestion


def test_refusal_marker():
    assert is_refusal("【无法回答】这个问题我查不到")
    assert is_refusal("  【无法回答】xx")
    assert not is_refusal("普通回答")


def test_marker_is_single_source():
    from app.prompts import SERVICE_PROMPT_TEMPLATE

    assert REFUSAL_MARKER in SERVICE_PROMPT_TEMPLATE.template


def test_insert_two_sources(db_session_factory):
    pool = LowConfidencePool(db_session_factory)
    pool.insert("retrieval_low_conf", 1, "邮费多少", "知识库无相关内容")
    pool.insert("self_check", None, "会飞的手机怎么买", "模型自评证据不足:……")
    with db_session_factory() as s:
        rows = s.query(LowConfidenceQuestion).order_by(LowConfidenceQuestion.id).all()
    assert [(r.source, r.conversation_id, r.raw_question) for r in rows] == [
        ("retrieval_low_conf", 1, "邮费多少"),
        ("self_check", None, "会飞的手机怎么买"),
    ]
