from sqlalchemy import select

from app.flywheel import FlywheelService, NormalizedQuestion
from app.models import KnowledgeChunk, LowConfidenceQuestion, ReviewQueue


def _normalizer(raw: str) -> NormalizedQuestion:
    return NormalizedQuestion("国际件运费怎么算", "国际件运费按目的地和重量计算,请联系客服查询。")


def test_capture_persists_source_and_retrieval_snapshot(db_session_factory):
    service = FlywheelService(db_session_factory, normalizer=_normalizer)
    review_id = service.capture(
        source="retrieval_low_conf", conversation_id=7, raw_question="国外快递费怎么算",
        reason="证据置信度低", retrieved_chunks=[
            {"rank": 1, "chunk_id": 12, "text": "国际件运费…", "score": 0.04}
        ],
    )

    with db_session_factory() as session:
        row = session.get(LowConfidenceQuestion, 1)
        review = session.get(ReviewQueue, review_id)
        assert row.source == "retrieval_low_conf"
        assert row.retrieved_chunks[0]["chunk_id"] == 12
        assert row.matched_review_id == review_id
        assert review.normalized_question == "国际件运费怎么算"
        assert review.occurrence_count == 1


def test_capture_semantic_match_accumulates_in_one_review_row(db_session_factory):
    service = FlywheelService(db_session_factory, normalizer=_normalizer)
    first = service.capture("self_check", 1, "国外运费怎么收", "模型自评不足", [])
    second = service.capture("user_feedback", 1, "国际快递收费标准", "用户点踩", [])

    assert second == first
    with db_session_factory() as session:
        review = session.get(ReviewQueue, first)
        assert review.occurrence_count == 2
        rows = session.scalars(select(LowConfidenceQuestion).order_by(LowConfidenceQuestion.id)).all()
        assert [row.source for row in rows] == ["self_check", "user_feedback"]
        assert all(row.matched_review_id == first for row in rows)


def test_approve_writes_pending_knowledge_chunk_and_rejects(db_session_factory):
    service = FlywheelService(db_session_factory, normalizer=_normalizer)
    review_id = service.capture("retrieval_low_conf", 1, "国外快递费怎么算", "低置信", [])

    chunk_id = service.approve(review_id, "国际件按目的地和重量计算,具体费用以结算页为准。")
    assert chunk_id is not None
    with db_session_factory() as session:
        review = session.get(ReviewQueue, review_id)
        chunk = session.get(KnowledgeChunk, chunk_id)
        assert review.review_status == "通过"
        assert review.approved_answer.startswith("国际件按")
        assert chunk.vectorize_status == "pending"
        assert chunk.questions == "国际件运费怎么算"

    rejected_id = service.capture("user_feedback", 1, "另一个缺口", "点踩", [])
    service.reject(rejected_id)
    with db_session_factory() as session:
        assert session.get(ReviewQueue, rejected_id).review_status == "驳回"
