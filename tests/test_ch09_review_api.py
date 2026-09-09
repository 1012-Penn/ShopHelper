"""待审队列后台 API 与 👎 回捞的接口级回归。"""
from sqlalchemy import select

from app.flywheel import FlywheelService
from app.models import KnowledgeChunk, LowConfidenceQuestion, ReviewQueue
from tests.helpers import fake_chat


async def make_flywheel_client(make_client, chat_text="好的"):
    client, app = await make_client(fake_chat(chat_text))
    # 生产 flywheel 绑定的是真实 MySQL factory,测试换成本用例的 SQLite factory
    app.state.flywheel = FlywheelService(app.state.session_factory)
    return client, app


async def test_review_queue_list_detail_approve_reject(make_client):
    client, app = await make_flywheel_client(make_client)

    app.state.flywheel.insert("retrieval_low_conf", 1, "会飞的手机怎么买", "证据置信度低",
                              [{"rank": 1, "chunk_id": 7, "score": 0.42, "question": "邮费",
                                "answer": "8 元"}])
    app.state.flywheel.insert("user_feedback", 2, "会飞的手机怎么买", "用户反馈未解决", [])

    listed = (await client.get("/api/review-queue")).json()["items"]
    assert len(listed) == 1  # 同义原话被确定性查重归并,一行一个缺口
    assert listed[0]["occurrence_count"] == 2

    detail = (await client.get(f"/api/review-queue/{listed[0]['id']}")).json()
    assert [q["source"] for q in detail["original_questions"]] == ["retrieval_low_conf", "user_feedback"]
    assert detail["original_questions"][0]["retrieved_chunks"][0]["chunk_id"] == 7

    missing = await client.get("/api/review-queue/999")
    assert missing.status_code == 404

    approved = await client.post(f"/api/review-queue/{listed[0]['id']}/approve",
                                 json={"approved_answer": "手机目前不能飞。"})
    assert approved.status_code == 200 and approved.json()["approved"] is True

    assert (await client.post(f"/api/review-queue/{listed[0]['id']}/reject")).status_code == 200


async def test_approve_writes_pending_knowledge_chunk(make_client):
    client, app = await make_flywheel_client(make_client)
    review_id = app.state.flywheel.insert("self_check", 1, "会员积分怎么兑换", "模型自评")

    await client.post(f"/api/review-queue/{review_id}/approve",
                      json={"approved_answer": "在会员页点兑换即可。"})

    with app.state.session_factory() as session:
        row = session.get(ReviewQueue, review_id)
        assert row.review_status == "通过" and row.approved_answer == "在会员页点兑换即可。"
        chunk = session.scalar(
            select(KnowledgeChunk)
            .where(KnowledgeChunk.section_path == f"review_queue:{review_id}"))
        assert chunk is not None and chunk.vectorize_status == "pending"


async def test_feedback_down_salvages_round_snapshot(make_client):
    client, app = await make_flywheel_client(make_client)
    snapshot = [{"rank": 1, "chunk_id": 3, "score": 0.61, "question": "退货", "answer": "7 天无理由"}]
    app.state.round_capture.record(9, "关税怎么算", snapshot)

    response = await client.post("/api/feedback", json={
        "session_id": 9, "question": "关税怎么算", "rating": "down"})
    assert response.status_code == 200

    with app.state.session_factory() as session:
        row = session.scalar(select(LowConfidenceQuestion))
        assert row.source == "user_feedback"
        assert row.retrieved_chunks == snapshot  # 回捞到当轮召回片段


async def test_feedback_down_without_round_lands_null_snapshot(make_client):
    client, app = await make_flywheel_client(make_client)

    await client.post("/api/feedback", json={
        "session_id": 9, "question": "没聊过的问题", "rating": "down"})

    with app.state.session_factory() as session:
        row = session.scalar(select(LowConfidenceQuestion))
        assert row.source == "user_feedback" and row.retrieved_chunks is None
