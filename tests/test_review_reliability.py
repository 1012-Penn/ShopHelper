"""审核保存与可重试索引的行为回归,所有外部服务均用替身。"""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.flywheel import FlywheelService
from app.kb import KnowledgeBaseStore
from app.models import Base, KnowledgeChunk, ReviewQueue
from tests.helpers import (FakeEmbedding, FakeVectorStore, admin_login, csrf_headers,
                           fake_chat, seed_admin)


def seed_review(factory):
    service = FlywheelService(factory)
    return service.capture("self_check", None, "海外运费多少", "", None)


def linked_chunks(factory):
    with factory() as session:
        return session.scalars(select(KnowledgeChunk)).all()


def test_approval_replay_reuses_original_chunk(db_session_factory):
    service = FlywheelService(db_session_factory)
    rid = seed_review(db_session_factory)
    first = service.approve(rid, "按目的地计算")
    second = service.approve(rid, " 按目的地计算 ")
    assert first == second
    assert len(linked_chunks(db_session_factory)) == 1


def test_changed_answer_does_not_overwrite_approved_knowledge(db_session_factory):
    service = FlywheelService(db_session_factory)
    rid = seed_review(db_session_factory)
    service.approve(rid, "原答案")
    with pytest.raises(ValueError, match="已核准"):
        service.approve(rid, "不同答案")
    assert [row.answer for row in linked_chunks(db_session_factory)] == ["原答案"]


@pytest.mark.parametrize("failure", ["embed", "upsert"])
def test_index_failure_is_durable_and_restart_retry_is_scoped(db_session_factory, failure):
    class FlakyEmbedding(FakeEmbedding):
        fail = failure == "embed"

        def embed(self, texts):
            if self.fail:
                self.fail = False
                raise ConnectionError("injected embedding outage")
            return super().embed(texts)

    class FlakyVectors(FakeVectorStore):
        fail = failure == "upsert"

        def upsert(self, rows):
            if self.fail:
                self.fail = False
                raise ConnectionError("injected upsert outage")
            return super().upsert(rows)

    rid = seed_review(db_session_factory)
    with db_session_factory() as session:
        unrelated = KnowledgeChunk(category="其他", questions="别的问题", answer="别的答案")
        session.add(unrelated)
        session.commit()
        unrelated_id = unrelated.id
    kb, vectors, embedder = KnowledgeBaseStore(db_session_factory), FlakyVectors(), FlakyEmbedding()
    service = FlywheelService(db_session_factory, kb=kb, vectors=vectors, embedder=embedder)
    cid = service.approve(rid, "按目的地计算")  # 失败只能影响索引,不可回滚已保存审核或抛错
    detail = service.detail(rid)
    assert detail["knowledge_chunk_id"] == cid and detail["index_status"] == "pending"
    assert detail["review_status"] == "通过"
    # 新 service 不依赖旧进程任务或缓存。
    restarted = FlywheelService(db_session_factory, kb=kb, vectors=vectors, embedder=embedder)
    assert restarted.retry_index(rid) == cid
    assert restarted.detail(rid)["index_status"] == "done"
    assert restarted.retry_index(rid) == cid
    assert restarted.approve(rid, "按目的地计算") == cid
    assert len(linked_chunks(db_session_factory)) == 2  # 仅目标条目和未索引的干扰条目
    assert vectors.count() == 1
    assert kb.get_chunks([unrelated_id])[0].vectorize_status == "pending"


def test_review_list_exposes_saved_index_state(db_session_factory):
    service = FlywheelService(db_session_factory)
    rid = seed_review(db_session_factory)
    pending = service.detail(rid)
    assert pending["knowledge_chunk_id"] is None and pending["index_status"] is None
    cid = service.approve(rid, "答案")
    row = service.list_reviews("通过")[0]
    assert row["knowledge_chunk_id"] == cid and row["index_status"] == "pending"


def test_concurrent_approvals_have_one_knowledge_chunk(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'reviews.db'}", connect_args={"timeout": 20})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    rid = seed_review(factory)
    barrier = Barrier(4)

    def approve():
        barrier.wait(timeout=5)
        return FlywheelService(factory).approve(rid, "同一核准答案")

    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            ids = list(pool.map(lambda _: approve(), range(4)))
        assert len(set(ids)) == 1
        assert len(linked_chunks(factory)) == 1
    finally:
        engine.dispose()


async def test_approval_api_pending_conflict_and_retry_contract(make_client):
    client, app = await make_client(fake_chat("好的"))
    app.state.flywheel = FlywheelService(app.state.session_factory)
    seed_admin(app, "operator", "operator-pass-123", "operator")
    csrf = await admin_login(client, "operator", "operator-pass-123")
    headers = csrf_headers(csrf)
    rid = seed_review(app.state.session_factory)
    url = f"/api/review-queue/{rid}"
    assert (await client.post(url + "/retry-index", headers=headers)).status_code == 409
    assert (await client.post("/api/review-queue/999/retry-index",
                              headers=headers)).status_code == 404
    first = await client.post(url + "/approve", json={"approved_answer": "核准答案"},
                              headers=headers)
    assert first.status_code == 200
    assert first.json()["index_status"] == "pending"
    repeated = await client.post(url + "/approve", json={"approved_answer": "核准答案"},
                                 headers=headers)
    assert repeated.json() == first.json()
    conflict = await client.post(url + "/approve", json={"approved_answer": "另一答案"},
                                 headers=headers)
    assert conflict.status_code == 409
    retried = await client.post(url + "/retry-index", headers=headers)
    assert retried.status_code == 200 and retried.json() == first.json()
    app.state.flywheel = FlywheelService(
        app.state.session_factory, kb=KnowledgeBaseStore(app.state.session_factory),
        vectors=FakeVectorStore(), embedder=FakeEmbedding())
    done = await client.post(url + "/retry-index", headers=headers)
    assert done.status_code == 200 and done.json()["index_status"] == "done"
    assert done.json()["knowledge_chunk_id"] == first.json()["knowledge_chunk_id"]
    actions = [row["action"] for row in app.state.admin_audit_store.list_logs()]
    assert "review.approve" in actions
    assert "review.retry_index" in actions
