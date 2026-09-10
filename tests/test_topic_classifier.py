"""ch10 主题分类器旁路服务:落库、批量归类、分布聚合、API 优雅降级。"""
import json

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from app.models import LowConfidenceQuestion, TopicClassification
from app.store import TopicStore
from app.topic_taxonomy import LABEL_COUNT, TOPIC_LABELS, validate_labels


def _seed_pool(factory, questions: list[str]) -> list[int]:
    ids = []
    with factory() as session:
        for q in questions:
            row = LowConfidenceQuestion(raw_question=q, source="retrieval_low_conf")
            session.add(row)
            session.flush()
            ids.append(row.id)
        session.commit()
    return ids


def test_pending_questions_only_unclassified_in_insertion_order(db_session_factory):
    factory = db_session_factory
    ids = _seed_pool(factory, ["怎么退货", "发票丢了", "快递几天到"])
    store = TopicStore(factory)
    store.save_results([(ids[1], ["发票"])])

    pending = store.pending_questions(limit=10)
    assert [row["id"] for row in pending] == [ids[0], ids[2]]

    limited = store.pending_questions(limit=1)
    assert [row["id"] for row in limited] == [ids[0]]


def test_save_results_upserts_one_row_per_question(db_session_factory):
    factory = db_session_factory
    (ids) = _seed_pool(factory, ["怎么退货"])
    store = TopicStore(factory)
    assert store.save_results([(ids[0], ["退换货"])]) == 1
    assert store.save_results([(ids[0], ["退换货", "尺码"])]) == 1  # 重跑覆盖,不新增

    with factory() as session:
        rows = session.scalars(select(TopicClassification)).all()
        assert len(rows) == 1
        assert rows[0].labels == ["退换货", "尺码"]


def test_distribution_counts_every_label_including_zero(db_session_factory):
    factory = db_session_factory
    ids = _seed_pool(factory, ["怎么退货", "尺码大了", "还没归类的"])
    store = TopicStore(factory)
    store.save_results([(ids[0], ["退换货"]), (ids[1], ["尺码", "退换货"])])

    result = store.distribution()
    counts = {item["label"]: item["count"] for item in result["items"]}
    assert len(result["items"]) == LABEL_COUNT
    assert counts["退换货"] == 2 and counts["尺码"] == 1 and counts["发票"] == 0
    assert result["classified"] == 2
    assert result["unclassified"] == 1


def test_classifier_service_cleans_and_validates_labels():
    from app.topic_classifier import StubTopicClassifier

    service = StubTopicClassifier({"手机[PHONE] 怎么退款": ["支付", "退换货"],
                                   "乱写的": ["不存在的类"]})
    labels = service.predict(["  手机13812345678 怎么退款 ", "乱写的"])
    assert labels[0] == ["支付", "退换货"]
    assert labels[1] == []  # 非法标签被术语表闸丢弃


def test_classify_batch_writes_results_and_top1_fallback(db_session_factory):
    from app.topic_classifier import StubTopicClassifier

    factory = db_session_factory
    ids = _seed_pool(factory, ["怎么退货", "瞎写的没有标签"])
    store = TopicStore(factory)
    service = StubTopicClassifier({"怎么退货": ["退换货"], "瞎写的没有标签": []})
    # 全类低于阈值时按 Top-1 兜底,保证池子里每条都归得到类
    service.top1_fallback = True

    classified = service.classify_batch(store, limit=10)
    assert classified == 2
    result = store.distribution()
    assert result["classified"] == 2 and result["unclassified"] == 0


def test_onnx_service_unavailable_without_model(tmp_path):
    from app.config import Settings
    from app.topic_classifier import TopicClassifierService

    settings = Settings(topic_model_dir=str(tmp_path / "absent"))
    service = TopicClassifierService(settings)
    assert service.available() is False
    with pytest.raises(FileNotFoundError):
        service.predict(["怎么退货"])


@pytest.mark.asyncio
async def test_topics_api_distribution_and_batch(db_session_factory, monkeypatch):
    from app.main import create_app
    from app.config import Settings

    app = create_app(Settings(openai_api_key="sk-test", embedding_api_key="sk-test"))
    factory = db_session_factory
    ids = _seed_pool(factory, ["怎么退货", "快递几天到"])
    app.state.topic_store = TopicStore(factory)
    from app.topic_classifier import StubTopicClassifier
    app.state.topic_classifier = StubTopicClassifier({"怎么退货": ["退换货"], "快递几天到": ["物流"]})

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
        before = (await client.get("/api/topics/distribution")).json()
        assert before["classified"] == 0 and before["unclassified"] == 2

        batched = (await client.post("/api/topics/classify-batch", json={"limit": 10})).json()
        assert batched["classified"] == 2

        after = (await client.get("/api/topics/distribution")).json()
        counts = {item["label"]: item["count"] for item in after["items"]}
        assert counts["退换货"] == 1 and counts["物流"] == 1
        assert after["unclassified"] == 0


@pytest.mark.asyncio
async def test_topics_api_returns_503_when_model_missing(db_session_factory):
    from app.main import create_app
    from app.config import Settings
    from app.topic_classifier import TopicClassifierService
    from app.store import TopicStore

    app = create_app(Settings(openai_api_key="sk-test", embedding_api_key="sk-test"))
    app.state.topic_store = TopicStore(db_session_factory)
    app.state.topic_classifier = TopicClassifierService(
        Settings(topic_model_dir="models/definitely-absent"))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
        response = await client.post("/api/topics/classify-batch", json={"limit": 5})
        assert response.status_code == 503
