import json

from sqlalchemy import select

from app.db import make_session_factory
from app.models import Faq, Ticket
from app.tools.definitions import TOOL_LABELS, build_tools
from tests.test_models import create_memory_engine


def _make():
    engine = create_memory_engine()
    factory = make_session_factory(engine)
    with factory() as s:
        s.add_all(Faq(question=q, answer=a, category=c) for q, a, c in [
            ("退货政策是什么?", "七天无理由退货。", "售后"),
            ("多久能发货?", "48 小时内发出。", "物流"),
        ])
        s.commit()
    return build_tools(factory), factory


def test_build_tools_names_and_labels():
    tools, _ = _make()
    names = {t.name for t in tools}
    assert names == {"query_order", "query_product", "query_logistics", "query_faq", "create_ticket"}
    assert TOOL_LABELS["query_logistics"] == "物流查询"


def test_mock_tools_return_structured_json():
    tools, _ = _make()
    by_name = {t.name: t for t in tools}
    order = json.loads(by_name["query_order"].invoke({"order_id": "1001"}))
    assert order["order_id"] == "1001" and "status" in order and "amount" in order
    product = json.loads(by_name["query_product"].invoke({"keyword": "手机"}))
    assert "name" in product and "price" in product
    logistics = json.loads(by_name["query_logistics"].invoke({"order_id": "1001"}))
    assert logistics["order_id"] == "1001" and len(logistics["traces"]) >= 1


def test_query_faq_hit_and_miss():
    tools, _ = _make()
    by_name = {t.name: t for t in tools}
    hit = json.loads(by_name["query_faq"].invoke({"keyword": "退货"}))
    assert len(hit["items"]) == 1 and "七天无理由" in hit["items"][0]["answer"]
    miss = json.loads(by_name["query_faq"].invoke({"keyword": "邮费"}))
    assert miss["items"] == []


def test_create_ticket_persists():
    tools, factory = _make()
    by_name = {t.name: t for t in tools}
    result = json.loads(by_name["create_ticket"].invoke({
        "conversation_id": 1, "description": "想退货", "ticket_type": "售后",
    }))
    assert result["ticket_no"].startswith("T")
    with factory() as s:
        ticket = s.get(Ticket, result["ticket_no"])
        assert ticket is not None
        assert ticket.status == "待处理"
        assert ticket.conversation_id == 1


def test_create_ticket_rejects_bad_type():
    tools, _ = _make()
    by_name = {t.name: t for t in tools}
    try:
        by_name["create_ticket"].invoke({
            "conversation_id": 1, "description": "x", "ticket_type": "砍价",
        })
        raised = False
    except Exception:
        raised = True
    assert raised
