"""端到端:选择器轮→resume 旁路→子流程→agent;GET /api/orders;畸形 resume 422。"""
from tests.helpers import EchoResolver, GraphChatModel, StubExpand, post_chat_sse


class ToollessGraphModel(GraphChatModel):
    """agent 轮恒返纯文本(无工具),意图 JSON 可配。"""


async def _mk(make_client, intent_reply, turns, resolver=None, expander=None):
    model = ToollessGraphModel(intent_reply=intent_reply, turns=turns)
    return await make_client(model,
                             resolver=resolver or EchoResolver({}),
                             expander=expander or StubExpand({}))


async def test_ask_order_round_then_resume_completes(make_client):
    from app.prompts import ResolvedQuestion

    # 补全问法不含订单号(历史上只有商品语境)→ 槽位缺失走选择器
    resolver = EchoResolver({"它能退吗": ResolvedQuestion(
        resolved="我买的无线耳机能退吗", changed=True)})
    expander = StubExpand({"我买的无线耳机能退吗": ["无线耳机退货条件", "七天无理由时效"]})
    client, app = await _mk(make_client, '{"intent": "退款退货", "confidence": 0.9}',
                            [("text", "这一单签收未超7天,可以退。")],
                            resolver=resolver, expander=expander)
    from tests.test_chat_toolflow import _seed_kb_chunk
    _seed_kb_chunk(app, "r.md", "无线耳机退货条件", "签收后7天内可退。", "退货政策.md > 退货条件")

    frames = await post_chat_sse(client, {"message": "它能退吗"})
    types = [f["type"] for f in frames]
    assert "resolved" in types and "order_selector" in types
    assert "token" not in types  # 选择器轮没有答案流
    sel = next(f for f in frames if f["type"] == "order_selector")
    assert sel["question"] == "我买的无线耳机能退吗"
    assert sel["original"] == "它能退吗"
    assert sel["intent"] == "退款退货"
    assert [o["order_id"] for o in sel["items"]] == ["1001", "1002", "1003"]

    frames2 = await post_chat_sse(client, {
        "message": "它能退吗",
        "resume": {"order_id": "1001", "question": "我买的无线耳机能退吗",
                   "intent": "退款退货"}})
    types2 = [f["type"] for f in frames2]
    assert "order_selector" not in types2
    assert "citations" in types2
    assert any(f["type"] == "token" and "可以退" in f.get("content", "") for f in frames2)
    actions = next(f for f in frames2 if f["type"] == "actions")
    assert actions["items"][0]["action"] == "refund_form"
    assert actions["items"][0]["order_id"] == "1001"
    assert frames2[-1]["type"] == "done"


async def test_order_number_in_text_skips_selector(make_client):
    client, app = await _mk(make_client, '{"intent": "退款退货", "confidence": 0.9}',
                            [("text", "订单1001可以退。")])
    frames = await post_chat_sse(client, {"message": "订单1001能退吗"})
    types = [f["type"] for f in frames]
    assert "order_selector" not in types and "token" in types


async def test_other_intent_goes_agent(make_client):
    client, app = await _mk(make_client, '{"intent": "其他", "confidence": 0.2}',
                            [("text", "这个问题我需要进一步了解,您能说详细些吗")])
    frames = await post_chat_sse(client, {"message": "量子力学怎么解释"})
    assert any(f["type"] == "token" and "详细" in f.get("content", "") for f in frames)


async def test_orders_endpoint(make_client):
    client, app = await _mk(make_client, '{"intent": "闲聊"}', [("text", "你好呀")])
    resp = await client.get("/api/orders")
    assert resp.status_code == 200
    assert [o["order_id"] for o in resp.json()["items"]] == ["1001", "1002", "1003"]


async def test_resume_invalid_order_id_422(make_client):
    client, app = await _mk(make_client, '{"intent": "闲聊"}', [("text", "你好呀")])
    resp = await client.post("/api/chat", json={
        "message": "x", "resume": {"order_id": "abc", "question": "q"}})
    assert resp.status_code == 422


def test_initial_state_supports_dict_resume():
    from app.graph.builder import initial_state

    st = initial_state(1, "帮我退货", [], resume={"order_id": "1002", "question": "机械键盘能退吗", "intent": "退款退货"})
    assert st["resume_order_id"] == "1002"
    assert st["resume_question"] == "机械键盘能退吗"
    assert st["intent"] == "退款退货"
