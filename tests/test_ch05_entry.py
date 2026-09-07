"""意图识别节点 + 写死分流四出口。"""
from tests.helpers import fake_chat

from app.graph.entry import INTENTS, intent_node, resolve_node, route
from app.graph.state import ChatState


def _state(**kw) -> ChatState:
    base = dict(session_id=1, user_message="退货政策是什么", resolved_message="", intent="",
                evidence=[], low_confidence=False, low_reason="", gate_passed=False,
                agent_steps=0, final_reply="", suggested_actions=[], trace=[],
                messages=[], history=[])
    base.update(kw)
    return base


async def test_intent_parses_llm_json():
    model = fake_chat('{"intent": "退款退货"}')
    out = await intent_node(_state(), model)
    assert out["intent"] == "退款退货"
    assert "intent=退款退货" in out["trace"][-1]


async def test_intent_fallback_on_garbage():
    model = fake_chat("我说不好")
    out = await intent_node(_state(), model)
    assert out["intent"] == "商品咨询"


async def test_intent_fallback_out_of_enum():
    model = fake_chat('{"intent": "聊天气"}')
    out = await intent_node(_state(), model)
    assert out["intent"] == "商品咨询"


def test_seven_intents_enum():
    assert INTENTS == ("物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊")


def test_route_four_outlets():
    s = _state()
    for intent, expect in [("商品咨询", "retrieve"), ("退款退货", "retrieve"),
                           ("物流", "agent"), ("订单", "agent"), ("售后", "agent"),
                           ("投诉", "complaint"), ("闲聊", "chitchat")]:
        assert route(_state(intent=intent)) == expect


def test_resolve_passthrough():
    out = resolve_node(_state())
    assert out["resolved_message"] == "退货政策是什么"
    assert "node=resolve" in out["trace"][-1]
