"""resolve 正式版 / intent 置信度与降级路 / route 八出口。"""
from langchain_core.messages import AIMessage

from app.graph.entry import INTENT_FALLBACK, make_intent_node, make_resolve_node, route
from app.prompts import ResolvedQuestion
from tests.helpers import EchoResolver, fake_chat


def _state(**kw):
    base = dict(session_id=1, user_message="它能退吗", resolved_message="", intent="",
                intent_confidence=0.0, evidence=[], low_confidence=False, low_reason="",
                gate_passed=False, agent_steps=0, final_reply="", suggested_actions=[],
                trace=[], messages=[], history=[], order={}, expand_queries=[],
                refund_flow=False, pending_order_id="", resume_order_id="", resume_question="")
    base.update(kw)
    return base


class _SeqModel:
    """按序返回预置文本的 ainvoke 替身(意图降级路用)。"""

    def __init__(self, replies):
        self.replies = list(replies)

    async def ainvoke(self, messages, **kwargs):
        return AIMessage(content=self.replies.pop(0))


async def test_resolve_rewrites_with_history():
    resolver = EchoResolver({"它能退吗": ResolvedQuestion(resolved="订单1001的无线耳机能退吗", changed=True)})
    frames = []
    node = make_resolve_node(resolver)
    out = await node(_state(history=[{"role": "user", "content": "SH-E300耳机多少钱"}]),
                     writer=frames.append)
    assert out["resolved_message"] == "订单1001的无线耳机能退吗"
    assert frames and frames[0]["type"] == "resolved" and frames[0]["changed"] is True
    assert "node=resolve changed=True" in out["trace"][-1]


async def test_resolve_passthrough_when_complete():
    node = make_resolve_node(EchoResolver({}))  # 未命中映射 → 原样透传 changed=False
    frames = []
    out = await node(_state(user_message="退货政策是什么"), writer=frames.append)
    assert out["resolved_message"] == "退货政策是什么"
    assert frames == []  # changed=False 不发帧
    assert "changed=False" in out["trace"][-1]


async def test_resolve_resume_bypass_skips_llm():
    node = make_resolve_node(EchoResolver({}))
    frames = []
    out = await node(_state(resume_question="订单1002的机械键盘能退吗",
                            resume_order_id="1002"), writer=frames.append)
    assert out["resolved_message"] == "订单1002的机械键盘能退吗"
    assert frames[0]["type"] == "resolved"
    assert "resume_bypass" in out["trace"][-1]


async def test_resolve_llm_error_falls_back_passthrough():
    class Boom:
        async def ainvoke(self, text):
            raise RuntimeError("上游挂了")

    node = make_resolve_node(Boom())
    out = await node(_state(), writer=None)
    assert out["resolved_message"] == "它能退吗"


async def test_intent_outputs_confidence_and_trace():
    node = make_intent_node(fake_chat('{"intent": "退款退货", "confidence": 0.9}'))
    out = await node(_state(resolved_message="订单1001的无线耳机能退吗"))
    assert out["intent"] == "退款退货" and out["intent_confidence"] == 0.9
    assert "confidence=0.90" in out["trace"][-1]


async def test_intent_malformed_goes_other_with_flag():
    node = make_intent_node(fake_chat("我说不好"))
    out = await node(_state())
    assert out["intent"] == "其他" and out["intent_confidence"] == 0.0
    assert "malformed=true" in out["trace"][-1]


async def test_intent_escalation_low_confidence_rejudges():
    small = _SeqModel(['{"intent": "商品咨询", "confidence": 0.3}'])
    big = _SeqModel(['{"intent": "退款退货", "confidence": 0.95}'])
    node = make_intent_node(small, escalator=big, floor=0.6)
    out = await node(_state())
    assert out["intent"] == "退款退货" and out["intent_confidence"] == 0.95
    assert "escalated=true" in out["trace"][-1]


async def test_intent_escalation_skipped_when_confident():
    small = fake_chat('{"intent": "闲聊", "confidence": 0.9}')
    big = _SeqModel(['{"intent": "退款退货", "confidence": 0.9}'])  # 不应被消费
    node = make_intent_node(small, escalator=big, floor=0.6)
    out = await node(_state())
    assert out["intent"] == "闲聊"
    assert big.replies  # 未触发重判


def test_route_refund_flows_to_prepare_order():
    assert route(_state(intent="退款退货")) == "prepare_order"
    assert route(_state(intent="售后")) == "prepare_order"


def test_route_other_and_unchanged_outlets():
    assert route(_state(intent="其他")) == "agent"
    for intent, expect in [("商品咨询", "retrieve"), ("物流", "agent"), ("订单", "agent"),
                           ("投诉", "complaint"), ("闲聊", "chitchat")]:
        assert route(_state(intent=intent)) == expect
    assert INTENT_FALLBACK == "其他"
