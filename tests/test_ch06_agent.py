"""agent 子流程窄化:订单数据+政策证据注入、refund_form action、n_offset 续排。"""
import json

from app.graph.agent import make_agent_node
from tests.helpers import GraphChatModel, graph_state


def _state(**kw):
    """子流程轮默认底座:refund_flow=True + 订单1001;显式传参可覆盖。"""
    kw.setdefault("intent", "退款退货")
    kw.setdefault("resolved_message", "订单1001的无线耳机能退吗")
    kw.setdefault("refund_flow", True)
    kw.setdefault("order", {"order_id": "1001", "product": "无线耳机", "amount": 299.0,
                            "status": "已签收", "created_at": "2026-08-30"})
    return graph_state(**kw)


def _settings():
    from app.config import Settings

    return Settings(openai_api_key="sk-test", _env_file=None,
                    max_agent_steps=4, agent_token_budget=2000)


class _Registry:
    tools = []

    def labels(self):
        return {}


_EVIDENCE2 = [{"n": 1, "chunk_id": 11, "question": "a", "answer": "a",
               "category": "退货政策", "section_path": "a"},
              {"n": 2, "chunk_id": 12, "question": "b", "answer": "b",
               "category": "退货政策", "section_path": "b"}]


async def test_narrowing_injects_order_and_instruction():
    model = GraphChatModel(intent_reply="{}", turns=[("text", "这一单可以退。")])
    node = make_agent_node(model, _Registry(), _settings(), pool=None)
    out = await node(_state(evidence=[{"n": 1, "chunk_id": 11, "question": "退货条件",
                                       "answer": "签收后7天内", "category": "退货政策",
                                       "section_path": "退货政策.md"}]), writer=None)
    system = model.prompts[0][0].content
    assert "订单数据" in system and "无线耳机" in system
    assert "能不能退" in system and "1001" in system
    assert "退货条件" in system  # 证据块照旧注入
    assert out["final_reply"] == "这一单可以退。"


async def test_aftersale_narrowing_wording():
    model = GraphChatModel(intent_reply="{}", turns=[("text", "可以换新。")])
    node = make_agent_node(model, _Registry(), _settings(), pool=None)
    await node(_state(intent="售后"), writer=None)
    assert "修/换/退" in model.prompts[0][0].content


async def test_non_refund_converge_emits_refund_form_action():
    model = GraphChatModel(intent_reply="{}", turns=[("text", "这一单可以退。")])
    node = make_agent_node(model, _Registry(), _settings(), pool=None)
    out = await node(_state(), writer=None)
    assert out["suggested_actions"] == ["refund_form"]


async def test_refusal_keeps_transfer_actions():
    from app.guard import REFUSAL_MARKER

    model = GraphChatModel(intent_reply="{}",
                           turns=[("text", REFUSAL_MARKER + "证据不足,建议转人工。")])
    node = make_agent_node(model, _Registry(), _settings(), pool=None)
    out = await node(_state(), writer=None)
    assert out["suggested_actions"] == ["transfer_human", "create_ticket"]


async def test_non_refund_flow_turn_gets_no_refund_form():
    model = GraphChatModel(intent_reply="{}", turns=[("text", "无线耳机 299 元。")])
    node = make_agent_node(model, _Registry(), _settings(), pool=None)
    out = await node(_state(refund_flow=False, order={}, intent="商品咨询"), writer=None)
    assert out["suggested_actions"] == []


async def test_query_faq_numbering_continues_after_evidence():
    """子流程证据已有 n=1..2 时,agent 内 query_faq 的引用编号从 3 续排(防撞号)。"""
    tool_chunks = [{"name": "query_faq", "args": '{"keyword": "退货"}', "id": "t1", "index": 0}]
    model = GraphChatModel(intent_reply="{}", turns=[
        ("tools", tool_chunks),
        ("text", "结合政策这一单可以退[3]。"),
    ])
    frames = []

    class Reg(_Registry):
        tools = ["query_faq"]

        def labels(self):
            return {"query_faq": "FAQ 检索"}

        async def execute(self, name, args):
            return json.dumps({"items": [{"n": 1, "chunk_id": 99, "question": "运费",
                                          "answer": "8 元", "category": "退货政策",
                                          "section_path": "p"}],
                               "low_confidence": False, "reason": "",
                               "filter_fallback": False, "degraded": False},
                              ensure_ascii=False)

    node = make_agent_node(model, Reg(), _settings(), pool=None)
    await node(_state(evidence=_EVIDENCE2), writer=frames.append)
    cite = [f for f in frames if f["type"] == "citations"][-1]
    assert [it["n"] for it in cite["items"]] == [1, 2, 3]  # 既有证据 + 续排的工具证据


async def test_unknown_order_gets_no_refund_form():
    """M-4 回归:订单未命中(error dict)时不发 refund_form。"""
    model = GraphChatModel(intent_reply="{}", turns=[("text", "这一单可以退。")])
    node = make_agent_node(model, _Registry(), _settings(), pool=None)
    out = await node(_state(order={"order_id": "9999", "error": "未找到该订单"}), writer=None)
    assert out["suggested_actions"] == ["refund_form"]  # agent 侧照常建议,由 log 层裁剪

    from app.graph.logging_node import make_log_node
    from tests.helpers import graph_state
    frames = []

    class _Store:
        async def append(self, sid, msgs):
            pass

    class _Pool:
        def insert(self, *a):
            pass

    st = graph_state(final_reply="这一单可以退。", intent="退款退货", refund_flow=True,
                     suggested_actions=["refund_form"],
                     order={"order_id": "9999", "error": "未找到该订单"},
                     messages=[{"role": "user", "content": "x"}], trace=[])
    await make_log_node(_Store(), _Pool())(st, writer=frames.append)
    assert frames == []  # log 层裁剪:未知订单无 actions 帧
