"""ch05 图契约下的工具流:检索节点、工具回灌、citations 帧、低置信池(语义沿 ch04,载体换图)。"""
import json

from app.models import Faq
from tests.helpers import FakeEmbedding, GraphChatModel, post_chat_sse, seed_kb_chunk as _seed_kb_chunk


def _tc(name, args, id_, index=0):
    return {"name": name, "args": json.dumps(args, ensure_ascii=False), "id": id_,
            "index": index, "type": "tool_call_chunk"}


def _seed_faq(app):
    """给测试库灌一条可被 LIKE 命中的 FAQ。"""
    with app.state.session_factory() as s:
        s.add(Faq(question="退货政策是什么?", answer="七天无理由退货。", category="售后"))
        s.commit()


async def test_tool_flow_frame_order_and_persistence(make_client):
    model = GraphChatModel('{"intent": "商品咨询"}', [
        ("tools", [_tc("query_faq", {"keyword": "退货政策是什么"}, "call_1")]),
        ("text", "支持|七天无理由退货"),
    ])
    client, app = await make_client(model)
    # 图语义:知识路径先强制检索,须灌可命中向量知识让闸放行(仅 Faq 表会被闸拦)
    _seed_kb_chunk(app, "d.md", "退货政策是什么", "七天无理由退货。", "d.md > 退货政策")
    events = await post_chat_sse(client, {"message": "退货政策是什么"})

    assert events[0]["type"] == "session"
    status = [e for e in events if e["type"] == "tool_status"]
    assert any(s["name"] == "query_faq" for s in status)
    tokens = [e["content"] for e in events if e["type"] == "token"]
    assert "".join(tokens) == "支持七天无理由退货"
    assert events[-1]["type"] == "done"

    conv_id = events[0]["session_id"]
    history = await app.state.store.get_history(conv_id)
    # ch07 落库契约:工具轨迹(assistant 纯工具调用行 + tool 行)不落 messages 表
    assert [m["role"] for m in history] == ["user", "assistant"]
    assert history[1]["content"] == "支持七天无理由退货"


async def test_unknown_tool_error_fed_back_not_500(make_client):
    """模型点了未注册工具:错误字符串作为 tool 消息回灌,流程正常收敛。"""
    model = GraphChatModel('{"intent": "订单"}', [
        ("tools", [_tc("nope", {}, "call_x")]),
        ("text", "抱歉,该查询暂时不可用。"),
    ])
    client, app = await make_client(model)
    events = await post_chat_sse(client, {"message": "帮我查一下火星天气"})
    assert events[-1]["type"] == "done"
    history = await app.state.store.get_history(events[0]["session_id"])
    # ch07:未注册工具的错误回灌只活在当轮,落库只剩 user/assistant 文本
    assert [m["role"] for m in history] == ["user", "assistant"]
    assert history[1]["content"] == "抱歉,该查询暂时不可用。"


async def test_multi_tool_calls_all_fed_back(make_client):
    """模型一轮点多个工具:逐个执行、逐个回灌,徽章帧逐个下发。"""
    model = GraphChatModel('{"intent": "订单"}', [
        ("tools", [_tc("query_order", {"order_id": "1001"}, "c1", index=0),
                   _tc("query_logistics", {"order_id": "1001"}, "c2", index=1)]),
        ("text", "两路结果都拿到了。"),
    ])
    client, app = await make_client(model)
    events = await post_chat_sse(client, {"message": "订单 1001 的物流到哪了"})
    status = [e for e in events if e["type"] == "tool_status"]
    assert [s["label"] for s in status] == ["订单查询", "物流查询"]
    history = await app.state.store.get_history(events[0]["session_id"])
    assert [m["role"] for m in history] == ["user", "assistant"]  # ch07:工具行不落库
    assert history[1]["content"] == "两路结果都拿到了。"


# ---------- ch04:citations 帧 + 低置信度池(载体换图,语义不变) ----------

def _pool_rows(factory):
    from app.models import LowConfidenceQuestion

    with factory() as s:
        return [{"source": r.source, "raw_question": r.raw_question, "reason": r.reason or ""}
                for r in s.query(LowConfidenceQuestion).all()]


async def test_citations_frame_before_tokens(make_client):
    model = GraphChatModel('{"intent": "商品咨询"}', [
        ("tools", [_tc("query_faq", {"keyword": "邮费是多少"}, "call_1")]),
        ("text", "邮费一般8元|满99包邮"),
    ])
    client, app = await make_client(model)
    _seed_kb_chunk(app, "d.md", "邮费是多少", "普通订单邮费 8 元,满 99 元包邮。", "d.md > 邮费与运费")

    events = await post_chat_sse(client, {"message": "邮费是多少"})
    cites = [e for e in events if e["type"] == "citations"]
    assert len(cites) == 1
    items = cites[0]["items"]
    assert items and items[0]["n"] == 1 and "section_path" in items[0]
    types = [e["type"] for e in events]
    assert types.index("citations") < types.index("token")  # 帧在答案流之前
    assert _pool_rows(app.state.session_factory) == []      # 正常轮不落池


async def test_low_confidence_tool_result_pools(make_client):
    model = GraphChatModel('{"intent": "商品咨询"}', [
        ("tools", [_tc("query_faq", {"keyword": "量子力学怎么退货"}, "call_1")]),
        ("text", "这个我查不到|建议转人工"),
    ])
    client, app = await make_client(model)  # 空向量库 → query_faq low_confidence=true
    events = await post_chat_sse(client, {"message": "量子力学怎么退货"})
    assert [e for e in events if e["type"] == "citations"] == []
    rows = _pool_rows(app.state.session_factory)
    assert any(r["source"] == "retrieval_low_conf" and r["raw_question"] == "量子力学怎么退货"
               for r in rows)


async def test_refusal_answer_pools_self_check_only(make_client):
    """检索有结果、但模型自评答不了 → self_check 一条(闸未落池)。"""
    model = GraphChatModel('{"intent": "商品咨询"}', [
        ("tools", [_tc("query_faq", {"keyword": "退货政策是什么"}, "call_1")]),
        ("text", "【无法回答】知识库中的依据不足以回答|建议转人工"),
    ])
    client, app = await make_client(model)
    _seed_kb_chunk(app, "d.md", "退货政策是什么", "支持七天无理由退货,商品需保持完好。", "d.md > 退货政策")
    await post_chat_sse(client, {"message": "退货政策是什么"})
    rows = _pool_rows(app.state.session_factory)
    assert [r["source"] for r in rows] == ["self_check"]
    assert "模型自评证据不足" in rows[0]["reason"]


async def test_refusal_without_tool_call_pools(make_client):
    """ch04 验收补丁:模型不调工具直接拒答同样落 self_check 池。

    图语义:知识路径先强制检索,须闸通过(强证据)才轮到 Agent 直答——故灌可命中的知识。
    """
    model = GraphChatModel('{"intent": "商品咨询"}', [
        ("text", "【无法回答】知识库中的依据不足以回答|建议转人工"),
    ])
    client, app = await make_client(model)
    _seed_kb_chunk(app, "d.md", "退货政策是什么", "支持七天无理由退货,商品需保持完好。", "d.md > 退货政策")
    events = await post_chat_sse(client, {"message": "退货政策是什么"})
    assert [e for e in events if e["type"] == "citations"] == []
    rows = _pool_rows(app.state.session_factory)
    assert [r["source"] for r in rows] == ["self_check"]


async def test_citations_renumbered_across_multiple_tool_calls(make_client):
    """ch04 评审修复 + ch06 契约演进:知识路径证据占 n=1..k(不发帧不回灌),agent 内两次
    query_faq 的编号自 k+1 续排不撞号;citations 帧继承既有证据重发全量,工具 JSON 与帧一致。"""
    model = GraphChatModel('{"intent": "商品咨询"}', [
        ("tools", [_tc("query_faq", {"keyword": "邮费是多少"}, "call_1", index=0),
                   _tc("query_faq", {"keyword": "怎么申请退货"}, "call_2", index=1)]),
        ("text", "综合两条|结果如下"),
    ])
    client, app = await make_client(model)
    _seed_kb_chunk(app, "d.md", "邮费是多少", "普通订单邮费 8 元,满 99 元包邮。", "d.md > 邮费与运费")
    _seed_kb_chunk(app, "d2.md", "怎么申请退货", "订单详情页点击「申请售后」提交。", "d2.md > 申请退货")

    events = await post_chat_sse(client, {"message": "邮费多少,退货怎么申请"})
    cites = [e for e in events if e["type"] == "citations"]
    assert len(cites) >= 1
    ns = [it["n"] for it in cites[-1]["items"]]
    assert ns == list(range(1, len(ns) + 1)) and len(ns) >= 4  # 帧 = 证据[1,2] + 工具续排[3..]
    # ch07:工具行不落库,编号续排的校验对象改为最终 citations 帧(帧 = 证据[1,2] + 工具续排[3..])
    assert ns == list(range(1, len(ns) + 1)) and len(ns) >= 4
