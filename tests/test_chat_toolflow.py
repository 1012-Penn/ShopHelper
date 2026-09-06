"""chat 单轮工具流:第一段绑工具 astream,有 tool_calls 则执行回灌,第二段不绑工具收敛。"""
import json

import pytest

from app.models import Faq
from app.store import ConversationStore
from tests.helpers import FakeEmbedding, fake_chat, post_chat_sse


def _seed_faq(app):
    """给测试库灌一条可被 LIKE 命中的 FAQ。"""
    with app.state.session_factory() as s:
        s.add(Faq(question="退货政策是什么?", answer="七天无理由退货。", category="售后"))
        s.commit()


class Chunk:
    """与 AIMessageChunk 契约一致的极简替身:.text / .tool_call_chunks。"""

    def __init__(self, text="", tool_call_chunks=None):
        self.text = text
        self.tool_call_chunks = tool_call_chunks or []


class FakeToolChatModel:
    """第 1 次 astream 吐 tool_call_chunks,第 2 次吐文本;记录 bind_tools / astream 次数。"""

    def __init__(self, second_reply="支持七天无理由退货", tool_args='{"keyword": "退货"}'):
        self.bind_calls = 0
        self.astream_calls = 0
        self.second_reply = second_reply
        self.tool_args = tool_args

    def bind_tools(self, tools):
        self.bind_calls += 1
        return self

    async def astream(self, messages):
        self.astream_calls += 1
        self.last_messages = list(messages)
        if self.astream_calls == 1:
            yield Chunk(tool_call_chunks=[{
                "name": "query_faq", "args": self.tool_args, "id": "call_1",
                "index": 0, "type": "tool_call_chunk",
            }])
        else:
            for piece in self.second_reply.split("|"):
                yield Chunk(text=piece)


def _install(client, model):
    client.app.state.chat_model = model


async def test_tool_flow_frame_order_and_persistence(make_client):
    model = FakeToolChatModel("支持|七天无理由退货")
    client, app = await make_client(model)
    _seed_faq(app)
    events = await post_chat_sse(client, {"message": "退货政策是什么"})

    assert events[0]["type"] == "session"
    status = [e for e in events if e["type"] == "tool_status"]
    assert status == [{"type": "tool_status", "name": "query_faq", "label": "FAQ 检索"}]
    tokens = [e["content"] for e in events if e["type"] == "token"]
    assert "".join(tokens) == "支持七天无理由退货"
    assert events[-1]["type"] == "done"

    conv_id = events[0]["session_id"]
    history = await app.state.store.get_history(conv_id)
    assert [m["role"] for m in history] == ["user", "assistant", "tool", "assistant"]
    assert history[1]["content"] is None
    assert history[1]["tool_calls"][0]["id"] == "call_1"
    assert history[1]["tool_calls"][0]["name"] == "query_faq"
    assert history[2]["tool_call_id"] == "call_1"
    assert "items" in history[2]["content"]  # FAQ 检索结果 JSON
    assert history[3]["content"] == "支持七天无理由退货"


async def test_tool_flow_second_call_not_bound(make_client):
    """单轮收敛:只有第一段绑工具,第二段裸调。"""
    model = FakeToolChatModel()
    client, _ = await make_client(model)
    await post_chat_sse(client, {"message": "退货政策是什么"})
    assert model.bind_calls == 1
    assert model.astream_calls == 2
    # 第二段 prompt 回灌了 assistant(tool_calls) 与 ToolMessage
    kinds = [type(m).__name__ for m in model.last_messages]
    assert "AIMessage" in kinds and "ToolMessage" in kinds


async def test_plain_path_no_tool_frames(make_client):
    client, _ = await make_client(fake_chat("每天 9 点到 21 点。"))
    events = await post_chat_sse(client, {"message": "你们几点营业?"})
    assert not [e for e in events if e["type"] == "tool_status"]
    assert events[-1]["type"] == "done"
    assert "".join(e["content"] for e in events if e["type"] == "token") == "每天 9 点到 21 点。"


async def test_unknown_tool_error_fed_back_not_500(make_client):
    """模型点了未注册工具:错误字符串作为 tool 消息回灌,流程正常收敛。"""
    model = FakeToolChatModel()
    client, app = await make_client(model)
    # 覆写第一段吐的 tool_call_chunks 为未注册工具
    original_astream = model.astream

    async def astream(messages):
        model.astream_calls += 1
        model.last_messages = list(messages)
        if model.astream_calls == 1:
            yield Chunk(tool_call_chunks=[{
                "name": "nope", "args": "{}", "id": "call_x",
                "index": 0, "type": "tool_call_chunk",
            }])
        else:
            yield Chunk(text="抱歉,该查询暂时不可用。")

    model.astream = astream
    events = await post_chat_sse(client, {"message": "帮我查一下火星天气"})
    assert events[-1]["type"] == "done"
    history = await app.state.store.get_history(events[0]["session_id"])
    tool_msg = history[2]
    assert "未注册" in tool_msg["content"]
    assert tool_msg["tool_call_id"] == "call_x"


async def test_multi_tool_calls_all_fed_back(make_client):
    """模型一轮点多个工具:逐个执行、逐个回灌,徽章帧逐个下发。"""
    model = FakeToolChatModel()
    client, app = await make_client(model)

    async def astream(messages):
        model.astream_calls += 1
        model.last_messages = list(messages)
        if model.astream_calls == 1:
            yield Chunk(tool_call_chunks=[
                {"name": "query_order", "args": '{"order_id": "1001"}', "id": "c1",
                 "index": 0, "type": "tool_call_chunk"},
                {"name": "query_logistics", "args": '{"order_id": "1001"}', "id": "c2",
                 "index": 1, "type": "tool_call_chunk"},
            ])
        else:
            yield Chunk(text="两路结果都拿到了。")

    model.astream = astream
    events = await post_chat_sse(client, {"message": "订单 1001 的物流到哪了"})
    status = [e for e in events if e["type"] == "tool_status"]
    assert [s["label"] for s in status] == ["订单查询", "物流查询"]
    history = await app.state.store.get_history(events[0]["session_id"])
    assert [m["role"] for m in history] == ["user", "assistant", "tool", "tool", "assistant"]
    assert [m["tool_call_id"] for m in history[2:4]] == ["c1", "c2"]


# ---------- ch04:citations 帧 + 低置信度池 ----------

def _pool_rows(factory):
    from app.models import LowConfidenceQuestion

    with factory() as s:
        return [{"source": r.source, "raw_question": r.raw_question, "reason": r.reason or ""}
                for r in s.query(LowConfidenceQuestion).all()]


def _seed_shipping_kb(app):
    """灌一条运费知识(questions 与 query 同形,FakeReranker 高分不误触低置信)。"""
    from app.chunking import Chunk
    from app.kb import KnowledgeBaseStore, vectorize_pending

    kb = KnowledgeBaseStore(app.state.session_factory)
    kb.replace_doc_chunks("d.md", [Chunk(category="售后政策", questions="邮费是多少",
                                         answer="普通订单邮费 8 元,满 99 元包邮。",
                                         section_path="d.md > 邮费与运费",
                                         content_type="faq", is_key_clause=False)])
    vectorize_pending(kb, app.state.vectors, FakeEmbedding())


def _seed_return_kb(app):
    """灌一条退货政策知识(配拒答用例:检索命中但模型自评拒答)。"""
    from app.chunking import Chunk
    from app.kb import KnowledgeBaseStore, vectorize_pending

    kb = KnowledgeBaseStore(app.state.session_factory)
    kb.replace_doc_chunks("d.md", [Chunk(category="售后政策", questions="退货政策是什么",
                                         answer="支持七天无理由退货,商品需保持完好。",
                                         section_path="d.md > 退货政策",
                                         content_type="faq", is_key_clause=False)])
    vectorize_pending(kb, app.state.vectors, FakeEmbedding())


async def test_citations_frame_before_tokens(make_client):
    model = FakeToolChatModel("邮费一般8元|满99包邮", tool_args='{"keyword": "邮费是多少"}')
    client, app = await make_client(model)
    _seed_shipping_kb(app)

    events = await post_chat_sse(client, {"message": "邮费是多少"})
    cites = [e for e in events if e["type"] == "citations"]
    assert len(cites) == 1
    items = cites[0]["items"]
    assert items and items[0]["n"] == 1 and "section_path" in items[0]
    types = [e["type"] for e in events]
    assert types.index("citations") < types.index("token")  # 帧在答案流之前
    assert _pool_rows(app.state.session_factory) == []      # 正常轮不落池


async def test_low_confidence_tool_result_pools(make_client):
    model = FakeToolChatModel("这个我查不到|建议转人工")
    client, app = await make_client(model)  # 空向量库 → query_faq low_confidence=true
    events = await post_chat_sse(client, {"message": "量子力学怎么退货"})
    assert [e for e in events if e["type"] == "citations"] == []
    rows = _pool_rows(app.state.session_factory)
    assert len(rows) == 1
    assert rows[0]["source"] == "retrieval_low_conf"
    assert rows[0]["raw_question"] == "量子力学怎么退货"


async def test_refusal_answer_pools_self_check(make_client):
    """检索有结果、但模型自评答不了 → 只落 self_check 一条。"""
    model = FakeToolChatModel("【无法回答】知识库中的依据不足以回答|建议转人工",
                              tool_args='{"keyword": "退货政策是什么"}')
    client, app = await make_client(model)
    _seed_return_kb(app)
    await post_chat_sse(client, {"message": "退货政策是什么"})
    rows = _pool_rows(app.state.session_factory)
    assert [r["source"] for r in rows] == ["self_check"]
    assert "模型自评证据不足" in rows[0]["reason"]


async def test_refusal_without_tool_call_pools(make_client):
    """ch04 验收补丁:模型不调工具直接拒答(超范围题)同样落 self_check 池。"""

    class DirectRefusalModel:
        """不吐 tool_calls,第一段直接流式给出拒答文本。"""

        def bind_tools(self, tools):
            return self

        async def astream(self, messages):
            yield Chunk("【无法回答】量子力学不在店铺服务范围内|建议咨询专业渠道")

    client, app = await make_client(DirectRefusalModel())
    events = await post_chat_sse(client, {"message": "量子力学怎么退货"})
    assert [e for e in events if e["type"] == "citations"] == []
    rows = _pool_rows(app.state.session_factory)
    assert [r["source"] for r in rows] == ["self_check"]
    assert "模型自评证据不足" in rows[0]["reason"]


class TwoCallToolModel:
    """第一段一次吐两个 tool_call(两次 query_faq),第二段收敛;测引用编号续排。"""

    def bind_tools(self, tools):
        return self

    async def astream(self, messages):
        self.last_messages = list(messages)
        if getattr(self, "_first_done", False):
            for piece in "综合两条|结果如下".split("|"):
                yield Chunk(text=piece)
        else:
            self._first_done = True
            yield Chunk(tool_call_chunks=[
                {"name": "query_faq", "args": '{"keyword": "邮费是多少"}', "id": "call_1",
                 "index": 0, "type": "tool_call_chunk"},
                {"name": "query_faq", "args": '{"keyword": "怎么申请退货"}', "id": "call_2",
                 "index": 1, "type": "tool_call_chunk"},
            ])


async def test_citations_renumbered_across_multiple_tool_calls(make_client):
    """评审修复:同轮两次 query_faq,第二次的 n 续排不与第一次冲突;回灌给模型的工具结果同步重写。"""
    client, app = await make_client(TwoCallToolModel())
    _seed_shipping_kb(app)
    from app.chunking import Chunk
    from app.kb import KnowledgeBaseStore, vectorize_pending

    kb = KnowledgeBaseStore(app.state.session_factory)
    kb.replace_doc_chunks("d2.md", [Chunk(category="售后政策", questions="怎么申请退货",
                                          answer="订单详情页点击「申请售后」提交。",
                                          section_path="d2.md > 申请退货",
                                          content_type="faq", is_key_clause=False)])
    # 同一向量库追加灌第二批(d2.md replace 会清掉同前缀,直接再灌一次并补第一份)
    from app.kb import KnowledgeBaseStore as KB
    kb2 = KB(app.state.session_factory)
    kb2.replace_doc_chunks("d.md", [Chunk(category="售后政策", questions="邮费是多少",
                                          answer="普通订单邮费 8 元,满 99 元包邮。",
                                          section_path="d.md > 邮费与运费",
                                          content_type="faq", is_key_clause=False)])
    vectorize_pending(kb2, app.state.vectors, FakeEmbedding())

    events = await post_chat_sse(client, {"message": "邮费多少,退货怎么申请"})
    cites = [e for e in events if e["type"] == "citations"]
    assert len(cites) == 1
    ns = [it["n"] for it in cites[0]["items"]]
    assert ns == list(range(1, len(ns) + 1)) and len(ns) >= 2  # 续排:从 1 连续无冲突
    # 回灌给模型的 tool 消息里 n 也已重写,与 citations 帧一致
    conv_id = events[0]["session_id"]
    history = await app.state.store.get_history(conv_id)
    tool_msgs = [m for m in history if m["role"] == "tool"]
    import json as _json
    ns_in_tool = [it["n"] for m in tool_msgs
                  for it in (_json.loads(m["content"]).get("items") or [])]
    assert ns_in_tool == list(range(1, len(ns_in_tool) + 1))
    assert set(ns_in_tool) == set(ns)
