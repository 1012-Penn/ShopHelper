"""chat 单轮工具流:第一段绑工具 astream,有 tool_calls 则执行回灌,第二段不绑工具收敛。"""
import json

import pytest

from app.models import Faq
from app.store import ConversationStore
from tests.helpers import fake_chat, post_chat_sse


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

    def __init__(self, second_reply="支持七天无理由退货"):
        self.bind_calls = 0
        self.astream_calls = 0
        self.second_reply = second_reply

    def bind_tools(self, tools):
        self.bind_calls += 1
        return self

    async def astream(self, messages):
        self.astream_calls += 1
        self.last_messages = list(messages)
        if self.astream_calls == 1:
            yield Chunk(tool_call_chunks=[{
                "name": "query_faq", "args": '{"keyword": "退货"}', "id": "call_1",
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
