import asyncio

import pytest
from langchain_core.messages import AIMessageChunk

from tests.helpers import fake_chat, post_chat_sse


async def test_stream_event_sequence(make_client):
    client, _ = await make_client(fake_chat("你好 欢迎 光临"))
    events = await post_chat_sse(client, {"message": "在吗"})
    assert events[0] == {"type": "session", "session_id": events[0]["session_id"]}
    assert all(e["type"] == "token" for e in events[1:-1])
    assert events[-1] == {"type": "done"}
    assert "".join(e["content"] for e in events[1:-1]) == "你好 欢迎 光临"


async def test_new_session_id_generated_per_session(make_client):
    client, _ = await make_client(fake_chat("嗯", "好"))
    e1 = await post_chat_sse(client, {"message": "a"})
    e2 = await post_chat_sse(client, {"message": "b"})
    assert e1[0]["session_id"] != e2[0]["session_id"]


async def test_client_session_id_confirmed_and_reused(make_client):
    client, _ = await make_client(fake_chat("答一", "答二"))
    e1 = await post_chat_sse(client, {"message": "一", "session_id": 101})
    e2 = await post_chat_sse(client, {"message": "二", "session_id": 101})
    assert e1[0]["session_id"] == 101
    assert e2[0]["session_id"] == 101


async def test_reply_persisted_after_done(make_client):
    client, app = await make_client(fake_chat("在的 亲"))
    events = await post_chat_sse(client, {"message": "在吗"})
    sid = events[0]["session_id"]
    history = await app.state.store.get_history(sid)
    assert [m["role"] for m in history] == ["user", "assistant"]
    assert history[1]["content"] == "在的 亲"


async def test_upstream_error_emits_error_and_not_persisted(make_client):
    class Exploding:
        async def astream(self, messages):
            yield AIMessageChunk(content="部分")
            raise RuntimeError("boom")

    client, app = await make_client(Exploding())
    events = await post_chat_sse(client, {"message": "你好"})
    types = [e["type"] for e in events]
    assert types[0] == "session"
    assert types[-1] == "error"
    assert "done" not in types
    err = next(e for e in events if e["type"] == "error")
    assert "boom" in err["message"]
    history = await app.state.store.get_history(events[0]["session_id"])
    assert history == []


async def test_message_over_budget_returns_400(make_client):
    client, _ = await make_client(fake_chat("x"))
    resp = await client.post("/api/chat", json={"message": "超" * 3001})
    assert resp.status_code == 400


async def test_empty_message_rejected_422(make_client):
    client, _ = await make_client(fake_chat("x"))
    resp = await client.post("/api/chat", json={"message": ""})
    assert resp.status_code == 422


def test_prompt_messages_include_system_and_history():
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    from app.routers.chat import _to_prompt_messages

    trimmed = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "旧问"},
        {"role": "assistant", "content": "旧答"},
    ]
    msgs = _to_prompt_messages(trimmed, "新问")
    assert isinstance(msgs[0], SystemMessage) and msgs[0].content == "SYS"
    assert isinstance(msgs[1], HumanMessage) and msgs[1].content == "旧问"
    assert isinstance(msgs[2], AIMessage) and msgs[2].content == "旧答"
    assert isinstance(msgs[-1], HumanMessage) and msgs[-1].content == "新问"


class RecordingChatModel:
    """记录每轮实际收到的 prompt;reply 可变,非迭代器型,每轮 astream 都重新取值。"""

    def __init__(self, reply: str):
        self.reply = reply
        self.seen: list = []

    async def astream(self, messages):
        self.seen = list(messages)
        yield AIMessageChunk(content=self.reply)


async def test_second_round_prompt_carries_system_and_history(make_client):
    from langchain_core.messages import HumanMessage, SystemMessage

    model = RecordingChatModel("第一答")
    client, _ = await make_client(model)
    await post_chat_sse(client, {"message": "第一问", "session_id": 202})
    model.reply = "第二答"  # RecordingChatModel 不是迭代器型,直接改返回值即可
    await post_chat_sse(client, {"message": "第二问", "session_id": 202})

    seen = model.seen
    assert isinstance(seen[0], SystemMessage) and "小帮" in seen[0].content
    contents = [(type(m).__name__, m.content) for m in seen]
    assert ("HumanMessage", "第一问") in contents
    assert ("AIMessage", "第一答") in contents
    assert isinstance(seen[-1], HumanMessage) and seen[-1].content == "第二问"


@pytest.mark.xfail(
    reason="httpx 0.28.1 ASGITransport 在返回 Response 前完整运行 app 并缓冲全部响应体,"
    "客户端提前断开无法传播给应用;断开不落库契约(spec §4.1)在真实 ASGI 服务器下成立,"
    "本测试传输层无法模拟。详见 .superpowers/sdd/2026-09-04-ch01-pure-chat/final-fix-report.md",
    strict=True,
)
async def test_client_disconnect_before_done_not_persisted(make_client):
    """spec §4.1:客户端提前断开,本轮 user/assistant 两条都不落库。"""

    class SlowUpstream:
        async def astream(self, messages):
            yield AIMessageChunk(content="开头")
            await asyncio.sleep(5)
            yield AIMessageChunk(content="结尾")

    client, app = await make_client(SlowUpstream())
    async with client.stream(
        "POST", "/api/chat", json={"message": "你好", "session_id": 303}
    ) as resp:
        assert resp.status_code == 200
        first_chunk = None
        async for chunk in resp.aiter_text():  # 只读第一个 chunk 即退出(提前断开)
            first_chunk = chunk
            break
        assert first_chunk is not None
    await asyncio.sleep(0.1)  # 留出取消传播时间

    history = await app.state.store.get_history(303)
    assert history == []
