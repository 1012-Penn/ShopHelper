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
    e1 = await post_chat_sse(client, {"message": "一", "session_id": "s-1"})
    e2 = await post_chat_sse(client, {"message": "二", "session_id": "s-1"})
    assert e1[0]["session_id"] == "s-1"
    assert e2[0]["session_id"] == "s-1"


async def test_reply_persisted_after_done(make_client):
    client, app = await make_client(fake_chat("在的 亲"))
    events = await post_chat_sse(client, {"message": "在吗"})
    sid = events[0]["session_id"]
    data = await app.state.sessions.get(sid)
    assert [m["role"] for m in data.messages] == ["user", "assistant"]
    assert data.messages[1]["content"] == "在的 亲"


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
    data = await app.state.sessions.get(events[0]["session_id"])
    assert data.messages == []


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
