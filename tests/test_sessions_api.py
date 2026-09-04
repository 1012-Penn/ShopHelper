# tests/test_sessions_api.py
from tests.helpers import fake_chat, post_chat_sse


async def test_history_unknown_session_404(make_client):
    client, _ = await make_client(fake_chat("x"))
    resp = await client.get("/api/sessions/424242/history")
    assert resp.status_code == 404


async def test_history_nonint_session_422(make_client):
    client, _ = await make_client(fake_chat("x"))
    resp = await client.get("/api/sessions/not-an-int/history")
    assert resp.status_code == 422


async def test_history_after_two_rounds_keeps_context(make_client):
    client, _ = await make_client(fake_chat("第一答", "第二答"))
    await post_chat_sse(client, {"message": "第一问", "session_id": 9})
    await post_chat_sse(client, {"message": "第二问", "session_id": 9})

    resp = await client.get("/api/sessions/9/history")
    assert resp.status_code == 200
    assert resp.json() == {
        "session_id": 9,
        "messages": [
            {"role": "user", "content": "第一问", "tool_calls": None, "tool_call_id": None},
            {"role": "assistant", "content": "第一答", "tool_calls": None, "tool_call_id": None},
            {"role": "user", "content": "第二问", "tool_calls": None, "tool_call_id": None},
            {"role": "assistant", "content": "第二答", "tool_calls": None, "tool_call_id": None},
        ],
    }
