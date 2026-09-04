# tests/test_sessions.py
import asyncio
import uuid

from app.sessions import SessionStore


async def test_resolve_creates_unique_ids():
    store = SessionStore()
    a = await store.resolve(None)
    b = await store.resolve(None)
    assert a != b
    uuid.UUID(a)  # 是合法 uuid hex,不抛即通过


async def test_resolve_registers_unknown_id():
    store = SessionStore()
    sid = await store.resolve("client-given")
    assert sid == "client-given"
    assert await store.get("client-given") is not None


async def test_get_unknown_returns_none():
    assert await SessionStore().get("nope") is None


async def test_append_roundtrip():
    store = SessionStore()
    sid = await store.resolve(None)
    await store.append(sid, "你好", "欢迎光临")
    data = await store.get(sid)
    assert data.messages == [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "欢迎光临"},
    ]


async def test_append_unknown_id_is_noop():
    store = SessionStore()
    await store.append("ghost", "你好", "嗨")  # 不抛异常即通过


async def test_concurrent_appends_keep_order():
    store = SessionStore()
    sid = await store.resolve(None)
    await asyncio.gather(*(store.append(sid, f"u{i}", f"a{i}") for i in range(5)))
    data = await store.get(sid)
    expected = [pair for i in range(5) for pair in (f"u{i}", f"a{i}")]
    assert [m["content"] for m in data.messages] == expected
