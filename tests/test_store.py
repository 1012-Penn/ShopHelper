import pytest

from app.store import ConversationStore
from tests.test_models import create_memory_engine


@pytest.fixture
def store():
    from sqlalchemy.orm import sessionmaker

    return ConversationStore(sessionmaker(bind=create_memory_engine(), expire_on_commit=False))


async def test_resolve_creates_new_conversation(store):
    conv_id = await store.resolve(None)
    assert isinstance(conv_id, int)
    assert await store.get_history(conv_id) == []


async def test_resolve_existing_id_returns_same(store):
    conv_id = await store.resolve(None)
    assert await store.resolve(conv_id) == conv_id


async def test_resolve_unknown_client_id_registers_it(store):
    """ch01 语义延续:客户端给的未知 id 一并注册,不报错。"""
    assert await store.resolve(777) == 777
    assert await store.get_history(777) == []


async def test_append_and_history_order(store):
    conv_id = await store.resolve(None)
    await store.append(conv_id, [
        {"role": "user", "content": "退货政策是什么"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"name": "query_faq", "args": {"keyword": "退货"}, "id": "call_1"}]},
        {"role": "tool", "content": '{"items":[]}', "tool_call_id": "call_1"},
        {"role": "assistant", "content": "支持七天无理由退货。"},
    ])
    history = await store.get_history(conv_id)
    assert [m["role"] for m in history] == ["user", "assistant", "tool", "assistant"]
    assert history[1]["tool_calls"][0]["name"] == "query_faq"
    assert history[2]["tool_call_id"] == "call_1"
    assert history[3]["content"] == "支持七天无理由退货。"
