"""ch07 管线重接线 e2e:回灌与 State 合并、log 切片落库契约、级联/触发接线、可观测日志、输入闸。"""
import logging

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.context import ContextBudgets
from app.graph.builder import initial_state
from app.graph.logging_node import make_log_node
from app.history import rows_to_messages
from tests.helpers import GraphChatModel, graph_state, post_chat_sse
from tests.test_ch07_summary import StubSummaryModel

pytestmark = pytest.mark.asyncio


# ---------- 单元:initial_state / log 节点 ----------

def test_initial_state_hydrates_with_ids_and_user_marker():
    rows = [{"id": 7, "role": "user", "content": "旧问"},
            {"id": 8, "role": "assistant", "content": "旧答"}]
    st = initial_state(1, "新问", history_rows=rows, layered={"k": "v"})
    assert [m.id for m in st["messages"][:2]] == ["7", "8"]
    assert st["messages"][-1].content == "新问"
    assert st["messages"][-1].id.startswith("u")
    assert st["turn_user_msg_id"] == st["messages"][-1].id
    assert st["layered"] == {"k": "v"}


def test_initial_state_without_hydration_only_user_message():
    st = initial_state(1, "新问")
    assert len(st["messages"]) == 1 and st["layered"] == {}


async def test_log_node_persists_only_user_and_text_assistant(db_store):
    sid = await db_store.resolve(None)
    msgs = rows_to_messages([{"id": 1, "role": "user", "content": "旧"}])
    state = graph_state(session_id=sid,
                        messages=[*msgs,
                                  HumanMessage(content="新问", id="u99"),
                                  AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "c1"}]),
                                  ToolMessage(content="工具结果", tool_call_id="c1"),
                                  AIMessage(content="终答")],
                        turn_user_msg_id="u99", final_reply="终答")
    await make_log_node(db_store, None)(state)
    hist = await db_store.get_history(sid)
    # 只落本轮新消息:「旧」行代表已回灌的既有历史,不重复落库
    assert [(m["role"], m["content"]) for m in hist] == [("user", "新问"),
                                                         ("assistant", "终答")]


async def test_log_node_missing_turn_marker_persists_nothing(db_store, caplog):
    sid = await db_store.resolve(None)
    state = graph_state(session_id=sid, messages=[HumanMessage(content="x", id="u1")],
                        turn_user_msg_id="absent", final_reply="")
    with caplog.at_level(logging.WARNING, logger="app.graph.logging_node"):
        await make_log_node(db_store, None)(state)
    assert await db_store.get_history(sid) == []
    assert any("未找到本轮 user 消息" in r.getMessage() for r in caplog.records)


# ---------- 替身 ----------

class RecordingSummaryService:
    """触发接线替身:同步记录 maybe_trigger 调用。"""

    def __init__(self):
        self.calls = []

    def maybe_trigger(self, session_id, upto_id):
        self.calls.append((session_id, upto_id))


def _small_budgets() -> ContextBudgets:
    """演示级小预算:层1 420 / 层2 180,两轮长消息即触发级联。"""
    return ContextBudgets(window=4000, fixed=1000, peak=1500,
                          sliding=1500, history=600, layer1=420, layer2=180)


# ---------- e2e ----------

async def test_e2e_hydration_second_turn_sees_history(make_client):
    """同会话连发两轮:第二轮模型输入里能看到第一轮的 user/assistant(层 1 原文)。"""
    model = GraphChatModel('{"intent": "其他"}', [("text", "答一。"), ("text", "答二。")])
    client, app = await make_client(model)
    sid = (await post_chat_sse(client, {"message": "第一问"}))[0]["session_id"]
    await post_chat_sse(client, {"message": "第二问", "session_id": sid})
    texts = [m.content for m in model.prompts[1]]
    assert "第一问" in texts and "答一。" in texts  # 层 1 原文回灌
    assert texts[-1] == "第二问"                    # 当前消息在最后(无背景块时)


async def test_e2e_history_ctx_logged_every_turn(make_client, caplog):
    model = GraphChatModel('{"intent": "闲聊"}', [])
    client, app = await make_client(model)
    with caplog.at_level(logging.INFO, logger="app.routers.chat"):
        await post_chat_sse(client, {"message": "你好呀"})  # 闲聊兜底轮:不进 Agent 也要打
    assert any("[history_ctx]" in r.getMessage() for r in caplog.records)


async def test_e2e_cascade_and_trigger_wiring(make_client):
    """小预算下连聊:层1 超预算→锚点推进(级联);层2 超预算→summary trigger 接线。"""
    model = GraphChatModel('{"intent": "其他"}', [("text", "回复。")] * 12)
    client, app = await make_client(model)
    app.state.budgets = _small_budgets()
    recorder = RecordingSummaryService()
    app.state.summary_service = recorder
    store = app.state.store
    sid = None
    for _ in range(6):
        frames = await post_chat_sse(client, {"message": "问题" * 150, "session_id": sid})
        sid = frames[0]["session_id"]
    a = await store.get_anchors(sid)
    assert a["layer1_from"] is not None                     # 级联发生过
    assert recorder.calls and recorder.calls[-1][0] == sid  # 摘要触发接线


async def test_e2e_model_ctx_log_carries_summary_and_window(make_client, caplog):
    """有摘要 + 层 2 有内容时,model_ctx 原样带出摘要全文与滑窗逐条。"""
    model = GraphChatModel('{"intent": "其他"}', [("text", "答。")] * 3)
    client, app = await make_client(model)
    app.state.summary_service = StubSummaryModel()  # 防真触发(默认预算不会触发,双保险)
    store = app.state.store
    sid = (await post_chat_sse(client, {"message": "第一问"}))[0]["session_id"]
    rows = await store.get_rows(sid)
    await store.append_summary(sid, seq=1, from_id=rows[0]["id"],
                               upto_id=rows[-1]["id"], content="用户问过订单1001")
    await store.set_layer1_from(sid, rows[-1]["id"])
    model.turns = [("text", "再答。")]
    with caplog.at_level(logging.INFO, logger="app.graph.agent"):
        await post_chat_sse(client, {"message": "第二问", "session_id": sid})
    model_ctx = [r.getMessage() for r in caplog.records if "[model_ctx]" in r.getMessage()]
    assert model_ctx
    joined = "\n".join(model_ctx)
    assert "摘要全文:用户问过订单1001" in joined
    assert "滑窗逐条:" in joined


async def test_e2e_input_gate_uses_max_user_input_tokens(make_client):
    from app.config import Settings
    model = GraphChatModel('{"intent": "其他"}', [])
    client, app = await make_client(model)
    app.state.settings = Settings(openai_api_key="sk-test", _env_file=None,
                                  max_user_input_tokens=10)
    resp = await client.post("/api/chat", json={"message": "这句话肯定超过十个token的预算上限了"})
    assert resp.status_code == 400 and "单条输入预算" in resp.json()["detail"]
