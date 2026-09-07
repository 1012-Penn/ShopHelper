"""投诉安抚 / 闲聊固定话术 / log 落库与 actions 帧。"""
from app.graph.logging_node import make_log_node
from app.graph.simple import COMPLAINT_REPLY, CHITCHAT_REPLY, chitchat_node, complaint_node
from app.guard import REFUSAL_MARKER
from tests.test_ch05_entry import _state


class Rec:
    def __init__(self):
        self.frames = []

    def __call__(self, f):
        self.frames.append(f)


async def test_complaint_no_llm_two_actions():
    rec = Rec()
    out = await complaint_node(_state(session_id=1, user_message="我要投诉",
                                      resolved_message="我要投诉"), writer=rec)
    assert out["suggested_actions"] == ["transfer_human", "create_ticket"]
    assert out["final_reply"] == COMPLAINT_REPLY
    assert [m["role"] for m in out["messages"]] == ["user", "assistant"]
    assert out["final_reply"] == "".join(f["content"] for f in rec.frames if f["type"] == "token")


async def test_chitchat_fixed_reply():
    rec = Rec()
    out = await chitchat_node(_state(user_message="你好呀", resolved_message="你好呀"), writer=rec)
    assert out["final_reply"] == CHITCHAT_REPLY
    assert out["suggested_actions"] == []
    assert [m["role"] for m in out["messages"]] == ["user", "assistant"]


class FakeStore:
    def __init__(self):
        self.appended = []

    async def append(self, cid, msgs):
        self.appended.append((cid, msgs))


class FakePool:
    def __init__(self):
        self.rows = []

    def insert(self, *a):
        self.rows.append(a)


async def test_log_persists_and_emits_actions():
    store, pool = FakeStore(), FakePool()
    node = make_log_node(store, pool)
    msgs = [{"role": "user", "content": "我要投诉"},
            {"role": "assistant", "content": COMPLAINT_REPLY}]
    rec = Rec()
    out = await node(_state(session_id=9, user_message="我要投诉", final_reply=COMPLAINT_REPLY,
                            messages=msgs,
                            suggested_actions=["transfer_human", "create_ticket"],
                            trace=["node=resolve passthrough", "node=intent intent=投诉"]), writer=rec)
    assert store.appended[0][0] == 9 and len(store.appended[0][1]) == 2
    actions = [f for f in rec.frames if f["type"] == "actions"]
    assert [(a["action"], a["label"]) for a in actions[0]["items"]] == [
        ("transfer_human", "转人工"), ("create_ticket", "建工单")]
    assert pool.rows == []  # 非拒答不落池


async def test_log_self_check_pool_on_refusal():
    store, pool = FakeStore(), FakePool()
    node = make_log_node(store, pool)
    reply = f"{REFUSAL_MARKER}知识库中没有相关依据。"
    await node(_state(session_id=3, user_message="量子力学怎么退货", final_reply=reply,
                      messages=[{"role": "user", "content": "量子力学怎么退货"},
                                {"role": "assistant", "content": reply}],
                      suggested_actions=[], trace=[]))
    assert [r[0] for r in pool.rows] == ["self_check"]
    assert "模型自评证据不足" in pool.rows[0][3]


async def test_log_empty_messages_safe():
    store, pool = FakeStore(), FakePool()
    node = make_log_node(store, pool)
    await node(_state(session_id=1, user_message="hi", final_reply="", messages=[],
                      suggested_actions=[], trace=[]))
    assert store.appended[0][1] == []
