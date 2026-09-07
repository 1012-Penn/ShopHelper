"""/api/chat 跑图:SSE 契约 + 四出口端到端 + 历史双写。模型替身 = 意图 JSON + Agent 脚本连发。"""
import json

from langchain_core.messages import AIMessage
from pydantic import Field

from tests.helpers import FakeChatWithTools, post_chat_sse
from tests.helpers import ChunkStub as Chunk
from tests.test_chat_toolflow import _pool_rows, _seed_faq


class IntentAgentModel(FakeChatWithTools):
    """第 1 次 ainvoke = 意图 JSON;之后 astream 按 turns 流式吐(("tools",[chunks])|("text",str))。"""

    intent_reply: str
    turns: list = Field(default_factory=list)
    prompts: list = Field(default_factory=list)

    def __init__(self, intent_reply, turns):
        super().__init__(messages=iter([AIMessage(content=intent_reply)]),
                         intent_reply=intent_reply, turns=turns)

    def bind_tools(self, tools, **kwargs):
        return self

    async def astream(self, messages, **kwargs):
        self.prompts.append(list(messages))
        kind, payload = self.turns.pop(0) if self.turns else ("text", "默认回答。")
        if kind == "tools":
            yield Chunk(tool_call_chunks=payload)
        else:
            for piece in payload.split("|"):
                yield Chunk(text=piece)


def _tc(name, args, id_):
    return {"name": name, "args": json.dumps(args, ensure_ascii=False), "id": id_,
            "index": 0, "type": "tool_call_chunk"}


async def test_chitchat_fixed_no_model_tools(make_client):
    client, app = await make_client(IntentAgentModel('{"intent": "闲聊"}', []))
    events = await post_chat_sse(client, {"message": "你好呀"})
    assert events[0]["type"] == "session" and events[-1]["type"] == "done"
    tokens = "".join(e["content"] for e in events if e["type"] == "token")
    assert "智能客服" in tokens  # 固定话术
    assert not [e for e in events if e["type"] == "tool_status"]


async def test_complaint_emits_two_independent_actions(make_client):
    client, _ = await make_client(IntentAgentModel('{"intent": "投诉"}', []))
    events = await post_chat_sse(client, {"message": "我要投诉"})
    actions = [e for e in events if e["type"] == "actions"]
    assert len(actions) == 1
    assert [(a["action"], a["label"]) for a in actions[0]["items"]] == [
        ("transfer_human", "转人工"), ("create_ticket", "建工单")]
    # 投诉不进 Agent:没有第二次模型调用(turns 未被消费)
    assert not [e for e in events if e["type"] == "tool_status"]


async def test_knowledge_path_forced_retrieval_then_agent(make_client):
    client, app = await make_client(IntentAgentModel('{"intent": "商品咨询"}', []))
    _seed_faq(app)  # LIKE 检索可命中的 FAQ(chunking 后入向量库)
    # 灌一条可检索知识(向量化)
    from app.chunking import Chunk as KChunk
    from app.kb import KnowledgeBaseStore, vectorize_pending
    from tests.helpers import FakeEmbedding

    kb = KnowledgeBaseStore(app.state.session_factory)
    kb.replace_doc_chunks("d.md", [KChunk(category="售后政策", questions="SH-E300 多少钱",
                                          answer="SH-E300 售价 299 元。",
                                          section_path="d.md > 商品型号",
                                          content_type="faq", is_key_clause=False)])
    vectorize_pending(kb, app.state.vectors, FakeEmbedding())

    events = await post_chat_sse(client, {"message": "SH-E300 多少钱"})
    assert any(e["type"] == "tool_status" and e["name"] == "query_faq" for e in events)
    tokens = "".join(e["content"] for e in events if e["type"] == "token")
    assert "默认回答。" in tokens


async def test_weak_evidence_blocked_before_agent(make_client):
    """空知识库 → 闸拦下:兜底话术直出,Agent 不被调用(无 token 之外的模型痕迹)。"""
    client, app = await make_client(IntentAgentModel('{"intent": "商品咨询"}', []))
    events = await post_chat_sse(client, {"message": "量子力学怎么退货"})
    tokens = "".join(e["content"] for e in events if e["type"] == "token")
    assert "无法回答" in tokens
    rows = _pool_rows(app.state.session_factory)
    sources = [r["source"] for r in rows]
    assert "retrieval_low_conf" in sources  # 闸落池(飞轮素材)
    assert "self_check" in sources  # 兜底话术带拒答标记,守卫公共路径同样落池
    # 证据弱不进 Agent:turns 空,若 Agent 被调用会吐"默认回答。"
    assert "默认回答" not in tokens


async def test_business_path_multi_step_persists(make_client):
    """业务数据类不预检索直达 Agent;两步 ReAct(查订单→查物流)后收敛并落库。"""
    model = IntentAgentModel('{"intent": "订单"}', [
        ("tools", [_tc("query_order", {"order_id": "1001"}, "c1")]),
        ("tools", [_tc("query_logistics", {"order_id": "1001"}, "c2")]),
        ("text", "包裹已到杭州。"),
    ])
    client, app = await make_client(model)
    events = await post_chat_sse(client, {"message": "先查订单1001再告诉我物流"})
    labels = [e["label"] for e in events if e["type"] == "tool_status"]
    assert labels == ["订单查询", "物流查询"]  # 不预检索:没有 FAQ 检索徽章
    assert "".join(e["content"] for e in events if e["type"] == "token") == "包裹已到杭州。"
    history = await app.state.store.get_history(events[0]["session_id"])
    assert [m["role"] for m in history] == ["user", "assistant", "tool", "assistant", "tool", "assistant"]
    assert history[-1]["content"] == "包裹已到杭州。"


async def test_stale_evidence_not_leaked_across_turns(make_client):
    """checkpointer 复用 thread:上一回合知识证据不得泄漏进下一回合业务路径。"""
    from app.chunking import Chunk as KChunk
    from app.kb import KnowledgeBaseStore, vectorize_pending
    from tests.helpers import FakeEmbedding

    client, app = await make_client(IntentAgentModel('{"intent": "商品咨询"}', []))
    kb = KnowledgeBaseStore(app.state.session_factory)
    kb.replace_doc_chunks("d.md", [KChunk(category="售后政策", questions="邮费是多少",
                                          answer="邮费 8 元。", section_path="d.md > 邮费",
                                          content_type="faq", is_key_clause=False)])
    vectorize_pending(kb, app.state.vectors, FakeEmbedding())
    await post_chat_sse(client, {"message": "邮费是多少"})

    # 第二回合:换业务意图模型并重建图(同 store/pool/session,checkpointer thread 复用)
    from app.graph.builder import build_graph

    model2 = IntentAgentModel('{"intent": "订单"}', [("text", "订单没问题。")])
    app.state.graph = build_graph(
        model2, app.state.registry, app.state.settings,
        app.state.store, app.state.pool, app.state.retrieval_service, app.state.retrieval_kb)
    events = await post_chat_sse(client, {"message": "查订单", "session_id": 7})
    system = model2.prompts[0][0]
    assert "邮费" not in system.content  # 上一回合证据不残留
    assert "".join(e["content"] for e in events if e["type"] == "token") == "订单没问题。"
