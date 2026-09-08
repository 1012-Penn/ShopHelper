"""ReAct 主力 Agent 节点:单步收敛 / 多步工具链 / 步数熔断 / 拒答带 actions 建议。"""
import json

from langchain_core.messages import AIMessage
from pydantic import Field

from app.config import Settings
from app.graph.agent import make_agent_node
from tests.helpers import FakeChatWithTools, FakeEmbedding, FakeVectorStore
from tests.helpers import graph_state as _state
from tests.helpers import ChunkStub as Chunk


class ScriptedToolModel(FakeChatWithTools):
    """按脚本逐轮流式吐:("tools", [tc_chunks]) 或 ("text", str);记录 bind 次数与 prompt。"""

    # GenericFakeChatModel 是 pydantic 模型:实例属性必须声明为字段
    turns: list
    bind_calls: int = 0
    prompts: list = Field(default_factory=list)

    def __init__(self, turns):
        super().__init__(messages=iter([AIMessage(content="unused")]), turns=turns)

    def bind_tools(self, tools, **kwargs):
        self.bind_calls += 1
        return self

    async def astream(self, messages, **kwargs):
        self.prompts.append(list(messages))
        kind, payload = self.turns.pop(0)
        if kind == "tools":
            yield Chunk(tool_call_chunks=payload)
        else:
            yield Chunk(text=payload)


def _tc(name, args, id_, index=0):
    return {"name": name, "args": json.dumps(args, ensure_ascii=False), "id": id_,
            "index": index, "type": "tool_call_chunk"}


class Recorder:
    def __init__(self):
        self.frames = []

    def __call__(self, frame):
        self.frames.append(frame)


class FakePool:
    def __init__(self):
        self.rows = []

    def insert(self, *a):
        self.rows.append(a)


def _registry():
    from app.tools.definitions import build_tools
    from app.tools.registry import ToolRegistry
    return ToolRegistry(build_tools(None, embedder=FakeEmbedding(), vectors=FakeVectorStore(), top_k=3))


def _settings(**kw):
    kw.setdefault("openai_api_key", "sk-test")
    kw.setdefault("_env_file", None)
    return Settings(**kw)


async def test_single_step_converges_without_tools():
    model = ScriptedToolModel([("text", "直接能答。")])
    node = make_agent_node(model, _registry(), _settings(), FakePool())
    out = await node(_state(session_id=1, user_message="你们几点营业",
                            resolved_message="你们几点营业", history=[]))
    assert out["final_reply"] == "直接能答。"
    assert out["agent_steps"] == 1
    assert [(type(m).__name__, m.content) for m in out["messages"]] == [("AIMessage", "直接能答。")]


async def test_react_multi_step_order_then_logistics():
    model = ScriptedToolModel([
        ("tools", [_tc("query_order", {"order_id": "1001"}, "c1")]),
        ("tools", [_tc("query_logistics", {"order_id": "1001"}, "c2")]),
        ("text", "包裹已到杭州转运中心。"),
    ])
    rec = Recorder()
    node = make_agent_node(model, _registry(), _settings(), FakePool())
    out = await node(_state(session_id=1, user_message="先查订单1001再告诉我物流",
                            resolved_message="先查订单1001再告诉我物流", history=[]), writer=rec)
    assert out["agent_steps"] == 3 and out["final_reply"] == "包裹已到杭州转运中心。"
    labels = [f["label"] for f in rec.frames if f["type"] == "tool_status"]
    assert labels == ["订单查询", "物流查询"]
    roles = [type(m).__name__ for m in out["messages"]]
    assert roles == ["AIMessage", "ToolMessage", "AIMessage", "ToolMessage", "AIMessage"]
    # 每步都重新 bind_tools(ReAct 每步都带工具,区别于 ch04 单轮两段式)
    assert model.bind_calls == 3
    # 第二步的 prompt 回灌了第一步的 AIMessage(tool_calls) 与 ToolMessage
    kinds = [type(m).__name__ for m in model.prompts[1]]
    assert "ToolMessage" in kinds


async def test_max_steps_cutoff():
    model = ScriptedToolModel([("tools", [_tc("query_order", {"order_id": "1"}, f"c{i}")])
                               for i in range(10)])
    node = make_agent_node(model, _registry(), _settings(max_agent_steps=2), FakePool())
    out = await node(_state(session_id=1, user_message="查订单", resolved_message="查订单", history=[]))
    assert out["agent_steps"] == 2 and model.bind_calls == 2
    # C1 修复:熔断时回溯 assistant 文本;全程只有工具轮 → 给引导语而非原始工具 JSON
    assert out["final_reply"] == "问题比较复杂,请您稍后再试或换种问法。"
    assert not out["final_reply"].lstrip().startswith("{")


async def test_agent_suggests_actions_on_refusal():
    model = ScriptedToolModel([("text", "【无法回答】这个问题需要人工处理。")])
    node = make_agent_node(model, _registry(), _settings(), FakePool())
    out = await node(_state(session_id=1, user_message="投诉", resolved_message="投诉", history=[]))
    assert set(out["suggested_actions"]) == {"transfer_human", "create_ticket"}


async def test_evidence_in_background_message_not_system():
    """ch07 组装纪律:system 纯静态,证据进背景块并挂在当前用户消息之后。"""
    model = ScriptedToolModel([("text", "带[n]的回答。")])
    node = make_agent_node(model, _registry(), _settings(), FakePool())
    evidence = [{"n": 1, "chunk_id": 7, "question": "退货政策是什么", "answer": "七天无理由退货。",
                 "category": "售后政策", "section_path": "d.md > 退货政策"}]
    out = await node(_state(session_id=1, user_message="退货政策", resolved_message="退货政策",
                            evidence=evidence))
    assert out["final_reply"] == "带[n]的回答。"
    prompts = model.prompts[0]
    assert type(prompts[0]).__name__ == "SystemMessage"
    assert "[1]" not in prompts[0].content          # system 不含证据
    assert prompts[1].content == "退货政策"          # 层2/层1 空,紧跟当前用户消息
    bg = prompts[-1]
    assert type(bg).__name__ == "HumanMessage" and bg is not prompts[1]
    assert "【参考知识】" in bg.content and "[1]" in bg.content and "七天无理由退货" in bg.content


async def test_query_faq_low_confidence_pools():
    pool = FakePool()
    model = ScriptedToolModel([
        ("tools", [_tc("query_faq", {"keyword": "量子力学"}, "c1")]),
        ("text", "【无法回答】知识库中没有相关依据。"),
    ])
    rec = Recorder()
    node = make_agent_node(model, _registry(), _settings(), pool)
    out = await node(_state(session_id=5, user_message="量子力学怎么退货",
                            resolved_message="量子力学怎么退货", history=[]), writer=rec)
    assert [r[0] for r in pool.rows] == ["retrieval_low_conf"]
    assert pool.rows[0][2] == "量子力学怎么退货"
    assert [f for f in rec.frames if f["type"] == "citations"] == []  # 低置信不出证据


async def test_cutoff_prefers_last_assistant_text():
    """熔断前有过部分文本(如"已为您查询到")时,收敛用该文本而非引导语。"""
    model = ScriptedToolModel([
        ("text", "已为您查询到订单,"),
        ("tools", [_tc("query_order", {"order_id": "1"}, "c9")]),
        ("tools", [_tc("query_order", {"order_id": "1"}, "c9")]),
    ])
    rec = Recorder()
    node = make_agent_node(model, _registry(), _settings(max_agent_steps=3), FakePool())
    out = await node(_state(session_id=1, user_message="查订单", resolved_message="查订单",
                            history=[]), writer=rec)
    assert out["final_reply"] == "已为您查询到订单,"
