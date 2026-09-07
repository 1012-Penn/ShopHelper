# ch05 Workflow 编排(LangGraph 图骨架)Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 LangGraph 确定性图重构 /api/chat:意图分流四出口、知识类强制检索 + 置信度闸、手写 ReAct 主力 Agent、投诉转人工/建工单前端自选。

**Architecture:** `app/graph/` 新包承载图骨架(State/节点/builder);`/api/chat` 内部替换为 `graph.astream(stream_mode="custom")`,节点内 `get_stream_writer()` 发 SSE 帧原样透传;MySQL store 保留双写,InMemorySaver 只承载图内运行态。Agent 节点手写 ReAct 循环(bind_tools → 流式 → 执行回灌 → 收敛)。

**Tech Stack:** langgraph 1.2.11(`StateGraph`/`InMemorySaver`/`StreamWriter`,API 已对官方文档与 venv 源码核对)、langchain 1.4.0、FastAPI、原生 JS 前端。

**Spec:** `docs/superpowers/specs/2026-09-07-ch05-workflow-orchestration-design.md`(计划从 spec 论证,执行者两份都读)

## Global Constraints

- langgraph 版本 `>=1.2.11,<2`(pyproject 已加);不新增其他运行时依赖。
- ch02 五工具原样复用(`registry.execute` 执行),本章不新写业务工具。
- 检索复用 ch03/04 `RetrievalService`,置信度闸语义 = `RetrievalResult.low_confidence`(候选空或 Top-1 < `rerank_score_floor`)。
- SSE 帧兼容 ch04 前端契约:`session/token/tool_status/citations/done/error`,新增 `actions`。
- 拒答标记唯一来源 `app.guard.REFUSAL_MARKER`;落池 source 沿用 `retrieval_low_conf` / `self_check`。
- 意图七类枚举:`物流/订单/商品咨询/退款退货/售后/投诉/闲聊`;解析失败或超纲 → `商品咨询`。
-投诉、闲聊、兜底零模型调用;`create_ticket` 仅前端按钮触发。
- 测试不出网:替身注入模式沿 `tests/conftest.py::make_client` / `tests/helpers.py`。
- 每任务 TDD:先写失败测试 → 跑红 → 最小实现 → 跑绿 → 提交。

---

### Task 1: 祛魅热身——裸 Agent 循环(无 LangGraph)

**Files:**
- Create: `scripts/naive_agent.py`
- Test: `tests/test_naive_agent.py`

**Interfaces:**
- Produces: `naive_agent_loop(model, tools, question: str) -> str`(模块级函数,可导入测试;不依赖 langgraph)
- Produces: dev-notes/ch05.md 记录一次真跑(演示脚本入口 `python -m scripts.naive_agent "问题"`)

- [ ] **Step 1: 写失败测试**

```python
"""裸 Agent 循环:调 LLM → 有工具调用就执行喂回 → 没有就收敛。"""
from langchain.tools import tool
from tests.helpers import FakeChatWithTools
from langchain_core.messages import AIMessage


@tool
def echo_city(city: str) -> str:
    """按城市查天气(测试替身)。"""
    return f"{city}:晴"


class ToolThenAnswerModel(FakeChatWithTools):
    """第一次 invoke 出 tool_calls,第二次出文本答案。"""

    def __init__(self):
        super().__init__(messages=iter([
            AIMessage(content="", tool_calls=[
                {"name": "echo_city", "args": {"city": "杭州"}, "id": "c1", "type": "tool_call"}]),
            AIMessage(content="杭州今天晴。"),
        ]))

    def bind_tools(self, tools, **kwargs):
        return self


async def test_loop_executes_tool_and_converges():
    from scripts.naive_agent import naive_agent_loop

    model = ToolThenAnswerModel()
    answer = await naive_agent_loop(model, [echo_city], "杭州天气怎么样")
    assert answer == "杭州今天晴。"
    assert model.call_count == 2


async def test_loop_no_tool_single_call():
    from scripts.naive_agent import naive_agent_loop

    model = FakeChatWithTools(messages=iter([AIMessage(content="直接回答。")]))
    answer = await naive_agent_loop(model.bind_tools([echo_city]), [echo_city], "你好")
    assert answer == "直接回答。"
```

- [ ] **Step 2: 跑红** — `.venv/bin/pytest tests/test_naive_agent.py -v` → FAIL(ModuleNotFoundError: scripts.naive_agent)

- [ ] **Step 3: 最小实现 `scripts/naive_agent.py`**

```python
"""祛魅热身:最裸的 Agent 循环——它就是个带工具的 while 循环。用法:
    .venv/bin/python -m scripts.naive_agent "订单 1001 的物流到哪了"
"""
import asyncio
import json
import sys

from langchain_core.messages import HumanMessage, ToolMessage

from app.config import Settings
from app.llm import make_chat_model
from app.tools.definitions import build_tools
from app.tools.registry import ToolRegistry


async def naive_agent_loop(model, tools, question: str, max_steps: int = 10) -> str:
    """核心就这十几行:LLM 返回 tool_calls 就执行并喂回去,返回纯文本就收敛。"""
    messages = [HumanMessage(content=question)]
    registry = ToolRegistry(tools)
    for _ in range(max_steps):
        resp = await model.ainvoke(messages)
        if not resp.tool_calls:
            return resp.content
        messages.append(resp)
        for tc in resp.tool_calls:
            result = await registry.execute(tc["name"], json.dumps(tc["args"], ensure_ascii=False))
            messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
    return "达到步数上限,未收敛。"


async def main() -> None:
    settings = Settings()
    model = make_chat_model(settings).bind_tools(build_tools(None))
    question = " ".join(sys.argv[1:]) or "订单 1001 的物流到哪了"
    print(f"[user] {question}")
    print(f"[agent] {await naive_agent_loop(model, build_tools(None), question)}")


if __name__ == "__main__":
    asyncio.run(main())
```

注意:`build_tools(None)` 走生产路径(真 embedder/Milvus),`naive_agent_loop` 内部把 `tools` 包进 `ToolRegistry` 只为复用其执行语义(校验/超时/重试),`bind_tools` 的 `tools` 参数传同一份。测试里传替身工具列表,`ToolRegistry(tools)` 直接可构造。`FakeChatWithTools` 需加 `self.call_count` 计数(在 `tests/helpers.py` 的 `__init__` 或测试子类里 `ainvoke` 计数——实现时在测试子类覆写 `ainvoke` 计数即可,不改 helpers 公共契约)。

- [ ] **Step 4: 跑绿** — `.venv/bin/pytest tests/test_naive_agent.py -v` → 2 passed
- [ ] **Step 5: dev-notes 追记本任务 + commit** — `git commit -m "ch05: 祛魅热身——裸 Agent 循环 scripts/naive_agent.py"`

### Task 2: State + 意图识别 + 写死分流

**Files:**
- Create: `app/graph/__init__.py`(空)、`app/graph/state.py`、`app/graph/entry.py`
- Modify: `app/prompts.py`(加 INTENT_PROMPT)
- Test: `tests/test_ch05_entry.py`

**Interfaces:**
- Produces: `ChatState(TypedDict)` 字段见 spec;`INTENTS: tuple[str, ...]` 七类
- Produces: `async intent_node(state) -> dict`、`def route(state) -> str`(返回 `"retrieve"|"agent"|"complaint"|"chitchat"`)、`def resolve_node(state) -> dict`

- [ ] **Step 1: 失败测试**

```python
"""意图识别节点 + 写死分流。"""
from tests.helpers import fake_chat
from app.graph.entry import intent_node, resolve_node, route
from app.graph.state import ChatState, INTENTS


def _state(**kw):
    base = dict(session_id=1, user_message="退货政策是什么", resolved_message="", intent="",
                evidence=[], gate_passed=False, agent_steps=0, final_reply="",
                suggested_actions=[], trace=[], messages=[], history=[])
    base.update(kw)
    return base


async def test_intent_parses_llm_json():
    model = fake_chat('{"intent": "退款退货"}')
    out = await intent_node(_state(), model)
    assert out["intent"] == "退款退货"
    assert "intent=退款退货" in out["trace"][-1]


async def test_intent_fallback_on_garbage():
    model = fake_chat("我说不好")
    out = await intent_node(_state(), model)
    assert out["intent"] == "商品咨询"


async def test_intent_fallback_out_of_enum():
    model = fake_chat('{"intent": "聊天气"}')
    out = await intent_node(_state(), model)
    assert out["intent"] == "商品咨询"


def test_route_four_outlets():
    s = _state()
    for intent, expect in [("商品咨询", "retrieve"), ("退款退货", "retrieve"),
                           ("物流", "agent"), ("订单", "agent"), ("售后", "agent"),
                           ("投诉", "complaint"), ("闲聊", "chitchat")]:
        assert route(_state(intent=intent)) == expect


def test_resolve_passthrough():
    out = resolve_node(_state())
    assert out["resolved_message"] == "退货政策是什么"
```

- [ ] **Step 2: 跑红**

- [ ] **Step 3: 实现**

`app/prompts.py` 追加:

```python
INTENT_PROMPT = """你是电商客服的意图分类器。把用户消息判成以下七类之一,只输出 JSON:
{"intent": "物流|订单|商品咨询|退款退货|售后|投诉|闲聊"}
判类口径:问包裹/快递到哪了→物流;查订单状态/信息→订单;商品参数价格库存→商品咨询;
退钱退货流程政策→退款退货;安装维修换货等售后问题→售后;不满要讨说法→投诉;寒暄无关→闲聊。"""
```

`app/graph/state.py`:

```python
"""ch05 图骨架:State 一路贯穿,checkpointer 只承载图内运行态(历史真源在 MySQL store)。"""
from typing import TypedDict


class ChatState(TypedDict):
    session_id: int
    user_message: str
    resolved_message: str
    intent: str
    evidence: list[dict]          # 检索证据(带全局引用编号 n)
    gate_passed: bool
    agent_steps: int
    final_reply: str
    suggested_actions: list[str]  # "transfer_human" / "create_ticket"
    trace: list[str]              # 每节点一行,验收观察 + logger 输出
    messages: list                # 本轮新增的落库 pending(dict,role/content/tool_calls/tool_call_id)
    history: list                 # 裁剪后的历史(dict),由路由层注入
```

`app/graph/entry.py`:

```python
"""入口节点:指代消解(本章透传)+ 意图识别(简单 prompt 出 JSON)+ 写死分流。"""
import json

from langchain_core.messages import HumanMessage, SystemMessage

from app.graph.state import ChatState
from app.prompts import INTENT_PROMPT

INTENTS = ("物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊")
INTENT_FALLBACK = "商品咨询"  # 解析失败/超纲走知识路径,检索闸可拦

KNOWLEDGE_INTENTS = {"商品咨询", "退款退货"}
BUSINESS_INTENTS = {"物流", "订单", "售后"}


def _safe_intent(raw: str) -> str:
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return INTENT_FALLBACK
    intent = obj.get("intent") if isinstance(obj, dict) else None
    return intent if intent in INTENTS else INTENT_FALLBACK


def resolve_node(state: ChatState) -> dict:
    return {"resolved_message": state["user_message"],
            "trace": [*state["trace"], "node=resolve passthrough"]}


async def intent_node(state: ChatState, model) -> dict:
    resp = await model.ainvoke([SystemMessage(content=INTENT_PROMPT),
                                HumanMessage(content=state["resolved_message"])])
    intent = _safe_intent(resp.content)
    return {"intent": intent, "trace": [*state["trace"], f"node=intent intent={intent}"]}


def route(state: ChatState) -> str:
    if state["intent"] in KNOWLEDGE_INTENTS:
        return "retrieve"
    if state["intent"] in BUSINESS_INTENTS:
        return "agent"
    return "complaint" if state["intent"] == "投诉" else "chitchat"
```

- [ ] **Step 4: 跑绿** `.venv/bin/pytest tests/test_ch05_entry.py -v`
- [ ] **Step 5: commit** `ch05: State+意图识别(简单prompt)+写死分流四出口`

### Task 3: 知识路径——强制检索节点 + 置信度闸 + 兜底

**Files:**
- Create: `app/graph/knowledge.py`
- Test: `tests/test_ch05_knowledge.py`

**Interfaces:**
- Consumes: Task 2 的 `ChatState`;`RetrievalService.retrieve(query, strategy="hybrid_rerank")`(构造注入,测试给替身);`KnowledgeBaseStore.get_chunks`;`LowConfidencePool.insert`
- Produces: `def make_knowledge_nodes(service, kb, pool)` → `(retrieve_node, gate_node, fallback_node)`(闭包注入,便于测试替身);`evidence_to_blocks(items, rows) -> list[dict]`(证据条目格式 n/chunk_id/question/answer/category/section_path,复用 ch04 前端契约)

- [ ] **Step 1: 失败测试**

```python
"""知识路径:强制检索 → 置信度闸 → 弱证据兜底不进 Agent。"""
from types import SimpleNamespace

import pytest

from app.graph.knowledge import make_knowledge_nodes
from tests.test_ch05_entry import _state


def _service(items=None, low=False, reason=""):
    """RetrievalService 替身:retrieve 返回 dataclass 样对象。"""
    return SimpleNamespace(retrieve=lambda q, strategy="hybrid_rerank", category=None, use_rewrite=True:
                           SimpleNamespace(items=items or [], low_confidence=low, reason=reason,
                                           filter_fallback=False, degraded=False))


def _kb(chunk_id=7):
    row = SimpleNamespace(questions="退货政策是什么\n怎么退", answer="七天无理由退货。",
                          category="售后政策", section_path="d.md > 退货政策")
    return SimpleNamespace(get_chunks=lambda ids: [row for _ in ids])


class FakePool:
    def __init__(self):
        self.rows = []

    def insert(self, source, conversation_id, question, reason=""):
        self.rows.append((source, conversation_id, question, reason))


async def test_retrieve_attaches_evidence_and_trace():
    retrieve, gate, fallback = make_knowledge_nodes(
        _service(items=[SimpleNamespace(chunk_id=7, score=0.9)]), _kb(), FakePool())
    out = await retrieve(_state(resolved_message="退货政策是什么"))
    assert out["evidence"][0]["n"] == 1 and out["evidence"][0]["chunk_id"] == 7
    assert any("node=retrieve" in t and "top1=0.90" in t for t in out["trace"])


async def test_gate_weak_evidence_falls_back_and_pools():
    pool = FakePool()
    retrieve, gate, fallback = make_knowledge_nodes(_service(low=True, reason="证据置信度低"), _kb(), pool)
    g = await gate(_state(evidence=[], trace=["node=retrieve strategy=hybrid_rerank top1=0.00"]),
                   pool=pool)
    assert g["gate_passed"] is False
    out = await fallback(_state(session_id=1, user_message="量子力学怎么退货", **g))
    assert "无法回答" in out["final_reply"]
    assert pool.rows and pool.rows[0][0] == "retrieval_low_conf"
    assert out["messages"][0]["role"] == "assistant"  # 兜底话术入落库 pending


async def test_gate_strong_evidence_passes():
    retrieve, gate, _ = make_knowledge_nodes(_service(items=[SimpleNamespace(chunk_id=7, score=0.9)]),
                                             _kb(), FakePool())
    st = _state(resolved_message="退货政策是什么")
    st = {**st, **(await retrieve(st))}
    out = await gate(st)
    assert out["gate_passed"] is True
```

注:`gate_node` 的落池依赖通过 `make_knowledge_nodes` 第三参闭包持有,测试直接传;`gate` 返回 `{"gate_passed": bool, "trace": [...]}`;弱证据分支 `fallback_node` 自己产 `messages`/`final_reply`(用 `REFUSAL_MARKER` 开头的固定兜底话术,如 `【无法回答】这个问题我暂时无法给出可靠答案,您可以换个问法或选择下面的方式继续。`)并经 writer 发 token 帧(直调节点时 writer 参数可缺省——用 `get_stream_writer()` 失败即 no-op,或节点签名 `writer=None` 判空)。

- [ ] **Step 2: 跑红** → **Step 3: 实现 `app/graph/knowledge.py`**(核心逻辑):

```python
"""知识路径:强制 RAG 检索 → 置信度闸 → 弱证据兜底。service/kb/pool 构造注入。"""
from app.guard import REFUSAL_MARKER
from app.kb import lost_in_middle_order  # 若未导出则从 app.retrieval 导入

FALLBACK_REPLY = f"{REFUSAL_MARKER}这个问题我暂时无法给出可靠答案,您可以换个问法,或选择下方方式继续。"


def _emit(writer, frame: dict) -> None:
    if writer is not None:
        writer(frame)


def make_knowledge_nodes(service, kb, pool):
    async def retrieve_node(state, writer=None):
        result = service.retrieve(state["resolved_message"], strategy="hybrid_rerank")
        rows = kb.get_chunks([r.chunk_id for r in result.items])
        evidence = [
            {"n": i + 1, "chunk_id": r.chunk_id,
             "question": row.questions.splitlines()[0], "answer": row.answer,
             "category": row.category, "section_path": row.section_path or ""}
            for i, (r, row) in enumerate(zip(result.items, rows))
        ]
        top1 = f"{result.items[0].score:.2f}" if result.items else "0.00"
        trace = [*state["trace"], f"node=retrieve strategy=hybrid_rerank top1={top1} "
                                   f"low_confidence={result.low_confidence} n={len(evidence)}"]
        _emit(writer, {"type": "tool_status", "name": "query_faq", "label": "知识检索"})
        return {"evidence": evidence,
                "low_confidence": result.low_confidence, "low_reason": result.reason,
                "trace": trace}
    # 注:low_confidence/low_reason 需同步加进 ChatState(state.py 在本任务一并补两字段)

    async def gate_node(state, writer=None):
        if state["low_confidence"]:
            pool.insert("retrieval_low_conf", state["session_id"],
                        state["user_message"], str(state.get("low_reason") or ""))
            return {"gate_passed": False,
                    "trace": [*state["trace"], "node=gate blocked=low_confidence"]}
        return {"gate_passed": True, "trace": [*state["trace"], "node=gate passed"]}

    async def fallback_node(state, writer=None):
        _emit(writer, {"type": "token", "content": FALLBACK_REPLY})
        return {"final_reply": FALLBACK_REPLY,
                "messages": [{"role": "assistant", "content": FALLBACK_REPLY}],
                "trace": [*state["trace"], "node=fallback"]}

    return retrieve_node, gate_node, fallback_node
```

(`lost_in_middle_order` 的反漏斗摆放本章不必重复——ch04 在工具层做;此处证据按精排名次原序编号即可,删掉未用导入。)

- [ ] **Step 4: 跑绿** → **Step 5: commit** `ch05: 知识路径——强制检索+置信度闸+弱证据兜底落池`

### Task 4: ReAct 主力 Agent 节点

**Files:**
- Create: `app/graph/agent.py`
- Modify: `app/config.py`(加 `agent_max_steps: int = 6`、`agent_token_budget: int = 3000`)
- Modify: `app/history.py`(把 ch04 `app/routers/chat.py::_to_prompt_messages` 迁来为公共 `to_prompt_messages`,chat 路由 Task 6 改引)
- Test: `tests/test_ch05_agent.py`

**Interfaces:**
- Consumes: `registry`(ToolRegistry)、`model`(chat model)、`settings.agent_max_steps/agent_token_budget`;`Chunk` 替身契约(`.text`/`.tool_call_chunks`)沿 `tests/test_chat_toolflow.py`
- Produces: `def make_agent_node(model, registry, settings)` → `async agent_node(state, writer=None) -> dict`(写 `final_reply/agent_steps/messages/suggested_actions/trace`);`_merge_tool_call_chunks` 从 chat.py 迁入 agent.py

- [ ] **Step 1: 失败测试**(替身复用 `tests/test_chat_toolflow.py` 的 `Chunk`;agent 节点 writer 用 list 收集)

```python
"""ReAct 主力 Agent:单步收敛 / 多步工具链 / 步数熔断 / actions 建议。"""
import json

from app.config import Settings
from app.graph.agent import make_agent_node
from tests.helpers import FakeEmbedding  # 仅为 conftest 一致性;本文件其实不需要,可删
from tests.test_ch05_entry import _state
from tests.test_chat_toolflow import Chunk


class ScriptedToolModel:
    """按脚本逐轮流式吐:每轮一个 Chunk 列表;记录 bind_tools 次数与每次 prompt。"""

    def __init__(self, turns):
        self.turns = list(turns)   # [("tools", [tc...]) 或 ("text", "...")]
        self.bind_calls = 0
        self.prompts = []

    def bind_tools(self, tools):
        self.bind_calls += 1
        return self

    async def astream(self, messages):
        self.prompts.append(list(messages))
        kind, payload = self.turns.pop(0)
        if kind == "tools":
            yield Chunk(tool_call_chunks=payload)
        else:
            yield Chunk(text=payload)


def _tc(name, args, id_):
    return {"name": name, "args": json.dumps(args, ensure_ascii=False), "id": id_,
            "index": 0, "type": "tool_call_chunk"}


class Recorder:
    def __init__(self):
        self.frames = []

    def __call__(self, frame):
        self.frames.append(frame)


def _settings():
    return Settings(openai_api_key="sk-test", _env_file=None)


async def test_single_step_converges_without_tools():
    model = ScriptedToolModel([("text", "直接能答。")])
    registry = _registry()
    node = make_agent_node(model, registry, _settings())
    out = await node(_state(session_id=1, user_message="你们几点营业",
                            resolved_message="你们几点营业", history=[]), writer=Recorder())
    assert out["final_reply"] == "直接能答。"
    assert out["agent_steps"] == 1


async def test_react_multi_step_order_then_logistics():
    model = ScriptedToolModel([
        ("tools", [_tc("query_order", {"order_id": "1001"}, "c1")]),
        ("tools", [_tc("query_logistics", {"order_id": "1001"}, "c2")]),
        ("text", "包裹已到杭州转运中心。"),
    ])
    node = make_agent_node(model, _registry(), _settings())
    rec = Recorder()
    out = await node(_state(session_id=1, user_message="先查订单1001再告诉我物流",
                            resolved_message="先查订单1001再告诉我物流", history=[]), writer=rec)
    assert out["agent_steps"] == 3 and out["final_reply"] == "包裹已到杭州转运中心。"
    labels = [f["label"] for f in rec.frames if f["type"] == "tool_status"]
    assert labels == ["订单查询", "物流查询"]
    # 回灌轨迹:assistant(tool_calls) → tool → assistant(tool_calls) → tool
    roles = [m["role"] for m in out["messages"]]
    assert roles == ["user", "assistant", "tool", "assistant", "tool", "assistant"]


async def test_max_steps_cutoff():
    model = ScriptedToolModel([("tools", [_tc("query_order", {"order_id": "1"}, f"c{i}")]
                                          ) for i in range(10)])
    model.turns = [("tools", [_tc("query_order", {"order_id": "1"}, f"c{i}")]) for i in range(10)]
    settings = Settings(openai_api_key="sk-test", agent_max_steps=2, _env_file=None)
    node = make_agent_node(model, _registry(), settings)
    out = await node(_state(session_id=1, user_message="查订单", resolved_message="查订单",
                            history=[]), writer=Recorder())
    assert out["agent_steps"] == 2 and model.bind_calls == 2  # 熔断:不再无限循环


async def test_agent_suggests_actions_in_final_reply():
    model = ScriptedToolModel([("text", "【无法回答】这个问题需要人工处理。")])
    node = make_agent_node(model, _registry(), _settings())
    out = await node(_state(session_id=1, user_message="投诉", resolved_message="投诉",
                            history=[]), writer=Recorder())
    # Agent 判断合适时带建议:由答复里的拒答标记触发 transfer_human + create_ticket
    assert set(out["suggested_actions"]) == {"transfer_human", "create_ticket"}


def _registry():
    from app.tools.registry import ToolRegistry
    from app.tools.definitions import build_tools
    return ToolRegistry(build_tools(None, embedder=FakeEmbedding(),
                                    vectors=_FakeVectors(), top_k=3))
```

`_FakeVectors` 用 `tests/helpers.py` 的 `FakeVectorStore`(测试内直接 import)。注意 `_registry()` 里 query_faq 走替身检索(空库 → low_confidence),不影响 query_order/query_logistics(mock 无外呼)。

- [ ] **Step 2: 跑红** → **Step 3: 实现 `app/graph/agent.py`** 核心:

```python
"""主力 Agent 节点:手写 ReAct 循环——bind_tools 流式 → tool_calls 执行回灌 → 收敛/熔断。"""
import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.guard import is_refusal
from app.history import estimate_tokens
from app.prompts import SERVICE_PROMPT_TEMPLATE

ACTIONS_ON_REFUSAL = ["transfer_human", "create_ticket"]


def _merge_tool_call_chunks(chunks):   # 自 ch04 chat.py 原样迁入
    ...


def make_agent_node(model, registry, settings):
    async def agent_node(state, writer=None):
        history = _history_messages(state["history"])          # 裁剪历史 → LC 消息(不含本轮 user)
        messages = [SystemMessage(content=SERVICE_PROMPT_TEMPLATE.format()), *history,
                    HumanMessage(content=state["resolved_message"])]
        if state.get("evidence"):
            block = "\n".join(f"[{e['n']}] {e['question']}:{e['answer']}" for e in state["evidence"])
            messages[0] = SystemMessage(content=SERVICE_PROMPT_TEMPLATE.format()
                                        + f"\n\n参考知识(回答须带 [n] 角标):\n{block}")
        pending = [{"role": "user", "content": state["user_message"]}]
        spent, steps = 0, 0
        while steps < settings.agent_max_steps and spent < settings.agent_token_budget:
            steps += 1
            bound = model.bind_tools(registry.tools) if steps == 1 else model
            parts, call_chunks = [], []
            async for chunk in bound.astream(messages):
                if chunk.text:
                    parts.append(chunk.text)
                    _emit(writer, {"type": "token", "content": chunk.text})
                call_chunks.extend(getattr(chunk, "tool_call_chunks", None) or [])
            text = "".join(parts)
            spent += estimate_tokens(text)
            tool_calls = _merge_tool_call_chunks(call_chunks)
            if not tool_calls:
                pending.append({"role": "assistant", "content": text})
                actions = ACTIONS_ON_REFUSAL if is_refusal(text) else []
                return {"final_reply": text, "agent_steps": steps, "messages": pending,
                        "suggested_actions": actions,
                        "trace": [*state["trace"], f"node=agent steps={steps} converged"]}
            pending.append({"role": "assistant", "content": text or None, "tool_calls": tool_calls})
            messages.append(AIMessage(content=text, tool_calls=tool_calls))
            for tc in tool_calls:
                _emit(writer, {"type": "tool_status", "name": tc["name"],
                               "label": registry.labels().get(tc["name"], tc["name"])})
                result = await registry.execute(tc["name"], json.dumps(tc["args"], ensure_ascii=False))
                # query_faq 证据收集 + 低置信落池 + 编号续排(逻辑自 ch04 路由迁入,n_offset 续排)
                ...
                pending.append({"role": "tool", "content": result, "tool_call_id": tc["id"]})
                messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
        # 熔断:有部分文本就收敛,否则给引导语
        cutoff = pending[-1]["content"] or "问题比较复杂,请您稍后再试或换种问法。"
        return {"final_reply": cutoff, "agent_steps": steps, "messages": pending,
                "suggested_actions": [], "trace": [*state["trace"], f"node=agent steps={steps} cutoff"]}
    return agent_node
```

(`_emit` 与 knowledge.py 同款;`estimate_tokens` 已在 `app/history.py`。query_faq 迁入段完整保留 ch04 语义:low_confidence → `pool.insert`、items 编号 `n_offset` 续排并重写回灌 JSON、citations 帧在最终答案前由路由层攒发——本章 citations 改为:首轮工具结果产生时即由 agent 节点经 writer 发 `{"type":"citations","items":[...]}`,后续轮续排后再发一次全量帧,前端 byN 覆盖语义天然消解(ch04 已验证)。故 `make_agent_node` 需第四参 `pool`:`make_agent_node(model, registry, settings, pool)`,测试同步。)

- [ ] **Step 4: 跑绿** `.venv/bin/pytest tests/test_ch05_agent.py -v`
- [ ] **Step 5: commit** `ch05: ReAct 主力 Agent 节点——手写循环+熔断+actions建议`

### Task 5: 投诉/闲聊固定节点 + log 落库节点

**Files:**
- Create: `app/graph/simple.py`、`app/graph/logging_node.py`
- Test: `tests/test_ch05_simple.py`

**Interfaces:**
- Consumes: `ChatState`;`ConversationStore.append`;`LowConfidencePool.insert`(self_check)
- Produces: `complaint_node(state, writer=None)`(安抚话术 + `suggested_actions=["transfer_human","create_ticket"]` + `messages=[user, assistant]`);`chitchat_node(state, writer=None)`(固定话术);`def make_log_node(store, pool)` → `async log_node(state, writer=None)`(整轮 pending 落库、refusal→self_check 落池、trace 逐行 `logger.info`、发 `actions` 帧)

- [ ] **Step 1: 失败测试**

```python
"""投诉安抚 / 闲聊固定话术 / log 落库与 actions 帧。"""
from app.graph.simple import complaint_node, chitchat_node
from app.graph.logging_node import make_log_node
from tests.test_ch05_entry import _state


class Rec:
    def __init__(self): self.frames = []
    def __call__(self, f): self.frames.append(f)


async def test_complaint_no_llm_two_actions():
    rec = Rec()
    out = await complaint_node(_state(session_id=1, user_message="我要投诉",
                                      resolved_message="我要投诉"), writer=rec)
    assert out["suggested_actions"] == ["transfer_human", "create_ticket"]
    assert "抱歉" in out["final_reply"]  # 安抚话术
    assert [m["role"] for m in out["messages"]] == ["user", "assistant"]
    assert out["final_reply"] == "".join(f["content"] for f in rec.frames if f["type"] == "token")


async def test_chitchat_fixed_reply():
    out = await chitchat_node(_state(user_message="你好呀", resolved_message="你好呀"))
    assert out["final_reply"]  # 固定话术非空
    assert out["suggested_actions"] == []


class FakeStore:
    def __init__(self): self.appended = []
    async def append(self, cid, msgs): self.appended.append((cid, msgs))


class FakePool:
    def __init__(self): self.rows = []
    def insert(self, *a): self.rows.append(a)


async def test_log_persists_and_emits_actions():
    store, pool = FakeStore(), FakePool()
    node = make_log_node(store, pool)
    msgs = [{"role": "user", "content": "我要投诉"},
            {"role": "assistant", "content": "非常抱歉给您带来不好的体验…"}]
    rec = Rec()
    out = await node(_state(session_id=9, user_message="我要投诉", final_reply="非常抱歉给您带来不好的体验…",
                            messages=msgs, suggested_actions=["transfer_human", "create_ticket"]),
                     writer=rec)
    assert store.appended[0][0] == 9 and len(store.appended[0][1]) == 2
    actions = [f for f in rec.frames if f["type"] == "actions"]
    assert [a["action"] for a in actions[0]["items"]] == ["transfer_human", "create_ticket"]
    assert pool.rows == []  # 非拒答不落池


async def test_log_self_check_pool_on_refusal():
    node = make_log_node(FakeStore(), FakePool := type("P", (), {
        "insert": staticmethod(lambda *a: _rows.append(a))})() if False else None)
    # 简化:直接构造 FakePool,断言 source=self_check
```

(第 5 个测试写干净版:`FakePool` 收 rows,final_reply 以 `REFUSAL_MARKER` 开头 → `pool.rows[0][0] == "self_check"`。)

- [ ] **Step 2: 跑红** → **Step 3: 实现**。`simple.py` 话术定稿:

```python
COMPLAINT_REPLY = ("非常抱歉给您带来了不好的体验,您的反馈我们很重视。"
                   "您可以选择下面的方式,我们会尽快为您处理。")
CHITCHAT_REPLY = ("我是本店智能客服,专注商品咨询、订单物流和售后问题;"
                  "闲聊虽然不太擅长,但有关购物的任何问题都可以随时问我哦。")
ACTION_LABELS = {"transfer_human": "转人工", "create_ticket": "建工单"}
```

`logging_node.py`:`log_node` 做四件事——① `await store.append(session_id, messages)`(messages 为空也安全);② `is_refusal(final_reply)` → `pool.insert("self_check", session_id, user_message, "模型自评证据不足:" + final_reply[:200])`;③ `for line in trace: logger.info("[ch05] %s", line)`;④ `suggested_actions` 非空 → `writer({"type":"actions","items":[{"action":a,"label":ACTION_LABELS[a]} for a in ...]} )`。

- [ ] **Step 4: 跑绿** → **Step 5: commit** `ch05: 投诉/闲聊固定节点+log落库与actions帧`

### Task 6: builder 组图 + /api/chat 替换(SSE 契约)

**Files:**
- Create: `app/graph/builder.py`
- Modify: `app/routers/chat.py`(整文件重写)、`app/main.py`(state 挂 `graph`)、`app/history.py`(Task 4 已迁 `to_prompt_messages`)
- Test: `tests/test_ch05_chat_api.py`

**Interfaces:**
- Consumes: Task 2–5 全部节点;`InMemorySaver`;`trim_history`/`to_prompt_messages`(ch04 语义不变,System 居首、预算内裁剪)
- Produces: `build_graph(model, registry, settings, store, pool)` → CompiledStateGraph(`compile(checkpointer=InMemorySaver())`);`/api/chat` SSE 帧序:`session → (tool_status|citations|token|actions)* → done|error`

- [ ] **Step 1: 失败测试**(端到端四出口 + 兼容帧)

```python
"""/api/chat 跑图:SSE 契约 + 四出口 + 历史双写。模型替身 = 意图 JSON + Agent 脚本连发。"""
from langchain_core.messages import AIMessage

from tests.helpers import FakeChatWithTools, post_chat_sse
from tests.test_ch05_knowledge import _seed_kb  # 抽出 Task 3/4 测试里灌知识的小工具为公共函数


class IntentThenAgent(FakeChatWithTools):
    """第 1 次 ainvoke=意图 JSON,之后 astream 按 replies 流式吐。"""
    def __init__(self, intent_json, replies):
        super().__init__(messages=iter([AIMessage(content=intent_json)]))
        self.replies = list(replies)

    async def ainvoke(self, messages):
        self._i = 0
        return await super().ainvoke(messages)

    async def astream(self, messages):
        self._i = getattr(self, "_i", 0) + 1
        yield Chunk(text=self.replies[min(self._i - 1, len(self.replies) - 1)])


async def test_chitchat_fixed_no_llm_after_intent(make_client):
    client, app = await make_client(IntentThenAgent('{"intent": "闲聊"}', []))
    events = await post_chat_sse(client, {"message": "你好呀"})
    assert events[0]["type"] == "session" and events[-1]["type"] == "done"
    tokens = "".join(e["content"] for e in events if e["type"] == "token")
    assert "智能客服" in tokens          # 固定话术
    assert not [e for e in events if e["type"] == "tool_status"]


async def test_complaint_emits_two_action_buttons(make_client):
    client, _ = await make_client(IntentThenAgent('{"intent": "投诉"}', []))
    events = await post_chat_sse(client, {"message": "我要投诉"})
    actions = [e for e in events if e["type"] == "actions"]
    assert len(actions) == 1
    assert [(a["action"], a["label"]) for a in actions[0]["items"]] == [
        ("transfer_human", "转人工"), ("create_ticket", "建工单")]


async def test_knowledge_path_retrieves_then_agent(make_client):
    client, app = await make_client(IntentThenAgent('{"intent": "商品咨询"}', ["带证据的|回答。"]))
    _seed_kb(app)
    events = await post_chat_sse(client, {"message": "SH-E300 多少钱"})
    assert any(e["type"] == "tool_status" and e["name"] == "query_faq" for e in events)
    assert "".join(e["content"] for e in events if e["type"] == "token") == "带证据的回答。"


async def test_weak_evidence_blocked_before_agent(make_client):
    """空知识库 → 闸拦下,Agent 不被调用(第二次 astream 不发生),落池 retrieval_low_conf。"""
    client, app = await make_client(IntentThenAgent('{"intent": "商品咨询"}', ["不该出现"]))
    events = await post_chat_sse(client, {"message": "量子力学怎么退货"})
    assert "无法回答" in "".join(e["content"] for e in events if e["type"] == "token")
    assert "不该出现" not in "".join(e["content"] for e in events if e["type"] == "token")
    rows = _pool_rows(app.state.session_factory)
    assert rows[-1]["source"] == "retrieval_low_conf"


async def test_business_path_agent_calls_tools_and_persists(make_client):
    client, app = await make_client(IntentThenAgent('{"intent": "订单"}', ["查到了。"]))
    events = await post_chat_sse(client, {"message": "订单 1001 的物流到哪了"})
    assert any(e["type"] == "token" and "查到了" in e["content"] for e in events)
    history = await app.state.store.get_history(events[0]["session_id"])
    assert history[0]["role"] == "user" and history[-1]["role"] == "assistant"
```

(注:业务路径下 Agent 若不调工具直接回答也成立——`IntentThenAgent` 的 astream 纯文本;调工具版本由 `ScriptedToolModel` 改造(ainvoke 返回意图、astream 走脚本)在 Task 4/6 各覆盖。`Chunk` 自 `tests.test_chat_toolflow` 导入。`_pool_rows` 自 `tests.test_chat_toolflow` 导入或复制。)

- [ ] **Step 2: 跑红** → **Step 3: 实现**

`app/graph/builder.py`:

```python
"""组图:START→resolve→intent→route 四出口;agent/fallback/complaint/chitchat→log→END。"""
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from app.graph.agent import make_agent_node
from app.graph.entry import intent_node, resolve_node, route
from app.graph.knowledge import make_knowledge_nodes
from app.graph.logging_node import make_log_node
from app.graph.simple import chitchat_node, complaint_node
from app.graph.state import ChatState


def build_graph(model, registry, settings, store, pool, retrieval_service, kb):
    g = StateGraph(ChatState)
    g.add_node("resolve", resolve_node)
    g.add_node("intent", lambda state: intent_node(state, model))  # writer 经 StreamWriter 注入会丢——
    # 改用 functools.partial / 闭包装:节点签名 (state, writer) 由 langgraph 注入 writer,
    # model 用 partial 绑定:partial(intent_node, model=model) 不行(位置参)→ 定义小闭包
    g.add_node("retrieve", retrieve_node)   # 均为 make_*_nodes 闭包产物
    g.add_node("gate", gate_node)
    g.add_node("fallback", fallback_node)
    g.add_node("agent", agent_node)
    g.add_node("complaint", complaint_node)
    g.add_node("chitchat", chitchat_node)
    g.add_node("log", make_log_node(store, pool))
    g.add_edge(START, "resolve")
    g.add_edge("resolve", "intent")
    g.add_conditional_edges("intent", route, {
        "retrieve": "retrieve", "agent": "agent", "complaint": "complaint", "chitchat": "chitchat"})
    g.add_edge("retrieve", "gate")
    g.add_conditional_edges("gate", lambda s: "fallback" if not s["gate_passed"] else "agent",
                            {"fallback": "fallback", "agent": "agent"})
    for n in ("fallback", "agent", "complaint", "chitchat"):
        g.add_edge(n, "log")
    g.add_edge("log", END)
    return g.compile(checkpointer=InMemorySaver())
```

实现时注意:**writer 注入只对"函数第一参后带 `writer` 参"的节点生效**;`intent_node(state, model)` 需包成 `async def _node(state, writer): return await intent_node(state, model, writer)` 风格的闭包(intent 不需要 writer,可 `(state)` 直传同理)。`make_knowledge_nodes`/`make_agent_node` 闭包产物签名已是 `(state, writer=None)`,直接 `add_node` 即可(langgraph 按参数名注入 writer)。make_client 的 app.state 需挂 `graph`(conftest 在 Task 6 一并改:`app.state.graph = build_graph(chat_model, registry, settings, store, pool, RetrievalService替身链, kb)`,其中检索链复用 `build_tools` 同一套构造——生产路径由 `build_graph` 内部按 `settings` 构造真 RetrievalService + KnowledgeBaseStore,与 `build_tools` 的生产分支同构;**测试注入替身**时 conftest 传替身)。

`app/routers/chat.py` 重写(保留 `sse_frame`;`_merge_tool_call_chunks`/`_to_prompt_messages` 迁往 `agent.py`/`history.py`):

```python
@router.post("/api/chat")
async def chat(body: ChatRequest, request: Request) -> StreamingResponse:
    settings, store, graph = request.app.state.settings, request.app.state.store, request.app.state.graph
    if estimate_tokens(body.message) > settings.history_token_budget:
        raise HTTPException(status_code=400, detail="消息过长,超出会话历史预算")
    session_id = await store.resolve(body.session_id)
    history = trim_history([{"role": "system", "content": SERVICE_PROMPT_TEMPLATE.format()},
                            *(await store.get_history(session_id))],
                           settings.history_token_budget)

    async def event_stream():
        yield sse_frame({"type": "session", "session_id": session_id})
        try:
            config = {"configurable": {"thread_id": str(session_id)}}
            inp = {"session_id": session_id, "user_message": body.message, "history": history[1:]}
            async for frame in graph.astream(inp, config, stream_mode="custom"):
                yield sse_frame(frame)
            yield sse_frame({"type": "done"})
        except Exception as exc:
            yield sse_frame({"type": "error", "message": str(exc)})

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
```

要点:`history` 传裁剪后去掉 System 的历史(dict 列表,Task 4 的 `_history_messages` 把 dict→LC 消息,含 tool_calls/tool_call_id 还原——即 ch04 `_to_prompt_messages` 主体);落库由 log 节点做(整轮成功才落库语义保持);error 帧后本轮不落库(agent 节点 pending 只在 state 里,没到 log 节点自然不落)。

- [ ] **Step 4: 跑绿**(全量 `.venv/bin/pytest`,旧 `test_chat_toolflow.py`/`test_chat_api.py` 中针对旧路由实现的用例按新契约迁移:帧序断言保留,bind_calls/两段式断言删除或改为 agent 节点单测已覆盖的语义;**历史双写、citations 帧序、落池语义的用例必须存活**)。
- [ ] **Step 5: commit** `ch05: builder 组图+/api/chat 替换跑图,SSE 契约兼容+actions 帧`

### Task 7: /api/tickets 路由(建工单按钮后端)

**Files:**
- Create: `app/routers/tickets.py`
- Modify: `app/schemas.py`(加 `TicketRequest`)、`app/main.py`(挂路由)
- Test: `tests/test_ch05_tickets.py`

**Interfaces:**
- Produces: `POST /api/tickets` body `{"session_id": int, "description": str, "ticket_type": "投诉"|"售后"|"咨询"=缺省"投诉"}` → `200 {"ticket_no", "status"}`;会话不存在 → 404

- [ ] **Step 1: 失败测试**

```python
async def test_create_ticket_writes_table(make_client):
    client, app = await make_client(fake_chat("x"))
    resp = await client.post("/api/tickets", json={"session_id": None, "description": "商家一直不发货"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ticket_no"].startswith("T")
    from app.models import Ticket
    with app.state.session_factory() as s:
        assert s.query(Ticket).count() == 1


async def test_create_ticket_unknown_session_404(make_client):
    client, _ = await make_client(fake_chat("x"))
    resp = await client.post("/api/tickets", json={"session_id": 999, "description": "x"})
    assert resp.status_code == 404
```

- [ ] **Step 2: 跑红** → **Step 3: 实现**(`registry.execute("create_ticket", json)` 复用,conversation_id 用 `store.resolve` 结果;`session_id` 缺省 resolve 新建不 404,显式给了但 `store.exists` False 才 404——上面测试 999 → 404、None → 建新会话成功,第二个测试不变、第一个里 None 路径成立):

```python
@router.post("/api/tickets")
async def create_ticket(body: TicketRequest, request: Request):
    store, registry = request.app.state.store, request.app.state.registry
    if body.session_id is not None and not await store.exists(body.session_id):
        raise HTTPException(status_code=404, detail="会话不存在")
    conversation_id = await store.resolve(body.session_id)
    raw = await registry.execute("create_ticket", json.dumps({
        "conversation_id": conversation_id, "description": body.description,
        "ticket_type": body.ticket_type}, ensure_ascii=False))
    parsed = json.loads(raw)
    if "ticket_no" not in parsed:
        raise HTTPException(status_code=502, detail="创建工单失败")
    return parsed
```

- [ ] **Step 4: 跑绿** → **Step 5: commit** `ch05: /api/tickets 路由——建工单按钮唯一后端入口`

### Task 8: 前端 actions 按钮 + 确认交互

**Files:**
- Modify: `static/index.html`(send() 事件分派加 `actions` 分支;新增 `renderActions`/`transferHuman`/`createTicket` 三个函数与按钮样式)

**Interfaces:**
- Consumes: `actions` 帧 `{"items":[{"action","label"}]}`;`POST /api/tickets`

- [ ] **Step 1: 实现**(本任务为纯前端静态改动,验证 = Task 9 真机验收;先写后验):

```javascript
function renderActions(bubble, items) {
  if (!items || !items.length) return;
  const bar = document.createElement("div");
  bar.className = "action-bar";
  for (const it of items) {
    const btn = document.createElement("button");
    btn.className = "action-btn";
    btn.textContent = it.label;
    btn.onclick = () => it.action === "transfer_human" ? transferHuman(bar) : createTicket(bar, btn);
    bar.appendChild(btn);
  }
  bubble.appendChild(bar);
}

function transferHuman(bar) {
  bar.remove();
  addSystemMsg("已转接人工客服");
  addBotMsg().querySelector(".content").textContent = "您好,我是客服小猫,请问有什么可以帮您的";
}

async function createTicket(bar, btn) {
  if (!confirm("确认为您创建工单吗?")) return;
  btn.disabled = true;
  const resp = await fetch("/api/tickets", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: state.sessionId, description: lastUserMessage }),
  });
  const body = await resp.json();
  btn.textContent = resp.ok ? "工单 " + body.ticket_no + " 已创建" : "创建失败";
}
```

配套:`addSystemMsg`(居中灰字样式,新增)、`lastUserMessage` 在 send() 里记录本轮输入、`.action-bar/.action-btn` 样式(沿现有配色)。两按钮互不绑定、不点即无动作。

- [ ] **Step 2: commit** `ch05: 前端转人工/建工单独立按钮+确认交互`

### Task 9: 真机验收 + 文档收口

**Files:**
- Modify: `dev-notes/ch05.md`、`README.md`(ch05 段)、`pyproject.toml`(如有遗漏)

- [ ] docker compose + seed + 真链路五条验收(spec「真机验收」),日志确认 `node=retrieve`、Agent 多步 trace
- [ ] dev-notes 逐任务已追记(每任务完成时即时补,不收尾一次性补)
- [ ] 全量 `.venv/bin/pytest` 绿;README ch05 段;commit `ch05: 真机验收+文档收口`

---

## Self-Review 结论

- Spec 覆盖:祛魅(Task1)/图骨架分流(Task2)/检索+闸(Task3)/ReAct(Task4)/投诉闲聊+actions(Task5)/替换路由(Task6)/建工单(Task7)/前端(Task8)/验收(Task9)——全覆盖。
- 遗漏修正:ChatState 需补 `low_confidence`/`low_reason` 两字段(Task 3 已注明回填 state.py);`make_agent_node` 第四参 pool 已在 Task 4 注明。
- 类型一致性:节点签名统一 `(state, writer=None)`;`evidence` 条目字段与 ch04 citations 契约一致(n/chunk_id/question/answer/category/section_path)。
