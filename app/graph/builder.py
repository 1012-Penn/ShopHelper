"""组图:START→resolve→intent→route 四出口;agent/fallback/complaint/chitchat→log→END。

流式说明(py3.10 兼容,见 spec 实现期偏差):langgraph 的 get_stream_writer 在
Python 3.10 async 上下文被官方守卫禁用,故节点发帧走 configurable.sink 队列——
路由层建 asyncio.Queue 注入 config,节点内 writer 闭包 put_nowait,路由层排空转 SSE。
"""
import asyncio

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from app.graph.agent import make_agent_node
from app.graph.entry import intent_node, resolve_node, route
from app.graph.knowledge import make_knowledge_nodes
from app.graph.logging_node import make_log_node
from app.graph.simple import chitchat_node, complaint_node
from app.graph.state import ChatState


def _sink_adapter(sink) -> object:
    """asyncio.Queue → 可调用 writer;None 时保持 None(节点 _emit 静默)。"""
    if sink is None:
        return None
    return lambda frame: sink.put_nowait(frame)


def _with_sink(fn):
    """把 configurable.sink 适配成节点 writer 参数;同时让节点可被 (state, writer) 直测。"""

    async def _wrapped(state, config=None):
        sink = (config or {}).get("configurable", {}).get("sink")
        return await fn(state, writer=_sink_adapter(sink))

    return _wrapped


def build_graph(model, registry, settings, store, pool, retrieval_service, kb):
    retrieve_node, gate_node, fallback_node = make_knowledge_nodes(retrieval_service, kb, pool)

    async def _intent(state):
        return await intent_node(state, model)

    g = StateGraph(ChatState)
    g.add_node("resolve", resolve_node)
    g.add_node("intent", _intent)
    # 一切发帧节点都套 _with_sink:py3.10 async 下 langgraph 的 writer 注入失效,
    # 帧经 configurable.sink 队列外送(spec 实现期偏差)
    g.add_node("retrieve", _with_sink(retrieve_node))
    g.add_node("gate", _with_sink(gate_node))
    g.add_node("fallback", _with_sink(fallback_node))
    g.add_node("agent", _with_sink(make_agent_node(model, registry, settings, pool)))
    g.add_node("complaint", _with_sink(complaint_node))
    g.add_node("chitchat", _with_sink(chitchat_node))
    g.add_node("log", _with_sink(make_log_node(store, pool)))
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


def initial_state(session_id: int, user_message: str, history: list) -> dict:
    """路由层每回合注入的全量默认值:防 checkpointer 上一回合残留字段(如 evidence)泄漏。"""
    return {"session_id": session_id, "user_message": user_message, "resolved_message": "",
            "intent": "", "evidence": [], "low_confidence": False, "low_reason": "",
            "gate_passed": False, "agent_steps": 0, "final_reply": "", "suggested_actions": [],
            "trace": [], "messages": [], "history": history}


def make_retrieval_chain(session_factory, settings):
    """生产路径检索链(与 build_tools 生产分支同构):真 embedder/向量库/rewriter/reranker。"""
    from app.embedding import make_embedder
    from app.kb import KnowledgeBaseStore
    from app.rerank import make_reranker
    from app.retrieval import RetrievalService
    from app.rewrite import make_rewriter
    from app.vector_store import KnowledgeVectorStore

    kb = KnowledgeBaseStore(session_factory)
    service = RetrievalService(
        make_embedder(settings),
        KnowledgeVectorStore(settings.milvus_db_path, dim=settings.embedding_dim),
        kb,
        rewriter=make_rewriter(settings) if settings.query_rewrite_enabled else None,
        reranker=make_reranker(settings),
        candidates=settings.hybrid_candidates,
        final_top_k=settings.retrieval_final_top_k,
        rerank_score_floor=settings.rerank_score_floor,
    )
    return service, kb
