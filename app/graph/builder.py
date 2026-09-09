"""组图:START→resolve→intent→route 多出口;退款/售后走确定性子流程;全部出口汇于 log→END。

流式说明(py3.10 兼容,见 spec 实现期偏差):langgraph 的 get_stream_writer 在
Python 3.10 async 上下文被官方守卫禁用,故节点发帧走 configurable.sink 队列——
路由层建 asyncio.Queue 注入 config,节点内 writer 闭包 put_nowait,路由层排空转 SSE。
interrupt() 同受该守卫禁用(spike 实测),订单选择器停走走无状态回传(resume 旁路)。
"""
import asyncio
import json
import logging
from pathlib import Path
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from app.graph.agent import make_agent_node
from app.graph.entry import make_intent_node, make_resolve_node, route
from app.graph.knowledge import make_knowledge_nodes
from app.graph.logging_node import make_log_node
from app.graph.refund import make_refund_nodes
from app.graph.simple import chitchat_node, complaint_node
from app.graph.state import ChatState
from app.history import rows_to_messages

logger = logging.getLogger(__name__)


def _load_evidence_calibration(settings):
    """加载 ch04 校准产物;产物不可用时回落到当前 rerank floor。"""
    from app.retrieval import EvidenceCalibration

    path = Path(settings.evidence_calibration_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        allowed = {"top1_floor", "effective_score_floor", "min_effective_count",
                   "margin_floor", "combined_floor", "source"}
        return EvidenceCalibration(**{key: value for key, value in payload.items() if key in allowed})
    except (FileNotFoundError, OSError, ValueError, TypeError) as exc:
        logger.info("证据置信度校准产物不可用,使用配置基线:%s", exc)
        return EvidenceCalibration(top1_floor=settings.rerank_score_floor,
                                   effective_score_floor=settings.rerank_score_floor)


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


def build_graph(model, engine, settings, store, pool, retrieval_service, kb, *,
                resolver=None, intent_model=None, escalator=None, expander=None):
    """resolver/expander 缺省时生产路径自建(结构化真模型);测试经 conftest 注入替身。
    intent 默认用传入 model;intent_escalation_enabled 时小模型先判、大模型 escalator 重判。"""
    from app.llm import make_extract_model
    from app.prompts import ExpandedQueries, ResolvedQuestion

    retrieve_node, gate_node, fallback_node = make_knowledge_nodes(
        retrieval_service, kb, pool, snapshot_top_k=settings.langfuse_retrieval_top_k)

    if resolver is None:
        resolver = make_extract_model(settings).with_structured_output(
            ResolvedQuestion, method="function_calling")
    if expander is None:
        expander = make_extract_model(settings).with_structured_output(
            ExpandedQueries, method="function_calling")
    if settings.intent_escalation_enabled:
        small = make_extract_model(settings, model=settings.intent_small_model or None)
        _intent = make_intent_node(small, escalator=escalator or make_extract_model(settings),
                                   floor=settings.intent_confidence_floor)
    else:
        # spec 定稿:意图默认温度 0(extract 口径)——传入的 model 是 chat 温度(0.7),不能直接当分类器
        _intent = make_intent_node(intent_model or make_extract_model(settings))

    _resolve = make_resolve_node(resolver)
    prep, ask, fetch, expand, policy = make_refund_nodes(
        expander, retrieval_service, kb, pool, top_k=settings.rerank_top_k)

    g = StateGraph(ChatState)
    # 一切发帧节点都套 _with_sink:py3.10 async 下 langgraph 的 writer 注入失效,
    # 帧经 configurable.sink 队列外送(spec 实现期偏差)
    g.add_node("resolve", _with_sink(_resolve))
    g.add_node("intent", _intent)
    g.add_node("retrieve", _with_sink(retrieve_node))
    g.add_node("gate", _with_sink(gate_node))
    g.add_node("fallback", _with_sink(fallback_node))
    g.add_node("agent", _with_sink(make_agent_node(model, engine, settings, pool)))
    g.add_node("complaint", _with_sink(complaint_node))
    g.add_node("chitchat", _with_sink(chitchat_node))
    g.add_node("prepare_order", _with_sink(prep))
    g.add_node("ask_order", _with_sink(ask))
    g.add_node("fetch_order", _with_sink(fetch))
    g.add_node("expand", _with_sink(expand))
    g.add_node("policy_retrieve", _with_sink(policy))
    g.add_node("log", _with_sink(make_log_node(store, pool)))

    g.add_edge(START, "resolve")
    # resume 旁路:订单选择器点选回传,补全问题已定,直达子流程不再过 resolve/intent 的 LLM
    g.add_conditional_edges("resolve",
                            lambda s: "prepare_order" if s.get("resume_question") else "intent",
                            {"intent": "intent", "prepare_order": "prepare_order"})
    g.add_conditional_edges("intent", route, {
        "retrieve": "retrieve", "agent": "agent", "complaint": "complaint",
        "chitchat": "chitchat", "prepare_order": "prepare_order"})
    g.add_conditional_edges("prepare_order",
                            lambda s: "fetch_order" if s.get("pending_order_id") else "ask_order",
                            {"fetch_order": "fetch_order", "ask_order": "ask_order"})
    g.add_edge("fetch_order", "expand")
    g.add_edge("expand", "policy_retrieve")
    g.add_edge("policy_retrieve", "agent")
    g.add_edge("retrieve", "gate")
    g.add_conditional_edges("gate", lambda s: "fallback" if not s["gate_passed"] else "agent",
                            {"fallback": "fallback", "agent": "agent"})
    for n in ("fallback", "agent", "complaint", "chitchat", "ask_order"):
        g.add_edge(n, "log")
    g.add_edge("log", END)
    return g.compile(checkpointer=InMemorySaver())


def initial_state(session_id: int, user_message: str, resume=None,
                  history_rows: list | None = None, layered: dict | None = None) -> dict:
    """路由层每回合注入的全量默认值:防 checkpointer 上一回合残留字段(如 evidence)泄漏;
    resume 非空时携带选择器点选回传的槽位(兼容对象与 dict)。
    history_rows 非空 = 该 thread 的 checkpoint 为空(进程重启/首 touch),从 MySQL 回灌完整历史
    (行带 id,add_messages 按 id 去重,并发首轮不会双灌);layered 是模型面分层精简版。"""
    resume_order_id = getattr(resume, "order_id", None) if resume else ""
    if not resume_order_id and isinstance(resume, dict):
        resume_order_id = resume.get("order_id", "")
    resume_question = getattr(resume, "question", None) if resume else ""
    if not resume_question and isinstance(resume, dict):
        resume_question = resume.get("question", "")
    resume_intent = getattr(resume, "intent", None) if resume else ""
    if not resume_intent and isinstance(resume, dict):
        resume_intent = resume.get("intent", "")

    msgs = rows_to_messages(history_rows) if history_rows else []
    user_msg_id = f"u{uuid4().hex}"
    msgs.append(HumanMessage(content=user_message, id=user_msg_id))

    return {"session_id": session_id, "user_message": user_message, "resolved_message": "",
            "intent": resume_intent, "intent_confidence": 0.0,
            "evidence": [], "retrieved_chunks": [], "evidence_confidence": None,
            "low_confidence": False,
            "low_reason": "", "gate_passed": False, "agent_steps": 0, "final_reply": "",
            "suggested_actions": [], "trace": [],
            "messages": msgs, "turn_user_msg_id": user_msg_id, "layered": layered or {},
            "order": {}, "expand_queries": [], "refund_flow": False,
            "pending_order_id": "", "resume_order_id": resume_order_id,
            "resume_question": resume_question}


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
        final_top_k=settings.rerank_top_k,
        rerank_score_floor=settings.rerank_score_floor,
        evidence_calibration=_load_evidence_calibration(settings),
    )
    return service, kb
