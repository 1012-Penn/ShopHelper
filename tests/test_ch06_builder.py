"""I-4 回归:build_graph 的 escalation 生产接线——小模型先判、低置信升级大模型重判一次。"""
from tests.conftest import make_session_factory
from tests.helpers import (EchoResolver, FakeEmbedding, FakeReranker, FakeRewriter,
                           FakeVectorStore, StubExpand, GraphChatModel)
from tests.test_chat_toolflow import _tc


class _Fixed:
    """恒定回复的 ainvoke 替身,计数调用。"""

    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    async def ainvoke(self, messages, **kwargs):
        from langchain_core.messages import AIMessage

        self.calls += 1
        return AIMessage(content=self.reply)


async def test_build_graph_escalation_low_confidence_escalates(monkeypatch):
    import app.llm as llm_mod

    from app.config import Settings
    from app.graph.builder import build_graph, initial_state
    from app.guard import LowConfidencePool
    from app.kb import KnowledgeBaseStore
    from app.retrieval import RetrievalService
    from app.store import ConversationStore
    from app.tools.definitions import build_tools
    from app.tools.registry import ToolRegistry

    settings = Settings(openai_api_key="sk-test", _env_file=None,
                        intent_escalation_enabled=True)
    small = _Fixed('{"intent": "商品咨询", "confidence": 0.2}')  # 低于 floor
    big = _Fixed('{"intent": "闲聊", "confidence": 0.9}')

    def fake_extract(s, model=None):
        assert model == (settings.intent_small_model or None)  # small 按配置构造
        return small

    monkeypatch.setattr(llm_mod, "make_extract_model", fake_extract)

    factory = make_session_factory()
    store = ConversationStore(factory)
    pool = LowConfidencePool(factory)
    vectors = FakeVectorStore()
    kb = KnowledgeBaseStore(factory)
    service = RetrievalService(FakeEmbedding(), vectors, kb, rewriter=FakeRewriter(),
                               reranker=FakeReranker(), candidates=50, final_top_k=3,
                               rerank_score_floor=0.30)
    registry = ToolRegistry(build_tools(factory, embedder=FakeEmbedding(), vectors=vectors,
                                        top_k=3, rewriter=FakeRewriter(), reranker=FakeReranker()))
    chat = GraphChatModel(intent_reply="{}", turns=[("text", "有购物问题随时问我")])
    graph = build_graph(chat, registry, settings, store, pool, service, kb,
                        resolver=EchoResolver({}), expander=StubExpand({}),
                        escalator=big)  # 大模型显式注入,避免真构造

    config = {"configurable": {"thread_id": "esc1"}}
    final = await graph.ainvoke(initial_state(1, "你好呀", None), config)
    assert small.calls == 1 and big.calls == 1  # 小模型先判,低置信触发大模型重判
    assert final["intent"] == "闲聊"
    assert any("escalated=true" in line for line in final["trace"])
