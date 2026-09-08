import os

# 模块级 app = create_app() 在导入期构造 Settings,而 openai_api_key 必填;
# setdefault 只在环境缺失时兜底,真实 .env / 显式环境变量不受影响
os.environ.setdefault("OPENAI_API_KEY", "sk-test")
os.environ.setdefault("EMBEDDING_API_KEY", "sk-test")

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import Settings
from app.guard import LowConfidencePool
from app.kb import KnowledgeBaseStore
from app.main import create_app
from app.retrieval import RetrievalService
from app.store import ConversationStore
from app.models import Base
from tests.helpers import (FakeEmbedding, FakeReranker, FakeRewriter, FakeVectorStore,
                           EchoResolver, StubExpand, StubExtractModel)


def make_session_factory():
    # StaticPool:内存库全线程共享一条连接——registry 用 to_thread 在工作线程执行工具,
    # 默认按线程分连接会拿到另一个空库
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


@pytest.fixture
def db_session_factory():
    return make_session_factory()


@pytest.fixture
def db_store(db_session_factory):
    return ConversationStore(db_session_factory)


@pytest.fixture
async def make_client():
    """构建测试应用:真 Settings 换测试值,存储用 SQLite 内存库,模型/嵌入/向量库全用替身。"""
    clients = []

    async def _make(chat_model, extract_model=None, resolver=None, expander=None):
        app = create_app(Settings(openai_api_key="sk-test", _env_file=None))
        factory = make_session_factory()
        app.state.session_factory = factory
        app.state.store = ConversationStore(factory)
        app.state.pool = LowConfidencePool(factory)
        # store 被替换,摘要服务必须随之重建(M-2):否则它仍持 create_app 时的 MySQL factory
        from app.llm import make_extract_model
        from app.summarizer import SummaryService

        app.state.summary_service = SummaryService(
            app.state.store, make_extract_model(app.state.settings))
        vectors_stub = FakeVectorStore()
        app.state.vectors = vectors_stub
        # ch08 工具系统装配:内置注册(替身检索)+ 执行引擎;测试不接 MCP(空 urls)
        from app.tools.audit import ToolAuditStore
        from app.tools.base import ToolContext, ToolRegistryV2
        from app.tools.builtin import register_builtin
        from app.tools.engine import ToolEngine
        from app.tools.mcp_client import McpService

        kb = KnowledgeBaseStore(factory)
        service = RetrievalService(
            FakeEmbedding(), vectors_stub, kb, rewriter=FakeRewriter(), reranker=FakeReranker(),
            candidates=50, final_top_k=3, rerank_score_floor=0.30,
        )
        registry = ToolRegistryV2()
        tool_ctx = ToolContext(session_factory=factory, service=service, kb=kb, top_k=3)
        register_builtin(registry, tool_ctx)
        app.state.registry = registry
        app.state.tool_audit_store = ToolAuditStore(factory)
        app.state.engine = ToolEngine(registry, app.state.tool_audit_store,
                                      timeout_seconds=app.state.settings.tool_timeout_seconds,
                                      retry_attempts=app.state.settings.tool_retry_attempts)
        app.state.mcp_service = McpService(registry, {})
        app.state.retrieval_service = service
        app.state.retrieval_kb = kb
        app.state.chat_model = chat_model
        app.state.extract_model = extract_model or StubExtractModel()
        # ch05:图骨架,检索链用与工具检索替身同参的 stub
        from app.graph.builder import build_graph

        # ch06:resolve/expand 替身缺省注入,测试不出网;intent 显式用 chat 替身(生产缺省是温度 0 的 extract 模型)
        app.state.graph = build_graph(
            chat_model, app.state.engine, app.state.settings,
            app.state.store, app.state.pool, service, kb,
            resolver=resolver or EchoResolver({}),
            expander=expander or StubExpand({}),
            intent_model=chat_model,
        )
        app.state.retrieval_service = service
        app.state.retrieval_kb = kb
        client = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
        clients.append(client)
        return client, app

    yield _make
    for c in clients:
        await c.aclose()
