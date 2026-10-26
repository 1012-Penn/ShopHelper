import os
import tempfile

# 模块级 app = create_app() 在导入期构造 Settings,而 openai_api_key 必填;
# setdefault 只在环境缺失时兜底,真实 .env / 显式环境变量不受影响
os.environ.setdefault("OPENAI_API_KEY", "sk-test")
os.environ.setdefault("EMBEDDING_API_KEY", "sk-test")
os.environ.setdefault("ADMIN_SESSION_SECRET", "test-admin-secret")

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

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
    # 独立文件 SQLite 让请求/工作线程使用不同连接;测试仍完全隔离,不依赖共享 MySQL。
    tmpdir = tempfile.TemporaryDirectory(prefix="shophelper-test-")
    db_path = os.path.join(tmpdir.name, "test.db")
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False, "timeout": 30},
        poolclass=NullPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    factory._test_tmpdir = tmpdir
    return factory


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

    async def _make(chat_model, extract_model=None, resolver=None, expander=None,
                    settings_overrides=None):
        app = create_app(Settings(
            openai_api_key="sk-test", _env_file=None,
            admin_session_secret="test-admin-secret",
            admin_session_ttl_seconds=3600,
            admin_cookie_secure=False,
            admin_password_iterations=2_000,
            **(settings_overrides or {}),
        ))
        factory = make_session_factory()
        app.state.session_factory = factory
        app.state.store = ConversationStore(factory)
        from app.store import UsageStore
        from app.execution.service import ExecutionService
        from app.execution.store import ExecutionStore

        app.state.usage_store = UsageStore(factory)
        from app.observability import ObservabilityService
        app.state.observability = ObservabilityService(
            app.state.settings, app.state.usage_store)
        app.state.execution_store = ExecutionStore(factory)
        app.state.execution_service = ExecutionService(app.state.execution_store)
        app.state.pool = LowConfidencePool(factory)
        # create_app 里 admin store 指向真实 MySQL factory;测试必须重绑到 SQLite
        from app.admin_store import AdminAuditStore, AdminStore
        app.state.admin_store = AdminStore(factory, password_iterations=2_000)
        app.state.admin_audit_store = AdminAuditStore(factory)
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
        from app.execution.tool_engine import ExecutionToolEngine

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
        app.state.engine = ExecutionToolEngine(
            app.state.engine, app.state.execution_store)
        app.state.mcp_service = McpService(registry, {})
        app.state.retrieval_service = service
        app.state.retrieval_kb = kb
        app.state.chat_model = chat_model
        app.state.extract_model = extract_model or StubExtractModel()
        # ch05:图骨架,检索链用与工具检索替身同参的 stub
        from app.graph.builder import build_graph
        from app.rag_steps import RagLoopStepStore

        # ch06:resolve/expand 替身缺省注入,测试不出网;intent 显式用 chat 替身(生产缺省是温度 0 的 extract 模型)
        rag_steps_store = RagLoopStepStore(factory)
        app.state.rag_steps_store = rag_steps_store
        app.state.graph = build_graph(
            chat_model, app.state.engine, app.state.settings,
            app.state.store, app.state.pool, service, kb,
            resolver=resolver or EchoResolver({}),
            expander=expander or StubExpand({}),
            intent_model=chat_model,
            rag_steps_store=rag_steps_store,
        )
        app.state.retrieval_service = service
        app.state.retrieval_kb = kb
        client = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
        clients.append(client)
        return client, app

    yield _make
    for c in clients:
        await c.aclose()
