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
from app.main import create_app
from app.store import ConversationStore
from app.tools.definitions import build_tools
from app.tools.registry import ToolRegistry
from app.models import Base
from tests.helpers import (FakeEmbedding, FakeReranker, FakeRewriter, FakeVectorStore,
                           StubExtractModel)


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

    async def _make(chat_model, extract_model=None):
        app = create_app(Settings(openai_api_key="sk-test", _env_file=None))
        factory = make_session_factory()
        app.state.session_factory = factory
        app.state.store = ConversationStore(factory)
        app.state.pool = LowConfidencePool(factory)
        vectors_stub = FakeVectorStore()
        app.state.vectors = vectors_stub
        app.state.registry = ToolRegistry(build_tools(
            factory, embedder=FakeEmbedding(), vectors=vectors_stub, top_k=3,
            rewriter=FakeRewriter(), reranker=FakeReranker(),
        ))
        app.state.chat_model = chat_model
        app.state.extract_model = extract_model or StubExtractModel()
        client = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
        clients.append(client)
        return client, app

    yield _make
    for c in clients:
        await c.aclose()
