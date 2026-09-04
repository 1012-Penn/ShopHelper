import os

# 模块级 app = create_app() 在导入期构造 Settings,而 openai_api_key 必填;
# setdefault 只在环境缺失时兜底,真实 .env / 显式环境变量不受影响
os.environ.setdefault("OPENAI_API_KEY", "sk-test")

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.main import create_app
from app.store import ConversationStore
from app.tools.definitions import build_tools
from app.tools.registry import ToolRegistry
from app.models import Base
from tests.helpers import StubExtractModel


def make_session_factory():
    engine = create_engine("sqlite:///:memory:")
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
    """构建测试应用:真 Settings 换测试值,存储用 SQLite 内存库,模型用替身。"""
    clients = []

    async def _make(chat_model, extract_model=None):
        app = create_app(Settings(openai_api_key="sk-test", _env_file=None))
        factory = make_session_factory()
        app.state.session_factory = factory
        app.state.store = ConversationStore(factory)
        app.state.registry = ToolRegistry(build_tools(factory))
        app.state.chat_model = chat_model
        app.state.extract_model = extract_model or StubExtractModel()
        client = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
        clients.append(client)
        return client, app

    yield _make
    for c in clients:
        await c.aclose()
