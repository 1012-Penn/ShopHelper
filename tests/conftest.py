import os

# 模块级 app = create_app() 在导入期构造 Settings,而 openai_api_key 必填;
# setdefault 只在环境缺失时兜底,真实 .env / 显式环境变量不受影响
os.environ.setdefault("OPENAI_API_KEY", "sk-test")

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import Settings
from app.main import create_app
from app.sessions import SessionStore
from tests.helpers import StubExtractModel


@pytest.fixture
async def make_client():
    """构建测试应用:真 Settings 换测试值,sessions/模型全部替换为替身。"""
    clients = []

    async def _make(chat_model, extract_model=None):
        app = create_app(Settings(openai_api_key="sk-test", _env_file=None))
        app.state.sessions = SessionStore()
        app.state.chat_model = chat_model
        app.state.extract_model = extract_model or StubExtractModel()
        client = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
        clients.append(client)
        return client, app

    yield _make
    for c in clients:
        await c.aclose()
