from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.config import Settings
from app.db import make_engine, make_session_factory
from app.guard import LowConfidencePool
from app.llm import make_chat_model, make_extract_model
from app.routers import chat, extract, sessions
from app.schemas import AfterSaleExtraction
from app.store import ConversationStore
from app.tools.definitions import build_tools
from app.tools.registry import ToolRegistry

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    app = FastAPI(title="ShopHelper")
    app.state.settings = settings
    # 引擎惰性连接;测试注入替身时直接覆写 state 上这几个对象即可
    session_factory = make_session_factory(make_engine(settings))
    app.state.session_factory = session_factory
    app.state.store = ConversationStore(session_factory)
    app.state.pool = LowConfidencePool(session_factory)
    # build_tools 生产路径:内部构造真 embedder/向量库(v2)/rewriter/reranker
    app.state.registry = ToolRegistry(build_tools(
        session_factory, top_k=settings.retrieval_final_top_k,
    ))
    app.state.chat_model = make_chat_model(settings)
    app.state.extract_model = make_extract_model(settings).with_structured_output(
        AfterSaleExtraction, method="function_calling"
    )
    app.include_router(chat.router)
    app.include_router(sessions.router)
    app.include_router(extract.router)

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
