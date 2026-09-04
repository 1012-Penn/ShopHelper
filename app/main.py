from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.config import Settings
from app.llm import make_chat_model, make_extract_model
from app.routers import chat, extract, sessions
from app.schemas import AfterSaleExtraction
from app.sessions import session_store

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    app = FastAPI(title="ShopHelper")
    app.state.settings = settings
    app.state.sessions = session_store
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
