from fastapi import FastAPI

from app.config import Settings
from app.llm import make_chat_model, make_extract_model
from app.routers import chat
from app.schemas import AfterSaleExtraction
from app.sessions import session_store


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
    return app


app = create_app()
