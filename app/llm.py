# app/llm.py
from langchain_openai import ChatOpenAI

from app.config import Settings


def make_chat_model(settings: Settings) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        temperature=settings.chat_temperature,
    )


def make_extract_model(settings: Settings, model: str | None = None) -> ChatOpenAI:
    """model 覆盖参数(ch06):意图降级路给小模型传 intent_small_model 时用。"""
    return ChatOpenAI(
        model=model or settings.openai_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        temperature=settings.extract_temperature,
    )
