# app/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """.env 优先级低于显式环境变量;_env_file=None 可在测试中隔离真实 .env。"""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str
    openai_base_url: str = "https://api.deepseek.com/v1"
    openai_model: str = "deepseek-chat"
    history_token_budget: int = 3000
    chat_temperature: float = 0.7
    extract_temperature: float = 0
