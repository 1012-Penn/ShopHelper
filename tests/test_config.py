# tests/test_config.py
from langchain_openai import ChatOpenAI

from app.config import Settings
from app.llm import make_chat_model


def test_settings_defaults():
    s = Settings(openai_api_key="sk-test", _env_file=None)
    assert s.openai_base_url == "https://api.deepseek.com/v1"
    assert s.openai_model == "deepseek-chat"
    assert s.history_token_budget == 3000
    assert s.chat_temperature == 0.7
    assert s.extract_temperature == 0


def test_ch03_defaults():
    s = Settings(openai_api_key="sk-test", _env_file=None)
    assert s.embedding_api_base == "https://api.siliconflow.cn/v1"
    assert s.embedding_model == "BAAI/bge-m3"
    assert s.embedding_dim == 1024
    assert s.embedding_batch_size == 16
    assert s.milvus_db_path == "./data/milvus_knowledge.db"
    assert s.retrieval_top_k == 3
    assert s.chunk_max_chars == 500
    assert s.chunk_overlap_chars == 80
    assert s.dedup_threshold == 0.88


def test_settings_reads_env(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "my-model")
    monkeypatch.setenv("HISTORY_TOKEN_BUDGET", "100")
    s = Settings(openai_api_key="sk-test", _env_file=None)
    assert s.openai_model == "my-model"
    assert s.history_token_budget == 100


def test_make_chat_model_uses_settings():
    s = Settings(
        openai_api_key="sk-test",
        openai_model="m1",
        openai_base_url="https://example.com/v1",
        chat_temperature=0.3,
        _env_file=None,
    )
    m = make_chat_model(s)
    assert isinstance(m, ChatOpenAI)
    assert m.model_name == "m1"
    assert m.temperature == 0.3
    assert m.openai_api_base == "https://example.com/v1"
