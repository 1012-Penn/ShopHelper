# app/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """.env 优先级低于显式环境变量;_env_file=None 可在测试中隔离真实 .env。"""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str
    openai_base_url: str = "https://api.deepseek.com/v1"
    openai_model: str = "deepseek-chat"
    database_url: str = "mysql+pymysql://shophelper:shophelper@127.0.0.1:3306/shophelper?charset=utf8mb4"
    history_token_budget: int = 3000
    chat_temperature: float = 0.7
    extract_temperature: float = 0

    # ch03:RAG 知识库(BGE-M3 走硅基流动 OpenAI 兼容接口;向量库 Milvus Lite 本地文件)
    embedding_api_base: str = "https://api.siliconflow.cn/v1"
    embedding_api_key: str = ""
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    embedding_batch_size: int = 16  # 硅基流动单请求上限 32,留余量
    milvus_db_path: str = "./data/milvus_knowledge.db"
    retrieval_top_k: int = 3
    chunk_max_chars: int = 500
    chunk_overlap_chars: int = 80
    dedup_threshold: float = 0.88

    # ch04:混合检索 + 重排(rerank 与嵌入同走硅基流动;key 缺省回落 embedding_api_key)
    rerank_api_base: str = "https://api.siliconflow.cn/v1"
    rerank_api_key: str = ""
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    # 精排 Top-1 低于此判「证据置信度低」。校准依据(ch04 评估集真模型分布,2026-09-06):
    # 合法题 Top-1 ∈ [0.039, 0.99],纯噪声 junk = 0.0,词面重叠 junk ≈ 0.17(交给生成自评兜底)
    rerank_score_floor: float = 0.03
    hybrid_candidates: int = 50       # dense/BM25 各自召回数(hybrid_search limit)
    retrieval_final_top_k: int = 10   # 精排后进入 prompt 的条数
    query_rewrite_enabled: bool = True
