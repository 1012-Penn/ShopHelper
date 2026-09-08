# app/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """.env 优先级低于显式环境变量;_env_file=None 可在测试中隔离真实 .env。"""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str
    openai_base_url: str = "https://api.deepseek.com/v1"
    openai_model: str = "deepseek-chat"
    database_url: str = "mysql+pymysql://shophelper:shophelper@127.0.0.1:3306/shophelper?charset=utf8mb4"
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
    rerank_top_k: int = 10            # 精排后进入 prompt 的条数(ch07 由 retrieval_final_top_k 改名,env RERANK_TOP_K)
    query_rewrite_enabled: bool = True

    # ch05:ReAct 主力 Agent 停止条件
    max_agent_steps: int = 6        # 单轮最多工具交互步数,超限强制收敛(ch07 由 agent_max_steps 改名)
    agent_token_budget: int = 3000  # 单轮 Agent 侧输出 token 预算(estimate_tokens 口径)

    # ch06:意图识别(默认直接主模型;escalation 开启后小模型先判、置信度低于阈值大模型重判一次)
    intent_escalation_enabled: bool = False
    intent_confidence_floor: float = 0.6
    intent_small_model: str = ""  # 缺省同 openai_model

    # ch07:上下文预算从模型窗口倒推(公式与验收数字见 spec「预算推导」;常数为固定开销组件的估算)
    model_context_window: int = 65536        # MODEL_CONTEXT_WINDOW
    max_output_tokens: int = 2000            # 输出预留
    max_user_input_tokens: int = 2000        # 单条输入上限(兼消息长度闸)
    tool_result_max_tokens: int = 2000       # 单条工具结果截断阈值
    ctx_keep_rounds: int = 30                # 想留住的轮数
    ctx_per_round_tokens: int = 250          # 每轮稳态占用估算
    ctx_prompt_overhead_tokens: int = 1800   # system 渲染 + 工具定义估算
    ctx_evidence_item_tokens: int = 250      # 单条检索证据占用估算
    ctx_summary_allowance_tokens: int = 200  # 注入梗概预留
    ctx_safety_margin_tokens: int = 1500     # 安全余量
    ctx_layer2_assistant_head_chars: int = 60  # 层 2 客服答复只留开头几十字
