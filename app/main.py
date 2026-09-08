import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.config import Settings
from app.context import derive_budgets
from app.db import make_engine, make_session_factory
from app.guard import LowConfidencePool
from app.llm import make_chat_model, make_extract_model
from app.routers import chat, conversations, extract, orders, sessions, tickets
from app.schemas import AfterSaleExtraction
from app.store import ConversationStore
from app.summarizer import SummaryService
from app.tools.definitions import build_tools
from app.tools.registry import ToolRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def _ensure_app_log_handler() -> None:
    """上下文可观测口径:实际发给模型的上下文原样落 logs/app.log;重复 create_app(测试)不挂重。"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    target = str((log_dir / "app.log").resolve())
    for h in logging.getLogger().handlers:
        if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == target:
            return
    fh = logging.FileHandler(log_dir / "app.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(fh)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    _ensure_app_log_handler()
    app = FastAPI(title="ShopHelper")
    app.state.settings = settings
    # 引擎惰性连接;测试注入替身时直接覆写 state 上这几个对象即可
    session_factory = make_session_factory(make_engine(settings))
    app.state.session_factory = session_factory
    app.state.store = ConversationStore(session_factory)
    app.state.pool = LowConfidencePool(session_factory)
    # ch07:上下文预算从模型窗口倒推 + 启动自检(连一轮稳态都装不下就报警)
    budgets = derive_budgets(settings)
    app.state.budgets = budgets
    if budgets.sliding < settings.ctx_per_round_tokens:
        logger.error("[ctx] 上下文预算不足:滑窗 %d 装不下稳态一轮 %d"
                     "(窗口 %d − 输出预留 %d − 单轮峰值 %d − 固定开销 %d)",
                     budgets.sliding, settings.ctx_per_round_tokens, budgets.window,
                     settings.max_output_tokens, budgets.peak, budgets.fixed)
    else:
        logger.info("[ctx] budget 窗口=%d 固定=%d 峰值=%d 滑窗=%d 历史=%d 层1=%d 层2=%d",
                    budgets.window, budgets.fixed, budgets.peak, budgets.sliding,
                    budgets.history, budgets.layer1, budgets.layer2)
    app.state.hydrated_threads = set()  # 已回灌完整历史的 thread(checkpoint 空时从 MySQL 灌一次)
    app.state.summary_service = SummaryService(app.state.store, make_extract_model(settings),
                                               settings)
    # build_tools 生产路径:内部构造真 embedder/向量库(v2)/rewriter/reranker
    app.state.registry = ToolRegistry(build_tools(
        session_factory, top_k=settings.rerank_top_k,
    ))
    app.state.chat_model = make_chat_model(settings)
    app.state.extract_model = make_extract_model(settings).with_structured_output(
        AfterSaleExtraction, method="function_calling"
    )
    # ch05:LangGraph 图骨架(检索链与 build_tools 生产分支同构,各自持有真实现)
    from app.graph.builder import build_graph, make_retrieval_chain

    service, kb = make_retrieval_chain(session_factory, settings)
    app.state.graph = build_graph(
        app.state.chat_model, app.state.registry, settings,
        app.state.store, app.state.pool, service, kb,
    )
    app.include_router(chat.router)
    app.include_router(orders.router)
    app.include_router(sessions.router)
    app.include_router(extract.router)
    app.include_router(tickets.router)
    app.include_router(conversations.router)

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
