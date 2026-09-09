import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.config import Settings
from app.context import derive_budgets
from app.db import make_engine, make_session_factory
from app.guard import LowConfidencePool
from app.llm import make_chat_model, make_extract_model
from app.routers import (chat, conversations, evaluation, extract, feedback, orders,
                          review_queue, sessions, tickets, usage)
from app.schemas import AfterSaleExtraction
from app.store import ConversationStore, UsageStore
from app.flywheel import (DedupDecisionOutput, FlywheelService, NormalizedQuestion,
                          NormalizedQuestionOutput)
from app.evaluation import EvalRunStore
from app.summarizer import SummaryService

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
    app.state.usage_store = UsageStore(session_factory)
    app.state.eval_store = EvalRunStore(session_factory)
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
    app.state.summary_service = SummaryService(app.state.store, make_extract_model(settings))
    # ch08 工具系统:注册中心(内置+插件 autoload)→ 统一执行引擎 → MCP 动态发现
    from app.graph.builder import build_graph, make_retrieval_chain
    from app.tools.audit import ToolAuditStore
    from app.tools.base import ToolContext, ToolRegistryV2
    from app.tools.builtin import register_builtin, register_debug
    from app.tools.engine import ToolEngine
    from app.tools.mcp_client import McpService
    from app.tools.plugins import load_plugins

    service, kb = make_retrieval_chain(session_factory, settings)
    normalize_model = make_extract_model(settings).with_structured_output(
        NormalizedQuestionOutput, method="function_calling")
    dedup_model = make_extract_model(settings).with_structured_output(
        DedupDecisionOutput, method="function_calling")

    def normalize_question(raw: str) -> NormalizedQuestion:
        result = normalize_model.invoke(
            "把下面客服用户原话改写成标准 FAQ 问题,并给出一条保守的示例答案。\n" + raw)
        return NormalizedQuestion(result.normalized_question, result.ai_suggested_answer)

    def dedup_question(normalized: str, candidates) -> int | None:
        if not candidates:
            return None
        prompt = ("判断标准问题是否与待审队列候选中的某一条是同一个意思。只返回结构化结果。\n"
                  f"标准问题:{normalized}\n候选:\n" +
                  "\n".join(f"id={row.id}: {row.normalized_question}" for row in candidates))
        result = dedup_model.invoke(prompt)
        valid_ids = {row.id for row in candidates}
        return result.matched_review_id if result.same_meaning and result.matched_review_id in valid_ids else None

    app.state.flywheel = FlywheelService(
        session_factory, normalizer=normalize_question, deduper=dedup_question,
        kb=kb, vectors=service._vectors, embedder=service._embedder)
    # 生产图统一使用飞轮 facade;测试可继续注入轻量 LowConfidencePool。
    app.state.pool = app.state.flywheel
    registry = ToolRegistryV2()
    tool_ctx = ToolContext(session_factory=session_factory, service=service, kb=kb,
                           top_k=settings.rerank_top_k)
    register_builtin(registry, tool_ctx)
    load_plugins(registry, tool_ctx)
    if settings.tools_debug:
        register_debug(registry, settings)
    app.state.registry = registry
    tool_audit_store = ToolAuditStore(session_factory)
    app.state.tool_audit_store = tool_audit_store
    app.state.engine = ToolEngine(registry, tool_audit_store,
                                  timeout_seconds=settings.tool_timeout_seconds,
                                  retry_attempts=settings.tool_retry_attempts)
    app.state.mcp_service = McpService(registry, {
        "logistics": settings.mcp_logistics_url,
        "aftersales": settings.mcp_aftersales_url,
    })
    app.state.chat_model = make_chat_model(settings)
    app.state.extract_model = make_extract_model(settings).with_structured_output(
        AfterSaleExtraction, method="function_calling"
    )
    # ch05:LangGraph 图骨架(检索链与工具检索生产分支同构,各自持有真实现)
    compiled_graph = build_graph(
        app.state.chat_model, app.state.engine, settings,
        app.state.store, app.state.pool, service, kb,
    )
    from app.observability import ObservabilityService
    app.state.observability = ObservabilityService(settings, app.state.usage_store)
    app.state.graph = app.state.observability.bind_graph(compiled_graph)
    app.include_router(chat.router)
    app.include_router(feedback.router)
    app.include_router(review_queue.router)
    app.include_router(evaluation.router)
    app.include_router(usage.router)
    app.include_router(orders.router)
    app.include_router(sessions.router)
    app.include_router(extract.router)
    app.include_router(tickets.router)
    app.include_router(conversations.router)

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/admin/review")
    async def review_admin() -> FileResponse:
        return FileResponse(STATIC_DIR / "review-queue.html")

    @app.get("/admin/evals")
    async def eval_admin() -> FileResponse:
        return FileResponse(STATIC_DIR / "eval-runs.html")

    return app


app = create_app()
