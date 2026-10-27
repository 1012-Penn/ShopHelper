"""ch08 内置工具注册:query_order / query_product / query_faq / create_ticket。

query_logistics 自本章起由物流 MCP Server 接管,不再内置(Spec「MCP 接入」)。
写入类:create_ticket(access="write"),执行引擎把门,直调一律拦截等用户确认。
检索依赖(service/kb)由接线层组装进 ToolContext 传入,这里不碰应用全局。
"""
import inspect
import json
import logging
import time
from typing import Literal

from langchain.tools import tool
from pydantic import BaseModel, Field

from app.tools.base import ToolContext, ToolRecord, ToolRegistryV2

logger = logging.getLogger(__name__)


class CreateTicketInput(BaseModel):
    conversation_id: int = Field(description="当前会话 id")
    description: str = Field(description="问题描述")
    ticket_type: Literal["售后", "投诉", "咨询"] = Field(description="工单类型")
    confirm_key: str | None = Field(
        default=None, description="一次用户确认的幂等键;同键重复提交返回同一工单(ADR-3)")


def register_builtin(registry: ToolRegistryV2, ctx: ToolContext) -> None:
    if ctx.service is None:
        raise ValueError("ToolContext.service 未注入(query_faq 依赖检索服务)")
    service = ctx.service
    kb = ctx.kb

    @tool
    def query_order(order_id: str) -> str:
        """按订单号查询订单的基础信息:商品、金额、状态;不包含物流轨迹。
        裸数字如 1001 表示 legacy_demo 演示订单;完整 ORD-1001 表示 corpus 订单,两者不是同一主体。
        用户问订单买了什么、多少钱、什么状态时使用;问包裹/物流到哪了时不要用本工具。"""
        from app.orders import get_order

        return json.dumps(get_order(order_id), ensure_ascii=False)

    @tool
    def query_product(keyword: str) -> str:
        """按关键词查询商品:名称、价格、库存。用户咨询商品信息时使用。"""
        from app.products import get_product

        return json.dumps(get_product(keyword), ensure_ascii=False)

    @tool
    def query_faq(keyword: str, category: str | None = None,
                  order_id: str | None = None,
                  policy_version: str | None = None,
                  as_of_date: str | None = None) -> str:
        """语义检索常见问题知识库并精排。用户问退货政策、发货时间、邮费运费、付款、发票、会员等常见问题,或问具体商品型号(如 SH-E300)的参数、价格时使用;能从用户话里明确判断品类时传 category,判断不了不要传。已知具体订单时传 order_id;只有明确知道适用政策版本或假想签收日期时才传 policy_version/as_of_date。"""
        try:
            from datetime import date

            from app.business_facts import policy_version_for_date
            from app.orders import get_order
            from app.retrieval import EvidenceScope

            subject = None
            derived_version = None
            derived_date = None
            order_error = None
            if order_id:
                order = get_order(order_id)
                subject = (order.get("subject") or {}).get("canonical_id")
                order_error = order.get("error")
                if order.get("signed_at"):
                    derived_date = str(order["signed_at"])[:10]
                    derived_version = policy_version_for_date(derived_date)

            selected_version = derived_version or (policy_version if not order_id else None)
            parsed_date = None
            if derived_date:
                parsed_date = date.fromisoformat(str(derived_date)[:10])
            elif not order_id and as_of_date:
                try:
                    parsed_date = date.fromisoformat(str(as_of_date)[:10])
                except ValueError:
                    parsed_date = None
            scope = EvidenceScope(
                subject=subject,
                policy_version=selected_version,
                reference_date=parsed_date,
            )
            scope_arg = scope if scope.has_constraints() else None
            parameters = inspect.signature(service.retrieve).parameters.values()
            supports_scope = any(
                item.name == "scope" or item.kind == inspect.Parameter.VAR_KEYWORD
                for item in parameters
            )
            if supports_scope:
                result = service.retrieve(
                    keyword,
                    strategy="hybrid_rerank",
                    category=category,
                    scope=scope_arg,
                )
            else:
                # Old injected services keep their three-argument contract.
                result = service.retrieve(
                    keyword,
                    strategy="hybrid_rerank",
                    category=category,
                )
            # 低置信候选可以被问题池快照保留,但不能作为 Agent 工具证据继续向上游传播。
            visible_items = [] if result.low_confidence else result.items
            rows = kb.get_chunks([r.chunk_id for r in visible_items])
            evidence = [
                {
                    "n": i + 1,  # n = 精排名次,[1] 恒为最强证据
                    "chunk_id": r.chunk_id,
                    "question": row.questions.splitlines()[0],
                    "answer": row.answer,
                    "category": row.category,
                    "section_path": row.section_path or "",
                }
                for i, (r, row) in enumerate(zip(visible_items, rows))
            ]
            snapshot = result.snapshot() if hasattr(result, "snapshot") else []
            rows_by_id = {row.id: row for row in kb.get_chunks([r.chunk_id for r in result.items])}
            for item in snapshot:
                row = rows_by_id.get(item["chunk_id"])
                if row is not None:
                    item.update({"question": row.questions.splitlines()[0], "answer": row.answer,
                                 "category": row.category, "section_path": row.section_path or "",
                                 "text": f"{row.questions}\n{row.answer}"})
            from app.retrieval import lost_in_middle_order

            selection_facets = list(getattr(result, "selection_facets", []) or [])
            selection_latency_ms = float(
                getattr(result, "selection_latency_ms", 0.0) or 0.0
            )
            rerank_pool_size = len(getattr(result, "rerank_candidates", []) or [])
            return json.dumps({
                "items": lost_in_middle_order(evidence),
                "retrieved_chunks": snapshot,
                "low_confidence": result.low_confidence,
                "reason": result.reason,
                "filter_fallback": result.filter_fallback,
                "degraded": result.degraded,
                "selection_facets": selection_facets,
                "selection_latency_ms": selection_latency_ms,
                "rerank_pool_size": rerank_pool_size,
                "scope_filter": {
                    "status": result.scope_status,
                    "subject": subject,
                    "policy_version": selected_version or "unknown",
                    "reference_date": (
                        parsed_date.isoformat() if parsed_date else "unknown"
                    ),
                    "order_error": order_error,
                    "excluded_count": len(result.scope_excluded),
                    "excluded": result.scope_excluded[:20],
                    "unknown_chunk_ids": result.scope_unknown_ids,
                    "matched_chunk_ids": result.scope_matched_ids,
                },
            }, ensure_ascii=False)
        except Exception:
            logger.warning("query_faq 检索失败", exc_info=True)
            return json.dumps({
                "items": [], "low_confidence": True,
                "reason": "检索服务暂不可用,请稍后重试", "filter_fallback": False,
            }, ensure_ascii=False)

    @tool(args_schema=CreateTicketInput)
    def create_ticket(conversation_id: int, description: str, ticket_type: str,
                      confirm_key: str | None = None) -> str:
        """创建人工工单转人工处理。问题超出自动客服能力、用户强烈要求人工时使用。"""
        from app.tickets import (
            CONFIRM_KEY_CONFLICT,
            ConfirmKeyConflict,
            create_ticket_idempotent,
        )

        try:
            payload = create_ticket_idempotent(
                ctx.session_factory, conversation_id=conversation_id,
                description=description, ticket_type=ticket_type, confirm_key=confirm_key)
        except ConfirmKeyConflict:
            # 引擎契约:不向上抛异常;路由按 error 码转 409,绝不把旧工单当成功返回
            return json.dumps({
                "error": CONFIRM_KEY_CONFLICT,
                "detail": "同一确认键的会话/描述/工单类型与已建工单不一致;未创建新工单",
            }, ensure_ascii=False)
        return json.dumps(payload, ensure_ascii=False)

    registry.register(ToolRecord(query_order, source="builtin", access="read", label="订单查询"))
    registry.register(ToolRecord(query_product, source="builtin", access="read", label="商品查询"))
    registry.register(ToolRecord(query_faq, source="builtin", access="read", label="FAQ 检索"))
    registry.register(ToolRecord(create_ticket, source="builtin", access="write", label="创建工单"))


def register_debug(registry: ToolRegistryV2, settings) -> None:
    """验收 6 用慢工具:TOOLS_DEBUG=true 时注册,人为制造超时观察重试/审计。"""

    @tool
    def debug_slow_query(seconds: int) -> str:
        """【调试】模拟慢查询,sleep 指定秒数后返回。用于演示工具超时与重试。"""
        time.sleep(seconds)
        return json.dumps({"ok": True, "slept": seconds}, ensure_ascii=False)

    @tool
    def debug_slow_write(seconds: int) -> str:
        """【调试】模拟慢写入,sleep 指定秒数后返回。用于演示写操作超时不自动重试。"""
        time.sleep(seconds)
        return json.dumps({"ok": True, "slept": seconds, "write": True}, ensure_ascii=False)

    registry.register(ToolRecord(debug_slow_query, source="builtin", access="read", label="慢查询演示"))
    registry.register(ToolRecord(debug_slow_write, source="builtin", access="write", label="慢写入演示"))


def build_default_registry(session_factory, embedder=None, vectors=None, top_k=None,
                           rewriter=None, reranker=None) -> ToolRegistryV2:
    """便捷装配(兼容旧 build_tools 语义):替身/生产检索链按需组装 + 内置注册,返回注册表。"""
    from app.kb import KnowledgeBaseStore
    from app.rerank import make_reranker
    from app.retrieval import RetrievalService
    from app.rewrite import make_rewriter

    production = embedder is None or vectors is None
    settings = None
    if production:
        from app.config import Settings
        from app.embedding import make_embedder
        from app.vector_store import KnowledgeVectorStore

        settings = Settings()
        embedder = embedder or make_embedder(settings)
        vectors = vectors or KnowledgeVectorStore(settings.milvus_db_path, dim=settings.embedding_dim)
    top_k = top_k or (settings.rerank_top_k if production else 3)
    if production:
        if settings.query_rewrite_enabled:
            rewriter = rewriter or make_rewriter(settings)
        reranker = reranker if reranker is not None else make_reranker(settings)

    evidence_judge = None
    if production and settings.evidence_entailment_enabled:
        from app.evidence import make_evidence_judge
        evidence_judge = make_evidence_judge(settings)

    kb = KnowledgeBaseStore(session_factory)
    service = RetrievalService(
        embedder, vectors, kb, rewriter=rewriter, reranker=reranker,
        candidates=settings.hybrid_candidates if production else 50,
        final_top_k=top_k,
        rerank_score_floor=settings.rerank_score_floor if production else 0.30,
        evidence_judge=evidence_judge,
        evidence_top_n=settings.evidence_entailment_top_n if production else 3,
        evidence_slate_enabled=(
            settings.retrieval_evidence_slate_enabled if production else False
        ),
        rerank_pool_k=settings.retrieval_rerank_pool_k if production else None,
    )
    registry = ToolRegistryV2()
    register_builtin(registry, ToolContext(
        session_factory=session_factory, service=service, kb=kb, top_k=top_k))
    return registry
