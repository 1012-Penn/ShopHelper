"""ch08 过渡兼容 shim:build_tools 委托 builtin.register_builtin,旧调用方(main/conftest)不断供。

检索依赖构造逻辑自旧 build_tools 原样搬入;Task 5 接线改造完成后本文件删除。
"""
from app.tools.base import ToolContext, ToolRegistryV2
from app.tools.builtin import register_builtin

TOOL_LABELS = {
    "query_order": "订单查询",
    "query_product": "商品查询",
    "query_logistics": "物流查询",
    "query_faq": "FAQ 检索",
    "create_ticket": "创建工单",
}


def build_tools(session_factory, embedder=None, vectors=None, top_k=None,
                rewriter=None, reranker=None) -> list:
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

    kb = KnowledgeBaseStore(session_factory)
    service = RetrievalService(
        embedder, vectors, kb, rewriter=rewriter, reranker=reranker,
        candidates=settings.hybrid_candidates if production else 50,
        final_top_k=top_k,
        rerank_score_floor=settings.rerank_score_floor if production else 0.30,
    )
    registry = ToolRegistryV2()
    register_builtin(registry, ToolContext(
        session_factory=session_factory, service=service, kb=kb, top_k=top_k))
    return [r.tool for r in registry.records()]
