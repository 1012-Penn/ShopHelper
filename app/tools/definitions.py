"""五个业务工具:三个 mock(不接真实 API、不建表)+ query_faq(混合检索精排)+ create_ticket(写表)。"""
import json
import logging
import random
from datetime import datetime
from typing import Literal

from langchain.tools import tool
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.kb import KnowledgeBaseStore
from app.models import Faq, Ticket
from app.rerank import make_reranker
from app.retrieval import RetrievalService, lost_in_middle_order
from app.rewrite import make_rewriter

logger = logging.getLogger(__name__)


class CreateTicketInput(BaseModel):
    conversation_id: int = Field(description="当前会话 id")
    description: str = Field(description="问题描述")
    ticket_type: Literal["售后", "投诉", "咨询"] = Field(description="工单类型")

TOOL_LABELS = {
    "query_order": "订单查询",
    "query_product": "商品查询",
    "query_logistics": "物流查询",
    "query_faq": "FAQ 检索",
    "create_ticket": "创建工单",
}

CITIES = ["北京", "上海", "广州", "杭州", "成都"]


def build_tools(session_factory, embedder=None, vectors=None, top_k=None,
                rewriter=None, reranker=None) -> list:
    """注入面(ch04 扩展):只有生产路径(embedder/vectors 未注入)才按 Settings 构造真实现,
    含真 rewriter/reranker;测试部分注入(embedder/vectors 给替身、rewriter/reranker 缺省)时
    保持 None → 无改写、hybrid_rerank 降级 RRF 序,测试不出网。top_k 显式传入优先。
    query_faq 自 ch04 起走 retrieval.RetrievalService(hybrid_rerank)。"""
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
        # 非生产路径(替身注入)固定 0.30:FakeReranker 的内容字覆盖率口径下
        # junk 题 ≤0.25、相关题≈1.0,该阈值可分;生产路径走 config(真模型校准值 0.03)。
        rerank_score_floor=settings.rerank_score_floor if production else 0.30,
    )

    @tool
    def query_order(order_id: str) -> str:
        """按订单号查询订单信息:商品、金额、状态。用户问订单相关问题时使用。"""
        from app.orders import get_order

        return json.dumps(get_order(order_id), ensure_ascii=False)

    @tool
    def query_product(keyword: str) -> str:
        """按关键词查询商品:名称、价格、库存。用户咨询商品信息时使用。"""
        return json.dumps({
            "keyword": keyword,
            "name": f"{keyword}精选款",
            "price": round(random.uniform(9.9, 499.0), 2),
            "stock": random.randint(0, 500),
        }, ensure_ascii=False)

    @tool
    def query_logistics(order_id: str) -> str:
        """按订单号查询物流轨迹:承运商与节点列表。用户问包裹到哪了时使用。"""
        node_count = random.randint(2, 4)
        traces = [
            f"{datetime.now().strftime('%m-%d %H:%M')} 包裹已到达{random.choice(CITIES)}转运中心"
            for _ in range(node_count)
        ]
        traces.append(f"{datetime.now().strftime('%m-%d %H:%M')} 派送中,快递员 {random.randint(100, 999)} 号")
        return json.dumps({
            "order_id": order_id,
            "carrier": random.choice(["顺丰速运", "中通快递", "京东物流"]),
            "traces": traces,
        }, ensure_ascii=False)

    @tool
    def query_faq(keyword: str, category: str | None = None) -> str:
        """语义检索常见问题知识库并精排。用户问退货政策、发货时间、邮费运费、付款、发票、会员等常见问题,或问具体商品型号(如 SH-E300)的参数、价格时使用;能从用户话里明确判断品类时传 category,判断不了不要传。"""
        try:
            result = service.retrieve(keyword, strategy="hybrid_rerank", category=category)
            rows = kb.get_chunks([r.chunk_id for r in result.items])
            evidence = [
                {
                    "n": i + 1,  # n = 精排名次,[1] 恒为最强证据
                    "chunk_id": r.chunk_id,
                    "question": row.questions.splitlines()[0],
                    "answer": row.answer,
                    "category": row.category,
                    "section_path": row.section_path or "",
                }
                for i, (r, row) in enumerate(zip(result.items, rows))
            ]
            return json.dumps({
                "items": lost_in_middle_order(evidence),
                "low_confidence": result.low_confidence,
                "reason": result.reason,
                "filter_fallback": result.filter_fallback,
                "degraded": result.degraded,
            }, ensure_ascii=False)
        except Exception:
            logger.warning("query_faq 检索失败", exc_info=True)
            return json.dumps({
                "items": [], "low_confidence": True,
                "reason": "检索服务暂不可用,请稍后重试", "filter_fallback": False,
            }, ensure_ascii=False)

    @tool(args_schema=CreateTicketInput)
    def create_ticket(conversation_id: int, description: str, ticket_type: str) -> str:
        """创建人工工单转人工处理。问题超出自动客服能力、用户强烈要求人工时使用。"""
        ticket_no = f"T{datetime.now().strftime('%Y%m%d')}{random.randint(0, 999):03d}"
        with session_factory() as session:
            session.add(Ticket(
                ticket_no=ticket_no,
                conversation_id=conversation_id,
                description=description,
                ticket_type=ticket_type,
            ))
            session.commit()
        return json.dumps({"ticket_no": ticket_no, "status": "待处理"}, ensure_ascii=False)

    return [query_order, query_product, query_logistics, query_faq, create_ticket]
