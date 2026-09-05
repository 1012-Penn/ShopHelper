"""五个业务工具:三个 mock(不接真实 API、不建表)+ query_faq(向量语义检索)+ create_ticket(写表)。"""
import json
import random
from datetime import datetime
from typing import Literal

from langchain.tools import tool
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.kb import KnowledgeBaseStore
from app.models import Faq, Ticket


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

# 相似度下限:低于此分的命中直接丢弃,不进 items(与 prompt 的「对不上就如实说明」双保险)
RETRIEVAL_SCORE_FLOOR = 0.5


def build_tools(session_factory, embedder=None, vectors=None, top_k: int | None = None) -> list:
    """session_factory 由调用方注入;embedder / vectors 缺省时按 Settings 构造真实现(测试注入替身)。

    query_faq 自 ch03 起走向量语义检索:问题 embed → Milvus Top-K → MySQL 回取原文。
    工具入参出参契约与 ch02 保持不变;top_k 缺省真实现走配置、替身默认 3。
    """
    if embedder is None or vectors is None:
        from app.config import Settings
        from app.embedding import make_embedder
        from app.vector_store import KnowledgeVectorStore

        settings = Settings()
        embedder = embedder or make_embedder(settings)
        vectors = vectors or KnowledgeVectorStore(settings.milvus_db_path, dim=settings.embedding_dim)
        top_k = top_k or settings.retrieval_top_k
    top_k = top_k or 3  # 注入替身但未传 top_k 的兜底:None 传进 search 会静默挂掉
    kb = KnowledgeBaseStore(session_factory)

    @tool
    def query_order(order_id: str) -> str:
        """按订单号查询订单信息:商品、金额、状态。用户问订单相关问题时使用。"""
        return json.dumps({
            "order_id": order_id,
            "product": random.choice(["无线耳机", "机械键盘", "硅胶手机壳", "智能手环"]),
            "amount": round(random.uniform(19.9, 999.0), 2),
            "status": random.choice(["待发货", "已发货", "已签收"]),
            "created_at": datetime.now().strftime("%Y-%m-%d"),
        }, ensure_ascii=False)

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
    def query_faq(keyword: str) -> str:
        """按语义检索常见问题知识库。用户问退货政策、发货时间、邮费运费、付款、发票、会员等常见问题时使用。"""
        try:
            qvec = embedder.embed([keyword])[0]
            hits = [(cid, score) for cid, score in vectors.search(qvec, top_k=top_k)
                    if score >= RETRIEVAL_SCORE_FLOOR]
            items = []
            if hits:
                rows = kb.get_chunks([h[0] for h in hits])
                items = [
                    {
                        "question": r.questions.splitlines()[0],
                        "answer": r.answer,
                        "category": r.category,
                    }
                    for r in rows
                ]
        except Exception:
            items = []  # 契约:任何异常收敛为空结果,不抛错
        return json.dumps({"items": items}, ensure_ascii=False)

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
