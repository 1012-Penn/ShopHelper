"""五个业务工具:三个 mock(不接真实 API、不建表)+ query_faq(LIKE 查表)+ create_ticket(写表)。"""
import json
import random
from datetime import datetime
from typing import Literal

from langchain.tools import tool
from pydantic import BaseModel, Field
from sqlalchemy import select

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


def build_tools(session_factory) -> list:
    """session_factory 由调用方注入;返回 tool 对象列表(供 bind_tools 与 registry)。"""

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
        """按关键词检索常见问题库(FAQ)。用户问退货政策、发货时间等常见问题时使用。"""
        with session_factory() as session:
            rows = session.scalars(
                select(Faq).where(Faq.question.like(f"%{keyword}%")).limit(3)
            ).all()
            return json.dumps(
                {"items": [{"question": r.question, "answer": r.answer, "category": r.category} for r in rows]},
                ensure_ascii=False,
            )

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
