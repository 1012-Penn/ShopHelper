"""ch08 内置工具注册:query_order / query_product / query_faq / create_ticket。

query_logistics 自本章起由物流 MCP Server 接管,不再内置(Spec「MCP 接入」)。
写入类:create_ticket(access="write"),执行引擎把门,直调一律拦截等用户确认。
检索依赖(service/kb)由接线层组装进 ToolContext 传入,这里不碰应用全局。
"""
import json
import logging
import random
import time
from datetime import datetime
from typing import Literal

from langchain.tools import tool
from pydantic import BaseModel, Field

from app.models import Ticket
from app.tools.base import ToolContext, ToolRecord, ToolRegistryV2

logger = logging.getLogger(__name__)


class CreateTicketInput(BaseModel):
    conversation_id: int = Field(description="当前会话 id")
    description: str = Field(description="问题描述")
    ticket_type: Literal["售后", "投诉", "咨询"] = Field(description="工单类型")


def register_builtin(registry: ToolRegistryV2, ctx: ToolContext) -> None:
    if ctx.service is None:
        raise ValueError("ToolContext.service 未注入(query_faq 依赖检索服务)")
    service = ctx.service
    kb = ctx.kb

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
            from app.retrieval import lost_in_middle_order

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
        with ctx.session_factory() as session:
            session.add(Ticket(
                ticket_no=ticket_no,
                conversation_id=conversation_id,
                description=description,
                ticket_type=ticket_type,
            ))
            session.commit()
        return json.dumps({"ticket_no": ticket_no, "status": "待处理"}, ensure_ascii=False)

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
