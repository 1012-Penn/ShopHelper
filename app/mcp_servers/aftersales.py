"""ch08 售后 MCP Server:独立进程,FastMCP + Streamable HTTP;mock 在保/退货进度,不建表。

启动:.venv/bin/python -m uvicorn app.mcp_servers.aftersales:app --port 8002
AFTERSALES_EXTRA_TOOLS=true 时额外注册 query_repair_shop(验收 3:Server 侧新加工具,客户端不重启现问现拿)。
"""
import os
import random

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("aftersales", host="127.0.0.1", port=8002)


@mcp.tool()
def query_warranty(order_id: str, product: str = "") -> str:
    """按订单号查询商品是否在保:在保状态与保修截止日。用户问能不能保修、过没过保修期时使用。"""
    in_warranty = random.random() < 0.7
    import json

    return json.dumps({
        "order_id": order_id,
        "product": product or "商品",
        "in_warranty": in_warranty,
        "warranty_until": f"{random.randint(2026, 2028)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
        "note": "以购物凭证日期为准" if in_warranty else "已过整机保修期",
    }, ensure_ascii=False)


@mcp.tool()
def query_return_progress(order_id: str) -> str:
    """按订单号查询退货/退款申请进度:当前环节与下一步。用户问退货办到哪一步时使用。"""
    import json

    stages = ["已提交申请", "商家审核中", "退货物流运输中", "验货中", "退款已发起"]
    current = random.randint(0, len(stages) - 1)
    return json.dumps({
        "order_id": order_id,
        "stage": stages[current],
        "next": stages[current + 1] if current + 1 < len(stages) else "完成",
        "eta_days": random.randint(1, 5),
    }, ensure_ascii=False)


if os.environ.get("AFTERSALES_EXTRA_TOOLS") == "true":

    @mcp.tool()
    def query_repair_shop(city: str) -> str:
        """按城市查询就近维修网点:网点名与地址。用户问哪里能送修时使用。"""
        import json

        return json.dumps({
            "city": city,
            "shops": [f"{city}授权维修{random.choice(['一', '二', '三'])}号店(高新区{random.randint(1, 99)}号)"],
        }, ensure_ascii=False)


app = mcp.streamable_http_app()
