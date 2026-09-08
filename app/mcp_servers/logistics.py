"""ch08 物流 MCP Server:独立进程,FastMCP + Streamable HTTP;mock 随机轨迹,不接真实系统、不建表。

启动:.venv/bin/python -m uvicorn app.mcp_servers.logistics:app --port 8001
"""
import random
from datetime import datetime

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("logistics", host="127.0.0.1", port=8001)

_CITIES = ["北京", "上海", "广州", "杭州", "成都", "武汉", "西安"]
_CARRIERS = ["顺丰速运", "中通快递", "京东物流", "圆通速递"]


@mcp.tool()
def logistics_tracker(order_id: str) -> str:
    """按订单号查询物流轨迹:承运商与节点列表。用户问包裹到哪了、物流进度时使用。"""
    node_count = random.randint(2, 4)
    now = datetime.now().strftime("%m-%d %H:%M")
    traces = [
        f"{now} 包裹已到达{random.choice(_CITIES)}转运中心" for _ in range(node_count)
    ]
    traces.append(f"{now} 派送中,快递员 {random.randint(100, 999)} 号")
    return __import__("json").dumps({
        "order_id": order_id,
        "carrier": random.choice(_CARRIERS),
        "status": random.choice(["运输中", "派送中", "已签收"]),
        "traces": traces,
    }, ensure_ascii=False)


app = mcp.streamable_http_app()
