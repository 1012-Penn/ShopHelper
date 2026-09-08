"""示范插件(验收 1):新写一个工具,只做注册动作、不动核心代码,Agent 即可在对话中使用。"""
import json
from datetime import datetime

from langchain.tools import tool


@tool
def query_server_time() -> str:
    """查询客服系统的当前服务器时间。用户问现在几点、服务器时间时使用。"""
    return json.dumps({
        "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "weekday": ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][datetime.now().weekday()],
    }, ensure_ascii=False)


def register(registry, ctx) -> None:
    from app.tools.base import ToolRecord

    registry.register(ToolRecord(query_server_time, source="builtin", access="read", label="服务器时间"))
