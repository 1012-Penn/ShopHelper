"""祛魅热身:最裸的 Agent 循环——它就是个带工具的 while 循环。用法:
    .venv/bin/python -m scripts.naive_agent "订单 1001 的物流到哪了"
"""
import asyncio
import json
import sys

from langchain_core.messages import HumanMessage, ToolMessage

from app.config import Settings
from app.llm import make_chat_model
from app.tools.audit import ToolAuditStore
from app.tools.builtin import build_default_registry
from app.tools.engine import ToolEngine


async def naive_agent_loop(model, tools, question: str, max_steps: int = 10) -> str:
    """核心就这十几行:LLM 返回 tool_calls 就执行并喂回去,返回纯文本就收敛。"""
    messages = [HumanMessage(content=question)]
    registry = build_default_registry(None)
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.models import Base

    sqlite_engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(sqlite_engine)
    engine = ToolEngine(registry, ToolAuditStore(sessionmaker(bind=sqlite_engine)))
    for _ in range(max_steps):
        resp = await model.ainvoke(messages)
        if not resp.tool_calls:
            return resp.content
        messages.append(resp)
        for tc in resp.tool_calls:
            result = await engine.execute(tc["name"], json.dumps(tc["args"], ensure_ascii=False))
            messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
    return "达到步数上限,未收敛。"


async def main() -> None:
    from app.tools.definitions import build_tools

    settings = Settings()
    tools = build_tools(None)
    model = make_chat_model(settings).bind_tools(tools)
    question = " ".join(sys.argv[1:]) or "订单 1001 的物流到哪了"
    print(f"[user] {question}")
    print(f"[agent] {await naive_agent_loop(model, tools, question)}")


if __name__ == "__main__":
    asyncio.run(main())
