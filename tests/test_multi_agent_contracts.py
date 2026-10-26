import json

import pytest
from httpx import ASGITransport, AsyncClient
from langchain.tools import tool
from langchain_core.messages import AIMessage, ToolMessage

from app.multi_agent.api import create_multi_agent_app
from app.multi_agent.models import AgentPlan, AgentTask, MultiAgentLimits
from app.multi_agent.runtime import MultiAgentRunner, build_multi_agent_graph
from app.tools.base import ToolRecord, ToolRegistryV2
from app.tools.engine import ToolEngine


class StaticPlanner:
    def __init__(self, plan):
        self.plan = plan

    def with_structured_output(self, _schema, **_kwargs):
        return self

    async def ainvoke(self, _messages, **_kwargs):
        return self.plan


class RetrievalModel:
    def bind_tools(self, _tools, **_kwargs):
        return self

    async def ainvoke(self, messages, **_kwargs):
        if isinstance(messages[-1], ToolMessage):
            return AIMessage(content="订单事实已获取")
        return AIMessage(content="", tool_calls=[{
            "name": "query_order", "args": {"order_id": "1001"}, "id": "read-1",
        }])


class WriteActionModel:
    def bind_tools(self, tools, **_kwargs):
        self.tools = [tool.name for tool in tools]
        return self

    async def ainvoke(self, messages, **_kwargs):
        assert self.tools == ["create_ticket"]
        if isinstance(messages[-1], ToolMessage):
            return AIMessage(content="工单预览已生成，请确认；当前没有创建工单。")
        return AIMessage(content="", tool_calls=[{
            "name": "create_ticket",
            "args": {
                "conversation_id": 1,
                "description": "订单1001耳机售后处理",
                "ticket_type": "售后",
            },
            "id": "write-1",
        }])


class FinalModel:
    async def ainvoke(self, _messages, **_kwargs):
        return AIMessage(content="订单1001已签收；工单预览已生成，等待确认。")


class FakeAudit:
    def __init__(self):
        self.rows = []

    def log(self, **kwargs):
        self.rows.append(kwargs)


class FakeWriteSink:
    def __init__(self):
        self.by_key = {}
        self.calls = []

    def __call__(self, conversation_id, description, ticket_type, confirm_key=None):
        self.calls.append(confirm_key)
        if confirm_key and confirm_key in self.by_key:
            return self.by_key[confirm_key]
        row = {"ticket_no": f"T{len(self.by_key) + 1}", "conversation_id": conversation_id,
               "description": description, "ticket_type": ticket_type,
               "confirm_key": confirm_key}
        if confirm_key:
            self.by_key[confirm_key] = row
        return row


def _registry(sink):
    @tool
    def query_order(order_id: str) -> str:
        """Read a simulated order."""
        rows = {
            "1001": {"order_id": "1001", "product": "无线耳机", "status": "已签收"},
        }
        return json.dumps(rows.get(order_id, {"error": "not_found"}), ensure_ascii=False)

    @tool
    def query_faq(keyword: str) -> str:
        """Read simulated policy evidence."""
        return json.dumps({"items": [{"n": 1, "chunk_id": 15001,
                                      "answer": "签收后15天内可无理由退货。"}],
                           "keyword": keyword}, ensure_ascii=False)

    @tool
    def create_ticket(conversation_id: int, description: str, ticket_type: str,
                      confirm_key: str | None = None) -> str:
        """Write a ticket in the isolated fake sink."""
        return json.dumps(sink(conversation_id, description, ticket_type, confirm_key),
                          ensure_ascii=False)

    registry = ToolRegistryV2()
    registry.register(ToolRecord(query_order, source="builtin", access="read"))
    registry.register(ToolRecord(query_faq, source="builtin", access="read"))
    registry.register(ToolRecord(create_ticket, source="builtin", access="write"))
    return registry


@pytest.mark.asyncio
async def test_fake_write_is_intercepted_and_confirmation_is_idempotent():
    plan = AgentPlan(disposition="execute", tasks=[
        AgentTask(task_id="r1", agent="retrieval", objective="查询订单1001状态"),
        AgentTask(task_id="a1", agent="action", objective="为订单1001生成售后工单草稿",
                  depends_on=["r1"]),
    ])
    sink = FakeWriteSink()
    audit = FakeAudit()
    engine = ToolEngine(_registry(sink), audit, timeout_seconds=1, retry_attempts=0)
    runner = MultiAgentRunner(
        planner_model=StaticPlanner(plan),
        retrieval_model=RetrievalModel(),
        action_model=WriteActionModel(),
        supervisor_model=FinalModel(),
        engine=engine,
        limits=MultiAgentLimits(max_seconds=3, child_timeout_seconds=1),
    )
    result = await build_multi_agent_graph(runner).ainvoke({
        "run_id": "write-run", "task_id": "write-task",
        "user_message": "请为订单1001的耳机创建售后工单",
        "conversation_history": [],
    })

    assert len(result["write_previews"]) == 1
    assert result["write_previews"][0]["type"] == "ticket_preview"
    assert result["write_previews"][0]["conversation_id"] == 1
    assert result["write_previews"][0]["ticket_type"] == "售后"
    assert result["write_previews"][0]["description"] == "订单1001耳机售后处理"
    assert sink.calls == []
    assert "权限拒绝" in [row["status"] for row in audit.rows]

    preview = result["write_previews"][0]
    args = dict(preview["arguments"])
    args["confirm_key"] = preview["confirm_key"]
    encoded = json.dumps(args, ensure_ascii=False)
    first = json.loads(await engine.execute_confirmed("create_ticket", encoded,
                                                       conversation_id=1))
    second = json.loads(await engine.execute_confirmed("create_ticket", encoded,
                                                        conversation_id=1))

    assert first == second
    assert sink.calls == [preview["confirm_key"], preview["confirm_key"]]
    assert len(sink.by_key) == 1


@pytest.mark.asyncio
async def test_optional_router_returns_graph_result_and_rejects_empty_message():
    plan = AgentPlan(disposition="clarify", clarification="请提供订单号后再判断。")
    runner = MultiAgentRunner(
        planner_model=StaticPlanner(plan),
        retrieval_model=RetrievalModel(),
        action_model=WriteActionModel(),
        supervisor_model=FinalModel(),
        engine=ToolEngine(_registry(FakeWriteSink()), FakeAudit()),
        limits=MultiAgentLimits(max_seconds=3, child_timeout_seconds=1),
    )
    app = create_multi_agent_app(runner)

    async with AsyncClient(transport=ASGITransport(app=app),
                           base_url="http://multi-agent.test") as client:
        response = await client.post("/api/multi-agent/chat",
                                     json={"message": "这个能退吗？"})
        invalid = await client.post("/api/multi-agent/chat", json={"message": " "})

    assert response.status_code == 200
    assert response.json()["final_reply"] == "请提供订单号后再判断。"
    assert response.json()["status"] == "success"
    assert "gold" not in response.text
    assert invalid.status_code == 422
