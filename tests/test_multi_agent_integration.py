import json

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from sqlalchemy import select

from app.models import Ticket
from app.multi_agent.models import AgentPlan, AgentTask, MultiAgentLimits
from app.multi_agent.runtime import MultiAgentRunner, build_multi_agent_runtime


class StaticPlanner:
    def with_structured_output(self, _schema, **_kwargs):
        return self

    async def ainvoke(self, _messages, **_kwargs):
        return AgentPlan(disposition="execute", tasks=[
            AgentTask(task_id="r1", agent="retrieval", objective="查询订单1001"),
            AgentTask(
                task_id="a1",
                agent="action",
                objective="生成订单1001售后工单预览",
                depends_on=["r1"],
            ),
        ])


class RetrievalModel:
    def bind_tools(self, _tools, **_kwargs):
        return self

    async def ainvoke(self, _messages, **_kwargs):
        return AIMessage(content="订单1001已签收，商品为无线耳机。")


class ActionModel:
    def bind_tools(self, _tools, **_kwargs):
        return self

    async def ainvoke(self, messages, **_kwargs):
        if isinstance(messages[-1], ToolMessage):
            return AIMessage(content="已生成工单预览，等待用户确认。")
        return AIMessage(content="", tool_calls=[{
            "name": "create_ticket",
            "args": {
                "description": "订单1001耳机售后处理",
                "ticket_type": "售后",
            },
            "id": "multi-agent-write",
        }])


class FinalModel:
    async def ainvoke(self, _messages, **_kwargs):
        return AIMessage(content="订单1001的售后工单预览已生成，请确认。")


def _runtime(engine):
    return MultiAgentRunner(
        planner_model=StaticPlanner(),
        retrieval_model=RetrievalModel(),
        action_model=ActionModel(),
        supervisor_model=FinalModel(),
        engine=engine,
        limits=MultiAgentLimits(max_seconds=3, child_timeout_seconds=1),
    )


@pytest.mark.asyncio
async def test_application_factory_exposes_session_bound_multi_agent_preview(make_client):
    client, app = await make_client(
        type("IntentModel", (), {
            "ainvoke": lambda self, *_args, **_kwargs: AIMessage(
                content=json.dumps({"intent": "其他"}, ensure_ascii=False)
            )
        })()
    )
    app.state.multi_agent = _runtime(app.state.engine)

    response = await client.post("/api/multi-agent/chat", json={
        "message": "请为订单1001创建售后工单",
        "session_id": 777,
    })

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["conversation_id"] == 777
    assert len(body["write_previews"]) == 1
    preview = body["write_previews"][0]
    assert preview["conversation_id"] == 777
    assert preview["arguments"]["conversation_id"] == 777

    confirmation = {
        "conversation_id": 777,
        "tool_name": "create_ticket",
        "description": preview["description"],
        "ticket_type": preview["ticket_type"],
        "confirm_key": preview["confirm_key"],
    }
    first = await client.post("/api/tickets/confirm", json=confirmation)
    retry = await client.post("/api/tickets/confirm", json=confirmation)

    assert first.status_code == 200, first.text
    assert retry.status_code == 200, retry.text
    assert first.json()["ticket_no"] == retry.json()["ticket_no"]
    assert retry.json()["deduplicated"] is True
    history = await app.state.store.get_history(777)
    assert [row["role"] for row in history] == ["user", "assistant"]


def test_application_runtime_builder_uses_existing_engine_and_model():
    engine = object()
    model = object()
    runtime = build_multi_agent_runtime(engine=engine, model=model)
    assert runtime.engine is engine
    assert runtime.planner_model is model
    assert runtime.retrieval_model is model
    assert runtime.action_model is model
    assert runtime.supervisor_model is model


@pytest.mark.asyncio
async def test_application_factory_wires_typed_handoff_as_explicit_opt_in(make_client):
    _, app = await make_client(
        type("IntentModel", (), {
            "ainvoke": lambda self, *_args, **_kwargs: AIMessage(
                content=json.dumps({"intent": "其他"}, ensure_ascii=False)
            )
        })(),
        settings_overrides={"multi_agent_handoff_mode": "typed"},
    )

    assert app.state.multi_agent.handoff_mode == "typed"
    assert app.state.settings.multi_agent_handoff_mode == "typed"


@pytest.mark.asyncio
async def test_multi_agent_cancel_does_not_create_ticket(make_client):
    client, app = await make_client(
        type("IntentModel", (), {
            "ainvoke": lambda self, *_args, **_kwargs: AIMessage(
                content=json.dumps({"intent": "其他"}, ensure_ascii=False)
            )
        })()
    )
    app.state.multi_agent = _runtime(app.state.engine)

    response = await client.post("/api/multi-agent/chat", json={
        "message": "请为订单1001创建售后工单",
        "session_id": 778,
    })
    assert response.status_code == 200, response.text
    cancelled = await client.post("/api/tickets/cancel", json={
        "conversation_id": 778,
        "reason": "用户取消",
    })
    assert cancelled.status_code == 200
    with app.state.session_factory() as session:
        assert session.scalars(select(Ticket)).all() == []
