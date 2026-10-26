import pytest

from app.multi_agent.models import AgentPlan, MultiAgentLimits
from app.multi_agent.runtime import MultiAgentRunner
from tests.helpers import GraphChatModel


class _ClarifyPlanner:
    def with_structured_output(self, _schema, **_kwargs):
        return self

    async def ainvoke(self, _messages, **_kwargs):
        return AgentPlan(
            disposition="clarify",
            clarification="请说明需要比较的具体售后条件。",
        )


class _UnusedRoleModel:
    def bind_tools(self, _tools, **_kwargs):
        return self

    async def ainvoke(self, *_args, **_kwargs):
        raise AssertionError("role model should not run for clarify")


class _UnusedFinal:
    async def ainvoke(self, *_args, **_kwargs):
        raise AssertionError("final model should not run for clarify")


@pytest.mark.asyncio
async def test_real_app_factory_mounts_adaptive_route_and_reuses_run_semantics(make_client):
    client, app = await make_client(
        GraphChatModel('{"intent": "其他"}', [("text", "你好，有什么可以帮你？")]),
        settings_overrides={
            "adaptive_routing_enabled": True,
            "evidence_entailment_enabled": False,
        },
    )

    response = await client.post("/api/adaptive-routing/chat", json={
        "message": "你好",
        "session_id": 909,
        "history": [],
        "visible_budget": {"remaining_seconds": 20, "remaining_cost_usd": 0.05},
    })

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["selected_arm"] == "single"
    assert body["response"]["reply"] == "你好，有什么可以帮你？", body
    assert body["response"]["conversation_id"] == 909
    run_id = body["response"]["run_id"]
    assert run_id
    run = (await client.get(f"/api/chat/runs/{run_id}")).json()
    assert run["status"] == "completed"
    assert run["result"]["answer"] == "你好，有什么可以帮你？"
    history = await app.state.store.get_history(909)
    assert [row["role"] for row in history] == ["user", "assistant"]


@pytest.mark.asyncio
async def test_adaptive_route_selects_multi_agent_for_cross_order_question(make_client):
    client, app = await make_client(
        GraphChatModel('{"intent": "其他"}', [("text", "single path must not run")]),
        settings_overrides={
            "adaptive_routing_enabled": True,
            "evidence_entailment_enabled": False,
        },
    )
    app.state.multi_agent = MultiAgentRunner(
        planner_model=_ClarifyPlanner(),
        retrieval_model=_UnusedRoleModel(),
        action_model=_UnusedRoleModel(),
        supervisor_model=_UnusedFinal(),
        engine=app.state.engine,
        limits=MultiAgentLimits(max_seconds=3, child_timeout_seconds=1),
    )

    response = await client.post("/api/adaptive-routing/chat", json={
        "message": "比较订单1001和订单1002能否直接取消。",
        "session_id": 910,
    })

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["selected_arm"] == "multi"
    assert body["response"]["reply"] == "请说明需要比较的具体售后条件。"
