import asyncio
import json
import time
from datetime import datetime, timezone

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import ValidationError

from app.multi_agent.models import AgentPlan, AgentTask, MultiAgentLimits
from app.multi_agent.api import create_multi_agent_app
from app.multi_agent.roles import ROLE_SPECS
from app.multi_agent.runtime import MultiAgentRunner, build_multi_agent_graph


def _claim(**overrides):
    from app.evidence_contracts import EvidenceClaim, EvidenceSource

    values = {
        "claim_id": "c1",
        "run_id": "run",
        "task_id": "r1",
        "subject_ids": ["1001"],
        "predicate": "status",
        "value": "已签收",
        "scope": "order",
        "observed_at": "2026-10-27T00:00:00+08:00",
        "observed_sequence": 1,
        "certainty": "verified",
        "source": EvidenceSource(
            tool_name="query_order",
            arguments={"order_id": "1001"},
            raw_result='{"order_id":"1001","status":"已签收"}',
            tool_call_id="t1",
        ),
    }
    values.update(overrides)
    return EvidenceClaim(**values)


def test_claim_roundtrip_marks_missing_fields_unknown_and_keeps_raw_source():
    claim = _claim()
    payload = claim.model_dump(mode="json")
    restored = type(claim).model_validate_json(json.dumps(payload, ensure_ascii=False))

    assert restored == claim
    assert restored.unit == "unknown"
    assert restored.policy_version == "unknown"
    assert restored.valid_from == "unknown"
    assert restored.source.raw_result == '{"order_id":"1001","status":"已签收"}'


def test_claim_rejects_unsourced_or_draft_derived_fact():
    from app.evidence_contracts import EvidenceClaim

    with pytest.raises(ValidationError, match="source"):
        EvidenceClaim(
            claim_id="c1", run_id="run", task_id="a1",
            subject_ids=["1001"], predicate="status", value="已签收",
            observed_at="2026-10-27T00:00:00+08:00", observed_sequence=1,
        )


def test_tool_result_mapping_creates_verified_order_and_policy_claims():
    from app.evidence_contracts import claims_from_tool_result

    order = claims_from_tool_result(
        "query_order", {"order_id": "1001"},
        json.dumps({"order_id": "1001", "product": "无线耳机", "status": "已签收"},
                   ensure_ascii=False),
        run_id="run", task_id="r1", tool_call_id="call-1",
    )
    assert {claim.predicate for claim in order} >= {"product", "status"}
    assert all("1001" in claim.subject_ids for claim in order)
    assert all(claim.certainty == "verified" for claim in order)
    assert all('"已签收"' in claim.source.raw_result for claim in order)

    policy = claims_from_tool_result(
        "query_faq", {"keyword": "退货政策"},
        json.dumps({
            "items": [{
                "chunk_id": "chunk-v2",
                "answer": "v2 政策：签收后7天可退。",
                "scope": "global",
                "policy_version": "v2",
                "valid_from": "2026-10-01",
                "valid_until": "2026-12-31",
            }],
        }, ensure_ascii=False),
        run_id="run", task_id="r1", tool_call_id="call-2",
    )
    assert len(policy) == 1
    claim = policy[0]
    assert claim.predicate == "policy_content"
    assert claim.scope == "global"
    assert claim.policy_version == "v2"
    assert claim.source.chunk_id == "chunk-v2"
    assert claim.subject_ids == ["global_policy"]


def test_tool_result_mapping_keeps_field_specific_validity_for_stale_logistics():
    from app.evidence_contracts import claims_from_tool_result

    claims = claims_from_tool_result(
        "query_order", {"order_id": "1001"},
        json.dumps({
            "order_id": "1001", "status": "已签收",
            "logistics_status": "派送中",
            "logistics_status_valid_until": "2026-10-21T00:00:00+00:00",
        }, ensure_ascii=False),
        run_id="run", task_id="r1", tool_call_id="call-1",
        observed_at="2026-10-20T00:00:00+00:00",
    )
    logistics = next(item for item in claims if item.predicate == "logistics_status")
    assert logistics.valid_until == "2026-10-21T00:00:00+00:00"


def test_projection_respects_dependencies_subjects_and_policy_scope():
    from app.evidence_contracts import EvidenceLedger, TaskHandoff

    ledger = EvidenceLedger(now=lambda: datetime(2026, 10, 27, tzinfo=timezone.utc))
    ledger.add_many([
        _claim(claim_id="c1001", task_id="r1", subject_ids=["1001"]),
        _claim(
            claim_id="c1002", task_id="r2", subject_ids=["1002"],
            value="已发货",
            source=_claim().source.model_copy(update={"arguments": {"order_id": "1002"}}),
        ),
        _claim(
            claim_id="scope", task_id="r2", subject_ids=["1002"],
            predicate="product", value="无线耳机", scope="sku:keyboard",
        ),
    ])
    action = AgentTask(
        task_id="a1", agent="action", objective="为订单1001生成工单",
        depends_on=["r1"],
    )
    handoff = TaskHandoff.role(
        run_id="run", from_agent="supervisor", to_agent="action",
        task=action, subject_ids=["1001"],
        observed_at="2026-10-27T00:00:00+00:00",
    )
    visible = ledger.project(task=action, handoff=handoff)
    assert [claim.claim_id for claim in visible] == ["c1001"]

    isolated = AgentTask(task_id="r3", agent="retrieval", objective="查询订单1001状态")
    handoff = TaskHandoff.role(
        run_id="run", from_agent="supervisor", to_agent="retrieval",
        task=isolated, subject_ids=["1001"],
        observed_at="2026-10-27T00:00:00+00:00",
    )
    assert ledger.project(task=isolated, handoff=handoff) == []


def test_newer_observation_supersedes_old_and_expired_claim_is_not_projected():
    from app.evidence_contracts import EvidenceLedger, TaskHandoff

    ledger = EvidenceLedger(now=lambda: datetime(2026, 10, 27, tzinfo=timezone.utc))
    first = _claim(claim_id="old", value="已发货", observed_sequence=1)
    second = _claim(claim_id="new", value="已签收", observed_sequence=2)
    expired = _claim(
        claim_id="expired", predicate="tracking", value="SF1",
        valid_until="2026-10-26T00:00:00+00:00",
    )
    ledger.add_many([first, second, expired])

    task = AgentTask(task_id="a1", agent="action", objective="处理订单1001",
                     depends_on=["r1"])
    handoff = TaskHandoff.role(
        run_id="run", from_agent="supervisor", to_agent="action",
        task=task, subject_ids=["1001"],
        observed_at="2026-10-27T00:00:00+00:00",
    )
    visible = ledger.project(task=task, handoff=handoff)
    assert [claim.claim_id for claim in visible] == ["new"]
    assert ledger.claims["old"].superseded_by == "new"


@pytest.mark.asyncio
async def test_tool_cache_coalesces_concurrent_equivalent_reads_and_never_caches_writes():
    from app.evidence_contracts import ToolResultCache

    cache = ToolResultCache(dynamic_ttl_seconds=60)
    calls = 0

    async def read():
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.02)
        return json.dumps({"order_id": "1001", "status": "已签收"}, ensure_ascii=False)

    results = await asyncio.gather(
        cache.execute("query_order", {"order_id": "1001"}, read=read, execute=read),
        cache.execute("query_order", {"order_id": "1001"}, read=read, execute=read),
    )
    assert calls == 1
    assert len(results) == 2
    assert cache.stats.attempted == 2
    assert cache.stats.executed == 1
    assert cache.stats.deduplicated == 1

    writes = 0

    async def write():
        nonlocal writes
        writes += 1
        return str(writes)

    await cache.execute("create_ticket", {"description": "x"}, read=False, execute=write)
    await cache.execute("create_ticket", {"description": "x"}, read=False, execute=write)
    assert writes == 2
    assert cache.stats.executed == 3
    assert cache.stats.deduplicated == 1


@pytest.mark.asyncio
async def test_dynamic_tool_cache_rechecks_after_ttl():
    from app.evidence_contracts import ToolResultCache

    cache = ToolResultCache(dynamic_ttl_seconds=0)
    calls = 0

    async def read():
        nonlocal calls
        calls += 1
        return str(calls)

    first = await cache.execute("query_order", {"order_id": "1001"},
                                read=True, execute=read)
    second = await cache.execute("query_order", {"order_id": "1001"},
                                 read=True, execute=read)
    assert (first.result, second.result) == ("1", "2")
    assert cache.stats.executed == 2


class _NoToolModel:
    def bind_tools(self, _tools, **_kwargs):
        return self

    async def ainvoke(self, messages, **_kwargs):
        self.last_messages = messages
        return AIMessage(content="draft")


class _Planner:
    def __init__(self, plan):
        self.plan = plan

    def with_structured_output(self, _schema, **_kwargs):
        return self

    async def ainvoke(self, _messages, **_kwargs):
        return self.plan


class _Final:
    def __init__(self):
        self.messages = None

    async def ainvoke(self, messages, **_kwargs):
        self.messages = messages
        return AIMessage(content="汇总")


@pytest.mark.asyncio
async def test_runner_defaults_to_raw_and_legacy_constructor_and_response_shape():
    plan = AgentPlan(disposition="execute", tasks=[
        AgentTask(task_id="r1", agent="retrieval", objective="查询"),
    ])
    runner = MultiAgentRunner(
        planner_model=_Planner(plan), retrieval_model=_NoToolModel(),
        action_model=_NoToolModel(), supervisor_model=_Final(), engine=object(),
        limits=MultiAgentLimits(max_seconds=3, child_timeout_seconds=1),
    )
    assert runner.handoff_mode == "raw"


@pytest.mark.asyncio
async def test_raw_mode_still_sends_legacy_shared_facts_projection():
    task = AgentTask(task_id="r1", agent="retrieval", objective="查询订单1001状态")
    retrieval = _NoToolModel()
    runner = MultiAgentRunner(
        planner_model=_Planner(AgentPlan(disposition="clarify", clarification="x")),
        retrieval_model=retrieval, action_model=_NoToolModel(),
        supervisor_model=_Final(), engine=object(),
        limits=MultiAgentLimits(max_seconds=3, child_timeout_seconds=1),
    )
    await runner.run_task(
        agent_name="retrieval", task=task, facts={}, evidence={},
        run_id="raw-run", deadline=time.monotonic() + 2,
    )
    assert "shared_facts={}" in retrieval.last_messages[1].content
    assert "evidence_claims" not in retrieval.last_messages[1].content


class _TypedEngine:
    def __init__(self):
        self.calls = []

    async def execute(self, name, args_json, **kwargs):
        self.calls.append((name, json.loads(args_json)))
        return json.dumps({
            "order_id": "1001", "product": "无线耳机", "status": "已签收",
        }, ensure_ascii=False)


class _OneReadModel:
    def bind_tools(self, _tools, **_kwargs):
        return self

    async def ainvoke(self, messages, **_kwargs):
        self.last_messages = messages
        if isinstance(messages[-1], ToolMessage):
            return AIMessage(content="事实已获取")
        return AIMessage(content="", tool_calls=[{
            "name": "query_order", "args": {"order_id": "1001"}, "id": "read",
        }])


@pytest.mark.asyncio
async def test_typed_role_prompt_uses_projected_claims_and_records_handoff_kinds():
    task = AgentTask(task_id="r1", agent="retrieval", objective="查询订单1001状态")
    retrieval = _OneReadModel()
    runner = MultiAgentRunner(
        planner_model=_Planner(AgentPlan(disposition="clarify", clarification="x")),
        retrieval_model=retrieval, action_model=_NoToolModel(),
        supervisor_model=_Final(), engine=_TypedEngine(),
        limits=MultiAgentLimits(max_seconds=3, child_timeout_seconds=1),
        handoff_mode="typed",
    )
    outcome = await runner.run_task(
        agent_name="retrieval", task=task, facts={}, evidence={},
        run_id="typed-run", deadline=time.monotonic() + 2,
    )

    prompt = json.dumps(outcome.messages, ensure_ascii=False)
    assert "subject_ids" in prompt
    assert "shared_facts" not in prompt
    assert "raw_result" not in prompt
    assert outcome.evidence_claims
    assert outcome.evidence_claims[0].claim_id.startswith("c:")
    assert outcome.evidence_claims[0].source.raw_result
    assert outcome.handoff.kind == "role_handoff"
    assert any(event.kind == "tool_call" for event in outcome.handoff_records)
    assert ROLE_SPECS["retrieval"].allowed_tools == ("query_order", "query_faq")


class _DependencyModel:
    def __init__(self, role, *, capture=None):
        self.role = role
        self.capture = capture

    def bind_tools(self, _tools, **_kwargs):
        return self

    async def ainvoke(self, messages, **_kwargs):
        prompt = "\n".join(str(getattr(item, "content", "")) for item in messages)
        if self.role == "retrieval" and not isinstance(messages[-1], ToolMessage):
            order_id = "1001" if "订单1001" in prompt else "1002"
            return AIMessage(content="", tool_calls=[{
                "name": "query_order", "args": {"order_id": order_id},
                "id": f"read-{order_id}",
            }])
        if self.capture is not None:
            self.capture.append(prompt)
        return AIMessage(content="typed action draft")


class _OrdersEngine:
    def __init__(self):
        self.calls = []

    async def execute(self, name, args_json, **kwargs):
        arguments = json.loads(args_json)
        self.calls.append((name, arguments))
        order_id = arguments["order_id"]
        status = "已签收" if order_id == "1001" else "已发货"
        return json.dumps({
            "order_id": order_id, "product": f"商品{order_id}", "status": status,
        }, ensure_ascii=False)


@pytest.mark.asyncio
async def test_typed_action_model_input_is_limited_to_its_dependency_subject():
    plan = AgentPlan(disposition="execute", tasks=[
        AgentTask(task_id="r1", agent="retrieval", objective="查询订单1001状态"),
        AgentTask(task_id="r2", agent="retrieval", objective="查询订单1002状态"),
        AgentTask(
            task_id="a1", agent="action", objective="处理订单1001",
            depends_on=["r1"],
        ),
    ])
    action_prompts = []
    runner = MultiAgentRunner(
        planner_model=_Planner(plan),
        retrieval_model=_DependencyModel("retrieval"),
        action_model=_DependencyModel("action", capture=action_prompts),
        supervisor_model=_Final(), engine=_OrdersEngine(),
        limits=MultiAgentLimits(max_seconds=3, child_timeout_seconds=1),
        handoff_mode="typed",
    )
    result = await build_multi_agent_graph(runner).ainvoke({
        "run_id": "dependency-run", "task_id": "task",
        "user_message": "比较订单1001和1002，处理订单1001",
        "conversation_history": [],
    })

    assert len(action_prompts) == 1
    assert "订单1001" in action_prompts[0]
    assert "商品1001" in action_prompts[0]
    assert "订单1002" not in action_prompts[0]
    assert "商品1002" not in action_prompts[0]
    assert "raw_result" not in action_prompts[0]
    assert result["tool_cache_stats"]["executed"] == 2


@pytest.mark.asyncio
async def test_human_handoff_is_recorded_separately_from_internal_role_handoff():
    runner = MultiAgentRunner(
        planner_model=_Planner(AgentPlan(
            disposition="handoff", reason="用户要求人工客服",
        )),
        retrieval_model=_NoToolModel(), action_model=_NoToolModel(),
        supervisor_model=_Final(), engine=object(),
        limits=MultiAgentLimits(max_seconds=3, child_timeout_seconds=1),
        handoff_mode="typed",
    )
    result = await build_multi_agent_graph(runner).ainvoke({
        "run_id": "human-run", "task_id": "task",
        "user_message": "我要人工客服", "conversation_history": [],
    })
    assert [item["kind"] for item in result["handoffs"]] == ["human_handoff"]
    assert result["handoffs"][0]["to_agent"] == "human"


@pytest.mark.asyncio
async def test_typed_internal_fields_do_not_change_public_api_shape():
    runner = MultiAgentRunner(
        planner_model=_Planner(AgentPlan(
            disposition="clarify", clarification="请提供订单号。",
        )),
        retrieval_model=_NoToolModel(), action_model=_NoToolModel(),
        supervisor_model=_Final(), engine=object(),
        limits=MultiAgentLimits(max_seconds=3, child_timeout_seconds=1),
        handoff_mode="typed",
    )
    app = create_multi_agent_app(runner)
    async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/multi-agent/chat", json={"message": "这个能退吗"})
    assert response.status_code == 200
    assert set(response.json()) == {
        "run_id", "task_id", "conversation_id", "status", "final_reply",
        "facts", "evidence", "errors", "stop_reasons", "trace",
        "write_previews", "steps", "tokens",
    }
