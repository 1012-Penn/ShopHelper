import asyncio
import json
import time

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import ValidationError

from app.multi_agent.models import AgentPlan, AgentTask, MultiAgentLimits
from app.multi_agent.runtime import MultiAgentRunner, build_multi_agent_graph


class StaticPlanner:
    def __init__(self, plan):
        self.plan = plan
        self.calls = []

    def with_structured_output(self, _schema, **_kwargs):
        return self

    async def ainvoke(self, messages, **_kwargs):
        self.calls.append(messages)
        return self.plan


class RuleAgentModel:
    """Small deterministic LangChain-compatible model for graph integration tests."""

    def __init__(self):
        self.visible_tools = []

    def bind_tools(self, tools, **_kwargs):
        self.visible_tools = [tool.name for tool in tools]
        return self

    async def ainvoke(self, messages, **_kwargs):
        if isinstance(messages[-1], ToolMessage):
            return AIMessage(content="工具结果已记录")
        prompt = "\n".join(str(getattr(item, "content", "")) for item in messages)
        if "订单1001" in prompt and "query_order" in self.visible_tools:
            return AIMessage(content="", tool_calls=[{
                "name": "query_order",
                "args": {"order_id": "1001"},
                "id": "call-1001",
            }])
        if "订单1002" in prompt and "query_order" in self.visible_tools:
            return AIMessage(content="", tool_calls=[{
                "name": "query_order",
                "args": {"order_id": "1002"},
                "id": "call-1002",
            }])
        if "政策" in prompt and "query_faq" in self.visible_tools:
            return AIMessage(content="", tool_calls=[{
                "name": "query_faq",
                "args": {"keyword": "退货政策"},
                "id": "call-faq",
            }])
        return AIMessage(content="行动草稿：依据共享事实比较两单。")


class FinalModel:
    def __init__(self):
        self.calls = []

    async def ainvoke(self, messages, **_kwargs):
        self.calls.append(messages)
        return AIMessage(content="订单1001已签收，订单1002已发货；请依据政策确认。")


class ConcurrentEngine:
    def __init__(self):
        self.active = 0
        self.max_active = 0
        self.calls = []
        self.preview = None

    async def execute(self, name, args_json, *, conversation_id=None,
                      tool_call_id=None, on_write_intercept=None, confirmed=False):
        self.calls.append({"name": name, "arguments": json.loads(args_json)})
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.08 if name == "query_order" else 0.01)
            if name == "query_order":
                order_id = json.loads(args_json)["order_id"]
                rows = {
                    "1001": {"order_id": "1001", "product": "无线耳机", "status": "已签收"},
                    "1002": {"order_id": "1002", "product": "机械键盘", "status": "已发货"},
                }
                return json.dumps(rows[order_id], ensure_ascii=False)
            return json.dumps({
                "items": [{
                    "n": 1,
                    "chunk_id": 15001,
                    "answer": "签收后15天内可无理由退货。",
                    "section_path": "模拟政策",
                }],
                "low_confidence": False,
            }, ensure_ascii=False)
        finally:
            self.active -= 1


class FlakyEngine(ConcurrentEngine):
    def __init__(self):
        super().__init__()
        self.failed_once = False

    async def execute(self, name, args_json, **kwargs):
        if not self.failed_once:
            self.failed_once = True
            raise RuntimeError("temporary read failure")
        return await super().execute(name, args_json, **kwargs)


def _limits(**overrides):
    values = {
        "max_tasks": 3,
        "max_replans": 1,
        "max_steps": 12,
        "max_tokens": 6000,
        "max_seconds": 5.0,
        "child_timeout_seconds": 2.0,
    }
    values.update(overrides)
    return MultiAgentLimits(**values)


def test_plan_rejects_action_that_can_race_policy_judgment():
    with pytest.raises(ValidationError, match="action task"):
        AgentPlan(disposition="execute", tasks=[
            AgentTask(task_id="write", agent="action", objective="判断订单能否退货并起草工单"),
        ])


def test_plan_rejects_more_than_three_tasks_or_unknown_dependencies():
    with pytest.raises(ValidationError, match="at most 3"):
        AgentPlan(disposition="execute", tasks=[
            AgentTask(task_id=f"r{i}", agent="retrieval", objective=f"查询{i}")
            for i in range(4)
        ])
    with pytest.raises(ValidationError, match="unknown dependency"):
        AgentPlan(disposition="execute", tasks=[
            AgentTask(task_id="r1", agent="retrieval", objective="查询订单1001"),
            AgentTask(task_id="a1", agent="action", objective="汇总", depends_on=["missing"]),
        ])


@pytest.mark.asyncio
async def test_retrieval_tasks_run_in_parallel_then_action_consumes_shared_facts():
    plan = AgentPlan(disposition="execute", tasks=[
        AgentTask(task_id="r1", agent="retrieval", objective="查询订单1001状态"),
        AgentTask(task_id="r2", agent="retrieval", objective="查询订单1002状态"),
        AgentTask(task_id="a1", agent="action", objective="比较两单并给出政策草稿",
                  depends_on=["r1", "r2"]),
    ])
    engine = ConcurrentEngine()
    runner = MultiAgentRunner(
        planner_model=StaticPlanner(plan),
        retrieval_model=RuleAgentModel(),
        action_model=RuleAgentModel(),
        supervisor_model=FinalModel(),
        engine=engine,
        limits=_limits(),
    )
    graph = build_multi_agent_graph(runner)
    started = time.perf_counter()
    result = await graph.ainvoke({
        "run_id": "run-1",
        "task_id": "task-1",
        "user_message": "比较订单1001和订单1002，并说明退货政策",
        "conversation_history": [],
        "facts": {},
        "evidence": {},
        "errors": [],
        "stop_reasons": [],
        "trace": [],
        "role_histories": {},
    })
    elapsed = time.perf_counter() - started

    assert engine.max_active >= 2
    assert elapsed < 0.30
    assert set(result["facts"]) >= {"r1", "r2", "a1"}
    assert result["facts"]["r1"]["order_id"] == "1001"
    assert result["facts"]["r2"]["order_id"] == "1002"
    assert result["task_agents"]["r1"] == "retrieval"
    assert result["task_agents"]["a1"] == "action"
    assert result["final_reply"].startswith("订单1001")
    assert any(event["event"] == "task_start" for event in result["trace"])
    assert any(event["event"] == "task_complete" and event["task_id"] == "a1"
               for event in result["trace"])
    assert runner._runs == {}


@pytest.mark.asyncio
async def test_agent_cannot_call_tools_outside_its_whitelist():
    class ForbiddenModel:
        def __init__(self):
            self.visible_tools = []

        def bind_tools(self, tools, **_kwargs):
            self.visible_tools = [tool.name for tool in tools]
            return self

        async def ainvoke(self, *_args, **_kwargs):
            return AIMessage(content="", tool_calls=[{
                "name": "create_ticket",
                "args": {"conversation_id": 1, "description": "x", "ticket_type": "售后"},
                "id": "forbidden-call",
            }])

    model = ForbiddenModel()
    engine = ConcurrentEngine()
    runner = MultiAgentRunner(
        planner_model=StaticPlanner(AgentPlan(disposition="clarify", clarification="请补充订单号")),
        retrieval_model=model,
        action_model=RuleAgentModel(),
        supervisor_model=FinalModel(),
        engine=engine,
        limits=_limits(),
    )
    outcome = await runner.run_task(
        agent_name="retrieval",
        task=AgentTask(task_id="r1", agent="retrieval", objective="查询订单1001"),
        facts={}, evidence={}, run_id="run-2", deadline=time.monotonic() + 2,
    )

    assert "query_order" in model.visible_tools
    assert "query_faq" in model.visible_tools
    assert "create_ticket" not in model.visible_tools
    assert outcome.error.startswith("工具权限拒绝")
    assert engine.calls == []


@pytest.mark.asyncio
async def test_child_timeout_is_recorded_as_task_failure():
    class SlowEngine(ConcurrentEngine):
        async def execute(self, *_args, **_kwargs):
            await asyncio.sleep(0.10)
            return "{}"

    runner = MultiAgentRunner(
        planner_model=StaticPlanner(AgentPlan(disposition="clarify", clarification="请补充")),
        retrieval_model=RuleAgentModel(),
        action_model=RuleAgentModel(),
        supervisor_model=FinalModel(),
        engine=SlowEngine(),
        limits=_limits(child_timeout_seconds=0.02),
    )
    outcome = await runner.run_task(
        agent_name="retrieval",
        task=AgentTask(task_id="r1", agent="retrieval", objective="查询订单1001"),
        facts={}, evidence={}, run_id="run-3", deadline=time.monotonic() + 2,
    )

    assert outcome.error == "child_timeout"
    assert outcome.stop_reason == "timeout"


@pytest.mark.asyncio
async def test_planner_failure_is_returned_as_failed_state_instead_of_crashing_graph():
    class BrokenPlanner:
        def with_structured_output(self, _schema, **_kwargs):
            return self

        async def ainvoke(self, *_args, **_kwargs):
            raise RuntimeError("provider unavailable")

    runner = MultiAgentRunner(
        planner_model=BrokenPlanner(),
        retrieval_model=RuleAgentModel(),
        action_model=RuleAgentModel(),
        supervisor_model=FinalModel(),
        engine=ConcurrentEngine(),
        limits=_limits(),
    )
    result = await build_multi_agent_graph(runner).ainvoke({
        "run_id": "run-planner-failure", "task_id": "task",
        "user_message": "订单1001状态", "conversation_history": [],
    })

    assert result["status"] == "failed"
    assert any("provider unavailable" in error for error in result["errors"])
    assert result["final_reply"].startswith("【无法回答】")


@pytest.mark.asyncio
async def test_task_failure_replans_at_most_once_and_then_finalizes():
    class SequentialPlanner:
        def __init__(self):
            self.calls = 0
            self.plans = [
                AgentPlan(disposition="execute", tasks=[
                    AgentTask(task_id="r1", agent="retrieval", objective="查询订单1001"),
                ]),
                AgentPlan(disposition="execute", tasks=[
                    AgentTask(task_id="r2", agent="retrieval", objective="重试订单1001状态"),
                ]),
            ]

        def with_structured_output(self, _schema, **_kwargs):
            return self

        async def ainvoke(self, _messages, **_kwargs):
            plan = self.plans[min(self.calls, len(self.plans) - 1)]
            self.calls += 1
            return plan

    planner = SequentialPlanner()
    result = await build_multi_agent_graph(MultiAgentRunner(
        planner_model=planner,
        retrieval_model=RuleAgentModel(),
        action_model=RuleAgentModel(),
        supervisor_model=FinalModel(),
        engine=FlakyEngine(),
        limits=_limits(),
    )).ainvoke({
        "run_id": "run-replan", "task_id": "task",
        "user_message": "订单1001状态", "conversation_history": [],
    })

    assert planner.calls == 2
    assert result["replan_count"] == 1
    assert sum(event["event"] == "replan" for event in result["trace"]) == 1
    assert result["status"] == "success"
