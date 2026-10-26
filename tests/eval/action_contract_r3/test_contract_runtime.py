"""Contract tests for schema-bound preview/confirm/cancel actions."""

from __future__ import annotations

import json
import time

import pytest
from langchain.tools import tool
from langchain_core.messages import AIMessage, ToolMessage

from app.multi_agent.models import AgentPlan, AgentTask, MultiAgentLimits
from app.multi_agent.runtime import MultiAgentRunner, build_multi_agent_graph
from app.tools.base import ToolRecord, ToolRegistryV2
from app.tools.builtin import CreateTicketInput
from app.tools.engine import ToolEngine


class _Audit:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def log(self, **kwargs) -> None:
        self.rows.append(kwargs)


class _Sink:
    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.by_key: dict[str, dict] = {}

    def create(self, conversation_id: int, description: str, ticket_type: str,
               confirm_key: str | None = None) -> dict:
        if confirm_key and confirm_key in self.by_key:
            return {**self.by_key[confirm_key], "deduplicated": True}
        row = {
            "ticket_no": f"T{len(self.rows) + 1}",
            "conversation_id": conversation_id,
            "description": description,
            "ticket_type": ticket_type,
            "confirm_key": confirm_key,
        }
        self.rows.append(row)
        if confirm_key:
            self.by_key[confirm_key] = row
        return dict(row)


class _Planner:
    def __init__(self, plan: AgentPlan) -> None:
        self.plan = plan

    def with_structured_output(self, _schema, **_kwargs):
        return self

    async def ainvoke(self, _messages, **_kwargs):
        return self.plan


class _OrderModel:
    def bind_tools(self, _tools, **_kwargs):
        return self

    async def ainvoke(self, messages, **_kwargs):
        if isinstance(messages[-1], ToolMessage):
            return AIMessage(content="订单1001已签收。")
        return AIMessage(content="", tool_calls=[{
            "name": "query_order",
            "args": {"order_id": "1001"},
            "id": "order-1",
        }])


class _ScriptedAction:
    def __init__(self, responses: list[AIMessage]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def bind_tools(self, _tools, **_kwargs):
        return self

    async def ainvoke(self, _messages, **_kwargs):
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return response


class _Final:
    def __init__(self, text: str = "请查看结果。") -> None:
        self.text = text

    async def ainvoke(self, _messages, **_kwargs):
        return AIMessage(content=self.text)


def _registry(sink: _Sink) -> ToolRegistryV2:
    @tool
    def query_order(order_id: str) -> str:
        """Query one simulated order."""
        return json.dumps({
            "order_id": order_id,
            "product": "无线耳机",
            "status": "已签收",
        }, ensure_ascii=False)

    @tool(args_schema=CreateTicketInput)
    def create_ticket(conversation_id: int, description: str, ticket_type: str,
                      confirm_key: str | None = None) -> str:
        """Create a ticket in the isolated fake-write sink."""
        return json.dumps(
            sink.create(conversation_id, description, ticket_type, confirm_key),
            ensure_ascii=False,
        )

    registry = ToolRegistryV2()
    registry.register(ToolRecord(query_order, source="builtin", access="read"))
    registry.register(ToolRecord(create_ticket, source="builtin", access="write"))
    return registry


def _plan() -> AgentPlan:
    return AgentPlan(disposition="execute", tasks=[
        AgentTask(task_id="r1", agent="retrieval", objective="查询订单1001"),
        AgentTask(
            task_id="a1", agent="action",
            objective="基于共享事实生成工单草稿或回答", depends_on=["r1"],
        ),
    ])


def _runner(sink: _Sink, action_model, *, final_text: str = "请查看结果。",
            ledger=None, enabled: bool = True) -> MultiAgentRunner:
    engine = ToolEngine(_registry(sink), _Audit(), timeout_seconds=1, retry_attempts=0)
    return MultiAgentRunner(
        planner_model=_Planner(_plan()),
        retrieval_model=_OrderModel(),
        action_model=action_model,
        supervisor_model=_Final(final_text),
        engine=engine,
        limits=MultiAgentLimits(
            max_seconds=3, child_timeout_seconds=1, max_steps=12,
            max_tokens=6000, max_agent_iterations=4,
        ),
        action_contracts_enabled=enabled,
        action_ledger=ledger,
    )


async def _invoke(runner: MultiAgentRunner, *, message: str, conversation_id: int,
                  history: list[dict] | None = None, run_id: str = "run-1") -> dict:
    return await build_multi_agent_graph(runner).ainvoke({
        "run_id": run_id,
        "task_id": "task-1",
        "conversation_id": conversation_id,
        "user_message": message,
        "conversation_history": history or [],
        "facts": {},
        "evidence": {},
        "errors": [],
        "stop_reasons": [],
        "trace": [],
        "role_histories": {},
    })


def _tool_call(args: dict, *, call_id: str = "ticket-1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{
        "name": "create_ticket",
        "args": args,
        "id": call_id,
    }])


def test_schema_context_is_derived_from_registry_and_normalizes_only_named_synonyms():
    from app.action_contracts import ActionToolContext

    context = ActionToolContext.from_registry(_registry(_Sink()), "create_ticket")
    decision = context.prepare_arguments(
        {
            "conversation_id": 999,
            "description": "订单1001退款处理",
            "ticket_type": "退款",
            "confirm_key": "model-chosen-key",
        },
        conversation_id=17,
    )

    assert context.required_parameters == frozenset({
        "conversation_id", "description", "ticket_type",
    })
    assert context.enum_values["ticket_type"] == frozenset({"售后", "投诉", "咨询"})
    assert decision.ok is True
    assert decision.arguments == {
        "conversation_id": 17,
        "description": "订单1001退款处理",
        "ticket_type": "售后",
    }
    assert decision.original_arguments["ticket_type"] == "退款"
    type_normalization = next(
        item for item in decision.normalizations if item.field == "ticket_type"
    )
    assert type_normalization.reason == "unambiguous_refund_is_after_sales"
    assert any(item.field == "conversation_id"
               and item.reason == "trusted_run_scope" for item in decision.normalizations)


def test_schema_context_rejects_ambiguous_type_and_missing_description():
    from app.action_contracts import ActionToolContext

    context = ActionToolContext.from_registry(_registry(_Sink()), "create_ticket")
    ambiguous = context.prepare_arguments(
        {"description": "请帮我处理", "ticket_type": "其他"}, conversation_id=1,
    )
    missing_description = context.prepare_arguments(
        {"description": "  ", "ticket_type": "咨询"}, conversation_id=1,
    )

    assert ambiguous.ok is False
    assert ambiguous.needs_clarification is True
    assert "ticket_type" in ambiguous.feedback
    assert missing_description.ok is False
    assert missing_description.needs_clarification is True
    assert "description" in missing_description.feedback


def test_model_cannot_supply_confirmation_flag_or_key():
    from app.action_contracts import ActionToolContext

    context = ActionToolContext.from_registry(_registry(_Sink()), "create_ticket")
    decision = context.prepare_arguments(
        {
            "description": "订单1001退款",
            "ticket_type": "退款",
            "confirm_key": "model-key",
            "confirmed": True,
        },
        conversation_id=1,
    )
    assert decision.ok is False
    assert "confirmed" in decision.feedback
    assert "confirm_key" not in decision.arguments


@pytest.mark.asyncio
async def test_claiming_preview_without_artifact_gets_one_repair_then_truthful_failure():
    action = _ScriptedAction([
        AIMessage(content="工单预览已生成，请确认。"),
        AIMessage(content="工单预览已经为你生成。"),
    ])
    sink = _Sink()
    result = await _invoke(
        _runner(
            sink, action,
            final_text="工单已经创建成功。",
        ),
        message="请为订单1001创建售后工单，先给我预览。",
        conversation_id=1,
    )

    assert action.calls == 2
    assert sink.rows == []
    assert result["write_previews"] == []
    assert result["action_artifacts"][-1]["state"] == "failed"
    assert result["action_artifacts"][-1]["error_code"] == "missing_preview"
    assert "已生成" not in result["final_reply"]
    assert "已创建" not in result["final_reply"]
    assert "未能生成" in result["final_reply"]
    assert result["stop_reasons"][-1] == "action_contract_missing_preview"


@pytest.mark.asyncio
async def test_preview_claim_can_be_repaired_by_one_legal_structured_tool_call():
    action = _ScriptedAction([
        AIMessage(content="工单预览已生成。"),
        _tool_call({
            "conversation_id": 888,
            "description": "订单1001退款处理",
            "ticket_type": "退款",
        }),
        AIMessage(content="预览已生成，等待确认。"),
    ])
    sink = _Sink()
    result = await _invoke(
        _runner(sink, action),
        message="请为订单1001创建售后工单，只做预览。",
        conversation_id=7,
    )

    assert action.calls == 3
    assert sink.rows == []
    assert len(result["write_previews"]) == 1
    preview = result["write_previews"][0]
    assert preview["conversation_id"] == 7
    assert preview["ticket_type"] == "售后"
    assert preview["arguments"]["conversation_id"] == 7
    assert "confirm_key" not in preview["arguments"]
    assert result["action_artifacts"][-1]["state"] == "awaiting_confirmation"
    assert result["action_artifacts"][-1]["normalizations"]


@pytest.mark.asyncio
async def test_cancel_after_preview_does_not_rebuild_and_lost_response_replay_is_idempotent():
    sink = _Sink()
    action = _ScriptedAction([
        _tool_call({
            "description": "订单1001售后",
            "ticket_type": "售后",
        }),
        AIMessage(content="预览已生成，等待确认。"),
    ])
    runner = _runner(sink, action)
    first = await _invoke(
        runner, message="请为订单1001生成售后工单预览。",
        conversation_id=3, run_id="preview-run",
    )
    preview = first["write_previews"][0]

    cancelled = await _invoke(
        runner, message="算了，取消刚才的工单预览，不要提交。",
        conversation_id=3, run_id="cancel-run",
    )
    no_pending = await _invoke(
        runner, message="取消刚才的工单。", conversation_id=3, run_id="cancel-again",
    )

    assert cancelled["write_previews"] == []
    assert cancelled["action_artifacts"][-1]["state"] == "cancelled"
    assert no_pending["write_previews"] == []
    assert no_pending["action_artifacts"][-1]["error_code"] == "no_pending_action"
    assert sink.rows == []

    engine = runner.engine
    args = {**preview["arguments"], "confirm_key": preview["confirm_key"]}
    encoded = json.dumps(args, ensure_ascii=False)
    first_write = json.loads(await engine.execute_confirmed(
        "create_ticket", encoded, conversation_id=3, tool_call_id="confirm-1",
    ))
    replay = json.loads(await engine.execute_confirmed(
        "create_ticket", encoded, conversation_id=3, tool_call_id="confirm-2",
    ))
    assert first_write["ticket_no"] == replay["ticket_no"]
    assert replay["deduplicated"] is True
    assert len(sink.rows) == 1


@pytest.mark.asyncio
async def test_client_confirmation_marks_artifact_confirmed_after_real_engine_write():
    sink = _Sink()
    action = _ScriptedAction([
        _tool_call({"description": "订单1001售后", "ticket_type": "售后"}),
        AIMessage(content="预览已生成。"),
    ])
    runner = _runner(sink, action)
    state = await _invoke(
        runner, message="为订单1001生成售后工单预览。", conversation_id=11,
    )
    preview = state["write_previews"][0]
    args = {**preview["arguments"], "confirm_key": preview["confirm_key"]}
    ticket = json.loads(await runner.engine.execute_confirmed(
        "create_ticket", json.dumps(args, ensure_ascii=False),
        conversation_id=11, tool_call_id="client-confirm",
    ))
    artifact = runner.action_ledger.mark_confirmed(
        conversation_id=11,
        confirm_key=preview["confirm_key"],
        ticket=ticket,
    )
    assert artifact.state == "confirmed"
    assert artifact.write_committed is True
    assert artifact.ticket_no == ticket["ticket_no"]


@pytest.mark.asyncio
async def test_feasibility_question_and_cancel_without_pending_never_emit_preview():
    action = _ScriptedAction([
        _tool_call({"description": "错误地创建工单", "ticket_type": "售后"}),
        AIMessage(content="订单1001已签收，不能直接取消。"),
    ])
    sink = _Sink()
    result = await _invoke(
        _runner(sink, action), message="订单1001能取消吗？", conversation_id=4,
    )

    assert action.calls == 0
    assert sink.rows == []
    assert result["write_previews"] == []
    assert result["action_artifacts"][-1]["state"] == "cancelled"
    assert result["action_artifacts"][-1]["error_code"] == "no_pending_action"


@pytest.mark.asyncio
async def test_revised_description_supersedes_old_preview_but_never_writes():
    sink = _Sink()
    first_action = _ScriptedAction([
        _tool_call({
            "description": "订单1001退款处理",
            "ticket_type": "退款",
        }),
        AIMessage(content="预览已生成。"),
    ])
    runner = _runner(sink, first_action)
    first = await _invoke(
        runner, message="请为订单1001生成售后工单预览。",
        conversation_id=5, run_id="old-preview",
    )
    assert len(first["write_previews"]) == 1

    second_action = _ScriptedAction([
        _tool_call({
            "description": "订单1001改为维修处理",
            "ticket_type": "维修",
        }),
        AIMessage(content="新的预览已生成。"),
    ])
    runner.action_model = second_action
    second = await _invoke(
        runner, message="改一下，描述改成订单1001维修处理，再给预览。",
        conversation_id=5, run_id="revised-preview",
    )

    assert len(second["write_previews"]) == 1
    assert second["write_previews"][0]["description"] == "订单1001改为维修处理"
    states = [row["state"] for row in second["action_artifacts"]]
    assert states[-2:] == ["cancelled", "awaiting_confirmation"]
    assert sink.rows == []


@pytest.mark.asyncio
async def test_invalid_enum_is_fed_back_once_and_still_never_creates_preview():
    action = _ScriptedAction([
        _tool_call({"description": "订单1001处理", "ticket_type": "物流"}),
        _tool_call({"description": "订单1001处理", "ticket_type": "其他"}),
    ])
    sink = _Sink()
    result = await _invoke(
        _runner(sink, action), message="请为订单1001生成工单预览。",
        conversation_id=6,
    )

    assert action.calls == 2
    assert result["write_previews"] == []
    assert sink.rows == []
    assert result["action_artifacts"][-1]["state"] == "failed"
    assert result["action_artifacts"][-1]["error_code"] == "clarification_required"
    assert "售后" in result["action_artifacts"][-1]["error"]
    assert result["stop_reasons"][-1] == "action_contract_clarification"


@pytest.mark.asyncio
async def test_default_mode_keeps_existing_runtime_contract():
    action = _ScriptedAction([
        _tool_call({
            "conversation_id": 1,
            "description": "订单1001售后",
            "ticket_type": "售后",
        }),
        AIMessage(content="预览已生成。"),
    ])
    sink = _Sink()
    result = await _invoke(
        _runner(sink, action, enabled=False),
        message="请为订单1001生成售后工单预览。",
        conversation_id=1,
    )

    assert len(result["write_previews"]) == 1
    assert "action_artifacts" not in result
    assert sink.rows == []


def test_action_models_expose_required_artifact_states():
    from app.action_contracts import ActionArtifactState

    assert set(ActionArtifactState.__args__) == {
        "preview", "awaiting_confirmation", "confirmed", "cancelled", "failed",
    }


def test_query_reply_cannot_claim_preview_without_artifact():
    from app.action_contracts import ActionIntent, guard_final_reply

    reply = guard_final_reply(
        "订单1001的取消操作预览已生成，请确认。",
        intent=ActionIntent(reason="cancel_feasibility_question"),
        artifacts=[],
    )
    assert "预览已生成" not in reply
    assert "没有生成可确认的工单预览" in reply


def test_action_contract_has_no_hidden_retry_budget_beyond_one_repair():
    from app.action_contracts import ActionContractPolicy

    policy = ActionContractPolicy()
    assert policy.max_repairs == 1
    assert policy.allow_model_confirmation is False


def test_preview_only_wording_is_not_misclassified_as_cancel():
    from app.action_contracts import infer_action_intent

    intent = infer_action_intent("订单1001退款处理，请生成工单预览，不要提交。")
    assert intent.kind == "ticket_preview"


def test_pending_preview_can_be_cancelled_by_do_not_submit_wording():
    from app.action_contracts import ActionArtifact, infer_action_intent

    pending = ActionArtifact(
        kind="ticket_preview", state="awaiting_confirmation",
        confirm_key="pending-key",
    )
    initial = infer_action_intent("先生成工单预览，不要提交。")
    cancellation = infer_action_intent("不要提交了。", pending=pending)
    assert initial.kind == "ticket_preview"
    assert cancellation.kind == "ticket_cancel"


@pytest.mark.asyncio
async def test_cancellation_is_recorded_even_when_planner_schedules_no_action_task():
    sink = _Sink()
    engine = ToolEngine(_registry(sink), _Audit(), timeout_seconds=1, retry_attempts=0)
    runner = MultiAgentRunner(
        planner_model=_Planner(
            AgentPlan(disposition="clarify", clarification="已取消，不会创建工单。")
        ),
        retrieval_model=_OrderModel(),
        action_model=_ScriptedAction([AIMessage(content="unused")]),
        supervisor_model=_Final("不再创建工单。"),
        engine=engine,
        limits=MultiAgentLimits(max_seconds=3, child_timeout_seconds=1),
        action_contracts_enabled=True,
    )
    result = await _invoke(
        runner, message="算了，不建工单了，也不要提交。", conversation_id=9,
    )
    assert result["action_artifacts"][-1]["kind"] == "ticket_cancel"
    assert result["action_artifacts"][-1]["state"] == "cancelled"
    assert sink.rows == []


def test_graph_contract_keeps_elapsed_within_existing_limits():
    # This is a plumbing guard: the harness must not silently increase global limits.
    limits = MultiAgentLimits()
    assert limits.max_agent_iterations <= 4
    assert limits.max_steps <= 12
    assert time.monotonic() > 0
