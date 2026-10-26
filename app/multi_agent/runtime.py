"""LangGraph orchestration for the optional Supervisor/Planner + Agent loop."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from app.history import estimate_tokens
from app.evidence_contracts import (
    EvidenceLedger,
    TaskHandoff,
    ToolResultCache,
    claims_from_tool_result,
    subjects_from_arguments,
    subjects_from_text,
)
from app.multi_agent.models import (
    AgentOutcome,
    AgentPlan,
    AgentTask,
    MultiAgentLimits,
    MultiAgentState,
)
from app.multi_agent.roles import ROLE_SPECS


class RunLimitError(RuntimeError):
    pass


@dataclass
class _RunContext:
    run_id: str
    started_at: float
    deadline: float
    steps: int = 0
    tokens: int = 0
    agent_histories: dict[str, list[dict[str, Any]]] | None = None
    evidence_ledger: EvidenceLedger | None = None
    tool_cache: ToolResultCache | None = None

    def __post_init__(self) -> None:
        if self.agent_histories is None:
            self.agent_histories = {}
        if self.evidence_ledger is None:
            self.evidence_ledger = EvidenceLedger()
        if self.tool_cache is None:
            self.tool_cache = ToolResultCache()


@dataclass(frozen=True)
class _ToolSpec:
    name: str
    description: str = "multi-agent role tool"
    args_schema: dict[str, Any] | None = None


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _serialize_messages(messages: list[Any]) -> list[dict[str, Any]]:
    rows = []
    for message in messages:
        row = {"role": message.__class__.__name__, "content": message.content or ""}
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            row["tool_calls"] = tool_calls
        tool_call_id = getattr(message, "tool_call_id", None)
        if tool_call_id:
            row["tool_call_id"] = tool_call_id
        rows.append(row)
    return rows


class MultiAgentRunner:
    """Owns the role models and executes one isolated task at a time.

    The graph instance may serve multiple runs. Mutable counters and histories are
    keyed by run ID rather than stored on shared agent instances.
    """

    def __init__(self, *, planner_model, retrieval_model, action_model,
                 supervisor_model, engine, limits: MultiAgentLimits | None = None,
                 handoff_mode: Literal["raw", "typed"] = "raw",
                 dynamic_status_ttl_seconds: int = 300) -> None:
        self.planner_model = planner_model
        self.retrieval_model = retrieval_model
        self.action_model = action_model
        self.supervisor_model = supervisor_model
        self.engine = engine
        self.limits = limits or MultiAgentLimits()
        self.handoff_mode = handoff_mode
        self.dynamic_status_ttl_seconds = max(0, int(dynamic_status_ttl_seconds))
        self._runs: dict[str, _RunContext] = {}

    def _context(self, run_id: str) -> _RunContext:
        params = {
            "run_id": run_id,
            "deadline": time.monotonic() + self.limits.max_seconds,
            "started_at": time.monotonic(),
        }
        context = self._runs.get(run_id)
        if context is None:
            context = _RunContext(**params)
            if self.handoff_mode == "typed":
                context.tool_cache = ToolResultCache(
                    dynamic_ttl_seconds=self.dynamic_status_ttl_seconds)
            self._runs[run_id] = context
        return context

    def _claim_step(self, context: _RunContext) -> None:
        if time.monotonic() >= context.deadline:
            raise RunLimitError("global_timeout")
        if context.steps >= self.limits.max_steps:
            raise RunLimitError("global_step_limit")
        context.steps += 1

    def _count_tokens(self, context: _RunContext, text: str) -> None:
        context.tokens += estimate_tokens(text or "")
        if context.tokens > self.limits.max_tokens:
            raise RunLimitError("global_token_limit")

    def _tool_specs(self, agent_name: str) -> list[Any]:
        allowed = set(ROLE_SPECS[agent_name].allowed_tools)
        registry = getattr(self.engine, "registry", None)
        if registry is not None:
            records = {record.name: record for record in registry.records()}
            return [records[name].tool for name in ROLE_SPECS[agent_name].allowed_tools
                    if name in records]
        return [_ToolSpec(name) for name in ROLE_SPECS[agent_name].allowed_tools
                if name in allowed]

    @staticmethod
    def _fact_prompt(task: AgentTask, facts: dict[str, Any],
                     evidence: dict[str, Any]) -> str:
        visible_deps = {
            key: value for key, value in facts.items()
            if not task.depends_on or key in task.depends_on
        }
        return (
            f"task_id={task.task_id}\nagent_id={task.agent}:{task.task_id}\n"
            f"objective={task.objective}\ndepends_on={_json(task.depends_on)}\n"
            f"shared_facts={_json(visible_deps)}\n"
            f"evidence_ids={_json(list(evidence))}"
        )

    @staticmethod
    def _typed_prompt(task: AgentTask, handoff: TaskHandoff,
                      claims: list[Any]) -> str:
        projected_claims = []
        for claim in claims:
            payload = claim.model_dump(mode="json")
            payload["source"] = {
                key: value for key, value in payload["source"].items()
                if key != "raw_result"
            }
            projected_claims.append(payload)
        return _json({
            "task_id": task.task_id,
            "agent_id": f"{task.agent}:{task.task_id}",
            "objective": task.objective,
            "dependency_task_ids": task.depends_on,
            "subject_ids": handoff.subject_ids,
            "evidence_claims": projected_claims,
        })

    def _tool_is_read(self, name: str) -> bool:
        registry = getattr(self.engine, "registry", None)
        record = registry.get(name) if registry is not None else None
        if record is not None:
            return getattr(record, "access", "write") == "read"
        return name in {"query_order", "query_faq"}

    async def _model_call(self, model: Any, messages: list[Any], context: _RunContext,
                          child_deadline: float) -> Any:
        remaining = min(context.deadline, child_deadline) - time.monotonic()
        if remaining <= 0:
            raise asyncio.TimeoutError
        return await asyncio.wait_for(model.ainvoke(messages), timeout=remaining)

    async def run_task(self, *, agent_name: str, task: AgentTask,
                       facts: dict[str, Any], evidence: dict[str, Any],
                       run_id: str, deadline: float,
                       conversation_id: int | None = None,
                       context: _RunContext | None = None) -> AgentOutcome:
        context = context or self._context(run_id)
        child_deadline = min(deadline, time.monotonic() + self.limits.child_timeout_seconds)
        role = ROLE_SPECS[agent_name]
        model = self.retrieval_model if agent_name == "retrieval" else self.action_model
        observed_at = datetime.now(timezone.utc).isoformat()
        task_subjects = subjects_from_text(task.objective)
        handoff = TaskHandoff.role(
            run_id=run_id, from_agent="supervisor", to_agent=agent_name,
            task=task, subject_ids=task_subjects, observed_at=observed_at,
        )
        if self.handoff_mode == "typed":
            assert context.evidence_ledger is not None
            projected = context.evidence_ledger.project(task=task, handoff=handoff)
            handoff.evidence_claim_ids = [claim.claim_id for claim in projected]
            initial_prompt = self._typed_prompt(task, handoff, projected)
            system_prompt = (
                role.system_prompt
                + " typed_evidence 中只使用 source-backed evidence_claims；"
                  "unknown 字段不得当作已证实事实，superseded/过期证据不可复用。"
            )
        else:
            projected = []
            initial_prompt = self._fact_prompt(task, facts, evidence)
            system_prompt = role.system_prompt
        messages: list[Any] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=initial_prompt),
        ]
        tool_calls: list[dict[str, Any]] = []
        task_facts: dict[str, Any] = {}
        evidence_ids: list[str] = []
        preview: dict[str, Any] | None = None
        steps = 0
        start_tokens = context.tokens
        task_claims = []
        handoff_records: list[TaskHandoff] = [handoff]

        try:
            bound = model.bind_tools(self._tool_specs(agent_name))
            for _ in range(self.limits.max_agent_iterations):
                self._claim_step(context)
                steps += 1
                request_text = "\n".join(str(item.content or "") for item in messages)
                self._count_tokens(context, request_text)
                response = await self._model_call(bound, messages, context, child_deadline)
                self._count_tokens(context, str(getattr(response, "content", "") or ""))
                messages.append(response)
                calls = list(getattr(response, "tool_calls", None) or [])
                if not calls:
                    outcome = AgentOutcome(
                        task_id=task.task_id, agent=agent_name,
                        objective=task.objective, status="success",
                        draft=str(response.content or ""),
                        facts=task_facts or {"draft": str(response.content or "")},
                        evidence_ids=evidence_ids, tool_calls=tool_calls,
                        messages=_serialize_messages(messages), write_preview=preview,
                        stop_reason="agent_completed", steps=steps,
                        tokens=context.tokens - start_tokens,
                        evidence_claims=task_claims,
                        handoff=handoff,
                        handoff_records=handoff_records,
                    )
                    context.agent_histories[task.task_id] = outcome.messages
                    return outcome

                for index, call in enumerate(calls):
                    name = str(call.get("name") or "")
                    if name not in role.allowed_tools:
                        raise PermissionError(
                            f"工具权限拒绝:{agent_name} 不可调用 {name or '<empty>'}"
                        )
                    args = dict(call.get("args") or {})
                    registry = getattr(self.engine, "registry", None)
                    record = registry.get(name) if registry is not None else None
                    properties = (
                        (getattr(record, "json_schema", None) or {}).get("properties") or {}
                    )
                    if conversation_id is not None and "conversation_id" in properties:
                        args["conversation_id"] = conversation_id
                    tool_calls.append({"name": name, "arguments": args})

                    def intercept(record, values):
                        nonlocal preview
                        preview = {
                            "type": "ticket_preview",
                            "tool_name": getattr(record, "name", name),
                            "conversation_id": values.get("conversation_id"),
                            "description": values.get("description", ""),
                            "ticket_type": values.get("ticket_type", "售后"),
                            "arguments": dict(values),
                            "confirm_key": uuid.uuid4().hex,
                            "task_id": task.task_id,
                            "agent_id": f"{agent_name}:{task.task_id}",
                        }

                    remaining = min(context.deadline, child_deadline) - time.monotonic()
                    if remaining <= 0:
                        raise asyncio.TimeoutError
                    tool_call_id = str(call.get("id") or f"{task.task_id}-{index}")
                    call_observed_at = datetime.now(timezone.utc).isoformat()
                    handoff_records.append(TaskHandoff.tool(
                        run_id=run_id, from_agent=f"{agent_name}:{task.task_id}",
                        task_id=task.task_id, tool_name=name,
                        observed_at=call_observed_at,
                    ))

                    async def execute_tool():
                        return await asyncio.wait_for(
                            self.engine.execute(
                                name, _json(args), conversation_id=None,
                                tool_call_id=tool_call_id,
                                on_write_intercept=intercept,
                            ),
                            timeout=remaining,
                        )

                    if self.handoff_mode == "typed" and context.tool_cache is not None:
                        cached = await context.tool_cache.execute(
                            name, args, read=self._tool_is_read(name),
                            execute=execute_tool,
                            scope={
                                key: args[key] for key in (
                                    "as_of", "valid_at", "knowledge_as_of",
                                    "policy_version", "scope",
                                ) if key in args
                            },
                            ttl_seconds=self.dynamic_status_ttl_seconds,
                        )
                        result = cached.result
                    else:
                        result = await execute_tool()
                    self._count_tokens(context, str(result))
                    if str(result).startswith("错误:"):
                        raise RuntimeError(str(result))
                    messages.append(ToolMessage(
                        content=str(result),
                        tool_call_id=str(call.get("id") or f"{task.task_id}-{index}"),
                    ))
                    try:
                        parsed = json.loads(result)
                    except (TypeError, json.JSONDecodeError):
                        parsed = {"text": str(result)}
                    if isinstance(parsed, dict):
                        task_facts.update(parsed)
                        if self.handoff_mode == "typed":
                            assert context.evidence_ledger is not None
                            new_claims = claims_from_tool_result(
                                name, args, str(result), run_id=run_id,
                                task_id=task.task_id, tool_call_id=tool_call_id,
                                observed_at=call_observed_at,
                                dynamic_ttl_seconds=self.dynamic_status_ttl_seconds,
                            )
                            context.evidence_ledger.add_many(new_claims)
                            task_claims.extend(new_claims)
                        for item in parsed.get("items") or []:
                            if isinstance(item, dict) and item.get("chunk_id") is not None:
                                chunk_id = str(item["chunk_id"])
                                if chunk_id not in evidence_ids:
                                    evidence_ids.append(chunk_id)
            if task_facts:
                outcome = AgentOutcome(
                    task_id=task.task_id, agent=agent_name, objective=task.objective,
                    status="success", draft="已获取工具事实，达到单Agent迭代上限。",
                    facts=task_facts, evidence_ids=evidence_ids, tool_calls=tool_calls,
                    messages=_serialize_messages(messages), write_preview=preview,
                    stop_reason="agent_iteration_limit_with_facts", steps=steps,
                    tokens=context.tokens - start_tokens,
                    evidence_claims=task_claims,
                    handoff=handoff,
                    handoff_records=handoff_records,
                )
                context.agent_histories[task.task_id] = outcome.messages
                return outcome
            raise RunLimitError("agent_iterations_exhausted")
        except asyncio.TimeoutError:
            error, status, stop = "child_timeout", "timeout", "timeout"
        except PermissionError as exc:
            error, status, stop = str(exc), "failed", "tool_permission_denied"
        except RunLimitError as exc:
            error, status, stop = str(exc), "failed", "global_limit"
        except Exception as exc:
            error, status, stop = f"{type(exc).__name__}:{exc}", "failed", "agent_error"

        context.agent_histories[task.task_id] = _serialize_messages(messages)
        return AgentOutcome(
            task_id=task.task_id, agent=agent_name, objective=task.objective,
            status=status, facts=task_facts, evidence_ids=evidence_ids,
            tool_calls=tool_calls, messages=_serialize_messages(messages),
            write_preview=preview, error=error, stop_reason=stop, steps=steps,
            tokens=context.tokens - start_tokens,
            evidence_claims=task_claims,
            handoff=handoff,
            handoff_records=handoff_records,
        )

    async def plan(self, *, run_id: str, message: str,
                   conversation_history: list[dict[str, Any]],
                   failure_context: str = "") -> AgentPlan:
        context = self._context(run_id)
        self._claim_step(context)
        prompt = {
            "history": conversation_history,
            "current_message": message,
            "failure_context": failure_context,
            "max_tasks": min(self.limits.max_tasks, 3),
        }
        messages = [
            SystemMessage(content=ROLE_SPECS["planner"].system_prompt),
            HumanMessage(content=_json(prompt)),
        ]
        self._count_tokens(context, _json(prompt))
        structured = self.planner_model.with_structured_output(
            AgentPlan, method="function_calling"
        )
        response = await self._model_call(
            structured, messages, context,
            min(context.deadline, time.monotonic() + self.limits.child_timeout_seconds),
        )
        self._count_tokens(context, _json(response))
        plan = response if isinstance(response, AgentPlan) else AgentPlan.model_validate(response)
        if len(plan.tasks) > self.limits.max_tasks:
            raise ValueError("planner exceeded configured task limit")
        return plan

    async def execute_tasks(self, *, run_id: str, plan: AgentPlan,
                            facts: dict[str, Any], evidence: dict[str, Any],
                            deadline: float,
                            conversation_id: int | None = None) -> dict[str, Any]:
        context = self._context(run_id)
        pending = {task.task_id: task for task in plan.tasks if task.task_id not in facts}
        completed = set(facts)
        errors: list[str] = []
        task_agents: dict[str, str] = {}
        tool_calls: list[dict[str, Any]] = []
        previews: list[dict[str, Any]] = []
        trace: list[dict[str, Any]] = []
        handoffs: list[TaskHandoff] = []
        evidence_claims: list[Any] = []

        while pending:
            ready = [
                task for task in pending.values()
                if set(task.depends_on).issubset(completed)
            ]
            if not ready:
                errors.append("dependency_cycle_or_missing_evidence")
                break
            retrieval = [task for task in ready if task.agent == "retrieval"]
            wave = retrieval or [min(ready, key=lambda item: item.task_id)]
            for task in wave:
                trace.append({
                    "event": "task_start", "run_id": run_id, "task_id": task.task_id,
                    "agent_id": f"{task.agent}:{task.task_id}",
                })

            async def execute_one(task: AgentTask) -> AgentOutcome:
                return await self.run_task(
                    agent_name=task.agent, task=task, facts=dict(facts),
                    evidence=dict(evidence), run_id=run_id, deadline=deadline,
                    conversation_id=conversation_id,
                    context=context,
                )

            if len(wave) > 1:
                outcomes = list(await asyncio.gather(*(execute_one(task) for task in wave)))
            else:
                outcomes = [await execute_one(wave[0])]

            for outcome in outcomes:
                task = pending.pop(outcome.task_id)
                task_agents[outcome.task_id] = task.agent
                tool_calls.extend(outcome.tool_calls)
                handoffs.extend(outcome.handoff_records)
                evidence_claims.extend(outcome.evidence_claims)
                if outcome.write_preview:
                    previews.append(outcome.write_preview)
                if outcome.status != "success":
                    error = f"{outcome.task_id}:{outcome.error}:{outcome.stop_reason}"
                    errors.append(error)
                    trace.append({
                        "event": "task_failed", "run_id": run_id,
                        "task_id": outcome.task_id,
                        "agent_id": f"{outcome.agent}:{outcome.task_id}",
                        "detail": error,
                    })
                    continue
                facts[outcome.task_id] = outcome.facts
                completed.add(outcome.task_id)
                for evidence_id in outcome.evidence_ids:
                    evidence[evidence_id] = {"chunk_id": evidence_id}
                trace.append({
                    "event": "task_complete", "run_id": run_id,
                    "task_id": outcome.task_id,
                    "agent_id": f"{outcome.agent}:{outcome.task_id}",
                })
                trace.append({
                    "event": "role_handoff", "run_id": run_id,
                    "task_id": outcome.task_id,
                    "agent_id": f"supervisor->{outcome.agent}:{outcome.task_id}",
                    "detail": _json({
                        "evidence_claim_ids": (
                            outcome.handoff.evidence_claim_ids
                            if outcome.handoff else []
                        ),
                        "dependency_task_ids": task.depends_on,
                    }),
                })
            if errors:
                break
        return {
            "facts": facts, "evidence": evidence, "errors": errors,
            "stop_reasons": [item.split(":")[-1] for item in errors],
            "trace": trace, "task_agents": task_agents,
            "tool_calls": tool_calls, "write_previews": previews,
            "role_histories": dict(context.agent_histories or {}),
            "handoffs": [item.model_dump(mode="json") for item in handoffs],
            "evidence_claims": [item.model_dump(mode="json")
                                for item in evidence_claims],
            "tool_cache_stats": (
                context.tool_cache.stats.as_dict()
                if context.tool_cache is not None else {}
            ),
            "steps": context.steps, "tokens": context.tokens,
        }

    async def finalize(self, *, run_id: str, message: str, plan: AgentPlan,
                       facts: dict[str, Any], evidence: dict[str, Any],
                       errors: list[str]) -> str:
        if plan.disposition == "clarify":
            return plan.clarification
        if plan.disposition == "refuse":
            return "【无法回答】" + (plan.reason or "当前请求超出可提供的店铺服务范围。")
        if plan.disposition == "handoff":
            return "已为您转人工客服。" + (plan.reason or "")

        context = self._context(run_id)
        self._claim_step(context)
        if self.handoff_mode == "typed":
            assert context.evidence_ledger is not None
            task_ids = [task.task_id for task in plan.tasks]
            subjects = subjects_from_text(message)
            claims = context.evidence_ledger.project_supervisor(
                run_id=run_id, task_ids=task_ids, subject_ids=subjects)
            prompt = _json({
                "user_message": message,
                "dependency_task_ids": task_ids,
                "subject_ids": subjects,
                "evidence_claims": [claim.model_dump(mode="json") for claim in claims],
                "plan_reason": plan.reason,
                "errors": errors,
            })
        else:
            prompt = (
                f"user_message={message}\nshared_facts={_json(facts)}\n"
                f"evidence={_json(evidence)}\nplan_reason={plan.reason}\n"
                f"errors={_json(errors)}"
            )
        messages = [
            SystemMessage(content=ROLE_SPECS["supervisor"].system_prompt),
            HumanMessage(content=prompt),
        ]
        self._count_tokens(context, prompt)
        response = await self._model_call(
            self.supervisor_model, messages, context,
            min(context.deadline, time.monotonic() + self.limits.child_timeout_seconds),
        )
        text = str(getattr(response, "content", response) or "")
        self._count_tokens(context, text)
        return text


def build_multi_agent_graph(runner: MultiAgentRunner):
    """Compile the real LangGraph control flow used by tests and the optional entry."""

    async def supervisor_plan(state: dict) -> dict:
        run_id = state["run_id"]
        context = runner._context(run_id)
        errors: list[str] = []
        try:
            plan = await runner.plan(
                run_id=run_id,
                message=state["user_message"],
                conversation_history=state.get("conversation_history") or [],
            )
        except Exception as exc:
            error = f"planner:{type(exc).__name__}:{exc}"
            errors.append(error)
            plan = AgentPlan(
                disposition="refuse",
                reason="多Agent规划暂时不可用，请稍后重试或转人工。",
            )
        return {
            "plan": plan.model_dump(), "disposition": plan.disposition,
            "clarification": plan.clarification, "replan_count": 0,
            "needs_replan": False, "errors": errors, "started_at": context.started_at,
            "deadline": context.deadline, "steps": context.steps,
            "tokens": context.tokens, "trace": [{
                "event": "plan", "run_id": run_id,
                "agent_id": "planner:supervisor",
                "detail": f"{plan.disposition}:{len(plan.tasks)}",
            }],
        }

    async def execute_tasks(state: dict) -> dict:
        plan = AgentPlan.model_validate(state["plan"])
        if plan.disposition != "execute":
            return {"needs_replan": False}
        updates = await runner.execute_tasks(
            run_id=state["run_id"], plan=plan,
            facts=dict(state.get("facts") or {}),
            evidence=dict(state.get("evidence") or {}),
            deadline=float(state.get("deadline") or time.monotonic() + runner.limits.max_seconds),
            conversation_id=state.get("conversation_id"),
        )
        updates["needs_replan"] = bool(updates["errors"]) and \
            int(state.get("replan_count", 0)) < runner.limits.max_replans
        return updates

    async def supervisor_replan(state: dict) -> dict:
        errors = list(state.get("errors") or [])
        try:
            plan = await runner.plan(
                run_id=state["run_id"], message=state["user_message"],
                conversation_history=state.get("conversation_history") or [],
                failure_context=_json(errors),
            )
        except Exception as exc:
            errors.append(f"replanner:{type(exc).__name__}:{exc}")
            plan = AgentPlan(
                disposition="refuse",
                reason="多Agent重新规划失败，请稍后重试或转人工。",
            )
        return {
            "plan": plan.model_dump(), "disposition": plan.disposition,
            "clarification": plan.clarification,
            "replan_count": int(state.get("replan_count", 0)) + 1,
            "needs_replan": False,
            "errors": errors,
            "trace": [{
                "event": "replan", "run_id": state["run_id"],
                "agent_id": "planner:supervisor",
                "detail": f"{plan.disposition}:{len(plan.tasks)}",
            }],
        }

    async def supervisor_finalize(state: dict) -> dict:
        plan = AgentPlan.model_validate(state["plan"])
        errors = list(state.get("errors") or [])
        try:
            final = await runner.finalize(
                run_id=state["run_id"], message=state["user_message"], plan=plan,
                facts=dict(state.get("facts") or {}),
                evidence=dict(state.get("evidence") or {}),
                errors=errors,
            )
        except Exception as exc:
            errors.append(f"finalize:{type(exc).__name__}:{exc}")
            final = "【无法回答】多Agent汇总未完成，请稍后重试或转人工。"
        context = runner._context(state["run_id"])
        runner._runs.pop(state["run_id"], None)
        updates = {
            "final_reply": final,
            "status": "failed" if errors else "success",
            "errors": errors,
            "steps": context.steps, "tokens": context.tokens,
            "role_histories": dict(context.agent_histories or {}),
            "trace": [{
                "event": "finalize", "run_id": state["run_id"],
                "agent_id": "supervisor",
                "detail": "failed" if errors else "success",
            }],
        }
        if plan.disposition == "handoff":
            updates["handoffs"] = [TaskHandoff.human(
                run_id=state["run_id"], from_agent="planner:supervisor",
                reason=plan.reason,
                observed_at=datetime.now(timezone.utc).isoformat(),
            ).model_dump(mode="json")]
        return updates

    graph = StateGraph(MultiAgentState)
    graph.add_node("supervisor_plan", supervisor_plan)
    graph.add_node("execute_tasks", execute_tasks)
    graph.add_node("supervisor_replan", supervisor_replan)
    graph.add_node("supervisor_finalize", supervisor_finalize)
    graph.add_edge(START, "supervisor_plan")
    graph.add_edge("supervisor_plan", "execute_tasks")
    graph.add_conditional_edges(
        "execute_tasks",
        lambda state: "replan" if state.get("needs_replan") else "finalize",
        {"replan": "supervisor_replan", "finalize": "supervisor_finalize"},
    )
    graph.add_edge("supervisor_replan", "execute_tasks")
    graph.add_edge("supervisor_finalize", END)
    return graph.compile()


def build_multi_agent_runtime(*, engine, model,
                              limits: MultiAgentLimits | None = None,
                              handoff_mode: Literal["raw", "typed"] = "raw",
                              dynamic_status_ttl_seconds: int = 300) -> MultiAgentRunner:
    """Production wiring helper: one approved chat model, four explicit role labels."""

    return MultiAgentRunner(
        planner_model=model,
        retrieval_model=model,
        action_model=model,
        supervisor_model=model,
        engine=engine,
        limits=limits,
        handoff_mode=handoff_mode,
        dynamic_status_ttl_seconds=dynamic_status_ttl_seconds,
    )
