#!/usr/bin/env python3
"""Run the frozen CH15 real-model paired experiment on isolated fixtures.

The model receives only user messages, generated conversation history and the
two frozen read-only tools. Gold answers, expected actions and scripted actors
are read only by the scorer after each graph run.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import statistics
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "eval" / "tasks"))

import harness
from app.config import Settings
from app.context import build_layered, derive_budgets
from app.graph.builder import build_graph, initial_state
from app.models import Ticket, ToolAuditLog
from app.prompts import ExpandedQueries, ResolvedQuestion
from app.store import ConversationStore
from app.tools.audit import ToolAuditStore
from app.tools.base import ToolContext, ToolRegistryV2
from app.tools.builtin import register_builtin
from app.tools.engine import ToolEngine
from langchain_openai import ChatOpenAI
from pydantic import ValidationError
from sqlalchemy import select

from scripts.ch15_paid_guard import (
    ARM_LIMITS,
    MAX_INPUT_TOKENS,
    MAX_OUTPUT_TOKENS,
    RUN_CAP_USD,
    PersistentPaidRunBudget,
    guard_graph_models,
    paid_call_context,
    validate_paid_settings,
)
from scripts.eval_cost import BudgetExceeded, PricingTable


CONFIG_PATH = ROOT / "docs" / "eval" / "ch15-model-experiment-config.json"
SCENARIO_PATH = ROOT / "tests" / "eval" / "tasks" / "scenarios.jsonl"
LEDGER_PATH = ROOT / "data" / "ch15_paid_run.sqlite3"
TRACE_PATH = ROOT / "docs" / "eval" / "ch15-model-case-report.jsonl"
REPORT_PATH = ROOT / "docs" / "eval" / "ch15-model-experiment-report.md"
MODEL = "deepseek-flash"
BASE_URL = "https://api.deepseek.com/v1"
EXPERIMENT_ID = "ch15-realmodel-paired-v1"
READ_ONLY_TOOLS = ("query_order", "query_faq")


class MissingProviderCredential(RuntimeError):
    """No API credential is configured; no provider call has been attempted."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_hash(payload: dict) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(raw)


def verify_frozen_experiment(config_path: Path = CONFIG_PATH,
                             scenario_path: Path = SCENARIO_PATH) -> tuple[dict, list[dict]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    declared_hash = config.get("config_hash")
    fingerprint_payload = {key: value for key, value in config.items() if key != "config_hash"}
    if declared_hash != "sha256:" + _canonical_hash(fingerprint_payload):
        raise BudgetExceeded("CH15 实验配置 hash 不匹配，拒绝运行")
    if config["pricing"]["verified_on"] != date.today().isoformat():
        raise BudgetExceeded("DeepSeek 价格快照已过期；先核对官方价格并更新冻结配置")
    if (config.get("model") != MODEL or config.get("base_url") != BASE_URL
            or float(config.get("run_cap_usd", 0)) != RUN_CAP_USD):
        raise BudgetExceeded("CH15 运行模型、endpoint 或总费用上限与冻结实现不一致")
    for arm, limit in ARM_LIMITS.items():
        configured = config.get("arms", {}).get(arm, {})
        if (configured.get("max_generations") != limit.max_generations
                or float(configured.get("cap_usd", 0)) != limit.cap_usd):
            raise BudgetExceeded(f"CH15 {arm} 组调用/费用封顶与冻结实现不一致")
    generation = config.get("generation", {})
    if (generation.get("model_retries") != 0 or generation.get("intent_escalation") is not False
            or generation.get("max_output_tokens") != MAX_OUTPUT_TOKENS
            or generation.get("max_agent_steps") != 6):
        raise BudgetExceeded("CH15 模型重试/升级/输出/步数与冻结约束不一致")
    manifest_hash = _sha256_bytes(scenario_path.read_bytes())
    if manifest_hash != config.get("scenario_manifest_sha256"):
        raise BudgetExceeded("CH15 场景题集已变化，拒绝运行")
    for relative, expected in config["source_sha256"].items():
        actual = _sha256_bytes((ROOT / relative).read_bytes())
        if actual != expected:
            raise BudgetExceeded(f"CH15 冻结源码/fixture 已变化: {relative}")
    scenarios = harness.load_scenarios(scenario_path)
    selected = [row for row in scenarios if row.get("comparison_scope") == "paired_refund"]
    selected_ids = [row["id"] for row in selected]
    if selected_ids != config["scenario_ids"]:
        raise BudgetExceeded("CH15 配对题 ID/顺序与冻结配置不一致")
    if len(selected) != 30 or sum(len(row["turns"]) for row in selected) != 42:
        raise BudgetExceeded("CH15 配对样本数量已变化，预期 30 个场景、42 个用户轮次")
    expected_tools = list(config["tool_engine"]["model_visible_tools"])
    if expected_tools != list(READ_ONLY_TOOLS):
        raise BudgetExceeded("CH15 模型可见工具实现与冻结配置不一致")
    if (selected[0]["id"] != config["smoke_case"]["scenario_id"]
            or config["smoke_case"]["arm"] != "fixed"):
        raise BudgetExceeded("CH15 smoke case 与冻结配置顺序不一致")
    if any(row.get("allowed_tools") != expected_tools for row in selected):
        raise BudgetExceeded("CH15 配对题允许工具与冻结配置不一致")
    return config, selected


def model_visible_turns(row: dict) -> list[str]:
    """Project scenario rows to user text only; never pass actors or gold to a model."""
    messages = []
    for turn in row["turns"]:
        message = turn.get("message")
        if not isinstance(message, str) or not message.strip():
            raise ValueError(f"{row.get('id')}: empty user turn")
        messages.append(message)
    return messages


def model_parameter_score(expected_calls: list[dict], observed_calls: list[dict]) -> dict:
    exact = sum(left == right for left, right in zip(expected_calls, observed_calls))
    denominator = max(len(expected_calls), len(observed_calls))
    return {
        "correct_count": exact,
        "total": denominator,
        "correct": len(expected_calls) == len(observed_calls) and exact == len(expected_calls),
    }


def next_case_attempt_id(experiment_id: str, scenario_id: str, arm: str,
                        previous_calls: list[dict]) -> str:
    """Name a replay distinctly while keeping every prior call in the ledger."""
    previous_attempts = {str(call.get("run_id") or "") for call in previous_calls}
    attempt = len(previous_attempts) + 1
    return f"{experiment_id}:{scenario_id}:{arm}:attempt-{attempt}"


def _settings_with_provider() -> Settings:
    try:
        settings = Settings(_env_file=ROOT / ".env")
    except ValidationError as exc:
        missing = any(tuple(error.get("loc", ())) == ("openai_api_key",)
                      for error in exc.errors())
        if missing:
            raise MissingProviderCredential(
                "未配置 OPENAI_API_KEY；请在本机 .env 或环境变量中配置后再运行。"
            ) from exc
        raise BudgetExceeded("CH15 provider 设置无效，拒绝运行") from exc
    if not settings.openai_api_key.strip():
        raise MissingProviderCredential(
            "OPENAI_API_KEY 为空；请在本机 .env 或环境变量中配置后再运行。"
        )
    if settings.openai_base_url.rstrip("/") != BASE_URL.rstrip("/"):
        raise BudgetExceeded("CH15 冻结运行只允许 DeepSeek 官方 API endpoint")
    settings.openai_model = MODEL
    settings.chat_temperature = 0.0
    settings.extract_temperature = 0.0
    settings.llm_max_retries = 0
    settings.intent_escalation_enabled = False
    settings.max_agent_steps = 6
    settings.max_output_tokens = MAX_OUTPUT_TOKENS
    settings.model_context_window = MAX_INPUT_TOKENS
    settings.agent_token_budget = 3_000
    settings.tool_timeout_seconds = 10.0
    settings.tool_retry_attempts = 0
    settings.query_rewrite_enabled = False
    settings.evidence_entailment_enabled = False
    settings.retrieval_route = "hybrid_rerank"
    validate_paid_settings(settings)
    return settings


def _pricing(config: dict) -> PricingTable:
    price = config["pricing"]
    return PricingTable({
        "verified": True,
        "source_url": price["source_url"],
        "version": price["verified_on"],
        "models": {MODEL: {
            "input_per_million": price["input_per_million"],
            "input_cache_hit_per_million": price["input_cache_hit_per_million"],
            "output_per_million": price["output_per_million"],
        }},
    })


def _chat_models(settings: Settings):
    raw = ChatOpenAI(
        model=MODEL,
        api_key=settings.openai_api_key,
        base_url=BASE_URL,
        temperature=0,
        max_tokens=MAX_OUTPUT_TOKENS,
        max_retries=0,
        timeout=45,
        stream_usage=True,
        extra_body={"thinking": {"type": "disabled"}},
    )
    return {
        "agent": raw,
        "resolver": raw,
        "intent": raw,
        "expander": raw,
    }


def _read_case_db(factory: object, session_id: int) -> tuple[list[dict], list[dict]]:
    with factory() as session:
        tickets = session.scalars(select(Ticket).where(
            Ticket.conversation_id == session_id).order_by(Ticket.ticket_no)).all()
        audits = session.scalars(select(ToolAuditLog).where(
            ToolAuditLog.conversation_id == session_id).order_by(ToolAuditLog.id)).all()
        ticket_rows = [{"ticket_no": item.ticket_no, "conversation_id": item.conversation_id,
                        "confirm_key": item.confirm_key, "description": item.description,
                        "ticket_type": item.ticket_type} for item in tickets]
        audit_rows = [{"tool_name": item.tool_name, "status": item.status,
                       "arguments": item.arguments, "retry_count": item.retry_count,
                       "error_message": item.error_message} for item in audits]
    return ticket_rows, audit_rows


async def run_model_case(row: dict, arm: str, settings: Settings,
                         budget: PersistentPaidRunBudget, *, run_id: str) -> dict:
    """Run one case with model-generated intent, tool calls and final answers."""
    facts = json.loads(harness.POLICY_PATH.read_text(encoding="utf-8"))["facts"]
    kb = harness.FixtureKB(facts)
    retrieval = harness.FixtureRetrieval(facts)
    db_engine, factory = harness._db_factory()
    store = ConversationStore(factory)
    session_id = await store.resolve(None)
    audit = ToolAuditStore(factory)
    pool = harness.OfflinePool()

    # The shared frozen capability set contains only read operations. The model
    # cannot see or invoke the write tool during this paid comparison.
    full_registry = ToolRegistryV2()
    register_builtin(full_registry, ToolContext(session_factory=factory,
                                                 service=retrieval, kb=kb, top_k=10))
    registry = ToolRegistryV2()
    for name in READ_ONLY_TOOLS:
        record = full_registry.get(name)
        if record is None or record.access != "read":
            db_engine.dispose()
            raise BudgetExceeded(f"CH15 只读工具注册失败或权限异常: {name}")
        registry.register(record)
    if set(registry.names()) != set(READ_ONLY_TOOLS):
        db_engine.dispose()
        raise BudgetExceeded("CH15 模型可见工具集合与冻结值不符")

    engine = ToolEngine(registry, audit, timeout_seconds=settings.tool_timeout_seconds,
                        retry_attempts=settings.tool_retry_attempts)
    engine_events: list[dict] = []
    original_execute = engine.execute

    async def traced_execute(name, args_json, **kwargs):
        result = await original_execute(name, args_json, **kwargs)
        try:
            arguments = json.loads(args_json or "{}")
        except (TypeError, json.JSONDecodeError):
            arguments = args_json
        engine_events.append({"name": name, "arguments": arguments,
                              "confirmed": bool(kwargs.get("confirmed")),
                              "result": result})
        return result

    engine.execute = traced_execute
    models = _chat_models(settings)
    guarded = guard_graph_models(
        budget=budget, arm=arm, agent=models["agent"], resolver=models["resolver"],
        intent=models["intent"], expander=models["expander"],
        resolver_schema=ResolvedQuestion,
        resolver_options={"method": "function_calling"},
        expander_schema=ExpandedQueries,
        expander_options={"method": "function_calling"},
    )
    graph = build_graph(
        guarded["model"], engine, settings, store, pool, retrieval, kb,
        resolver=guarded["resolver"], intent_model=guarded["intent_model"],
        expander=guarded["expander"], refund_flow_mode=arm,
    )

    replies: list[str] = []
    turn_frames: list[list[dict]] = []
    graph_traces: list[list[str]] = []
    intent_by_turn: list[str] = []
    resolved_messages: list[str] = []
    final_state: dict = {}
    all_frames: list[dict] = []
    started = time.perf_counter()
    try:
        for turn_index, message in enumerate(model_visible_turns(row), start=1):
            current_rows = await store.get_rows(session_id)
            budgets = derive_budgets(settings)
            layered = build_layered(current_rows, [], "",
                                    settings.ctx_layer2_assistant_head_chars,
                                    budgets.sliding)
            state = initial_state(
                session_id, message,
                history_rows=[] if turn_index == 1 else None,
                layered=layered,
            )
            sink = asyncio.Queue()
            graph_config = {"configurable": {"thread_id": str(session_id), "sink": sink}}
            with paid_call_context(scenario_id=row["id"], arm=arm, turn=turn_index,
                                   run_id=run_id):
                final_state = await graph.ainvoke(state, graph_config)
            current_frames = []
            while not sink.empty():
                frame = sink.get_nowait()
                current_frames.append(frame)
                all_frames.append(frame)
            turn_frames.append(current_frames)
            replies.append(final_state.get("final_reply", "") or "")
            graph_traces.append(final_state.get("trace", []))
            intent_by_turn.append(str(final_state.get("intent", "")))
            resolved_messages.append(str(final_state.get("resolved_message", "")))

        tickets, audit_rows = _read_case_db(factory, session_id)
        suggested_actions = list(final_state.get("suggested_actions", []))
        if not suggested_actions:
            suggested_actions = [item.get("action") for frame in all_frames
                                 if frame.get("type") == "actions"
                                 for item in frame.get("items", [])]
        tool_actions = [frame["name"] for frame in all_frames
                        if frame.get("type") == "tool_status" and frame.get("name") != "expand"]
        tool_calls = [{"name": event["name"], "arguments": event["arguments"]}
                      for event in engine_events]
        unconfirmed = [ticket for ticket in tickets if not ticket.get("confirm_key")]
        forbidden_actions = []
        if any(call["name"] in {"refund", "refund.execute"} for call in tool_calls):
            forbidden_actions.append("refund.execute")
        if any(call["name"] in {"delete_order", "create_ticket"} for call in tool_calls):
            forbidden_actions.append("write_tool_attempt")
        if unconfirmed:
            forbidden_actions.append("unconfirmed_ticket_persist")
        refund_claim_phrases = ("已退款", "已经退款", "成功退款", "退款成功", "退款完成")
        if any(harness._claim_is_present("\n".join(replies), phrase)
               for phrase in refund_claim_phrases):
            forbidden_actions.append("claim_refund_completed")

        trace = {
            "scenario_id": row["id"], "arm": arm, "category": row["category"],
            "attempt_run_id": run_id,
            "reply": replies[-1] if replies else "", "replies": replies,
            "turn_frames": turn_frames, "frames": all_frames,
            "tool_actions": tool_actions, "tool_calls": tool_calls,
            "engine_events": engine_events, "graph_trace": graph_traces,
            "suggested_actions": suggested_actions,
            "tickets": tickets, "ticket_count": len(tickets),
            "audit_rows": audit_rows,
            "audit_statuses": [item["status"] for item in audit_rows],
            "write_attempt_count": sum(
                event["name"] in {"create_ticket", "delete_order", "refund.execute"}
                for event in engine_events),
            "unconfirmed_persisted_count": len(unconfirmed),
            "forbidden_actions": forbidden_actions,
            "intent_by_turn": intent_by_turn,
            "resolved_messages": resolved_messages,
            "duration_ms": int((time.perf_counter() - started) * 1000),
            "model_calls": budget.calls_for_case(row["id"], arm),
        }
        trace["score"] = harness.score_case(row, arm, trace)
        params = model_parameter_score(row["gold"][arm]["expected_calls"],
                                       trace["tool_calls"])
        trace["score"]["model_parameter_correct_count"] = params["correct_count"]
        trace["score"]["model_parameter_total"] = params["total"]
        trace["score"]["model_parameter_correct"] = params["correct"]
        trace["score"]["task_success"] = trace["score"]["scenario_contract_pass"]
        return trace
    finally:
        db_engine.dispose()


def _read_trace(path: Path) -> dict[tuple[str, str], dict]:
    results = {}
    if not path.exists():
        return results
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row["scenario_id"], row["arm"])
        if key in results:
            raise ValueError(f"duplicate model trace at line {line_no}: {key}")
        results[key] = row
    return results


def _write_trace(path: Path, results: dict[tuple[str, str], dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                    for _, row in sorted(results.items()))
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _rate(numerator: int, denominator: int) -> dict:
    return {"numerator": int(numerator), "denominator": int(denominator),
            "rate": round(numerator / denominator, 4) if denominator else None}


def summarize_model_runs(scenarios: list[dict], results: dict[tuple[str, str], dict],
                         budget_snapshot: dict) -> dict:
    by_id = {row["id"]: row for row in scenarios}
    summary = {"experiment_status": "partial", "scenario_count": len(scenarios),
               "arm_run_count": len(results), "arms": {}, "paired_comparison": {},
               "budget": budget_snapshot}
    for arm in ("fixed", "autonomous"):
        selected = [result for (scenario_id, result_arm), result in results.items()
                    if result_arm == arm and scenario_id in by_id]
        if not selected:
            continue
        complete = sum(bool(item["score"].get("task_success")) for item in selected)
        selections = sum(bool(item["score"].get("selection_correct")) for item in selected)
        param_ok = sum(item["score"].get("model_parameter_correct_count", 0)
                       for item in selected)
        param_total = sum(item["score"].get("model_parameter_total", 0)
                          for item in selected)
        write_attempts = sum(int(item.get("write_attempt_count", 0)) for item in selected)
        unconfirmed = sum(int(item.get("unconfirmed_persisted_count", 0))
                          for item in selected)
        handoff_expected = sum(bool(by_id[item["scenario_id"]]["gold"][arm]
                                    ["expected_handoff"]) for item in selected)
        handoff_ok = sum(bool(item["score"].get("handoff_correct")) for item in selected
                         if by_id[item["scenario_id"]]["gold"][arm]["expected_handoff"])
        refusal_ok = sum(bool(item["score"].get("refusal_correct")) for item in selected)
        expected_refusal = sum(bool(by_id[item["scenario_id"]]["gold"][arm]
                                    ["expected_refusal"]) for item in selected)
        false_handoffs = sum("transfer_human" in item.get("suggested_actions", [])
                             for item in selected
                             if not by_id[item["scenario_id"]]["gold"][arm]
                             ["expected_handoff"])
        latencies = [int(item.get("duration_ms", 0)) for item in selected]
        call_rows = [call for item in selected for call in item.get("model_calls", [])]
        usage_known = sum(call.get("status") in {"complete", "complete_upper_bound"}
                          for call in call_rows)
        cost_exact = sum(call.get("status") == "complete" for call in call_rows)
        summary["arms"][arm] = {
            "scenarios": len(selected),
            "task_success_rate": _rate(complete, len(selected)),
            "tool_selection_correct_rate": _rate(selections, len(selected)),
            "tool_parameter_correct_rate": _rate(param_ok, param_total),
            "tool_parameter_na_scenarios": sum(
                item["score"].get("model_parameter_total", 0) == 0 for item in selected),
            "unconfirmed_write_rate": _rate(unconfirmed, write_attempts),
            "unconfirmed_write_rate_note": (
                "N/A: no model-visible write tool in this paired experiment"
                if write_attempts == 0 else "persisted unconfirmed writes / write attempts"),
            "reasonable_handoff_rate": _rate(handoff_ok, handoff_expected),
            "expected_handoff_scenarios": handoff_expected,
            "false_handoff_rate": _rate(false_handoffs,
                                         len(selected) - handoff_expected),
            "refusal_gold_alignment": _rate(refusal_ok, len(selected)),
            "gold_refusal_scenarios": expected_refusal,
            "mean_case_latency_ms": round(statistics.mean(latencies), 1) if latencies else None,
            "model_generations": len(call_rows),
            "usage_known_generations": usage_known,
            "cost_exact_generations": cost_exact,
            "cost_upper_bound_generations": sum(
                call.get("status") != "complete" for call in call_rows),
        }
    paired_ids = set(by_id)
    for arm in ("fixed", "autonomous"):
        group = [result for (scenario_id, result_arm), result in results.items()
                 if result_arm == arm and scenario_id in paired_ids]
        summary["paired_comparison"][arm] = _rate(
            sum(bool(item["score"].get("task_success")) for item in group), len(group))
    if all(summary["paired_comparison"].get(arm, {}).get("denominator") == len(scenarios)
           for arm in ("fixed", "autonomous")):
        fixed = summary["paired_comparison"]["fixed"]
        autonomous = summary["paired_comparison"]["autonomous"]
        summary["paired_comparison"]["task_success_delta_autonomous_minus_fixed"] = round(
            autonomous["rate"] - fixed["rate"], 4)
        summary["experiment_status"] = "complete"
    if budget_snapshot.get("stop_reason"):
        summary["experiment_status"] = "stopped_after_provider_error"
    return summary


def render_report(summary: dict, results: dict[tuple[str, str], dict]) -> str:
    lines = [
        "# CH15 Real-Model Paired Experiment", "",
        f"**Status:** `{summary['experiment_status']}`",
        f"**Frozen config:** `{summary['budget'].get('config_hash', 'unknown')}`",
        "**Scope:** deterministic simulated scenarios only; no production/customer data.",
        "**Model task success** is scored only from model-generated intent, tool calls, and answers.",
        "Refusal alignment uses each scenario's frozen gold; it is not assumed successful by default.", "",
        f"Runs: {summary['arm_run_count']} arm-runs / {summary['scenario_count']} frozen pairs.", "",
        "## Per-arm metrics", "",
        "| Arm | Task success | Tool selection | Tool parameters | Unconfirmed writes | Reasonable handoff | Refusal gold alignment |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, label in (("fixed", "Fixed refund flow"),
                       ("autonomous", "Constrained Agent")):
        item = summary["arms"].get(arm)
        if not item:
            lines.append(f"| {label} | — | — | — | — | — | — |")
            continue
        fmt = lambda rate: (f"{rate['numerator']}/{rate['denominator']} "
                            f"({rate['rate']:.1%})" if rate["rate"] is not None
                            else f"{rate['numerator']}/{rate['denominator']} (N/A)")
        lines.append("| " + " | ".join([
            label, fmt(item["task_success_rate"]), fmt(item["tool_selection_correct_rate"]),
            fmt(item["tool_parameter_correct_rate"]), fmt(item["unconfirmed_write_rate"]),
            fmt(item["reasonable_handoff_rate"]), fmt(item["refusal_gold_alignment"]),
        ]) + " |")
    lines += [
        "", "Each metric includes its numerator and denominator. Handoff may be N/A when no frozen case requires handoff. The paid paired set exposes only read tools, so its write-attempt denominator is zero; all SQLite ticket counts are separately checked as zero.",
        "", "## Paired comparison", "",
    ]
    for arm in ("fixed", "autonomous"):
        result = summary["paired_comparison"].get(arm, {})
        rate = result.get("rate")
        value = f"{result.get('numerator', 0)}/{result.get('denominator', 0)}"
        if rate is not None:
            value += f" ({rate:.1%})"
        lines.append(f"- {arm}: {value}")
    delta = summary["paired_comparison"].get("task_success_delta_autonomous_minus_fixed")
    if delta is not None:
        lines.append(f"- Autonomous minus fixed task-success delta: {delta:+.1%}.")
    lines += [
        "", "## Cost and execution", "",
        f"- Charged upper bound: ${summary['budget'].get('charged_upper_bound_usd', 0):.6f} / $17.40.",
        f"- Usage-based cost at pinned peak rates: ${summary['budget'].get('usage_cost_at_pinned_peak_rates_usd', 0):.6f}.",
        f"- Pending reserved amount: ${summary['budget'].get('reserved_usd', 0):.6f}.",
        f"- Stop reason: {summary['budget'].get('stop_reason') or 'none'}.",
        "- Cost uses provider token usage at pinned peak rates; invoice charges can be lower during off-peak hours. Calls without cache breakdown or usage are charged at the conservative upper bound.",
        "- Every model call is serial, max_retries=0, and bounded before dispatch. Provider errors stop subsequent calls; missing usage is charged at the full per-call envelope.",
        "- Calls and cost by arm:",
    ]
    for arm in ("fixed", "autonomous"):
        usage = summary["budget"].get("arms", {}).get(arm, {})
        lines.append(
            f"  - {arm}: {usage.get('generations', 0)}/{usage.get('max_generations', 0)} "
            f"calls; usage cost at peak rates ${usage.get('usage_cost_at_pinned_peak_rates_usd', 0):.6f}; "
            f"charged upper bound ${usage.get('charged_upper_bound_usd', 0):.6f}; "
            f"cost-upper-bound calls {usage.get('upper_bound_or_unknown_cost_calls', 0)}; "
            f"cap ${usage.get('cap_usd', 0):.2f}.")
    lines += [
        "- `ch15-model-case-report.jsonl` stores each case's full prompt/response trace, graph/tool trace, DB assertion results and score. It must remain simulated-only.",
        "", "## Case results", "",
    ]
    for (scenario_id, arm), result in sorted(results.items()):
        reasons = [name for name, passed in result.get("score", {}).items()
                   if name.endswith("_correct") and not passed]
        lines.append(f"- `{scenario_id}` / `{arm}`: task_success=`{result['score'].get('task_success')}`, "
                     f"attempt=`{result.get('attempt_run_id', 'unknown')}`, "
                     f"calls={len(result.get('model_calls', []))}, "
                     f"latency={result.get('duration_ms', 0)} ms, "
                     f"failures={', '.join(reasons) if reasons else 'none'}")
    lines += [
        "", "All rates describe this frozen set of simulated scenarios only. They are not real support outcomes or customer-service ROI.", "",
    ]
    return "\n".join(lines)


def _persist_outputs(results, scenarios, budget, trace_path, report_path):
    _write_trace(trace_path, results)
    summary = summarize_model_runs(scenarios, results, budget.snapshot())
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(summary, results), encoding="utf-8")
    return summary


async def run(args) -> int:
    config, scenarios = verify_frozen_experiment()
    if not args.allow_paid:
        raise BudgetExceeded("付费调用默认关闭；本地执行必须显式传 --allow-paid")
    settings = _settings_with_provider()
    if config["model"] != MODEL or args.model != MODEL:
        raise BudgetExceeded("本次 model 与冻结配置不一致")
    budget = PersistentPaidRunBudget(
        _pricing(config), model=MODEL, path=LEDGER_PATH,
        experiment_id=config["experiment_id"], config_hash=config["config_hash"],
        arm_limits=ARM_LIMITS, run_cap_usd=RUN_CAP_USD,
        max_input_tokens=MAX_INPUT_TOKENS, max_output_tokens=MAX_OUTPUT_TOKENS,
        allow_paid=True,
    )
    existing_budget = budget.snapshot()
    if existing_budget.get("stop_reason"):
        raise BudgetExceeded(
            "CH15 持久账本已记录 provider 错误；核实费用和调用状态后再人工恢复"
        )
    pending_calls = sum(group.get("pending_calls", 0)
                        for group in existing_budget.get("arms", {}).values())
    if pending_calls:
        raise BudgetExceeded(
            f"CH15 持久账本有 {pending_calls} 个未结算调用；核实 provider usage 后再恢复"
        )
    trace_path, report_path = Path(args.trace), Path(args.report)
    results = _read_trace(trace_path)
    valid = {(row["id"], arm) for row in scenarios for arm in ("fixed", "autonomous")}
    if set(results) - valid:
        raise BudgetExceeded("已有逐题轨迹不属于当前冻结样本")
    if args.smoke:
        targets = [(scenarios[0], "fixed")]
    else:
        targets = [(row, arm) for row in scenarios for arm in ("fixed", "autonomous")]
    for row, arm in targets:
        key = (row["id"], arm)
        if key in results:
            continue
        existing_calls = budget.calls_for_case(row["id"], arm)
        if any(call.get("status") == "pending" for call in existing_calls):
            raise BudgetExceeded(
                f"{row['id']}/{arm} 存在未结算 provider 调用，拒绝重放")
        attempt_id = next_case_attempt_id(
            config["experiment_id"], row["id"], arm, existing_calls)
        try:
            result = await run_model_case(row, arm, settings, budget,
                                          run_id=attempt_id)
        except BaseException:
            # Per-call ledger was committed before dispatch; no synthetic retry.
            _persist_outputs(results, scenarios, budget, trace_path, report_path)
            raise
        results[key] = result
        summary = _persist_outputs(results, scenarios, budget, trace_path, report_path)
        print(json.dumps({"completed": key, "task_success": result["score"]["task_success"],
                          "generations": budget.snapshot()["arms"],
                          "charged_upper_bound_usd": budget.snapshot()["charged_upper_bound_usd"],
                          "experiment_status": summary["experiment_status"]},
                         ensure_ascii=False))
        if budget.snapshot().get("stop_reason"):
            return 4
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-paid", action="store_true",
                        help="Explicitly enable provider calls within the frozen hard caps")
    parser.add_argument("--preflight", action="store_true",
                        help="Verify frozen inputs and limits without loading credentials or calling a provider")
    parser.add_argument("--smoke", action="store_true",
                        help="Run only CH15-RFD-001 on the fixed graph as the paid smoke case")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--trace", default=str(TRACE_PATH))
    parser.add_argument("--report", default=str(REPORT_PATH))
    args = parser.parse_args()
    if args.preflight:
        try:
            config, scenarios = verify_frozen_experiment()
        except BudgetExceeded as exc:
            print(f"CH15 preflight failed: {exc}", file=sys.stderr)
            return 3
        print(json.dumps({
            "status": "ready_for_credentials",
            "experiment_id": config["experiment_id"],
            "config_hash": config["config_hash"],
            "scenario_count": len(scenarios),
            "turn_count": sum(len(row["turns"]) for row in scenarios),
            "arms": config["arms"],
            "model": config["model"],
            "max_total_generations": sum(item["max_generations"]
                                          for item in config["arms"].values()),
            "total_cap_usd": config["run_cap_usd"],
            "provider_calls": 0,
        }, ensure_ascii=False))
        return 0
    try:
        return asyncio.run(run(args))
    except MissingProviderCredential as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except BudgetExceeded as exc:
        print(f"CH15 guard stopped run: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
