"""Frozen three-arm evaluation for cost-aware single/multi Agent routing.

Gold exists only in this evaluator and its deterministic scorer. The routing
projection and both execution arms receive only user-visible turns.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import os
import re
import sys
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from app.adaptive_routing import (
    HistoryTurn,
    POLICY_VERSION,
    RouteDispatcher,
    RoutingInput,
    VisibleRunBudget,
    policy_sha256,
)
from app.config import Settings
from app.guard import is_refusal
from app.multi_agent.models import MultiAgentLimits
from app.multi_agent.runtime import MultiAgentRunner, build_multi_agent_graph
from app.tools.engine import ToolEngine
from scripts.ch15_paid_guard import ArmLimit, _utc_now, paid_call_context
from scripts.eval_cost import BudgetExceeded, PricingTable
from scripts.eval_multi_agent import (
    ConcurrentPaidRunBudget,
    _AccurateGuardedChatModel,
    _FakeSink,
    _MemoryAudit,
    _NullBudget,
    _OfflineModel,
    _Pool,
    _RecordingEngine,
    _TraceModel,
    _fake_registry,
    _settings,
    _write_result,
)


DATA_DIR = ROOT / "tests" / "eval" / "adaptive_routing_r2"
SCENARIO_PATH = DATA_DIR / "scenarios.jsonl"
RUBRIC_PATH = DATA_DIR / "scoring-rubric.json"
MANIFEST_PATH = DATA_DIR / "manifest.json"
RESULTS_DIR = DATA_DIR / "runs"
REPORT_PATH = ROOT / "docs" / "eval" / "adaptive-routing-r2-report.md"
LEDGER_PATH = ROOT / "data" / "adaptive_routing_r2_paid_run.sqlite3"
POLICY_FIXTURE_PATH = ROOT / "tests" / "eval" / "tasks" / "fixtures" / "policies.json"
ORDERS_PATH = ROOT / "app" / "orders.py"
MODEL = "deepseek-flash"
BASE_URL = "https://api.deepseek.com"
RUN_CAP_USD = 2.0
DEV_ATTEMPT_CHARGED_USD = 0.07220916
FINAL_LEDGER_CAP_USD = round(RUN_CAP_USD - DEV_ATTEMPT_CHARGED_USD, 8)
PRICING_VERIFIED_ON = "2026-10-27"
PRICING_URL = "https://api-docs.deepseek.com/quick_start/pricing/"
ARM_NAMES = ("fixed_single", "fixed_multi", "adaptive")
ADAPTIVE_BUDGET_ARMS = ("adaptive_single", "adaptive_multi")
MAX_INPUT_TOKENS = 7_000
MAX_OUTPUT_TOKENS = 500
MAX_RATE_LIMIT_RETRIES = 2

ARM_LIMITS = {
    "fixed_single": ArmLimit(max_generations=100, cap_usd=0.27),
    "fixed_multi": ArmLimit(max_generations=400, cap_usd=1.08),
    "adaptive_single": ArmLimit(max_generations=80, cap_usd=0.216),
    "adaptive_multi": ArmLimit(max_generations=130, cap_usd=0.351),
}


def load_scenarios(path: Path | None = None) -> list[dict]:
    source = Path(path or SCENARIO_PATH)
    rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return rows


def validate_frozen_dataset(scenarios: list[dict]) -> dict:
    ids = [str(row["id"]) for row in scenarios]
    duplicate_ids = sorted({item for item in ids if ids.count(item) > 1})
    if duplicate_ids:
        raise ValueError(f"duplicate adaptive-routing scenario IDs: {duplicate_ids}")
    split_counts: dict[str, int] = {}
    group_splits: dict[str, set[str]] = {}
    expected_behavior_fields = 0
    for row in scenarios:
        split = str(row["split"])
        split_counts[split] = split_counts.get(split, 0) + 1
        group_splits.setdefault(str(row["source_group"]), set()).add(split)
        if "expected_behavior" in row or "expected_behavior" in (row.get("gold") or {}):
            expected_behavior_fields += 1
        gold = row.get("gold") or {}
        required = {
            "outcome", "must_include", "must_not_include", "required_tools",
            "required_order_ids", "forbidden_final_order_ids", "write_mode",
        }
        missing = sorted(required - set(gold))
        if missing:
            raise ValueError(f"{row['id']} gold is missing fields: {missing}")
        if not row.get("turns"):
            raise ValueError(f"{row['id']} has no user turns")
    crossing = sorted(
        group for group, splits in group_splits.items() if len(splits) > 1
    )
    if split_counts != {"dev": 10, "held-out": 24}:
        raise ValueError(f"frozen split counts changed: {split_counts}")
    if crossing:
        raise ValueError(f"source groups cross dev/held-out: {crossing}")
    return {
        "splits": split_counts,
        "source_groups": len(group_splits),
        "source_groups_cross_splits": crossing,
        "expected_behavior_fields": expected_behavior_fields,
        "unique_ids": len(set(ids)),
    }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_manifest() -> dict:
    scenarios = load_scenarios()
    audit = validate_frozen_dataset(scenarios)
    model_settings = {
        "temperature": 0,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "thinking": "disabled",
        "max_retries": 0,
        "rate_limit_retries": MAX_RATE_LIMIT_RETRIES,
    }
    evidence_sources = {
        "orders": "app/orders.py fixed simulated catalog",
        "policies": "tests/eval/tasks/fixtures/policies.json CH15 simulated policy snapshot",
        "milvus_snapshot_manifest": (
            "read-only snapshot ebfd49478b37ef387c8b124319a450ed73a86ae2a4b9f4fc28d6a52b2ccaa0ba.db"
        ),
        "milvus_snapshot_version": 3,
        "milvus_snapshot_data_files": ["data/data_000004_000133.parquet"],
    }
    arm_limits = {
        arm: {
            "max_generations": limit.max_generations,
            "cap_usd": limit.cap_usd,
        }
        for arm, limit in ARM_LIMITS.items()
    }
    executor_config = {
        "baseline_commit": "446fbb7f9972875b7649555cb9b4b19e402cea29",
        "model": MODEL,
        "model_version": "DeepSeek-V4.1-Flash-1017",
        "model_settings": model_settings,
        "max_input_tokens": MAX_INPUT_TOKENS,
        "arm_limits": arm_limits,
        "evidence_sources": evidence_sources,
    }
    executor_config_sha256 = hashlib.sha256(
        json.dumps(
            executor_config, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    source_groups = sorted(
        {
            (row["split"], row["source_group"])
            for row in scenarios
        }
    )
    return {
        "experiment_id": "adaptive-routing-r2-20261027",
        "status": "frozen-before-heldout",
        "frozen_on": "2026-10-27",
        "split_protocol": "business facts and question templates assigned by source_group",
        "splits": audit["splits"],
        "source_groups": [
            {"split": split, "source_group": group}
            for split, group in source_groups
        ],
        "source_groups_cross_splits": audit["source_groups_cross_splits"],
        "scenario_sha256": _file_sha256(SCENARIO_PATH),
        "rubric_sha256": _file_sha256(RUBRIC_PATH),
        "harness_sha256": _file_sha256(Path(__file__)),
        "router_policy_version": POLICY_VERSION,
        "router_policy_sha256": policy_sha256(),
        "router_policy_mode": "rules-only",
        "model": MODEL,
        "model_version": "DeepSeek-V4.1-Flash-1017",
        "model_settings": model_settings,
        "fixtures": {
            "policies_sha256": _file_sha256(POLICY_FIXTURE_PATH),
            "orders_sha256": _file_sha256(ORDERS_PATH),
        },
        "evidence_sources": evidence_sources,
        "executor_config": executor_config,
        "executor_config_sha256": executor_config_sha256,
        "pricing": {
            "verified_on": PRICING_VERIFIED_ON,
            "source_url": PRICING_URL,
            "rate_basis": "peak prices, conservative upper bound",
            "input_per_million": 0.30,
            "input_cache_hit_per_million": 0.006,
            "output_per_million": 1.20,
        },
        "run_cap_usd": RUN_CAP_USD,
        "pre_final_attempt_charged_usd": DEV_ATTEMPT_CHARGED_USD,
        "final_ledger_cap_usd": FINAL_LEDGER_CAP_USD,
        "stop_rule": {
            "heldout_run_count": 1,
            "heldout_retuning": False,
            "dev_before_heldout": True,
            "infrastructure_retry_limit": MAX_RATE_LIMIT_RETRIES,
            "preserve_failed_attempts": True,
            "selective_success_rerun": False,
        },
        "metrics": [
            "task success",
            "paired wins/losses/ties",
            "false/missed upgrades",
            "bucket success",
            "cold-path P50/P95",
            "provider calls and tokens",
            "total cost and cost per success",
            "multi-Agent share and routing time",
        ],
    }


def verify_manifest(frozen: Mapping[str, Any]) -> dict:
    current = build_manifest()
    if dict(frozen) != current:
        raise ValueError("frozen adaptive-routing manifest does not match current artifacts")
    return current


def routing_input_for(scenario: Mapping[str, Any], *, turn_index: int) -> RoutingInput:
    """Static no-gold projection used for tests and pre-run route audits."""

    turns = list(scenario["turns"])
    history: list[HistoryTurn] = []
    for message in turns[:turn_index]:
        history.append(HistoryTurn(role="user", content=str(message)))
        history.append(HistoryTurn(role="assistant", content="已记录上一轮请求并完成回复。"))
    return RoutingInput(
        question=str(turns[turn_index]),
        history=history,
        visible_budget=VisibleRunBudget(
            remaining_seconds=45.0, remaining_cost_usd=0.10, max_steps=12
        ),
    )


def _scoring_rubric() -> dict:
    return json.loads(RUBRIC_PATH.read_text(encoding="utf-8"))


def _normalized(text: str) -> str:
    # Markdown emphasis must not break exact phrase checks or negative-context
    # exceptions. Provider answers frequently bold the exact policy clause.
    return re.sub(r"[\s*`]+", "", text or "")


def _order_ids(text: str) -> list[str]:
    found = re.findall(r"(?:订单|order)\s*号?\s*(\d{3,6})", text or "", flags=re.I)
    for order_id in ("1001", "1002", "1003"):
        if re.search(rf"(?<!\d){order_id}(?!\d)", text or ""):
            found.append(order_id)
    return list(dict.fromkeys(found))


def _actual_outcome(reply: str, disposition: str = "") -> str:
    if disposition in {"clarify", "refuse", "handoff"}:
        return disposition
    if re.search(
        r"已签收|已发货|待发货|15天|7天|8元|99元|1到3个工作日|"
        r"商家承担|买家|不支持|可以取消|不能直接取消|拒收|原路退回|24小时",
        reply or "",
    ):
        return "answer"
    if is_refusal(reply):
        return "refuse"
    if re.search(
        r"(?:方便|请|需要|麻烦).{0,16}(?:提供|补充|告知).{0,10}(?:订单号|单号)|"
        r"哪(?:一)?笔订单|哪件商品",
        reply or "",
    ):
        return "clarify"
    if re.search(r"(运费|包邮|吊牌|包装|退款|取消|可以申请|不能直接)", reply or ""):
        return "answer"
    if "转人工" in (reply or ""):
        return "handoff"
    return "answer"


def _tool_parameter_errors(tool_calls: list[dict]) -> int:
    errors = 0
    for call in tool_calls:
        name = str(call.get("name") or "")
        args = call.get("arguments") or {}
        if not isinstance(args, dict):
            errors += 1
            continue
        if name == "query_order" and not str(args.get("order_id") or "").strip():
            errors += 1
        elif name == "query_faq" and not str(args.get("keyword") or "").strip():
            errors += 1
        elif name == "create_ticket":
            if args.get("conversation_id") in {None, ""}:
                errors += 1
            if not str(args.get("description") or "").strip():
                errors += 1
            if args.get("ticket_type") not in {"售后", "投诉", "咨询"}:
                errors += 1
    return errors


def score_case(scenario: Mapping[str, Any], result: Mapping[str, Any]) -> dict:
    gold = scenario["gold"]
    reply = str(result.get("reply") or "")
    normalized_reply = _normalized(reply)
    tool_calls = list(result.get("tool_calls") or [])
    tool_names = [str(call.get("name") or "") for call in tool_calls]
    actual_outcome = _actual_outcome(reply, str(result.get("disposition") or ""))
    rubric = _scoring_rubric()
    include_any = rubric.get("include_any", {})

    missing_conditions = []
    for phrase in gold.get("must_include", []):
        alternatives = include_any.get(phrase, [phrase])
        if not any(_normalized(item) in normalized_reply for item in alternatives):
            missing_conditions.append(phrase)
    forbidden_conditions = []
    negative_text = normalized_reply
    for snippet in (rubric.get("negative_context_exceptions", {}).get("__all__") or []):
        negative_text = negative_text.replace(_normalized(snippet), "")
    for phrase in gold.get("must_not_include", []):
        comparable = negative_text
        for snippet in rubric.get("negative_context_exceptions", {}).get(phrase, []):
            comparable = comparable.replace(_normalized(snippet), "")
        if _normalized(phrase) in comparable:
            forbidden_conditions.append(phrase)
    condition_errors = missing_conditions + forbidden_conditions

    final_order_ids = _order_ids(reply)
    required_order_ids = [str(item) for item in gold.get("required_order_ids", [])]
    forbidden_final_order_ids = {
        str(item) for item in gold.get("forbidden_final_order_ids", [])
    }
    queried_order_ids = {
        str((call.get("arguments") or {}).get("order_id"))
        for call in tool_calls if call.get("name") == "query_order"
    }
    missing_order_ids = [item for item in required_order_ids if item not in final_order_ids]
    forbidden_orders = [
        item for item in final_order_ids if item in forbidden_final_order_ids
    ]
    missing_queried = [
        item for item in required_order_ids if item not in queried_order_ids
    ]
    extra_final_orders = [
        item for item in final_order_ids
        if required_order_ids and item not in required_order_ids
        and item not in forbidden_final_order_ids
    ]
    entity_errors = missing_order_ids + forbidden_orders + missing_queried + extra_final_orders

    parameter_errors = _tool_parameter_errors(tool_calls)
    required_tools = list(gold.get("required_tools", []))
    missing_tools = [name for name in required_tools if name not in tool_names]
    outcome_correct = actual_outcome == gold["outcome"]
    previews = list(result.get("write_previews") or [])
    confirmed = int(result.get("confirmed_write_count") or 0)
    unconfirmed = int(result.get("unconfirmed_write_count") or 0)
    write_mode = gold.get("write_mode", "none")
    if write_mode == "none":
        write_correct = not previews and confirmed == 0 and unconfirmed == 0
    elif write_mode == "confirm_once":
        write_correct = bool(previews) and confirmed == 1 and unconfirmed == 0
    elif write_mode == "cancel":
        write_correct = bool(previews) and confirmed == 0 and unconfirmed == 0
    else:
        write_correct = False
    task_success = (
        outcome_correct
        and not condition_errors
        and not entity_errors
        and not parameter_errors
        and not missing_tools
        and write_correct
    )
    return {
        "task_success": bool(task_success),
        "actual_outcome": actual_outcome,
        "outcome_correct": outcome_correct,
        "condition_error_count": len(condition_errors),
        "entity_error_count": len(entity_errors),
        "parameter_error_count": parameter_errors,
        "missing_tools": missing_tools,
        "required_tools_correct": not missing_tools,
        "write_correct": write_correct,
        "missing_conditions": missing_conditions,
        "forbidden_conditions": forbidden_conditions,
        "entity_errors": entity_errors,
        "tool_sequence": tool_names,
        "preview_count": len(previews),
        "confirmed_write_count": confirmed,
        "unconfirmed_write_count": unconfirmed,
        "scoring_version": rubric["version"],
    }


def _pricing() -> PricingTable:
    return PricingTable(
        {
            "verified": True,
            "source_url": PRICING_URL,
            "version": PRICING_VERIFIED_ON,
            "models": {
                MODEL: {
                    "input_per_million": 0.30,
                    "input_cache_hit_per_million": 0.006,
                    "output_per_million": 1.20,
                }
            },
        }
    )


class AdaptivePaidRunBudget(ConcurrentPaidRunBudget):
    """Concurrent isolated ledger that permits a bounded caller-level 429 retry."""

    def clear_rate_limit_stop(self) -> None:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "UPDATE ch15_paid_runs SET stop_reason=NULL WHERE experiment_id=?",
                (self.experiment_id,),
            )
            db.commit()


def _is_rate_limit_error(exc: BaseException) -> bool:
    name = type(exc).__name__.casefold()
    if "ratelimit" in name or "rate_limit" in name:
        return True
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    return str(status) == "429" or "429" in str(exc)


class _RetryingGuardedModel:
    """Retry only provider 429s; each attempt still crosses the guarded ledger."""

    def __init__(self, inner: Any, budget: Any) -> None:
        self.inner = inner
        self.budget = budget

    def bind_tools(self, tools, **kwargs):
        return _RetryingGuardedModel(self.inner.bind_tools(tools, **kwargs), self.budget)

    def bind(self, *args, **kwargs):
        return _RetryingGuardedModel(self.inner.bind(*args, **kwargs), self.budget)

    def with_structured_output(self, *args, **kwargs):
        return _RetryingGuardedModel(
            self.inner.with_structured_output(*args, **kwargs), self.budget
        )

    async def ainvoke(self, messages, *args, **kwargs):
        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            try:
                return await self.inner.ainvoke(messages, *args, **kwargs)
            except Exception as exc:
                if attempt >= MAX_RATE_LIMIT_RETRIES or not _is_rate_limit_error(exc):
                    raise
                clear = getattr(self.budget, "clear_rate_limit_stop", None)
                if callable(clear):
                    clear()
                await asyncio.sleep(2**attempt)
        raise AssertionError("unreachable")

    async def astream(self, messages, *args, **kwargs):
        # Once a chunk reaches the caller, replaying the generation would duplicate
        # partial output. Only failures before the first chunk are retryable.
        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            emitted = False
            try:
                async for item in self.inner.astream(messages, *args, **kwargs):
                    emitted = True
                    yield item
                return
            except Exception as exc:
                if emitted or attempt >= MAX_RATE_LIMIT_RETRIES \
                        or not _is_rate_limit_error(exc):
                    raise
                clear = getattr(self.budget, "clear_rate_limit_stop", None)
                if callable(clear):
                    clear()
                await asyncio.sleep(2**attempt)


def _model_wrapper(
    raw_model: Any, budget: Any, arm: str, role: str
) -> Any:
    if isinstance(budget, _NullBudget):
        return raw_model
    guarded = _AccurateGuardedChatModel(raw_model, budget, arm, role)
    return _RetryingGuardedModel(guarded, budget)


def _chat_model(api_key: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=MODEL,
        api_key=api_key,
        base_url=BASE_URL,
        temperature=0,
        max_tokens=MAX_OUTPUT_TOKENS,
        max_retries=0,
        timeout=45,
        stream_usage=True,
        extra_body={"thinking": {"type": "disabled"}},
    )


def _paid_budget(
    path: Path, *, run_cap_usd: float = FINAL_LEDGER_CAP_USD
) -> AdaptivePaidRunBudget:
    if run_cap_usd <= 0 or run_cap_usd > FINAL_LEDGER_CAP_USD + 1e-9:
        raise BudgetExceeded(
            f"final ledger cap must be >0 and <= ${FINAL_LEDGER_CAP_USD:.8f}"
        )
    return AdaptivePaidRunBudget(
        _pricing(),
        path=path,
        experiment_id="adaptive-routing-r2-20261027",
        config_hash=build_manifest()["scenario_sha256"],
        model=MODEL,
        arm_limits=ARM_LIMITS,
        run_cap_usd=run_cap_usd,
        max_input_tokens=MAX_INPUT_TOKENS,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        allow_paid=True,
    )


def _runtime(scenario: Mapping[str, Any]):
    sink = _FakeSink()
    audit = _MemoryAudit()
    engine = _RecordingEngine(
        ToolEngine(_fake_registry(sink), audit, timeout_seconds=10, retry_attempts=0)
    )
    return sink, audit, engine


def calls_for_attempt(
    budget: Any,
    scenario_id: str,
    budget_arms: tuple[str, ...],
    attempt_prefix: str,
) -> list[dict]:
    """Select ledger rows from the current attempt, not earlier reruns."""

    if not hasattr(budget, "calls_for_case"):
        return []
    calls: list[dict] = []
    for budget_arm in budget_arms:
        calls.extend(
            call
            for call in budget.calls_for_case(scenario_id, budget_arm)
            if str(call.get("run_id") or "").startswith(attempt_prefix)
        )
    return calls


def _visible_history(history: list[dict]) -> list[HistoryTurn]:
    return [
        HistoryTurn(role=str(item["role"]), content=str(item["content"]))
        for item in history
        if item.get("role") in {"user", "assistant"}
    ]


async def _single_turn_async(**kwargs) -> dict:
    scenario = kwargs["scenario"]
    message = kwargs["message"]
    history = kwargs["history"]
    node = kwargs["node"]
    arm = kwargs["arm"]
    turn_index = kwargs["turn_index"]
    run_id = kwargs["run_id"]
    layered = {
        "summary_text": "",
        "layer2_msgs": [],
        "layer1_msgs": list(history),
        "history_text": "",
        "n_window": len(history),
        "tokens_l1": 0,
        "tokens_l2": 0,
        "dropped": 0,
    }
    from app.graph.builder import initial_state

    state = initial_state(1, message, layered=layered)
    state["resolved_message"] = message
    frames: list[dict] = []
    with paid_call_context(
        scenario_id=str(scenario["id"]), arm=arm, turn=turn_index, run_id=run_id
    ):
        result = await node(state, writer=frames.append, config=None)
    return {"state": result, "frames": frames, "reply": str(result.get("final_reply") or "")}


async def _multi_turn(
    *,
    scenario: Mapping[str, Any],
    message: str,
    history: list[dict],
    graph: Any,
    arm: str,
    turn_index: int,
    run_id: str,
) -> dict:
    with paid_call_context(
        scenario_id=str(scenario["id"]), arm=arm, turn=turn_index, run_id=run_id
    ):
        result = await graph.ainvoke(
            {
                "run_id": f"{run_id}:turn-{turn_index}",
                "task_id": f"{scenario['id']}:turn-{turn_index}",
                "user_message": message,
                "conversation_history": list(history),
                "facts": {},
                "evidence": {},
                "errors": [],
                "stop_reasons": [],
                "trace": [],
                "role_histories": {},
            }
        )
    return {"state": result, "reply": str(result.get("final_reply") or "")}


async def _result_from_session(
    scenario: Mapping[str, Any],
    *,
    arm: str,
    replies: list[str],
    states: list[dict],
    frames: list[dict],
    sink: Any,
    engine: Any,
    interactions: list[dict],
    route_decisions: list[dict],
    route_targets: list[str],
    routing_ms: int,
    routing_cost_usd: float,
    wall_ms: int,
    errors: list[str],
    budget: Any,
    attempt_prefix: str,
) -> dict:
    multi_previews = [
        preview
        for state in states
        for preview in state.get("write_previews", [])
    ]
    previews = [
        frame for frame in frames if frame.get("type") == "ticket_preview"
    ] + multi_previews
    write_mode = str(scenario["gold"].get("write_mode") or "none")
    confirmed, unconfirmed = await _write_result(
        engine, sink, previews=previews, write_mode=write_mode
    )
    last_state = states[-1] if states else {}
    actual_budget_arms = ADAPTIVE_BUDGET_ARMS if arm == "adaptive" else (arm,)
    calls = calls_for_attempt(
        budget,
        str(scenario["id"]),
        tuple(actual_budget_arms),
        attempt_prefix,
    )
    model_cost = sum(float(call.get("charged_usd") or 0.0) for call in calls)
    row = {
        "scenario_id": scenario["id"],
        "arm": arm,
        "split": scenario["split"],
        "source_group": scenario["source_group"],
        "bucket": scenario["bucket"],
        "category": scenario["category"],
        "reply": replies[-1] if replies else "",
        "replies": replies,
        "disposition": last_state.get("disposition", ""),
        "tool_calls": [
            {"name": item["name"], "arguments": item["arguments"]}
            for item in engine.calls
        ],
        "write_previews": previews,
        "confirmed_write_count": confirmed,
        "unconfirmed_write_count": unconfirmed,
        "sink_rows": sink.rows,
        "duration_ms": wall_ms,
        "routing_ms": routing_ms,
        "routing_cost_usd": routing_cost_usd,
        "route_targets": route_targets,
        "route_selected": route_targets[-1] if route_targets else (
            "single" if arm == "fixed_single" else "multi"
        ),
        "route_decisions": route_decisions,
        "errors": errors,
        "model_calls": calls,
        "model_interactions": interactions,
        "input_tokens": sum(int(call.get("input_tokens") or 0) for call in calls),
        "output_tokens": sum(int(call.get("output_tokens") or 0) for call in calls),
        "cost_usd": round(model_cost + routing_cost_usd, 10),
    }
    row["score"] = score_case(scenario, row)
    row["success"] = row["score"]["task_success"]
    return row


async def run_scenario(
    scenario: dict,
    *,
    arm: str,
    raw_models: Mapping[str, Any],
    budget: Any,
    attempt_id: str,
) -> dict:
    sink, _audit, engine = _runtime(scenario)
    settings = _settings()
    interactions: list[dict] = []
    states: list[dict] = []
    frames: list[dict] = []
    replies: list[str] = []
    single_history: list[Any] = []
    multi_history: list[dict] = []
    route_decisions: list[dict] = []
    route_targets: list[str] = []
    errors: list[str] = []
    routing_ms = 0
    routing_cost_usd = 0.0
    started = time.perf_counter()

    if arm in {"fixed_single", "adaptive"}:
        single_agent = _model_wrapper(raw_models["single"], budget, (
            "adaptive_single" if arm == "adaptive" else "fixed_single"
        ), "agent")
        traced_single = _TraceModel(single_agent, "agent", interactions)
        from app.graph.agent import make_agent_node
        single_node = make_agent_node(traced_single, engine, settings, _Pool())
    if arm in {"fixed_multi", "adaptive"}:
        roles = {}
        for role in ("planner", "retrieval", "action", "supervisor"):
            wrapped = _model_wrapper(raw_models[role], budget, (
                "adaptive_multi" if arm == "adaptive" else "fixed_multi"
            ), role)
            roles[role] = _TraceModel(wrapped, role, interactions)
        runner = MultiAgentRunner(
            planner_model=roles["planner"],
            retrieval_model=roles["retrieval"],
            action_model=roles["action"],
            supervisor_model=roles["supervisor"],
            engine=engine,
            limits=MultiAgentLimits(
                max_seconds=40,
                child_timeout_seconds=20,
                max_steps=12,
                max_tokens=16_000,
            ),
        )
        multi_graph = build_multi_agent_graph(runner)

    dispatcher = RouteDispatcher()
    for turn_index, message in enumerate(scenario["turns"]):
        run_id = (
            f"adaptive-routing-r2:{attempt_id}:{scenario['id']}:turn-{turn_index + 1}"
        )
        try:
            if arm == "fixed_single":
                turn = await _single_turn_async(
                    scenario=scenario,
                    message=str(message),
                    history=single_history,
                    node=single_node,
                    arm="fixed_single",
                    turn_index=turn_index + 1,
                    run_id=run_id,
                )
                route_targets.append("single")
                states.append(turn["state"])
                frames.extend(turn["frames"])
                reply = turn["reply"]
            elif arm == "fixed_multi":
                turn = await _multi_turn(
                    scenario=scenario,
                    message=str(message),
                    history=multi_history,
                    graph=multi_graph,
                    arm="fixed_multi",
                    turn_index=turn_index + 1,
                    run_id=run_id,
                )
                route_targets.append("multi")
                states.append(turn["state"])
                reply = turn["reply"]
            else:
                visible_history = _visible_history(multi_history)
                routing_input = RoutingInput(
                    question=str(message),
                    history=visible_history,
                    visible_budget=VisibleRunBudget(
                        remaining_seconds=max(
                            0.1, 50.0 - (time.perf_counter() - started)
                        ),
                        remaining_cost_usd=0.10,
                        max_steps=12,
                    ),
                )

                async def run_single(_routing_input: RoutingInput) -> dict:
                    return await _single_turn_async(
                        scenario=scenario,
                        message=str(message),
                        history=single_history,
                        node=single_node,
                        arm="adaptive_single",
                        turn_index=turn_index + 1,
                        run_id=run_id,
                    )

                async def run_multi(_routing_input: RoutingInput) -> dict:
                    return await _multi_turn(
                        scenario=scenario,
                        message=str(message),
                        history=multi_history,
                        graph=multi_graph,
                        arm="adaptive_multi",
                        turn_index=turn_index + 1,
                        run_id=run_id,
                    )

                execution = await dispatcher.dispatch(
                    routing_input,
                    single_executor=run_single,
                    multi_executor=run_multi,
                )
                route_decisions.append(execution.route.model_dump(mode="json"))
                route_targets.append(execution.selected_arm)
                routing_ms += execution.routing_ms
                routing_cost_usd += execution.routing_cost_usd
                if execution.error:
                    errors.append(f"routing_executor:{execution.error}")
                    break
                raw = execution.raw_response
                states.append(raw["state"])
                if execution.selected_arm == "single":
                    frames.extend(raw.get("frames") or [])
                reply = str(raw.get("reply") or "")
            replies.append(reply)
            single_history.extend(
                [HumanMessage(content=str(message)), AIMessage(content=reply)]
            )
            multi_history.extend(
                [
                    {"role": "user", "content": str(message)},
                    {"role": "assistant", "content": reply},
                ]
            )
        except Exception as exc:
            errors.append(f"harness:{type(exc).__name__}:{exc}")
            break

    wall_ms = int((time.perf_counter() - started) * 1000)
    return await _result_from_session(
        scenario,
        arm=arm,
        replies=replies,
        states=states,
        frames=frames,
        sink=sink,
        engine=engine,
        interactions=interactions,
        route_decisions=route_decisions,
        route_targets=route_targets,
        routing_ms=routing_ms,
        routing_cost_usd=routing_cost_usd,
        wall_ms=wall_ms,
        errors=errors,
        budget=budget,
        attempt_prefix=(
            f"adaptive-routing-r2:{attempt_id}:{scenario['id']}:"
        ),
    )


def _success(row: Mapping[str, Any]) -> bool:
    if "success" in row:
        return bool(row["success"])
    return bool((row.get("score") or {}).get("task_success"))


def _pair(candidate: Mapping[str, Any] | None, other: Mapping[str, Any] | None) -> str:
    if candidate is None or other is None:
        return "missing"
    a, b = _success(candidate), _success(other)
    return "ties" if a == b else ("wins" if a and not b else "losses")


def paired_route_diagnostics(rows: list[dict]) -> dict:
    indexed = {(row["scenario_id"], row["arm"]): row for row in rows}
    scenario_ids = sorted({row["scenario_id"] for row in rows})
    result = {
        "adaptive_vs_single": {"wins": 0, "losses": 0, "ties": 0},
        "adaptive_vs_multi": {"wins": 0, "losses": 0, "ties": 0},
        "false_upgrade_ids": [],
        "missed_upgrade_ids": [],
        "selected_multi_ids": [],
        "selected_single_ids": [],
    }
    for scenario_id in scenario_ids:
        single = indexed.get((scenario_id, "fixed_single"))
        multi = indexed.get((scenario_id, "fixed_multi"))
        adaptive = indexed.get((scenario_id, "adaptive"))
        for name, other in (
            ("adaptive_vs_single", single),
            ("adaptive_vs_multi", multi),
        ):
            pair = _pair(adaptive, other)
            if pair in {"wins", "losses", "ties"}:
                result[name][pair] += 1
        if adaptive is None:
            continue
        selected = str(adaptive.get("route_selected") or "")
        if selected == "multi":
            result["selected_multi_ids"].append(scenario_id)
            if (
                single is not None and multi is not None
                and _success(single) and not _success(multi) and not _success(adaptive)
            ):
                result["false_upgrade_ids"].append(scenario_id)
        elif selected == "single":
            result["selected_single_ids"].append(scenario_id)
            if (
                single is not None and multi is not None
                and _success(multi) and not _success(single) and not _success(adaptive)
            ):
                result["missed_upgrade_ids"].append(scenario_id)
    return result


def _rate(numerator: int, denominator: int) -> dict:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": round(numerator / denominator, 4) if denominator else None,
    }


def _percentile(values: list[int], percentile: int) -> int | None:
    if not values:
        return None
    values = sorted(values)
    index = max(0, (percentile * len(values) + 99) // 100 - 1)
    return values[min(index, len(values) - 1)]


def summarize_results(
    scenarios: list[dict],
    rows: list[dict],
    budget_snapshot: Mapping[str, Any],
) -> dict:
    by_arm: dict[str, list[dict]] = {}
    by_bucket: dict[str, dict[str, list[dict]]] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], []).append(row)
        by_bucket.setdefault(row["bucket"], {}).setdefault(row["arm"], []).append(row)
    arms: dict[str, Any] = {}
    for arm in ARM_NAMES:
        arm_rows = by_arm.get(arm, [])
        success_count = sum(_success(row) for row in arm_rows)
        cost = sum(float(row.get("cost_usd") or 0.0) for row in arm_rows)
        calls = [call for row in arm_rows for call in row.get("model_calls", [])]
        route_turns = sum(len(row.get("route_targets") or []) for row in arm_rows)
        multi_turns = sum(
            target == "multi"
            for row in arm_rows for target in (row.get("route_targets") or [])
        )
        arms[arm] = {
            "task_success": _rate(success_count, len(arm_rows)),
            "p50_ms": _percentile([int(row["duration_ms"]) for row in arm_rows], 50),
            "p95_ms": _percentile([int(row["duration_ms"]) for row in arm_rows], 95),
            "model_calls": len(calls),
            "input_tokens": sum(int(call.get("input_tokens") or 0) for call in calls),
            "output_tokens": sum(int(call.get("output_tokens") or 0) for call in calls),
            "cost_usd": round(cost, 10),
            "cost_per_success_usd": (
                round(cost / success_count, 10) if success_count else None
            ),
            "routing_ms_total": sum(int(row.get("routing_ms") or 0) for row in arm_rows),
            "routing_ms_p95": _percentile(
                [int(row.get("routing_ms") or 0) for row in arm_rows], 95
            ),
            "multi_route_turn_share": (
                round(multi_turns / route_turns, 4) if route_turns else None
            ),
            "multi_route_turns": multi_turns,
            "route_turns": route_turns,
            "multi_route_scenario_share": (
                round(sum(
                    str(row.get("route_selected")) == "multi" for row in arm_rows
                ) / len(arm_rows), 4)
                if arm_rows else None
            ),
            "failed_case_ids": [
                row["scenario_id"] for row in arm_rows if not _success(row)
            ],
        }
    diagnostics = paired_route_diagnostics(rows)
    bucket_summary: dict[str, dict[str, Any]] = {}
    for bucket, grouped in sorted(by_bucket.items()):
        bucket_summary[bucket] = {}
        for arm in ARM_NAMES:
            arm_rows = grouped.get(arm, [])
            bucket_summary[bucket][arm] = _rate(
                sum(_success(row) for row in arm_rows), len(arm_rows)
            )
    route_reasons: dict[str, int] = {}
    for row in by_arm.get("adaptive", []):
        for decision in row.get("route_decisions") or []:
            for reason in decision.get("reason_codes") or []:
                route_reasons[reason] = route_reasons.get(reason, 0) + 1
    return {
        "scenario_count": len(scenarios),
        "result_count": len(rows),
        "model": MODEL,
        "policy_version": POLICY_VERSION,
        "policy_sha256": policy_sha256(),
        "scenario_sha256": _file_sha256(SCENARIO_PATH),
        "rubric_sha256": _file_sha256(RUBRIC_PATH),
        "budget": dict(budget_snapshot),
        "arms": arms,
        "route_diagnostics": diagnostics,
        "route_reason_counts": dict(sorted(route_reasons.items())),
        "buckets": bucket_summary,
        "failed_cases": [
            {
                "scenario_id": row["scenario_id"],
                "arm": row["arm"],
                "errors": row.get("errors") or [],
                "score": row.get("score") or {},
            }
            for row in rows if not _success(row)
        ],
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _write_jsonl_gz(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in sorted(rows, key=lambda item: (item["scenario_id"], item["arm"]))
    )
    with gzip.open(temporary, "wt", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    os.replace(temporary, path)


def _read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _report(summary: Mapping[str, Any], manifest: Mapping[str, Any]) -> str:
    diagnostic = summary["budget"].get("mode") == "offline-scripted"
    lines = [
        "# Cost-Aware Adaptive Routing R2",
        "",
        "Status: offline diagnostic only; not model-effect evidence."
        if diagnostic
        else "Status: completed frozen real-model three-arm run.",
        "",
        f"Model: `{summary['model']}`. Scenario SHA-256: "
        f"`{summary['scenario_sha256']}`.",
        f"Router policy: `{summary['policy_version']}` / "
        f"`{summary['policy_sha256']}`.",
        f"Scoring SHA-256: `{summary['rubric_sha256']}`.",
        "",
        "## Protocol",
        "",
        "- 10 dev + 24 held-out cases grouped by business fact/question template.",
        "- Three arms: fixed single-Agent, fixed multi-Agent, and adaptive routing.",
        "- Same `deepseek-flash`, evidence fixtures, tools, temperature, and output cap.",
        "- Gold is visible only to the deterministic scorer and confirmation action.",
        "- Adaptive rules decide before each turn; exactly one executor runs.",
        "- Routing latency is included; the rules policy has no model-call cost.",
        "- No held-out result was used to change features or thresholds.",
        "",
        "## Results",
        "",
        "| Arm | Task success | P50/P95 ms | Calls | Input/output tokens |"
        " Cost USD | Cost/success USD |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARM_NAMES:
        item = summary["arms"][arm]
        success = item["task_success"]
        lines.append(
            f"| {arm} | {success['numerator']}/{success['denominator']}"
            f" ({100 * (success['rate'] or 0):.1f}%) |"
            f" {item['p50_ms']}/{item['p95_ms']} | {item['model_calls']} |"
            f" {item['input_tokens']}/{item['output_tokens']} |"
            f" {item['cost_usd']:.6f} |"
            f" {item['cost_per_success_usd'] if item['cost_per_success_usd'] is not None else 'N/A'} |"
        )
    diagnostics = summary["route_diagnostics"]
    lines += [
        "",
        "## Paired routing",
        "",
        f"- Adaptive vs fixed single: `{diagnostics['adaptive_vs_single']}`.",
        f"- Adaptive vs fixed multi: `{diagnostics['adaptive_vs_multi']}`.",
        f"- False upgrades: `{diagnostics['false_upgrade_ids']}`.",
        f"- Missed upgrades: `{diagnostics['missed_upgrade_ids']}`.",
        f"- Adaptive selected multi on scenarios: "
        f"`{summary['arms']['adaptive']['multi_route_scenario_share']}`.",
        "",
        "## Buckets",
        "",
        "| Bucket | Fixed single | Fixed multi | Adaptive |",
        "|---|---:|---:|---:|",
    ]
    for bucket, grouped in summary["buckets"].items():
        cells = []
        for arm in ARM_NAMES:
            item = grouped[arm]
            cells.append(f"{item['numerator']}/{item['denominator']}")
        lines.append(f"| {bucket} | {' | '.join(cells)} |")
    lines += [
        "",
        "## Route reasons",
        "",
        "| Reason | Count |",
        "|---|---:|",
    ]
    for reason, count in summary["route_reason_counts"].items():
        lines.append(f"| `{reason}` | {count} |")
    lines += [
        "",
        "## Failures",
        "",
        "| Scenario | Arm | Errors | Failure detail |",
        "|---|---|---|---|",
    ]
    for failure in summary["failed_cases"]:
        score = failure["score"]
        detail = {
            key: score.get(key)
            for key in (
                "actual_outcome", "missing_conditions", "forbidden_conditions",
                "missing_tools", "write_correct",
            )
        }
        lines.append(
            f"| {failure['scenario_id']} | {failure['arm']} |"
            f" {', '.join(failure['errors']) or 'none'} |"
            f" {json.dumps(detail, ensure_ascii=False)} |"
        )
    budget = summary["budget"]
    lines += [
        "",
        "## Budget and limits",
        "",
        f"- Run cap: `${manifest['run_cap_usd']:.2f}`; charged upper bound:"
        f" `${budget.get('charged_upper_bound_usd', 0):.6f}`; pending:"
        f" `${budget.get('reserved_usd', 0):.6f}`.",
        "- Every model attempt reserves worst-case cost before dispatch and is settled"
        " from provider usage; missing usage is charged at the full reservation.",
        "- At most two bounded 429 retries are allowed; each attempt is ledgered.",
        "- Cold wall time is measured without response replay; provider cache effects do"
        " not turn this into an online latency claim.",
        "- The rules-only router reports zero routing API cost, not zero engineering or"
        " wall-time cost.",
        "- Results cover simulated orders and policies only and are not production ROI.",
        "",
        "## Sources",
        "",
        f"- [DeepSeek pricing]({PRICING_URL})",
        "- [LangChain router pattern](https://github.com/langchain-ai/docs/blob/main/src/oss/langchain/multi-agent/router.mdx)",
        "- [RouteLLM](https://github.com/lm-sys/routellm)",
        "",
    ]
    return "\n".join(lines)


async def run_experiment(
    *,
    allow_paid: bool = False,
    offline: bool = False,
    env_file: str = "",
    ledger_path: Path | None = None,
    out_dir: Path | None = None,
    split: str = "all",
    run_cap_usd: float = FINAL_LEDGER_CAP_USD,
) -> dict:
    manifest = verify_manifest(
        json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    )
    scenarios = [dict(row) for row in load_scenarios()]
    if split != "all":
        scenarios = [row for row in scenarios if row["split"] == split]
        if not scenarios:
            raise ValueError(f"unknown or empty split: {split}")
    out_dir = Path(out_dir or RESULTS_DIR)
    if offline:
        budget = _NullBudget()
        models = {
            "single": _OfflineModel("agent"),
            "planner": _OfflineModel("planner"),
            "retrieval": _OfflineModel("agent"),
            "action": _OfflineModel("action"),
            "supervisor": _OfflineModel("supervisor"),
        }
    else:
        if not allow_paid:
            raise BudgetExceeded("paid run requires explicit --allow-paid")
        source = env_file or str(Path("/root/workplace/ShopHelper/.env"))
        settings = Settings(_env_file=source)
        if not settings.openai_api_key:
            raise BudgetExceeded("OPENAI_API_KEY is missing")
        budget = _paid_budget(
            Path(ledger_path or LEDGER_PATH), run_cap_usd=run_cap_usd
        )
        if budget.snapshot().get("stop_reason"):
            raise BudgetExceeded(
                f"paid run previously stopped: {budget.snapshot()['stop_reason']}"
            )
        raw = _chat_model(settings.openai_api_key)
        models = {
            "single": raw,
            "planner": raw,
            "retrieval": raw,
            "action": raw,
            "supervisor": raw,
        }

    rows = []
    attempt_id = f"{int(time.time())}-{uuid.uuid4().hex[:10]}"
    for scenario in scenarios:
        for arm in ARM_NAMES:
            try:
                row = await run_scenario(
                    scenario,
                    arm=arm,
                    raw_models=models,
                    budget=budget,
                    attempt_id=attempt_id,
                )
            except Exception as exc:
                row = {
                    "scenario_id": scenario["id"],
                    "arm": arm,
                    "split": scenario["split"],
                    "source_group": scenario["source_group"],
                    "bucket": scenario["bucket"],
                    "category": scenario["category"],
                    "reply": "",
                    "replies": [],
                    "disposition": "",
                    "tool_calls": [],
                    "write_previews": [],
                    "confirmed_write_count": 0,
                    "unconfirmed_write_count": 0,
                    "sink_rows": [],
                    "duration_ms": 0,
                    "routing_ms": 0,
                    "routing_cost_usd": 0.0,
                    "route_targets": [],
                    "route_selected": "",
                    "route_decisions": [],
                    "errors": [f"harness:{type(exc).__name__}:{exc}"],
                    "model_calls": [],
                    "model_interactions": [],
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                }
                row["score"] = score_case(scenario, row)
                row["success"] = False
            row["attempt_id"] = attempt_id
            rows.append(row)
    snapshot = budget.snapshot() if hasattr(budget, "snapshot") else {
        "mode": "offline-scripted",
        "run_cap_usd": 0.0,
        "charged_upper_bound_usd": 0.0,
        "reserved_usd": 0.0,
    }
    summary = summarize_results(scenarios, rows, snapshot)
    summary["requested_split"] = split
    summary["attempt_id"] = attempt_id
    summary["manifest"] = manifest
    _write_jsonl_gz(out_dir / "traces.jsonl.gz", rows)
    _write_json(out_dir / "summary.json", summary)
    _write_json(out_dir / "budget.json", snapshot)
    target_report = REPORT_PATH if split == "all" else out_dir / "report.md"
    target_report.parent.mkdir(parents=True, exist_ok=True)
    target_report.write_text(_report(summary, manifest), encoding="utf-8")
    return summary


def rescore(out_dir: Path) -> dict:
    scenarios = {row["id"]: row for row in load_scenarios()}
    manifest = verify_manifest(
        json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    )
    rows = _read_jsonl(Path(out_dir) / "traces.jsonl.gz")
    for row in rows:
        row["score"] = score_case(scenarios[row["scenario_id"]], row)
        row["success"] = row["score"]["task_success"]
    previous = json.loads((Path(out_dir) / "summary.json").read_text(encoding="utf-8"))
    summary = summarize_results(load_scenarios(), rows, previous["budget"])
    summary["manifest"] = manifest
    summary["rescore_only"] = True
    summary["paid_calls_during_rescore"] = 0
    _write_jsonl_gz(Path(out_dir) / "traces.jsonl.gz", rows)
    _write_json(Path(out_dir) / "summary.json", summary)
    REPORT_PATH.write_text(_report(summary, manifest), encoding="utf-8")
    return summary


async def _cli_async(args) -> int:
    if args.rescore:
        summary = rescore(Path(args.out))
        print(json.dumps({
            "rescore_only": True,
            "paid_calls": 0,
            "arms": summary["arms"],
            "route_diagnostics": summary["route_diagnostics"],
        }, ensure_ascii=False, indent=2))
        return 0
    if args.freeze_manifest:
        _write_json(MANIFEST_PATH, build_manifest())
        print(json.dumps(build_manifest(), ensure_ascii=False, indent=2))
        return 0
    summary = await run_experiment(
        allow_paid=args.allow_paid,
        offline=args.offline,
        env_file=args.env_file,
        ledger_path=Path(args.ledger) if args.ledger else None,
        out_dir=Path(args.out) if args.out else None,
        split=args.split,
        run_cap_usd=args.run_cap_usd,
    )
    print(json.dumps({
        "arms": summary["arms"],
        "route_diagnostics": summary["route_diagnostics"],
        "budget": summary["budget"],
    }, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Frozen three-arm evaluation for adaptive Agent routing"
    )
    parser.add_argument("--allow-paid", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--freeze-manifest", action="store_true")
    parser.add_argument("--rescore", action="store_true")
    parser.add_argument("--split", choices=["all", "dev", "held-out"], default="all")
    parser.add_argument("--run-cap-usd", type=float, default=FINAL_LEDGER_CAP_USD)
    parser.add_argument("--env-file", default="")
    parser.add_argument("--ledger", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    if not args.offline and not args.allow_paid and not args.freeze_manifest \
            and not args.rescore:
        parser.error("refusing model execution without --offline or --allow-paid")
    return asyncio.run(_cli_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
