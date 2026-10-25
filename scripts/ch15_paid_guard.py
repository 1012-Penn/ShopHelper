"""Pre-call caps for a future CH15 paid-model comparison.

Each accepted generation reserves its worst-case cost at the configured
input/output envelope before the provider is called. Failed or timed-out calls
remain charged at that envelope, which is intentionally conservative.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from scripts.eval_cost import BudgetExceeded, PricingTable

MAX_INPUT_TOKENS = 65_536
MAX_OUTPUT_TOKENS = 2_000
RUN_CAP_USD = 17.40


@dataclass(frozen=True)
class ArmLimit:
    max_generations: int
    cap_usd: float


ARM_LIMITS = {
    "fixed": ArmLimit(max_generations=378, cap_usd=9.20),
    "autonomous": ArmLimit(max_generations=336, cap_usd=8.20),
}


def validate_paid_settings(settings: Any) -> None:
    """Reject settings that invalidate the published call-count envelope."""
    if int(getattr(settings, "llm_max_retries", -1)) != 0:
        raise BudgetExceeded("CH15付费对照要求 llm_max_retries=0，禁止隐藏重试")
    if bool(getattr(settings, "intent_escalation_enabled", True)):
        raise BudgetExceeded("CH15付费对照关闭意图升级，禁止额外模型调用")
    if int(getattr(settings, "max_agent_steps", MAX_OUTPUT_TOKENS)) > 6:
        raise BudgetExceeded("CH15付费对照的 Agent 步数上限为 6")
    if int(getattr(settings, "max_output_tokens", MAX_OUTPUT_TOKENS)) > MAX_OUTPUT_TOKENS:
        raise BudgetExceeded("CH15付费对照的输出 token 上限为 2000")


class PaidRunBudget:
    """In-memory CH15 run/arm envelopes; does not read or alter CH11 ledger.

    One instance must be shared by both serial arms. State is not durable across
    process restarts, so a resumed paid run must reconcile provider spend first.
    """

    def __init__(
        self,
        pricing: PricingTable,
        *,
        model: str,
        arm_limits: dict[str, ArmLimit] = ARM_LIMITS,
        run_cap_usd: float = RUN_CAP_USD,
        max_input_tokens: int = MAX_INPUT_TOKENS,
        max_output_tokens: int = MAX_OUTPUT_TOKENS,
        allow_paid: bool = False,
    ) -> None:
        self.pricing = pricing
        self.model = model
        self.arm_limits = dict(arm_limits)
        self.run_cap_usd = float(run_cap_usd)
        self.max_input_tokens = int(max_input_tokens)
        self.max_output_tokens = int(max_output_tokens)
        self.allow_paid = bool(allow_paid)
        self.max_call_cost_usd = pricing.cost(
            model, self.max_input_tokens, self.max_output_tokens)
        self._generations = {arm: 0 for arm in self.arm_limits}
        self._reserved_usd = {arm: 0.0 for arm in self.arm_limits}

        if self.run_cap_usd <= 0 or self.max_input_tokens <= 0 or self.max_output_tokens <= 0:
            raise ValueError("CH15付费预算和 token 上限必须为正数")
        if sum(limit.cap_usd for limit in self.arm_limits.values()) > self.run_cap_usd:
            raise ValueError("各组费用上限之和超过 CH15 总费用上限")
        for arm, limit in self.arm_limits.items():
            if limit.max_generations <= 0 or limit.cap_usd <= 0:
                raise ValueError(f"{arm} 组调用数和费用上限必须为正数")
            if self.max_call_cost_usd * limit.max_generations > limit.cap_usd + 1e-9:
                raise ValueError(f"{arm} 组硬调用上限无法落入费用封顶")

    def reserve_generation(self, arm: str, *, request_bytes: int) -> float:
        """Reserve one maximum-cost generation before dispatching it upstream."""
        if not self.allow_paid:
            raise BudgetExceeded("付费调用默认关闭；需要显式 allow_paid=True")
        if not self.pricing.verified:
            raise BudgetExceeded("价格未核实，拒绝付费调用")
        if arm not in self.arm_limits:
            raise BudgetExceeded(f"未知 CH15 评测组: {arm}")
        if request_bytes < 0 or request_bytes > self.max_input_tokens:
            raise BudgetExceeded(
                f"请求输入上限: {request_bytes} > {self.max_input_tokens} bytes/token-envelope")

        limit = self.arm_limits[arm]
        next_count = self._generations[arm] + 1
        if next_count > limit.max_generations:
            raise BudgetExceeded(f"{arm} 组调用次数上限 {limit.max_generations} 已达")
        next_arm_cost = self._reserved_usd[arm] + self.max_call_cost_usd
        next_run_cost = sum(self._reserved_usd.values()) + self.max_call_cost_usd
        if next_arm_cost > limit.cap_usd + 1e-9:
            raise BudgetExceeded(f"{arm} 组硬费用上限 ${limit.cap_usd:.2f} 将被突破")
        if next_run_cost > self.run_cap_usd + 1e-9:
            raise BudgetExceeded(f"CH15 总费用上限 ${self.run_cap_usd:.2f} 将被突破")

        self._generations[arm] = next_count
        self._reserved_usd[arm] = next_arm_cost
        return self.max_call_cost_usd

    def snapshot(self) -> dict:
        return {
            "model": self.model,
            "allow_paid": self.allow_paid,
            "pricing_verified": self.pricing.verified,
            "run_cap_usd": self.run_cap_usd,
            "reserved_usd": round(sum(self._reserved_usd.values()), 8),
            "max_call_cost_usd": round(self.max_call_cost_usd, 10),
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens": self.max_output_tokens,
            "arms": {
                arm: {
                    "generations": self._generations[arm],
                    "max_generations": limit.max_generations,
                    "cap_usd": limit.cap_usd,
                    "reserved_usd": round(self._reserved_usd[arm], 8),
                }
                for arm, limit in self.arm_limits.items()
            },
        }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    schema_builder = getattr(value, "model_json_schema", None)
    if callable(schema_builder):
        return _json_safe(schema_builder())
    if hasattr(value, "args_schema"):
        schema = getattr(value, "args_schema", None)
        schema_builder = getattr(schema, "model_json_schema", None)
        if callable(schema_builder):
            schema = schema_builder()
        return {
            "name": getattr(value, "name", ""),
            "description": getattr(value, "description", ""),
            "args_schema": _json_safe(schema),
        }
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _json_safe(model_dump())
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


def _encoded_request_size(value: Any) -> int:
    try:
        encoded = json.dumps(_json_safe(value), ensure_ascii=False).encode("utf-8")
    except (TypeError, ValueError):
        encoded = repr(value).encode("utf-8", errors="replace")
    # Reserve room for provider message framing and tool-call metadata.
    return len(encoded) + 512


class GuardedChatModel:
    """Guard LangChain-style invoke/stream entrypoints used by the CH15 graph."""

    def __init__(self, inner, budget: PaidRunBudget, arm: str, call_kind: str,
                 *, tool_schema: Any = None) -> None:
        self.inner = inner
        self.budget = budget
        self.arm = arm
        self.call_kind = call_kind
        self.tool_schema = tool_schema

    def _reserve(self, value: Any) -> None:
        request_bytes = _encoded_request_size(value)
        if self.tool_schema is not None:
            request_bytes += _encoded_request_size(self.tool_schema) - 512
        self.budget.reserve_generation(self.arm, request_bytes=request_bytes)

    def _validate_options(self, options: dict) -> None:
        pending = [options]
        while pending:
            current = pending.pop()
            for key, value in current.items():
                if key in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
                    if value is not None and int(value) > self.budget.max_output_tokens:
                        raise BudgetExceeded(
                            f"请求输出 token 上限: {key}={value} > "
                            f"{self.budget.max_output_tokens}")
                elif isinstance(value, dict):
                    pending.append(value)

    def bind_tools(self, tools, **kwargs):
        self._validate_options(kwargs)
        bound = self.inner.bind_tools(tools, **kwargs)
        schemas = [self.tool_schema, tools]
        return GuardedChatModel(bound, self.budget, self.arm, self.call_kind,
                                tool_schema=[schema for schema in schemas if schema is not None])

    def bind(self, *args, **kwargs):
        self._validate_options(kwargs)
        return GuardedChatModel(self.inner.bind(*args, **kwargs), self.budget,
                                self.arm, self.call_kind, tool_schema=self.tool_schema)

    def with_structured_output(self, *args, **kwargs):
        return GuardedChatModel(self.inner.with_structured_output(*args, **kwargs),
                                self.budget, self.arm, self.call_kind,
                                tool_schema=self.tool_schema)

    async def ainvoke(self, value, *args, **kwargs):
        self._validate_options(kwargs)
        self._reserve(value)
        return await self.inner.ainvoke(value, *args, **kwargs)

    async def astream(self, value, *args, **kwargs):
        self._validate_options(kwargs)
        self._reserve(value)
        async for item in self.inner.astream(value, *args, **kwargs):
            yield item

    def invoke(self, value, *args, **kwargs):
        self._validate_options(kwargs)
        self._reserve(value)
        return self.inner.invoke(value, *args, **kwargs)

    def stream(self, value, *args, **kwargs):
        self._validate_options(kwargs)
        self._reserve(value)
        yield from self.inner.stream(value, *args, **kwargs)


def guard_graph_models(*, budget: PaidRunBudget, arm: str, agent, resolver,
                       intent, expander) -> dict[str, GuardedChatModel]:
    """Wrap every LLM generation source injected into the CH15 graph.

    Call this after applying `with_structured_output` to resolver/expander. Returning
    a complete mapping avoids accidentally leaving an entry or Agent model unguarded.
    """
    components = {
        "model": (agent, "agent"),
        "resolver": (resolver, "resolve"),
        "intent_model": (intent, "intent"),
        "expander": (expander, "expand"),
    }
    missing = [name for name, (model, _kind) in components.items() if model is None]
    if missing:
        raise BudgetExceeded("CH15 付费图缺少待守卫模型: " + ", ".join(missing))
    return {
        name: GuardedChatModel(model, budget, arm, kind)
        for name, (model, kind) in components.items()
    }
