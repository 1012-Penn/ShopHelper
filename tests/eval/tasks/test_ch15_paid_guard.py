from types import SimpleNamespace

import pytest

from scripts.eval_cost import BudgetExceeded, PricingTable
from scripts.ch15_paid_guard import (
    ArmLimit,
    GuardedChatModel,
    PaidRunBudget,
    guard_graph_models,
    validate_paid_settings,
)


def _pricing(*, verified=True):
    return PricingTable({
        "verified": verified,
        "models": {"test-model": {
            "input_per_million": 0.30,
            "output_per_million": 1.20,
        }},
    })


def _budget(*, allow_paid=True, max_calls=1, max_input_tokens=2048):
    return PaidRunBudget(
        _pricing(),
        model="test-model",
        arm_limits={"fixed": ArmLimit(max_generations=max_calls, cap_usd=1.0)},
        run_cap_usd=1.0,
        max_input_tokens=max_input_tokens,
        max_output_tokens=32,
        allow_paid=allow_paid,
    )


class _ProviderFake:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, value, *args, **kwargs):
        self.calls += 1
        return {"content": "ok"}

    async def astream(self, value, *args, **kwargs):
        self.calls += 1
        yield "ok"

    def bind_tools(self, tools, **kwargs):
        return self


class _LargeToolSpec:
    name = "query_order"
    description = "d" * 1500
    args_schema = {"type": "object", "properties": {"order_id": {"type": "string"}}}

    def __repr__(self):
        return "<tool>"


async def collect(stream):
    return [item async for item in stream]


def test_paid_guard_rejects_oversized_request_before_provider_call():
    provider = _ProviderFake()
    model = GuardedChatModel(provider, _budget(max_input_tokens=512), "fixed", "agent")

    with pytest.raises(BudgetExceeded, match="输入上限"):
        import asyncio
        asyncio.run(model.ainvoke("x" * 200))

    assert provider.calls == 0


def test_paid_guard_counts_bound_tool_description_and_schema_in_request_envelope():
    provider = _ProviderFake()
    model = GuardedChatModel(provider, _budget(max_input_tokens=1024), "fixed", "agent")
    bound = model.bind_tools([_LargeToolSpec()])

    with pytest.raises(BudgetExceeded, match="输入上限"):
        import asyncio
        asyncio.run(bound.ainvoke("short prompt"))

    assert provider.calls == 0


def test_paid_guard_blocks_generation_after_arm_limit_even_through_bound_stream():
    import asyncio

    provider = _ProviderFake()
    model = GuardedChatModel(provider, _budget(max_calls=1), "fixed", "agent")
    bound = model.bind_tools([{"name": "query_order", "parameters": {}}])

    assert asyncio.run(collect(bound.astream("order 1001"))) == ["ok"]
    with pytest.raises(BudgetExceeded, match="调用次数上限"):
        asyncio.run(collect(bound.astream("order 1001")))

    assert provider.calls == 1


def test_paid_guard_rejects_output_override_before_provider_call():
    import asyncio

    provider = _ProviderFake()
    model = GuardedChatModel(provider, _budget(), "fixed", "agent")

    with pytest.raises(BudgetExceeded, match="输出 token 上限"):
        asyncio.run(model.ainvoke("short prompt", max_tokens=2048))

    assert provider.calls == 0


def test_paid_guard_rejects_nested_provider_output_override():
    import asyncio

    provider = _ProviderFake()
    model = GuardedChatModel(provider, _budget(), "fixed", "agent")

    with pytest.raises(BudgetExceeded, match="输出 token 上限"):
        asyncio.run(model.ainvoke("short prompt", model_kwargs={"max_tokens": 2048}))

    assert provider.calls == 0


def test_graph_model_factory_guards_resolve_intent_expand_and_agent_calls():
    import asyncio

    provider = _ProviderFake()
    budget = _budget(max_calls=4)
    models = guard_graph_models(
        budget=budget, arm="fixed", agent=provider, resolver=provider,
        intent=provider, expander=provider)

    for model in models.values():
        asyncio.run(model.ainvoke("short prompt"))

    assert set(models) == {"model", "resolver", "intent_model", "expander"}
    assert provider.calls == 4
    assert budget.snapshot()["arms"]["fixed"]["generations"] == 4


def test_paid_guard_requires_explicit_opt_in_and_verified_prices():
    guarded = _budget(allow_paid=False)

    with pytest.raises(BudgetExceeded, match="allow_paid"):
        guarded.reserve_generation("fixed", request_bytes=4)

    unverified = PaidRunBudget(
        _pricing(verified=False),
        model="test-model",
        arm_limits={"fixed": ArmLimit(max_generations=1, cap_usd=1.0)},
        run_cap_usd=1.0,
        max_input_tokens=2048,
        max_output_tokens=32,
        allow_paid=True,
    )
    with pytest.raises(BudgetExceeded, match="价格未核实"):
        unverified.reserve_generation("fixed", request_bytes=4)


def test_default_envelope_reserves_worst_case_and_fits_each_group_cap():
    budget = PaidRunBudget(_pricing(), model="test-model", allow_paid=True)

    assert budget.snapshot()["max_call_cost_usd"] == pytest.approx(0.0220608)
    assert budget.snapshot()["arms"]["fixed"]["max_generations"] == 378
    assert budget.snapshot()["arms"]["autonomous"]["max_generations"] == 336
    assert budget.snapshot()["run_cap_usd"] == 17.40


def test_paid_settings_reject_retries_escalation_and_larger_call_envelopes():
    settings = SimpleNamespace(
        llm_max_retries=1,
        intent_escalation_enabled=False,
        max_agent_steps=6,
        max_output_tokens=2000,
    )

    with pytest.raises(BudgetExceeded, match="重试"):
        validate_paid_settings(settings)

    settings.llm_max_retries = 0
    settings.intent_escalation_enabled = True
    with pytest.raises(BudgetExceeded, match="升级"):
        validate_paid_settings(settings)

    settings.intent_escalation_enabled = False
    settings.max_agent_steps = 7
    with pytest.raises(BudgetExceeded, match="Agent 步数"):
        validate_paid_settings(settings)
