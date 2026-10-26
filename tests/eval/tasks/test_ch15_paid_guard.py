import json
from types import SimpleNamespace

import pytest

from scripts.eval_cost import BudgetExceeded, PricingTable
from scripts.ch15_paid_guard import (
    ArmLimit,
    GuardedChatModel,
    PaidRunBudget,
    PersistentPaidRunBudget,
    guard_graph_models,
    paid_call_context,
    validate_paid_settings,
)


def _pricing(*, verified=True):
    return PricingTable({
        "verified": verified,
        "models": {"test-model": {
            "input_per_million": 0.30,
            "input_cache_hit_per_million": 0.006,
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

    def with_structured_output(self, schema, **kwargs):
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


def test_graph_model_factory_includes_structured_output_schema_in_guarded_request():
    import asyncio

    provider = _ProviderFake()
    budget = _budget(max_input_tokens=1024, max_calls=4)
    models = guard_graph_models(
        budget=budget, arm="fixed", agent=provider, resolver=provider,
        intent=provider, expander=provider, resolver_schema=_LargeToolSpec())

    with pytest.raises(BudgetExceeded, match="输入上限"):
        asyncio.run(models["resolver"].ainvoke("short prompt"))

    assert provider.calls == 0
    assert budget.snapshot()["arms"]["fixed"]["generations"] == 0


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


def test_persistent_paid_budget_reconciles_actual_usage_and_survives_restart(tmp_path):
    path = tmp_path / "ch15.sqlite3"
    kwargs = dict(
        pricing=_pricing(), model="test-model", path=path, experiment_id="exp-v1",
        config_hash="sha256:config", arm_limits={"fixed": ArmLimit(2, 0.05)},
        run_cap_usd=0.05, max_input_tokens=2048, max_output_tokens=32,
        allow_paid=True,
    )
    budget = PersistentPaidRunBudget(**kwargs)
    call_id = budget.begin_generation(
        "fixed", request_bytes=400, scenario_id="CH15-RFD-001", turn=1,
        call_kind="agent", request={"messages": ["case prompt"]}, max_output_tokens=32,
    )
    assert budget.snapshot()["reserved_usd"] == pytest.approx(0.0006528)
    budget.complete_generation(call_id, usage={"input_tokens": 100, "output_tokens": 20,
                                              "cache_hit_tokens": 0,
                                              "cache_miss_tokens": 100})
    budget.close()

    reopened = PersistentPaidRunBudget(**kwargs)
    snapshot = reopened.snapshot()
    assert snapshot["arms"]["fixed"]["generations"] == 1
    assert snapshot["usage_cost_at_pinned_peak_rates_usd"] == pytest.approx(0.000054)
    assert snapshot["reserved_usd"] == 0
    reopened.close()


def test_persistent_pending_reservation_blocks_restart_and_counts_full_reserve(tmp_path):
    path = tmp_path / "ch15.sqlite3"
    kwargs = dict(
        pricing=_pricing(), model="test-model", path=path, experiment_id="exp-v1",
        config_hash="sha256:config", arm_limits={"fixed": ArmLimit(2, 0.05)},
        run_cap_usd=0.05, max_input_tokens=2048, max_output_tokens=32,
        allow_paid=True,
    )
    budget = PersistentPaidRunBudget(**kwargs)
    budget.begin_generation("fixed", request_bytes=400, scenario_id="CH15-RFD-001",
                            turn=1, call_kind="agent", request={"messages": ["prompt"]},
                            max_output_tokens=32)
    budget.close()

    reopened = PersistentPaidRunBudget(**kwargs)
    with pytest.raises(BudgetExceeded, match="未结算"):
        reopened.begin_generation("fixed", request_bytes=400, scenario_id="CH15-RFD-002",
                                 turn=1, call_kind="agent", request={"messages": ["prompt"]},
                                 max_output_tokens=32)
    assert reopened.snapshot()["reserved_usd"] == pytest.approx(0.0006528)
    reopened.close()


def test_persistent_provider_error_stops_future_paid_calls(tmp_path):
    path = tmp_path / "ch15.sqlite3"
    kwargs = dict(
        pricing=_pricing(), model="test-model", path=path, experiment_id="exp-v1",
        config_hash="sha256:config", arm_limits={"fixed": ArmLimit(2, 0.05)},
        run_cap_usd=0.05, max_input_tokens=2048, max_output_tokens=32,
        allow_paid=True,
    )
    budget = PersistentPaidRunBudget(**kwargs)
    call_id = budget.begin_generation(
        "fixed", request_bytes=400, scenario_id="CH15-RFD-001", turn=1,
        call_kind="agent", request={"messages": ["prompt"]}, max_output_tokens=32,
    )
    budget.complete_generation(call_id, error_type="TimeoutError", duration_ms=45000)
    snapshot = budget.snapshot()
    assert snapshot["stop_reason"] == "provider_call_failed:TimeoutError"
    assert snapshot["charged_upper_bound_usd"] == pytest.approx(0.0006528)
    with pytest.raises(BudgetExceeded, match="已因 provider 错误停止"):
        budget.begin_generation(
            "fixed", request_bytes=400, scenario_id="CH15-RFD-002", turn=1,
            call_kind="agent", request={"messages": ["another prompt"]},
            max_output_tokens=32,
        )
    budget.close()


def test_persistent_usage_prices_cache_hits_and_falls_back_to_all_miss_upper_bound(tmp_path):
    kwargs = dict(
        pricing=_pricing(), model="test-model", path=tmp_path / "ch15.sqlite3",
        experiment_id="exp-v1", config_hash="sha256:config",
        arm_limits={"fixed": ArmLimit(2, 0.05)}, run_cap_usd=0.05,
        max_input_tokens=2048, max_output_tokens=32, allow_paid=True,
    )
    budget = PersistentPaidRunBudget(**kwargs)
    exact_id = budget.begin_generation(
        "fixed", request_bytes=400, scenario_id="CH15-RFD-001", turn=1,
        call_kind="agent", request={"messages": ["prompt"]}, max_output_tokens=32,
    )
    budget.complete_generation(exact_id, usage={
        "input_tokens": 100, "output_tokens": 20,
        "cache_hit_tokens": 50, "cache_miss_tokens": 50,
    })
    upper_id = budget.begin_generation(
        "fixed", request_bytes=400, scenario_id="CH15-RFD-002", turn=1,
        call_kind="agent", request={"messages": ["prompt"]}, max_output_tokens=32,
    )
    budget.complete_generation(upper_id, usage={"input_tokens": 100, "output_tokens": 20})

    exact, upper = budget.calls_for_case("CH15-RFD-001", "fixed")[0], \
        budget.calls_for_case("CH15-RFD-002", "fixed")[0]
    snapshot = budget.snapshot()
    assert exact["status"] == "complete"
    assert exact["charged_usd"] == pytest.approx(0.0000393)
    assert upper["status"] == "complete_upper_bound"
    assert upper["charged_usd"] == pytest.approx(0.000054)
    assert snapshot["usage_cost_at_pinned_peak_rates_usd"] == pytest.approx(0.0000393)
    assert snapshot["charged_upper_bound_usd"] == pytest.approx(0.0000933)
    budget.close()


def test_guarded_chat_model_persists_only_model_visible_input_and_provider_usage(tmp_path):
    import asyncio
    from langchain_core.messages import AIMessage

    class ProviderWithUsage:
        async def ainvoke(self, value, *args, **kwargs):
            return AIMessage(content="真实模型答复",
                             usage_metadata={"input_tokens": 100, "output_tokens": 20,
                                             "total_tokens": 120,
                                             "input_token_details": {"cache_read": 0}})

    budget = PersistentPaidRunBudget(
        _pricing(), model="test-model", path=tmp_path / "ch15.sqlite3",
        experiment_id="exp-v1", config_hash="sha256:config",
        arm_limits={"fixed": ArmLimit(2, 0.05)}, run_cap_usd=0.05,
        max_input_tokens=2048, max_output_tokens=32, allow_paid=True,
    )
    model = GuardedChatModel(ProviderWithUsage(), budget, "fixed", "intent")

    async def run():
        with paid_call_context(scenario_id="CH15-RFD-001", arm="fixed", turn=1,
                               run_id="run-v1"):
            return await model.ainvoke({"messages": ["订单1001退款时效是什么？"]})

    result = asyncio.run(run())
    calls = budget.calls_for_case("CH15-RFD-001", "fixed")
    assert result.content == "真实模型答复"
    assert len(calls) == 1
    assert calls[0]["status"] == "complete"
    assert calls[0]["input_tokens"] == 100 and calls[0]["output_tokens"] == 20
    assert calls[0]["cache_hit_tokens"] == 0 and calls[0]["cache_miss_tokens"] == 100
    assert "订单1001退款时效是什么？" in json.dumps(calls[0]["request"], ensure_ascii=False)
    assert "gold" not in json.dumps(calls[0]["request"], ensure_ascii=False)
    assert budget.snapshot()["usage_cost_at_pinned_peak_rates_usd"] == pytest.approx(0.000054)
    budget.close()


def test_structured_output_guard_records_raw_provider_usage_and_returns_parsed_value(tmp_path):
    import asyncio
    from langchain_core.messages import AIMessage

    class StructuredProvider:
        include_raw = False

        def with_structured_output(self, schema, **kwargs):
            self.include_raw = kwargs.get("include_raw", False)
            return self

        async def ainvoke(self, value, *args, **kwargs):
            return {
                "raw": AIMessage(content='{"answer":"yes"}', usage_metadata={
                    "input_tokens": 100, "output_tokens": 20,
                    "total_tokens": 120,
                    "input_token_details": {"cache_read": 0},
                }),
                "parsed": SimpleNamespace(answer="yes"),
                "parsing_error": None,
            }

    provider = StructuredProvider()
    budget = PersistentPaidRunBudget(
        _pricing(), model="test-model", path=tmp_path / "ch15.sqlite3",
        experiment_id="exp-structured", config_hash="sha256:config",
        arm_limits={"fixed": ArmLimit(2, 0.05)}, run_cap_usd=0.05,
        max_input_tokens=4096, max_output_tokens=32, allow_paid=True,
    )
    guarded = GuardedChatModel(provider, budget, "fixed", "resolve").with_structured_output(
        _LargeToolSpec(), method="function_calling")

    async def run():
        with paid_call_context(scenario_id="CH15-RFD-001", arm="fixed", turn=1,
                               run_id="run-v1"):
            return await guarded.ainvoke("question")

    parsed = asyncio.run(run())
    calls = budget.calls_for_case("CH15-RFD-001", "fixed")
    assert provider.include_raw is True
    assert parsed.answer == "yes"
    assert len(calls) == 1
    assert calls[0]["status"] == "complete"
    assert calls[0]["input_tokens"] == 100 and calls[0]["output_tokens"] == 20
    assert "yes" in json.dumps(calls[0]["response"], ensure_ascii=False)
    assert budget.snapshot()["usage_cost_at_pinned_peak_rates_usd"] == pytest.approx(0.000054)
    budget.close()
