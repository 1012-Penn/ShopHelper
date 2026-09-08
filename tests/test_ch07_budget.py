"""ch07 预算推导:演示配置逐位复现验收数字;只调窗口归零;默认配置不触发压缩。"""
from app.config import Settings
from app.context import derive_budgets


def _settings(**kw) -> Settings:
    return Settings(openai_api_key="sk-test", _env_file=None, **kw)


def test_demo_config_matches_acceptance_exactly():
    b = derive_budgets(_settings(model_context_window=18000, max_output_tokens=2000,
                                 max_user_input_tokens=2000, max_agent_steps=3,
                                 tool_result_max_tokens=1200, rerank_top_k=5))
    assert b.peak == 5600            # 2000 + 3*1200
    assert b.fixed == 4750           # 1800 + 5*250 + 200 + 1500
    assert b.sliding == 5650
    assert b.layer1 == 3954          # int(5650*0.7) 浮点截断语义
    assert b.layer2 == 1695


def test_window_only_collapses_to_zero():
    b = derive_budgets(_settings(model_context_window=18000))
    assert b.sliding == 0
    assert b.sliding < _settings().ctx_per_round_tokens  # 启动自检报警条件成立


def test_default_config_roomy_no_compression():
    b = derive_budgets(_settings())
    assert b.sliding == 43536
    assert b.history == 7500 and b.layer1 == 5250 and b.layer2 == 2250


def test_negative_clamps_to_zero():
    b = derive_budgets(_settings(model_context_window=1, ctx_safety_margin_tokens=100000))
    assert b.sliding == 0 and b.layer1 == 0 and b.layer2 == 0
