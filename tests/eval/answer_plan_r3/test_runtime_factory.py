from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.answer_plan.runtime import build_optional_answer_plan_runtime


class _Pricing:
    def cost(self, *_args, **_kwargs):
        return 0.001


def _settings(**updates):
    values = {
        "answer_plan_enabled": False,
        "answer_plan_ledger_path": "",
        "answer_plan_max_cost_usd": 2.0,
        "openai_model": "deepseek-flash",
        "structured_output_mode": "json",
        "extract_temperature": 0,
        "max_output_tokens": 1200,
        "llm_max_retries": 0,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def test_optional_factory_keeps_default_path_unmounted():
    assert build_optional_answer_plan_runtime(
        _settings(), base_model=object(), pricing=_Pricing()
    ) is None


def test_optional_factory_fails_closed_without_independent_ledger(tmp_path):
    with pytest.raises(ValueError, match="ledger"):
        build_optional_answer_plan_runtime(
            _settings(answer_plan_enabled=True),
            base_model=object(),
            pricing=_Pricing(),
        )


def test_optional_factory_builds_explicit_budgeted_runtime(tmp_path):
    runtime = build_optional_answer_plan_runtime(
        _settings(
            answer_plan_enabled=True,
            answer_plan_ledger_path=str(tmp_path / "plan.sqlite3"),
        ),
        base_model=object(),
        pricing=_Pricing(),
    )

    assert runtime is not None
    assert runtime.ledger.committed_usd() == 0.0
    assert callable(runtime.make_node())
