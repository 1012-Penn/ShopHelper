from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from scripts.answer_plan_r3.budget import (
    AnswerPlanBudgetLedger,
    BudgetedModel,
    BudgetExceeded,
)


class _Pricing:
    def cost(self, *_args, **_kwargs):
        return 0.002


class _FlakyModel:
    def __init__(self):
        self.calls = 0

    def invoke(self, _request, **_kwargs):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("429 rate limit")
        return AIMessage(
            content="ok",
            usage_metadata={
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
            },
        )


def test_ledger_reserves_before_call_and_releases_pending_on_settle(tmp_path):
    ledger = AnswerPlanBudgetLedger(tmp_path / "ledger.sqlite3", limit_usd=0.1)
    first = ledger.reserve(0.06, model="m", kind="plan", case_id="D1")
    with pytest.raises(BudgetExceeded):
        ledger.reserve(0.05, model="m", kind="plan", case_id="D1")

    ledger.settle(
        first,
        actual_usd=0.01,
        usage={"input_tokens": 10, "output_tokens": 1},
    )
    second = ledger.reserve(0.05, model="m", kind="plan", case_id="D2")

    assert second
    assert ledger.committed_usd() == pytest.approx(0.06)


def test_budgeted_model_counts_bounded_retry_attempts(tmp_path):
    ledger = AnswerPlanBudgetLedger(tmp_path / "ledger.sqlite3", limit_usd=0.05)
    model = BudgetedModel(
        _FlakyModel(),
        model="deepseek-flash",
        pricing=_Pricing(),
        ledger=ledger,
        max_tokens=100,
        backoff_seconds=0,
    )

    response = model.invoke("hello", context={"case_id": "D1", "kind": "answer"})

    assert response.content == "ok"
    records = ledger.records()
    assert len(records) == 2
    assert [row["status"] for row in records] == ["provider_error", "succeeded"]
    assert sum(float(row["actual_usd"]) for row in records) > 0
    assert ledger.committed_usd() == pytest.approx(
        sum(float(row["actual_usd"]) for row in records)
    )
