"""预算表、台账与调用前护栏。"""
from __future__ import annotations

import json

import pytest

from scripts.eval_cost import (
    BudgetExceeded,
    BudgetLedger,
    CostTracker,
    PricingTable,
)


def _pricing(tmp_path, *, verified=True):
    path = tmp_path / "pricing.yaml"
    path.write_text(
        f"""
version: "test"
currency: USD
verified: {str(verified).lower()}
source_url: "https://example.test"
models:
  deepseek-chat:
    input_per_million: 1.0
    output_per_million: 2.0
""",
        encoding="utf-8",
    )
    return PricingTable.load(path)


def test_pricing_cost_math(tmp_path):
    pricing = _pricing(tmp_path)
    assert pricing.cost("deepseek-chat", 1_000_000, 1_000_000) == 3.0


def test_allow_paid_false_raises(tmp_path):
    tracker = CostTracker(_pricing(tmp_path), BudgetLedger(tmp_path / "ledger.json"), 10)
    with pytest.raises(BudgetExceeded):
        tracker.guard(10, "deepseek-chat")
    with pytest.raises(BudgetExceeded):
        tracker.add("deepseek-chat", 10, 10)


def test_unverified_pricing_raises(tmp_path):
    tracker = CostTracker(
        _pricing(tmp_path, verified=False),
        BudgetLedger(tmp_path / "ledger.json"),
        10,
        allow_paid=True,
    )
    with pytest.raises(BudgetExceeded):
        tracker.guard(10, "deepseek-chat")


def test_run_cap_and_total_ledger(tmp_path):
    ledger = BudgetLedger(tmp_path / "ledger.json", total_limit_usd=1.0)
    tracker = CostTracker(_pricing(tmp_path), ledger, max_cost_usd=0.5, allow_paid=True)
    with pytest.raises(BudgetExceeded):
        tracker.guard(1_000_000, "deepseek-chat")
    tracker.add("deepseek-chat", 100_000, 100_000)
    assert ledger.total_usd() == pytest.approx(0.3)
    assert json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))["limit_usd"] == 1.0


def test_ledger_cap_raises(tmp_path):
    ledger = BudgetLedger(tmp_path / "ledger.json", total_limit_usd=0.1)
    ledger.record(0.09, "prior")
    tracker = CostTracker(_pricing(tmp_path), ledger, max_cost_usd=1.0, allow_paid=True)
    with pytest.raises(BudgetExceeded):
        tracker.guard(1_000_000, "deepseek-chat")


def test_snapshot_reports_limits(tmp_path):
    tracker = CostTracker(
        _pricing(tmp_path), BudgetLedger(tmp_path / "ledger.json"), 2.0, allow_paid=True
    )
    snapshot = tracker.snapshot()
    assert snapshot["run_limit_usd"] == 2.0
    assert snapshot["ledger_limit_usd"] == 10.0
    assert snapshot["pricing_verified"] is True


def test_corrupt_ledger_fails_closed(tmp_path):
    path = tmp_path / "ledger.json"
    path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(BudgetExceeded):
        BudgetLedger(path)
