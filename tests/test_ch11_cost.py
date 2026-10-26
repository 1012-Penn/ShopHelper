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


def test_unbounded_budget_is_explicit_and_records_attempts(tmp_path):
    from scripts.eval_cost import AttemptLedger

    attempts = AttemptLedger(tmp_path / "attempts.jsonl")
    ledger = BudgetLedger(tmp_path / "ledger.json", total_limit_usd=None)
    tracker = CostTracker(
        _pricing(tmp_path), ledger, max_cost_usd=None, allow_paid=True,
        attempt_ledger=attempts,
    )
    tracker.guard(100, "deepseek-chat", kind="answer", request_text="private prompt")
    tracker.add("deepseek-chat", 100, 50, kind="answer")

    rows = [json.loads(line) for line in (tmp_path / "attempts.jsonl").read_text().splitlines()]
    assert [row["event"] for row in rows] == ["started", "succeeded"]
    assert rows[0]["request_sha256"] != "private prompt"
    assert "private prompt" not in (tmp_path / "attempts.jsonl").read_text()
    assert rows[1]["amount_usd"] == pytest.approx(0.0002)
    assert tracker.snapshot()["run_limit_usd"] is None


def test_attempt_ledger_blocks_resume_with_ambiguous_started_request(tmp_path):
    from scripts.eval_cost import AttemptLedger

    path = tmp_path / "attempts.jsonl"
    ledger = AttemptLedger(path)
    ledger.begin(model="deepseek-chat", kind="answer", estimated_input_tokens=50,
                 request_text="q", context={})

    with pytest.raises(BudgetExceeded, match="未闭合"):
        AttemptLedger(path)


def test_rag_worker_inherits_attempt_context(tmp_path):
    from app.rag_loop import _call_bounded
    from scripts.eval_cost import AttemptLedger, attempt_context

    attempts_path = tmp_path / "attempts.jsonl"
    tracker = CostTracker(
        _pricing(tmp_path), BudgetLedger(tmp_path / "ledger.json", None), None,
        allow_paid=True, attempt_ledger=AttemptLedger(attempts_path),
    )
    with attempt_context(split="dev", experiment="rag", group="controlled_loop",
                         case_id="DEV-MULTI-01"):
        _call_bounded(
            lambda: tracker.guard(100, "deepseek-chat", kind="answer", request_text="q"),
            timeout=2,
        )
    tracker.add("deepseek-chat", 100, 50, kind="answer")

    started = json.loads(attempts_path.read_text().splitlines()[0])
    assert {key: started[key] for key in ("split", "experiment", "group", "case_id")} == {
        "split": "dev", "experiment": "rag", "group": "controlled_loop",
        "case_id": "DEV-MULTI-01",
    }


def test_failed_attempt_is_persisted_and_redacts_api_key(tmp_path, monkeypatch):
    from scripts.eval_cost import AttemptLedger

    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    tracker = CostTracker(
        _pricing(tmp_path), BudgetLedger(tmp_path / "ledger.json", None), None,
        allow_paid=True, attempt_ledger=AttemptLedger(tmp_path / "attempts.jsonl"),
    )
    tracker.guard(10, "deepseek-chat", kind="answer", request_text="q")
    tracker.fail_attempt(RuntimeError("provider rejected test-secret"))

    rows = [json.loads(line) for line in (tmp_path / "attempts.jsonl").read_text().splitlines()]
    assert rows[-1]["event"] == "failed"
    assert "test-secret" not in str(rows)
    assert rows[-1]["retry_count"] == 0


def test_audited_chat_replays_exact_durable_response_without_second_call(tmp_path):
    from langchain_core.messages import AIMessage
    from scripts.eval_cost import AttemptLedger, AuditedChat, CallResultJournal

    class StubChat:
        def __init__(self):
            self.calls = 0

        def invoke(self, prompt):
            self.calls += 1
            return AIMessage(
                content="answer",
                usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
            )

    journal = CallResultJournal(tmp_path / "results.json")
    stub = StubChat()
    attempts_path = tmp_path / "attempts.jsonl"
    ledger_path = tmp_path / "costs.json"
    first_cost = CostTracker(
        _pricing(tmp_path), BudgetLedger(ledger_path, None), None, allow_paid=True,
        attempt_ledger=AttemptLedger(attempts_path),
    )
    first_cost.guard(100, "deepseek-chat", kind="answer", request_text="prompt")
    first = AuditedChat(stub, first_cost, journal, "deepseek-chat", "answer")
    assert first.invoke("prompt").content == "answer"
    first_cost.add("deepseek-chat", 100, 20, kind="answer")

    second_cost = CostTracker(
        _pricing(tmp_path), BudgetLedger(ledger_path, None), None, allow_paid=True,
        attempt_ledger=AttemptLedger(attempts_path),
    )
    second_cost.guard(100, "deepseek-chat", kind="answer", request_text="prompt")
    second = AuditedChat(stub, second_cost, journal, "deepseek-chat", "answer")
    assert second.invoke("prompt").content == "answer"
    assert second.last_cache_hit is True
    assert second_cost.add("deepseek-chat", 100, 20, kind="answer") == 0.0
    assert stub.calls == 1
    events = [json.loads(line)["event"] for line in attempts_path.read_text().splitlines()]
    assert events == ["started", "request_bound", "succeeded", "started", "request_bound", "cache_hit"]
    assert second_cost.api_attempts == 0


def test_corrupt_ledger_fails_closed(tmp_path):
    path = tmp_path / "ledger.json"
    path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(BudgetExceeded):
        BudgetLedger(path)
