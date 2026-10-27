from __future__ import annotations

import pytest

from scripts.answer_quality_depth.budget import (
    AnswerQualityBudgetExceeded,
    AnswerQualityLedger,
    CappedProvider,
)


def test_ledger_counts_pending_reservations_against_cap(tmp_path) -> None:
    ledger = AnswerQualityLedger(tmp_path / "ledger.sqlite3", total_limit_usd=2.0)
    ledger.reserve(1.25, model="deepseek-flash", kind="answer", note="first")

    with pytest.raises(AnswerQualityBudgetExceeded):
        ledger.reserve(0.80, model="deepseek-flash", kind="judge", note="second")


def test_ledger_settlement_releases_unused_reservation(tmp_path) -> None:
    ledger = AnswerQualityLedger(tmp_path / "ledger.sqlite3", total_limit_usd=2.0)
    first = ledger.reserve(1.25, model="deepseek-flash", kind="answer")
    ledger.settle(first, actual_usd=0.1, usage={"output_tokens": 10})
    ledger.reserve(1.8, model="deepseek-flash", kind="judge")

    assert ledger.total_usd() == pytest.approx(1.9)


class _Response:
    def __init__(self, content: str = "ok") -> None:
        self.id = "response-id"
        self.model = "deepseek-flash-2026"
        self.choices = [
            type(
                "Choice", (), {"message": type("Message", (), {"content": content})()}
            )()
        ]
        self.usage = type(
            "Usage",
            (),
            {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "prompt_tokens_details": type("Details", (), {"cached_tokens": 0})(),
            },
        )()

    def model_dump(self, mode="json"):
        return {"id": self.id, "model": self.model}


class _RetryableError(Exception):
    status_code = 429


class _Client:
    def __init__(self, outcomes) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0
        self.chat = type("Chat", (), {"completions": self})()

    def create(self, **_kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def test_capped_provider_retries_at_most_twice_and_settles_each_attempt(
    tmp_path,
) -> None:
    ledger = AnswerQualityLedger(tmp_path / "ledger.sqlite3", total_limit_usd=2.0)
    client = _Client(
        [_RetryableError("429"), _RetryableError("429"), _Response("recovered")]
    )
    provider = CappedProvider(
        client,
        ledger=ledger,
        pricing={"input_per_million": 0.30, "output_per_million": 1.20},
        model="deepseek-flash",
        max_retries=2,
        retry_backoff_seconds=0,
    )

    response = provider.call(
        [{"role": "user", "content": "hello"}],
        kind="answer",
        max_tokens=100,
    )

    assert response.text == "recovered"
    assert client.calls == 3
    assert len(ledger.records()) == 3
    assert all(record["state"] == "settled" for record in ledger.records())
