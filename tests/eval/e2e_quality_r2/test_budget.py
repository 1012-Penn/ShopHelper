from __future__ import annotations

import threading

import pytest

from scripts.e2e_quality_r2.budget import (
    BudgetExceeded,
    ExactCallCache,
    R2Budget,
)


def test_multiple_pending_reservations_are_supported_and_capped(tmp_path):
    budget = R2Budget(tmp_path / "budget.json", limit_usd=2.0)
    first = budget.reserve({"case_id": "D1", "kind": "generation"}, 0.8)
    second = budget.reserve({"case_id": "D2", "kind": "judge"}, 0.7)

    with pytest.raises(BudgetExceeded):
        budget.reserve({"case_id": "D3", "kind": "generation"}, 0.6)

    budget.settle(first, actual_usd=0.2, status="succeeded")
    third = budget.reserve({"case_id": "D3", "kind": "generation"}, 0.6)
    assert third
    budget.settle(second, actual_usd=0.7, status="succeeded")
    budget.settle(third, actual_usd=0.6, status="succeeded")
    assert budget.used_usd() == pytest.approx(1.5)


def test_concurrent_reservations_do_not_overcommit(tmp_path):
    budget = R2Budget(tmp_path / "budget.json", limit_usd=1.0)
    accepted: list[str] = []
    rejected: list[Exception] = []

    def worker(index: int) -> None:
        try:
            accepted.append(
                budget.reserve({"case_id": f"C{index}", "kind": "generation"}, 0.05)
            )
        except BudgetExceeded as exc:
            rejected.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(24)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(accepted) == 20
    assert len(rejected) == 4
    assert budget.pending_usd() == pytest.approx(1.0)


def test_exact_cache_keys_prompt_tools_schema_and_split(tmp_path):
    cache = ExactCallCache(tmp_path / "cache.json")
    base = cache.key(
        namespace="e2e-r2",
        model="deepseek-flash",
        request="question",
        schema={"type": "object"},
        tools=["query_faq"],
        split="dev",
        turn=0,
    )
    assert base == cache.key(
        namespace="e2e-r2",
        model="deepseek-flash",
        request="question",
        schema={"type": "object"},
        tools=["query_faq"],
        split="dev",
        turn=0,
    )
    assert base != cache.key(
        namespace="e2e-r2",
        model="deepseek-flash",
        request="question",
        schema={"type": "object"},
        tools=["query_faq"],
        split="heldout",
        turn=0,
    )
    assert base != cache.key(
        namespace="e2e-r2",
        model="deepseek-flash",
        request="question",
        schema={"type": "object"},
        tools=["query_order"],
        split="dev",
        turn=0,
    )
    cache.put(base, {"answer": "cached"})
    assert cache.get(base) == {"answer": "cached"}
