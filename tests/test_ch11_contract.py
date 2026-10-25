"""评测契约:必填字段、稳定哈希、diff 与失败枚举。"""
from __future__ import annotations

import pytest

from scripts.eval_contract import (
    FAILURE_LABELS,
    ContractError,
    build_contract,
    check_contract,
    contract_diff,
    contract_sha256,
)


def _contract(**overrides):
    payload = dict(
        dataset={"split": "frozen-v1", "sha256": "d", "n": 90},
        corpus={"docs_sha256": "docs", "facts_sha256": "facts", "doc_count": 8, "chunk_count": 130},
        index={"backend": "milvus-lite", "build_id": "b", "embedding_model": "m", "dim": 1024},
        retrieval={"strategy": "hybrid_rerank", "candidates": 50, "top_k": 10},
        rewrite={"model": "m", "mode": "frozen"},
        llm={"model": "deepseek-chat", "prompt_sha256": "p"},
        judge={"model": "deepseek-chat", "prompt_sha256": "j"},
        pricing={"version": "v", "currency": "USD", "verified": False},
        run={"git_commit": "abc", "seed": 1, "started_at": "t1", "host": "h1"},
        harness={"version": "1", "mode": "full"},
    )
    payload.update(overrides)
    return build_contract(**payload)


def test_contract_hash_ignores_run_started_at_and_host():
    left = _contract()
    right = _contract(run={"git_commit": "abc", "seed": 1, "started_at": "t2", "host": "h2"})
    assert contract_sha256(left) == contract_sha256(right)


def test_contract_diff_detects_model_change():
    left = _contract()
    right = _contract(llm={"model": "other", "prompt_sha256": "p"})
    assert "llm.model" in contract_diff(left, right)


def test_check_contract_rejects_missing_fields():
    contract = _contract()
    del contract["dataset"]["n"]
    with pytest.raises(ContractError):
        check_contract(contract)


def test_failure_labels_are_stable():
    assert FAILURE_LABELS == (
        "retrieval_miss", "entity_confusion", "version_conflict_missed",
        "evidence_insufficient", "gate_false_accept", "gate_false_block",
        "answer_wrong_fact", "answer_missing_condition", "forbidden_claim",
        "citation_unsupported", "over_refusal", "under_refusal",
        "clarify_missing_slot", "clarify_over_asking", "verification_overclaim",
        "tool_error", "timeout", "cost_over_budget", "data_error",
    )
