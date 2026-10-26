import json
import hashlib
from pathlib import Path

import pytest
import scripts.eval_rag_ablation as ch16

from scripts.ch16_analysis import (
    ContractError,
    build_experiment_contract,
    check_cross_experiment_locks,
    check_phase_lock,
    protocol_sha256,
    validate_budget_request,
    validate_group_contracts,
)
from scripts.eval_cost import BudgetExceeded
from scripts.eval_rag_ablation import cache_only_reranker, estimate_group_costs


def _source(*, strategy="hybrid_rerank", rag_mode="baseline", model="deepseek-flash"):
    contract = {
        "contract_version": 1,
        "dataset": {"split": "dev", "sha256": "d", "n": 60},
        "corpus": {"docs_sha256": "docs", "facts_sha256": "facts", "chunk_count": 130},
        "index": {"backend": "milvus-lite", "build_id": "b", "snapshot_sha256": "s",
                  "embedding_model": "BAAI/bge-m3", "dim": 1024},
        "retrieval": {"strategy": strategy, "candidates": 50, "top_k": 10},
        "rewrite": {"mode": "frozen", "cache_sha256": "rewrite"},
        "llm": {"model": model, "temperature": 0.0, "prompt_sha256": "answer"},
        "judge": {"model": model, "prompt_sha256": "judge", "evidence_enabled": True},
        "pricing": {"version": "v", "currency": "USD", "verified": True},
        "run": {"git_commit": "abc", "seed": 1, "started_at": "time", "host": "host"},
        "harness": {"version": "ch11-harness-1", "mode": "full", "rag_mode": rag_mode},
    }
    if rag_mode == "baseline":
        contract["rag"] = {"mode": "baseline", "max_rounds": 3, "analyzer_enabled": False}
    else:
        contract["harness"]["version"] = "ch11-harness-1-rag"
        contract["rag"] = {"mode": rag_mode,
                           "max_rounds": 1 if rag_mode == "decompose_once" else 3,
                           "analyzer_enabled": True}
    return contract


def _group(experiment, strategy="hybrid_rerank", rag_mode="baseline"):
    return build_experiment_contract(
        split="dev",
        experiment=experiment,
        strategy=strategy,
        rag_mode=rag_mode,
        source_contract=_source(strategy=strategy, rag_mode=rag_mode),
    )


def test_retrieval_group_contracts_allow_only_strategy_to_change():
    contracts = [
        _group("retrieval", strategy=strategy)
        for strategy in ("dense", "bm25", "hybrid", "hybrid_rerank")
    ]

    validate_group_contracts(contracts)
    assert len({contract["protocol_sha256"] for contract in contracts}) == 1
    assert {contract["group"]["strategy"] for contract in contracts} == {
        "dense", "bm25", "hybrid", "hybrid_rerank"
    }


def test_rag_group_contracts_allow_only_rag_configuration_to_change():
    contracts = [
        _group("rag", rag_mode=mode)
        for mode in ("baseline", "decompose_once", "controlled_loop")
    ]

    validate_group_contracts(contracts)


def test_historical_contract_validation_can_skip_current_code_rehash(monkeypatch):
    contracts = [
        _group("retrieval", strategy=strategy)
        for strategy in ("dense", "bm25", "hybrid", "hybrid_rerank")
    ]

    def unexpected_rehash(_source):
        raise AssertionError("historical validation must use the stored protocol")

    monkeypatch.setattr(ch16, "protocol_sha256", unexpected_rehash)
    validate_group_contracts(contracts, verify_current_code=False)


def test_protocol_lock_allows_split_specific_rewrite_cache_hashes():
    dev = _source()
    frozen = _source()
    frozen["dataset"]["split"] = "frozen-v1"
    frozen["rewrite"]["cache_sha256"] = "frozen-rewrite"

    assert protocol_sha256(dev) == protocol_sha256(frozen)


def test_contract_rejects_model_or_candidate_cap_drift():
    contracts = [_group("retrieval", strategy="dense"), _group("retrieval", strategy="bm25")]
    contracts[1]["source_contract"]["llm"]["model"] = "different-model"
    with pytest.raises(ContractError, match="llm.model"):
        validate_group_contracts(contracts)

    contracts = [_group("retrieval", strategy="dense"), _group("retrieval", strategy="bm25")]
    contracts[1]["source_contract"]["retrieval"]["candidates"] = 49
    with pytest.raises(ContractError, match="候选上限"):
        validate_group_contracts(contracts)


def test_formal_frozen_run_requires_matching_dev_lock(tmp_path):
    lock = tmp_path / "dev-lock.json"
    lock.write_text(json.dumps({"protocol_sha256": "fixed"}), encoding="utf-8")

    check_phase_lock("dev", lock, "fixed")
    check_phase_lock("frozen-v1", lock, "fixed")
    with pytest.raises(ContractError, match="dev lock"):
        check_phase_lock("frozen-v1", lock, "changed")
    with pytest.raises(ContractError, match="dev lock"):
        check_phase_lock("frozen-v1", tmp_path / "missing.json", "fixed")


def test_dev_allows_refreshing_old_cross_experiment_lock(tmp_path):
    (tmp_path / "rag.json").write_text(
        json.dumps({"protocol_sha256": "other"}), encoding="utf-8"
    )

    check_cross_experiment_locks("dev", tmp_path, "fixed")


def test_frozen_run_requires_both_matching_dev_experiment_locks(tmp_path):
    (tmp_path / "retrieval.json").write_text(
        json.dumps({"protocol_sha256": "fixed"}), encoding="utf-8"
    )
    with pytest.raises(ContractError, match="必须先锁定 retrieval 与 rag 两组 dev 协议"):
        check_cross_experiment_locks("frozen-v1", tmp_path, "fixed")

    (tmp_path / "rag.json").write_text(
        json.dumps({"protocol_sha256": "different"}), encoding="utf-8"
    )
    with pytest.raises(ContractError, match="跨实验 dev lock"):
        check_cross_experiment_locks("frozen-v1", tmp_path, "fixed")


def test_paid_budget_requires_explicit_allowance_and_global_ledger():
    validate_budget_request(allow_paid=False, max_cost_usd=0.0, ledger=None)
    with pytest.raises(ContractError, match="allow-paid"):
        validate_budget_request(allow_paid=False, max_cost_usd=0.01, ledger=None)
    with pytest.raises(ContractError, match="ledger"):
        validate_budget_request(allow_paid=True, max_cost_usd=0.01, ledger=None)
    validate_budget_request(allow_paid=True, max_cost_usd=0.01, ledger="global-ledger.json")
    validate_budget_request(
        allow_paid=True, max_cost_usd=None, ledger="ch16-ledger.json", unbounded=True
    )
    with pytest.raises(ContractError, match="无限预算模式需要显式 --allow-paid"):
        validate_budget_request(
            allow_paid=False, max_cost_usd=None, ledger="ch16-ledger.json", unbounded=True
        )
    with pytest.raises(ContractError, match="无限模式需显式"):
        validate_budget_request(allow_paid=True, max_cost_usd=None, ledger="ledger.json")


@pytest.mark.parametrize("amount", [float("nan"), float("inf"), float("-inf")])
def test_paid_budget_rejects_non_finite_amounts(amount):
    with pytest.raises(ContractError, match="必须是有限数字"):
        validate_budget_request(
            allow_paid=True, max_cost_usd=amount, ledger="global-ledger.json"
        )


def test_rerank_cache_only_replays_cache_and_blocks_misses(tmp_path):
    model = "rerank-model"
    reranker = cache_only_reranker(tmp_path / "rerank.json", model)
    key = reranker.cache.key(model, "q", json.dumps(["doc"], ensure_ascii=False), "1")
    reranker.cache.put(key, {"results": [[0, 0.9]]})

    assert reranker.rerank("q", ["doc"], top_n=1) == [(0, 0.9)]
    with pytest.raises(BudgetExceeded, match="未发起外部调用"):
        reranker.rerank("other", ["doc"], top_n=1)


def test_estimates_are_per_group_and_never_authorize_spending():
    estimate = estimate_group_costs("frozen-v1")

    assert estimate["paid_calls_executed"] == 0
    assert estimate["total_max_cost_usd_if_all_groups_run"] > 0
    assert {row["group"] for row in estimate["groups"]} == {
        "vector", "bm25", "rrf", "rrf_rerank", "baseline", "decompose_once", "controlled_loop"
    }
    assert all(row["max_cost_usd_per_group"] >= 0 for row in estimate["groups"])


def test_retrieval_comparability_flags_mixed_answer_gate_policies():
    outputs = {
        "vector": [{"gate": {"evidence_entailment": None}}],
        "rrf_rerank": [{"gate": {"evidence_entailment": True}}],
    }

    result = ch16._answer_comparability("retrieval", outputs)

    assert result["retrieval_ranking_comparable"] is True
    assert result["answer_gate_policy_comparable"] is False
    assert result["gate_entailment_measured_cases"] == {
        "vector": 0, "rrf_rerank": 1
    }


def test_answer_gate_comparability_uses_case_membership_not_only_counts():
    outputs = {
        "vector": [
            {"case_id": "A", "gate": {"evidence_entailment": None}},
            {"case_id": "B", "gate": {"evidence_entailment": True}},
        ],
        "bm25": [
            {"case_id": "A", "gate": {"evidence_entailment": True}},
            {"case_id": "B", "gate": {"evidence_entailment": None}},
        ],
    }

    result = ch16._answer_comparability("retrieval", outputs)

    assert result["answer_gate_policy_comparable"] is False


def test_budget_audit_does_not_treat_estimate_as_enforced_approval():
    contracts = [_group("retrieval", strategy="dense")]
    contracts[0]["source_contract"]["run"].update({
        "ch16_allow_paid": True,
        "ch16_unbounded_budget": True,
        "ch16_max_cost_usd": None,
    })

    audit = ch16._budget_audit(contracts)

    assert audit["contract_mode"] == "unbounded"
    assert audit["enforced_group_caps"] is False
    assert audit["approval_artifact_recorded"] is False


def test_scoring_code_hashes_cover_reproduction_sources():
    hashes = ch16._scoring_code_hashes()

    assert set(hashes) == {
        "scripts/eval_checks.py", "scripts/ch11_harness.py",
        "scripts/ch16_analysis.py", "scripts/eval_rag_ablation.py",
    }
    assert hashes["scripts/eval_checks.py"] == hashlib.sha256(
        Path("scripts/eval_checks.py").read_bytes()
    ).hexdigest()


def test_snapshot_identity_accepts_only_matching_sha256(tmp_path):
    snapshot = tmp_path / "snapshot.db"
    snapshot.write_bytes(b"canonical snapshot fixture")
    expected = hashlib.sha256(snapshot.read_bytes()).hexdigest()

    assert ch16.validate_snapshot_identity(snapshot, expected_sha256=expected) == expected
    with pytest.raises(ContractError, match="快照 SHA-256 与 CH11 canonical 不一致"):
        ch16.validate_snapshot_identity(snapshot, expected_sha256="0" * 64)


def test_snapshot_identity_rejects_missing_file(tmp_path):
    with pytest.raises(ContractError, match="评测快照不存在"):
        ch16.validate_snapshot_identity(tmp_path / "missing.db", expected_sha256="0" * 64)


def test_case_checkpoint_is_atomic_and_bound_to_exact_group_contract(tmp_path):
    group = tmp_path / "group"
    contract = {
        "protocol_sha256": "protocol", "experiment": "retrieval", "split": "dev",
        "group": {"strategy": "dense"},
    }
    cases = [{"id": "DEV-01"}, {"id": "DEV-02"}]
    ch16._checkpoint_state(group, contract, cases)
    trace = {
        "case_id": "DEV-01", "retrieval_status": "measured",
        "answer_status": "measured",
    }
    ch16._checkpoint_append(group, trace)

    assert ch16._checkpoint_state(group, contract, cases) == [trace]
    changed = {**contract, "protocol_sha256": "changed"}
    with pytest.raises(ContractError, match="检查点契约不匹配"):
        ch16._checkpoint_state(group, changed, cases)


def test_checkpoint_resume_retries_unmeasured_cases_and_removes_old_rows(tmp_path):
    group = tmp_path / "group"
    contract = {
        "protocol_sha256": "protocol", "experiment": "rag", "split": "dev",
        "group": {"rag_mode": "controlled_loop"},
    }
    cases = [{"id": "DEV-01"}, {"id": "DEV-02"}]
    ch16._checkpoint_state(group, contract, cases)
    ch16._checkpoint_append(group, {
        "case_id": "DEV-01", "rag_status": "measured",
        "retrieval_status": "measured", "answer_status": "measured",
    })
    ch16._checkpoint_append(group, {
        "case_id": "DEV-02", "rag_status": "unmeasured",
        "retrieval_status": "unmeasured", "answer_status": "unmeasured",
    })

    traces = ch16._checkpoint_state(group, contract, cases)

    assert [trace["case_id"] for trace in traces] == ["DEV-01"]
    checkpoint_rows = [json.loads(row) for row in
                       (group / "case-checkpoint.jsonl").read_text().splitlines()]
    assert [trace["case_id"] for trace in checkpoint_rows] == ["DEV-01"]
