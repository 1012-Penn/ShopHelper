from concurrent.futures import ThreadPoolExecutor

import pytest

from scripts.e2e_quality_r2.budget import R2Budget
from scripts.eval_e2e_quality_r2 import (
    DEFAULT_RESULTS,
    ROOT,
    _build_parser,
    _CallCappedBudget,
    _run_contract,
)


def test_parser_defaults_to_shared_budget_and_accepts_independent_ledger(tmp_path):
    parser = _build_parser()

    defaults = parser.parse_args([])
    custom = parser.parse_args(["--session-budget", str(tmp_path / "smoke-budget.json")])

    assert defaults.session_budget == ROOT / "data/e2e-quality-r2/session-budget.json"
    assert custom.session_budget == tmp_path / "smoke-budget.json"
    assert defaults.output == DEFAULT_RESULTS


def test_parser_accepts_fixed_smoke_case_and_stop_rules():
    parser = _build_parser()

    args = parser.parse_args([
        "--case-ids", "DEV-DIRECT-E300,DEV-POLICY-V2,DEV-SHIP-REMOTE",
        "--max-provider-calls", "40",
        "--max-wall-seconds", "600",
    ])

    assert args.case_ids == "DEV-DIRECT-E300,DEV-POLICY-V2,DEV-SHIP-REMOTE"
    assert args.max_provider_calls == 40
    assert args.max_wall_seconds == 600


def test_parser_keeps_rag_depth_assertion_opt_in():
    parser = _build_parser()

    defaults = parser.parse_args([])
    smoke = parser.parse_args(["--require-rag-depth-integration"])

    assert defaults.require_rag_depth_integration is False
    assert smoke.require_rag_depth_integration is True


def test_contract_records_actual_budget_cap():
    contract = _run_contract(
        splits=["dev"],
        cases={"dev": []},
        arm="baseline",
        answer_verification_enabled=False,
        structured_mode="production",
        budget_limit_usd=0.5,
    )

    assert contract["budget_limit_usd"] == 0.5


def test_call_cap_counts_new_records_not_prior_ledger_history(tmp_path):
    prior = R2Budget(tmp_path / "budget.json", limit_usd=2)
    prior_attempt = prior.reserve({}, estimated_usd=0)
    prior.settle(prior_attempt, actual_usd=0, status="succeeded")

    budget = _CallCappedBudget(
        tmp_path / "budget.json", limit_usd=2, max_calls=1
    )
    first_new = budget.reserve({}, estimated_usd=0)
    budget.settle(first_new, actual_usd=0, status="succeeded")

    with pytest.raises(RuntimeError, match="provider-call stop rule reached: 1"):
        budget.reserve({}, estimated_usd=0)


def test_call_cap_counts_concurrent_pending_reservations(tmp_path):
    budget = _CallCappedBudget(tmp_path / "budget.json", limit_usd=2, max_calls=3)

    def reserve():
        try:
            return budget.reserve({}, estimated_usd=0)
        except RuntimeError:
            return None

    with ThreadPoolExecutor(max_workers=8) as executor:
        attempts = list(executor.map(lambda _: reserve(), range(8)))

    assert sum(attempt is not None for attempt in attempts) == 3
