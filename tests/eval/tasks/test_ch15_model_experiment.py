import json
import copy
from pathlib import Path

from scripts.ch15_model_experiment import (
    model_parameter_score,
    model_visible_turns,
    next_case_attempt_id,
    render_report,
    rescore_result,
    verify_frozen_experiment,
)


MODEL_TRACE = Path(__file__).resolve().parents[3] / "docs/eval/ch15-model-case-report.jsonl"


def model_result(scenario_id, arm="autonomous"):
    rows = [json.loads(line) for line in MODEL_TRACE.read_text(encoding="utf-8").splitlines()]
    return next(row for row in rows
                if row["scenario_id"] == scenario_id and row["arm"] == arm)


def test_frozen_paid_experiment_has_thirty_pairs_and_forty_two_user_turns():
    config, scenarios = verify_frozen_experiment()

    assert len(scenarios) == 30
    assert sum(len(row["turns"]) for row in scenarios) == 42
    assert len(config["scenario_ids"]) == 30
    assert config["arms"]["fixed"]["max_generations"] == 378
    assert config["arms"]["autonomous"]["max_generations"] == 336
    assert config["run_cap_usd"] == 17.40


def test_model_visible_projection_contains_user_text_only():
    row = {
        "id": "CH15-REV-001",
        "turns": [{"message": "订单1001想退款", "intents": {"fixed": "退款退货"},
                   "actors": {"fixed": {"reply": "预写回答"}},
                   "expected": {"fixed": {"tools": ["query_order"]}}}],
        "gold": {"fixed": {"must_include": ["不可退款"]}},
    }

    visible = model_visible_turns(row)
    serialized = json.dumps(visible, ensure_ascii=False)

    assert visible == ["订单1001想退款"]
    assert "gold" not in serialized
    assert "actors" not in serialized
    assert "expected" not in serialized
    assert "预写回答" not in serialized


def test_model_parameter_denominator_counts_missing_and_extra_calls():
    expected = [{"name": "query_order", "arguments": {"order_id": "1001"}}]
    good = model_parameter_score(expected, expected)
    missing = model_parameter_score(expected, [])
    extra = model_parameter_score([], [{"name": "delete_order", "arguments": {}}])

    assert good == {"correct_count": 1, "total": 1, "correct": True}
    assert missing == {"correct_count": 0, "total": 1, "correct": False}
    assert extra == {"correct_count": 0, "total": 1, "correct": False}


def test_incomplete_case_replay_gets_a_new_attempt_id_without_dropping_calls():
    prior = [
        {"run_id": "ch15-realmodel-paired-v1"},
        {"run_id": "ch15-realmodel-paired-v1"},
        {"run_id": "ch15-realmodel-paired-v1"},
    ]
    assert next_case_attempt_id(
        "ch15-realmodel-paired-v1", "CH15-RFD-001", "fixed", prior
    ) == "ch15-realmodel-paired-v1:CH15-RFD-001:fixed:attempt-2"


def test_model_report_metrics_table_renders_as_a_markdown_table():
    summary = {
        "experiment_status": "partial",
        "scenario_count": 0,
        "arm_run_count": 0,
        "arms": {},
        "paired_comparison": {},
        "budget": {
            "charged_upper_bound_usd": 0,
            "usage_cost_at_pinned_peak_rates_usd": 0,
            "reserved_usd": 0,
            "stop_reason": None,
            "arms": {},
        },
    }

    report = render_report(summary, {})

    assert "| Arm | Semantic task success | Original strict contract | Unconfirmed writes | Reasonable handoff | Refusal gold alignment |\n|---|" in report


def test_rescore_preserves_legacy_score_and_applies_semantic_rubric():
    config, scenarios = verify_frozen_experiment()
    row = next(row for row in scenarios if row["id"] == "CH15-RFD-001")
    result = copy.deepcopy(model_result("CH15-RFD-001"))

    rescored = rescore_result(row, "autonomous", result, config)

    assert rescored["legacy_score"]["task_success"] is False
    assert rescored["score"]["task_success"] is True
    assert rescored["score"]["scoring_version"] == "semantic_task_v2"


def test_report_explains_original_and_corrected_task_success():
    summary = {
        "experiment_status": "complete",
        "scoring_version": "semantic_task_v2",
        "scenario_count": 1,
        "arm_run_count": 2,
        "arms": {},
        "paired_comparison": {},
        "budget": {
            "charged_upper_bound_usd": 0,
            "usage_cost_at_pinned_peak_rates_usd": 0,
            "reserved_usd": 0,
            "stop_reason": None,
            "arms": {},
        },
    }

    report = render_report(summary, {})

    assert "Scoring correction" in report
    assert "original strict contract" in report
    assert "semantic task success" in report


def test_case_report_separates_nonblocking_diagnostics_from_task_failures():
    summary = {
        "experiment_status": "complete",
        "scoring_version": "semantic_task_v2",
        "scenario_count": 1,
        "arm_run_count": 1,
        "arms": {},
        "paired_comparison": {},
        "budget": {
            "charged_upper_bound_usd": 0,
            "usage_cost_at_pinned_peak_rates_usd": 0,
            "reserved_usd": 0,
            "stop_reason": None,
            "arms": {},
        },
    }
    result = {
        "score": {
            "task_success": True,
            "required_tools_correct": True,
            "parameter_correct": True,
            "refusal_correct": True,
            "handoff_correct": False,
            "outcome_correct": True,
            "includes_correct": True,
            "excludes_correct": True,
            "database_correct": True,
        },
        "attempt_run_id": "attempt-1",
        "model_calls": [],
        "duration_ms": 1,
    }

    report = render_report(summary, {("CH15-RFD-001", "autonomous"): result})

    assert "failures=none" in report
    assert "diagnostics=handoff_correct" in report
