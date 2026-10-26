import json

from scripts.ch15_model_experiment import (
    model_parameter_score,
    model_visible_turns,
    next_case_attempt_id,
    render_report,
    verify_frozen_experiment,
)


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

    assert "| Arm | Task success | Tool selection | Tool parameters | Unconfirmed writes | Reasonable handoff | Refusal gold alignment |\n|---|" in report
