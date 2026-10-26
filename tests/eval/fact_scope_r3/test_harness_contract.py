from tests.eval.fact_scope_r3.harness import (
    ARMS,
    MAX_CONTEXT_CHARS,
    AnswerOutput,
    _knowledge_service,
    build_context,
    load_split,
    preflight,
    relevant_chunk_ids,
    score_output,
)


def test_frozen_split_contract_has_eight_dev_and_twenty_heldout():
    result = preflight()

    assert result["status"] == "ok"
    assert result["dev_cases"] == 8
    assert result["heldout_cases"] == 20
    assert result["groups_disjoint"] is True
    assert result["arms"] == list(ARMS)


def test_prompts_do_not_contain_gold_and_have_shared_context_cap():
    cases, _ = load_split("dev")
    kb, service = _knowledge_service()
    case = next(row for row in cases if row["id"] == "FSR3-DEV-005")

    contexts = [
        build_context(case, arm, kb, service)
        for arm in ARMS
    ]

    for context in contexts:
        assert context.prompt_chars <= MAX_CONTEXT_CHARS + 1000
        assert "expected_behavior" not in context.prompt
        assert "window_days" not in context.prompt
        assert "gold" not in context.prompt.casefold()


def test_scope_arm_is_the_only_arm_with_scope_exclusions():
    cases, _ = load_split("dev")
    kb, service = _knowledge_service()
    case = next(row for row in cases if row["id"] == "FSR3-DEV-005")

    baseline = build_context(case, "baseline", kb, service)
    identity = build_context(case, "identity", kb, service)
    scope = build_context(case, "scope", kb, service)

    assert baseline.retrieval["scope_excluded_count"] == 0
    assert identity.retrieval["scope_excluded_count"] == 0
    assert scope.retrieval["scope_status"] in {"partial", "verified", "unknown"}
    assert scope.retrieval["scope_excluded_count"] > 0


def test_scorer_keeps_subject_and_condition_metrics_separate():
    cases, gold = load_split("dev")
    kb, service = _knowledge_service()
    case = next(row for row in cases if row["id"] == "FSR3-DEV-005")
    context = build_context(case, "scope", kb, service)
    relevant = relevant_chunk_ids(kb, gold[case["id"]]["relevant_markers"])
    output = AnswerOutput(
        action="answer",
        order_subject="corpus:ORD-1004",
        policy_version="v2",
        window_days=7,
        answer="适用 v2，窗口 7 天。",
        cited_chunk_ids=[],
    )

    score = score_output(output, gold[case["id"]], context, relevant)

    assert score["subject_correct"] is True
    assert score["conditions_correct"] is False
    assert score["task_success"] is False
    assert score["field_mismatches"]["window_days"]["expected"] == 15
