from app.query_guard import (
    audit_history_resolution,
    audit_query_rewrite,
    needs_history_resolution,
    safe_resolve_prompt,
)


def test_rewrite_guard_falls_back_when_explicit_order_is_removed():
    decision = audit_query_rewrite(
        "ORD-1002 未发货，能直接取消吗？",
        "订单未发货时能否直接取消",
    )

    assert decision.accepted is False
    assert decision.text == "ORD-1002 未发货，能直接取消吗？"
    assert "missing_anchor:ORD-1002" in decision.violations


def test_history_guard_rejects_hallucinated_or_swapped_entity():
    decision = audit_history_resolution(
        "它能取消吗？",
        "用户：SH-E300 多少钱？\n客服：299 元。",
        "SH-E120 能取消吗？",
    )

    assert decision.accepted is False
    assert decision.text == "它能取消吗？"
    assert "new_anchor:SH-E120" in decision.violations


def test_history_guard_accepts_anchor_grounded_in_history():
    decision = audit_history_resolution(
        "它续航多久？",
        "用户：SH-E300 多少钱？\n客服：299 元。",
        "SH-E300 续航多久？",
    )

    assert decision.accepted is True
    assert decision.text == "SH-E300 续航多久？"
    assert decision.violations == ()


def test_history_guard_allows_negative_condition_grounded_in_history():
    decision = audit_history_resolution(
        "那能取消吗？",
        "用户：ORD-1002 是什么状态？\n客服：已支付，尚未发货。",
        "ORD-1002 未发货时能取消吗？",
    )

    assert decision.accepted is True
    assert decision.text == "ORD-1002 未发货时能取消吗？"


def test_rewrite_guard_rejects_negation_flip():
    decision = audit_query_rewrite(
        "SH-E120 是不是不支持主动降噪？",
        "SH-E120 支持主动降噪吗？",
    )

    assert decision.accepted is False
    assert "polarity_changed" in decision.violations


def test_rewrite_guard_allows_neutral_question_rewrite():
    decision = audit_query_rewrite(
        "能不能告诉我 SH-E120 的续航",
        "SH-E120 续航多少小时？",
    )

    assert decision.accepted is True
    assert decision.text == "SH-E120 续航多少小时？"


def test_rewrite_guard_rejects_invented_policy_version():
    decision = audit_query_rewrite(
        "签收后多久能无理由退货？",
        "v2 版签收后多久能无理由退货？",
    )

    assert decision.accepted is False
    assert "new_anchor:v2" in decision.violations


def test_history_resolution_detector_distinguishes_ellipsis_from_complete_question():
    assert needs_history_resolution("它续航多久？") is True
    assert needs_history_resolution("那这个能取消吗？") is True
    assert needs_history_resolution("ORD-1002 当前是什么状态？") is False


def test_safe_resolve_prompt_makes_latest_message_authoritative():
    prompt = safe_resolve_prompt(
        "不要用旧型号，按 SH-E200 查续航",
        "用户：SH-E300 续航多久？",
    )

    assert "最新消息" in prompt
    assert "否定" in prompt
    assert "SH-E200" in prompt
