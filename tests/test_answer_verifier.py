from app.answer_verifier import AnswerVerdict, AnswerVerifier


class FakeSemantic:
    def __init__(self, verdicts):
        self.verdicts = list(verdicts)
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        return self.verdicts.pop(0)


def _evidence():
    return [
        {
            "n": 1,
            "question": "订单状态总览",
            "answer": "| ORD-1004 | 2026-06-08 16:05 | 399 元 | 金卡会员 | 已签收 |",
        },
        {
            "n": 2,
            "question": "v1 七天无理由",
            "answer": "签收日期早于 2026-06-01 的订单适用 v1，7 天内无理由退货。",
        },
        {
            "n": 3,
            "question": "v2 十五天无理由",
            "answer": "签收日期为 2026-06-01 或之后的订单适用 v2，15 天内无理由退货。",
        },
    ]


def test_unknown_citation_fails_before_semantic_judge():
    semantic = FakeSemantic([])
    verifier = AnswerVerifier(semantic)
    result = verifier.verify(
        "ORD-1004 签收于 2026-06-08，适用 v2，15 天内可退[9]。",
        _evidence(),
    )
    assert result.status == "fail"
    assert any("引用编号" in issue for issue in result.issues)
    assert semantic.calls == 0


def test_current_order_answer_cannot_use_another_orders_receipt_date():
    semantic = FakeSemantic([AnswerVerdict(status="pass", reason="语义一致")])
    verifier = AnswerVerifier(semantic)
    result = verifier.verify(
        "ORD-1004 签收于 2026-06-12，适用 v2，15 天内可退[1][2].",
        _evidence(),
    )
    assert result.status == "fail"
    assert any("ORD-1004" in issue and "签收日期" in issue for issue in result.issues)


def test_answer_cannot_introduce_order_id_absent_from_question():
    semantic = FakeSemantic([AnswerVerdict(status="pass", reason="忽略实体引入")])
    verifier = AnswerVerifier(semantic)
    result = verifier.verify(
        "ORD-1006 的商品为 SH-E300，签收日期为 2026-06-08，适用 v2[1][3]。",
        _evidence(),
        query="SH-E300 签收第 6 天，包装拆了但商品完好，能退吗？",
    )
    assert result.status == "fail"
    assert any("未提供的订单" in issue for issue in result.issues)
    assert semantic.calls == 0


def test_order_date_conflict_requires_explicit_correction_of_user_premise():
    semantic = FakeSemantic([AnswerVerdict(status="pass", reason="事实一致")])
    verifier = AnswerVerifier(semantic)
    unresolved = verifier.verify(
        "ORD-1004 签收于 2026-06-08，适用 v2，15 天内可退[1][3]。",
        _evidence(),
        query="ORD-1004 签收于 2026-06-05，退货窗口是多久？",
    )
    assert unresolved.status == "fail"
    assert any("用户提供的签收日期" in issue for issue in unresolved.issues)

    clarified = verifier.verify(
        "ORD-1004 的实际签收日期为 2026-06-08，不是用户所说的 2026-06-05，"
        "适用 v2，15 天内可退[1][3]。",
        _evidence(),
        query="ORD-1004 签收于 2026-06-05，退货窗口是多久？",
    )
    assert clarified.status == "pass"


def test_policy_version_and_window_must_match_signed_date():
    semantic = FakeSemantic([AnswerVerdict(status="pass", reason="语义一致")])
    verifier = AnswerVerifier(semantic)
    result = verifier.verify(
        "ORD-1004 签收于 2026-05-20，适用 v2，15 天内可退[2][3]。",
        _evidence(),
    )
    assert result.status == "fail"
    assert any("v2" in issue or "15 天" in issue for issue in result.issues)


def test_semantic_pass_is_required_even_when_entities_and_numbers_match():
    semantic = FakeSemantic([AnswerVerdict(status="pass", reason="事实一致")])
    verifier = AnswerVerifier(semantic)
    result = verifier.verify(
        "ORD-1004 签收于 2026-06-08，适用 v2，15 天内可退[1][3]。",
        _evidence(),
    )
    assert result.status == "pass"
    assert semantic.calls == 1


def test_review_revises_at_most_once_and_accepts_clean_revision():
    semantic = FakeSemantic([AnswerVerdict(status="pass", reason="修订正确")])
    verifier = AnswerVerifier(semantic, max_revisions=1)
    calls = []

    def revise(answer, issues):
        calls.append((answer, issues))
        return "ORD-1004 签收于 2026-06-08，适用 v2，15 天内可退[1][3]。"

    result = verifier.review(
        "ORD-1004 签收于 2026-06-08，适用 v2，30 天内可退[1][3]。",
        _evidence(),
        revise=revise,
    )
    assert result.status == "pass"
    assert len(calls) == 1
    assert result.revision_calls == 1
    assert result.final_answer.endswith("[1][3]。")


def test_review_uses_safe_refusal_after_one_failed_revision():
    semantic = FakeSemantic([
        AnswerVerdict(status="fail", reason="实体错误"),
        AnswerVerdict(status="fail", reason="修订仍错误"),
    ])
    verifier = AnswerVerifier(semantic, max_revisions=1)
    result = verifier.review(
        "错误答案[1]。",
        _evidence(),
        revise=lambda _answer, _issues: "仍然错误的答案[1]。",
    )
    assert result.status == "refuse"
    assert result.revision_calls == 1
    assert "无法给出可靠答案" in result.final_answer


def test_refusal_is_not_automatically_passed_when_semantic_checker_fails():
    semantic = FakeSemantic([AnswerVerdict(status="fail", reason="证据本可回答")])
    verifier = AnswerVerifier(semantic)
    result = verifier.verify("【无法回答】证据不足。", _evidence())
    assert result.status == "fail"
    assert semantic.calls == 1
