from __future__ import annotations

from app.answer_plan import (
    AnswerPlan,
    AnswerPlanPipeline,
    EvidenceCandidate,
    PolicyScope,
    render_answer_plan,
    validate_answer_plan,
)


def _candidate(**updates):
    payload = {
        "evidence_id": "EV-1",
        "text": "SH-K68 87 键机械键盘，售价 149 元，一年质保。",
        "source_id": "catalog#k68",
        "source_title": "商品目录 > SH-K68",
        "entities": ["SH-K68"],
    }
    payload.update(updates)
    return EvidenceCandidate(**payload)


def _plan(
    *,
    quote="售价 149 元",
    claim="SH-K68 售价 149 元",
    subject="SH-K68",
    value="149",
    unit="元",
    dates=None,
    policy_version="",
    **updates,
):
    payload = {
        "disposition": "answer",
        "claims": [
            {
                "claim_id": "C1",
                "text": claim,
                "subject": subject,
                "value": value,
                "unit": unit,
                "dates": dates or [],
                "policy_version": policy_version,
                "evidence": [{"evidence_id": "EV-1", "quote": quote}],
            }
        ],
    }
    payload.update(updates)
    return AnswerPlan.model_validate(payload)


def test_validation_normalizes_an_exact_quote_to_its_verified_span():
    candidate = _candidate()
    result = validate_answer_plan(_plan(), [candidate])

    claim = result.claims[0]
    assert result.status == "pass"
    assert claim.status == "supported"
    binding = claim.claim.evidence[0]
    assert candidate.text[binding.start:binding.end] == binding.quote
    assert binding.quote == "售价 149 元"


def test_validation_rejects_quote_not_present_in_named_evidence():
    result = validate_answer_plan(_plan(quote="售价 199 元"), [_candidate()])

    assert result.status == "fail"
    assert result.claims[0].status == "unsupported"
    assert "quote" in result.claims[0].issues[0].message


def test_validation_rejects_wrong_model_number_even_when_quote_exists():
    result = validate_answer_plan(
        _plan(claim="SH-K870 售价 149 元"), [_candidate()]
    )

    assert result.status == "fail"
    assert result.claims[0].status == "unsupported"
    assert "主体" in result.claims[0].issues[0].message


def test_validation_rejects_number_unit_conflict():
    result = validate_answer_plan(
        _plan(
            quote="售价 149 元，一年质保",
            claim="SH-K68 售价 149 天",
            unit="天",
        ),
        [_candidate()],
    )

    assert result.status == "fail"
    assert result.claims[0].status == "unsupported"
    assert "数值" in result.claims[0].issues[0].message


def test_validation_rejects_omitted_applicability_condition():
    candidate = _candidate(
        text="质量问题退货时，商品需保持完好，商家承担退回运费。",
        entities=[],
        required_conditions=["商品需保持完好"],
    )
    plan = _plan(
        quote="商家承担退回运费",
        claim="商家承担退回运费",
        subject="",
        value="",
        unit="",
    )

    result = validate_answer_plan(plan, [candidate])

    assert result.status == "fail"
    assert result.claims[0].status == "unsupported"
    assert "必要条件" in result.claims[0].issues[0].message


def test_validation_keeps_unanchored_claim_uncertain_instead_of_supported():
    plan = _plan(
        quote="一年质保",
        claim="SH-K68 售价 149 元",
    )

    result = validate_answer_plan(plan, [_candidate()])

    assert result.status == "uncertain"
    assert result.claims[0].status == "uncertain"


def test_validation_accepts_chinese_number_unit_when_quote_uses_same_form():
    candidate = _candidate(
        text="SH-K68 机械键盘一年质保。",
        entities=["SH-K68"],
    )
    plan = _plan(
        quote="一年质保",
        claim="SH-K68 质保一年",
        value="一",
        unit="年",
    )

    result = validate_answer_plan(plan, [candidate])

    assert result.status == "pass"
    assert result.claims[0].status == "supported"


def test_validation_accepts_chinese_number_with_unit_embedded_in_value():
    candidate = _candidate(
        text="SH-B7 智能手环一年质保。",
        entities=["SH-B7"],
    )
    plan = _plan(
        quote="一年质保",
        claim="SH-B7 手环提供一年质保",
        subject="SH-B7",
        value="一年",
        unit="",
    )

    result = validate_answer_plan(plan, [candidate])

    assert result.status == "pass"
    assert result.claims[0].status == "supported"


def test_validation_does_not_treat_free_text_value_as_numeric_anchor():
    candidate = _candidate(
        text="SH-M10 当前为清仓状态，不参与会员 95 折。",
        entities=["SH-M10"],
    )
    plan = _plan(
        quote="不参与会员 95 折",
        claim="SH-M10 当前为清仓状态，不参与会员 95 折",
        subject="SH-M10",
        value="不参与会员 95 折",
        unit="",
    )

    result = validate_answer_plan(plan, [candidate])

    assert result.status == "pass"
    assert result.claims[0].status == "supported"


def test_validation_accepts_declared_condition_alias_without_losing_scope():
    candidate = _candidate(
        text="签收日期早于 2026-06-01 的订单适用 v1，窗口为 7 天。",
        entities=[],
        required_conditions=["签收日期早于 2026-06-01"],
        condition_aliases={
            "签收日期早于 2026-06-01": [
                "签收日期在 v1 有效期内（2024-01-01 至 2026-05-31）"
            ]
        },
    )
    plan = _plan(
        quote="窗口为 7 天",
        claim="v1 有效期内签收适用 7 天窗口",
        subject="",
        value="7",
        unit="天",
        policy_version="v1",
        dates=["2026-05-20"],
    )
    plan.claims[0].necessary_conditions = [
        "签收日期在 v1 有效期内（2024-01-01 至 2026-05-31）"
    ]

    result = validate_answer_plan(plan, [candidate])

    assert result.status == "pass"
    assert result.claims[0].status == "supported"


def test_no_claim_limited_help_preserves_clarifying_question():
    plan = AnswerPlan(
        disposition="limited_help",
        claims=[],
        clarifying_question="请提供具体商品型号。",
    )

    checked = validate_answer_plan(plan, [])
    rendered = render_answer_plan(checked)

    assert checked.status == "pass"
    assert "请提供具体商品型号" in rendered.text


def test_validation_uses_policy_scope_for_date_version_conflict():
    candidate = _candidate(
        text="v2 自 2026-06-01 起生效，无理由退货窗口为签收后 15 天。",
        source_id="policy#window",
        source_title="政策版本说明",
        entities=[],
        policy_scope=PolicyScope(
            version="v2",
            effective_from="2026-06-01",
            effective_to=None,
        ),
    )
    plan = _plan(
        quote="无理由退货窗口为签收后 15 天",
        claim="2026-05-20 签收适用 v2 的 15 天窗口",
        subject="",
        value="15",
        unit="天",
        dates=["2026-05-20"],
        policy_version="v2",
    )

    result = validate_answer_plan(plan, [candidate])

    assert result.status == "fail"
    assert result.claims[0].status == "unsupported"
    assert "政策" in result.claims[0].issues[0].message


def test_renderer_assigns_citations_only_to_verified_claims_and_keeps_conditions():
    candidate = _candidate()
    plan = _plan()
    plan.claims[0].necessary_conditions = ["商品需保持完好"]
    plan.limitations = ["未核验保修起算日"]
    result = validate_answer_plan(plan, [candidate])

    rendered = render_answer_plan(result)

    assert "SH-K68 售价 149 元 [1]。" in rendered.text
    assert "。 [1]" not in rendered.text
    assert "商品需保持完好" in rendered.text
    assert "未核验保修起算日" in rendered.text
    assert rendered.citations == {
        1: {
            "evidence_id": "EV-1",
            "source_id": "catalog#k68",
            "source_title": "商品目录 > SH-K68",
        }
    }


class _FakePlanner:
    def __init__(self, plans):
        self.plans = list(plans)
        self.calls = 0

    def create(self, query, candidates):
        self.calls += 1
        return self.plans.pop(0)

    def repair(self, query, candidates, plan, issues):
        self.calls += 1
        return self.plans.pop(0)


def test_pipeline_allows_one_bounded_repair_and_revalidates():
    bad = _plan(quote="售价 199 元")
    good = _plan()
    planner = _FakePlanner([bad, good])
    pipeline = AnswerPlanPipeline(planner, max_corrections=9)

    result = pipeline.answer("SH-K68 多少钱？", [_candidate()])

    assert planner.calls == 2
    assert result.correction_attempts == 1
    assert result.validation.status == "pass"
    assert "售价 149 元 [1]" in result.rendered.text


def test_pipeline_does_not_render_unspecified_claims_after_failed_repair():
    bad = _plan(quote="售价 199 元")
    planner = _FakePlanner([bad, bad])
    pipeline = AnswerPlanPipeline(planner)

    result = pipeline.answer("SH-K68 多少钱？", [_candidate()])

    assert planner.calls == 2
    assert result.validation.status == "fail"
    assert result.rendered.text.startswith("目前只能提供有限帮助")
    assert "售价 149 元" not in result.rendered.text
    assert "引用无法核验" in result.rendered.text
