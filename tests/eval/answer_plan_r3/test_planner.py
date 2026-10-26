from __future__ import annotations

from app.answer_plan import AnswerPlan, EvidenceCandidate, PlanIssue
from app.answer_plan.planner import StructuredAnswerPlanner


class _Extractor:
    def __init__(self):
        self.prompts = []

    def invoke(self, prompt, **_kwargs):
        self.prompts.append(str(prompt))
        return AnswerPlan.model_validate({
            "disposition": "answer",
            "claims": [{
                "claim_id": "C1",
                "text": "SH-K68 售价 149 元",
                "subject": "SH-K68",
                "value": "149",
                "unit": "元",
                "evidence": [{
                    "evidence_id": "EV-1",
                    "quote": "售价 149 元",
                }],
            }],
        })


def test_create_prompt_contains_only_visible_inputs_and_evidence_metadata():
    extractor = _Extractor()
    planner = StructuredAnswerPlanner(extractor)
    candidates = [
        EvidenceCandidate(
            evidence_id="EV-1",
            text="SH-K68 售价 149 元。",
            source_id="catalog#k68",
            source_title="商品目录",
            entities=["SH-K68"],
            required_conditions=["在售"],
        )
    ]

    plan = planner.create("SH-K68 多少钱？", candidates)

    prompt = extractor.prompts[0]
    assert plan.claims[0].evidence[0].evidence_id == "EV-1"
    assert "SH-K68 多少钱？" in prompt
    assert "SH-K68 售价 149 元。" in prompt
    assert "如果现有证据不足" in prompt
    for forbidden in ("expected_behavior", "gold_facts", "scoring", "reference_answer"):
        assert forbidden not in prompt


def test_repair_prompt_supplies_validation_issues_without_relaxing_rules():
    extractor = _Extractor()
    planner = StructuredAnswerPlanner(extractor)
    candidate = EvidenceCandidate(
        evidence_id="EV-1",
        text="SH-K68 售价 149 元。",
        source_id="catalog#k68",
    )
    plan = planner.create("问", [candidate])

    planner.repair(
        "问",
        [candidate],
        plan,
        [PlanIssue(code="quote_not_found", message="原文 quote 不存在", claim_id="C1")],
    )

    repair_prompt = extractor.prompts[-1]
    assert "quote_not_found" in repair_prompt
    assert "不得删除适用条件" in repair_prompt
    assert "最多修正一次" in repair_prompt or "仅返回修正后的计划" in repair_prompt
