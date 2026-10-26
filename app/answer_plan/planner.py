"""Structured answer-plan generation using the central extractor factory."""
from __future__ import annotations

import json

from app.answer_plan.models import (
    AnswerPlan,
    EvidenceCandidate,
    PlanIssue,
)

PLAN_CREATE_PROMPT_VERSION = "answer-plan-r3-create-v1"
PLAN_REPAIR_PROMPT_VERSION = "answer-plan-r3-repair-v1"

_CREATE_PROMPT = """你是电商客服的证据绑定计划器。只依据用户当前问题和下方候选证据生成回答计划。

【用户问题】
{query}

【候选证据】
{evidence}

要求：
1. 只把回答问题所必需的实质事实写成 claim；每个 claim 必须引用候选 evidence_id。
2. evidence.quote 必须逐字复制对应证据中的连续原文，不得改写，不得引用整段文档。
3. 主体、数值、单位、日期、政策版本必须与原文一致。
4. 原文中的适用范围、前提和例外是必要条件，不得删除；写入 necessary_conditions，并在 claim.text 中保留。
5. 如果现有证据不足，不要补造事实。disposition 使用 clarify、limited_help 或 refuse，并在 clarifying_question / limitations 中说明缺口。
6. 证据只能支持事实绑定，不能把不确定关系强行判成确定；无法绑定的说法不要写入确定 claim。
7. 不得处理与用户当前问题无关的信息。
"""

_REPAIR_PROMPT = """你是电商客服的证据绑定计划修正器。仅返回修正后的计划。

【用户问题】
{query}

【候选证据】
{evidence}

【上一版计划】
{plan}

【校验问题】
{issues}

修正要求：
1. 只修复下列问题，不扩大回答范围。
2. quote 必须逐字出现在对应证据中；引用 ID 只能来自候选证据。
3. 不得删除适用条件、例外、数值单位或政策范围。
4. 无法修复时改为 clarify、limited_help 或 refuse，不要猜测。
"""


def _candidate_payload(candidate: EvidenceCandidate) -> dict:
    return {
        "evidence_id": candidate.evidence_id,
        "text": candidate.text,
        "source_id": candidate.source_id,
        "source_title": candidate.source_title,
        "entities": candidate.entities,
        "required_conditions": candidate.required_conditions,
        "condition_aliases": candidate.condition_aliases,
        "policy_scope": (
            candidate.policy_scope.model_dump(mode="json")
            if candidate.policy_scope is not None
            else None
        ),
    }


def _evidence_text(candidates: list[EvidenceCandidate]) -> str:
    return json.dumps(
        [_candidate_payload(candidate) for candidate in candidates],
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


class StructuredAnswerPlanner:
    def __init__(self, extractor) -> None:
        self._extractor = extractor

    def create(self, query: str, candidates: list[EvidenceCandidate]) -> AnswerPlan:
        prompt = _CREATE_PROMPT.format(
            query=query,
            evidence=_evidence_text(candidates),
        )
        return self._extractor.invoke(prompt)

    def repair(
        self,
        query: str,
        candidates: list[EvidenceCandidate],
        plan: AnswerPlan,
        issues: list[PlanIssue],
    ) -> AnswerPlan:
        prompt = _REPAIR_PROMPT.format(
            query=query,
            evidence=_evidence_text(candidates),
            plan=json.dumps(
                plan.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
            ),
            issues=json.dumps(
                [issue.model_dump(mode="json") for issue in issues],
                ensure_ascii=False,
                sort_keys=True,
            ),
        )
        return self._extractor.invoke(prompt)


def make_answer_plan_planner(settings):
    from app.llm import make_structured_extractor

    return StructuredAnswerPlanner(
        make_structured_extractor(settings, AnswerPlan)
    )
