from __future__ import annotations

from typing import Protocol

from app.answer_plan.models import (
    AnswerPlan,
    AnswerPlanResult,
    EvidenceCandidate,
    PlanIssue,
)
from app.answer_plan.renderer import render_with_candidates
from app.answer_plan.verification import validate_answer_plan


class AnswerPlanner(Protocol):
    def create(self, query: str, candidates: list[EvidenceCandidate]) -> AnswerPlan:
        ...

    def repair(
        self,
        query: str,
        candidates: list[EvidenceCandidate],
        plan: AnswerPlan,
        issues: list[PlanIssue],
    ) -> AnswerPlan:
        ...


class AnswerPlanPipeline:
    def __init__(self, planner: AnswerPlanner, *, max_corrections: int = 1) -> None:
        self._planner = planner
        self._max_corrections = max(0, min(1, int(max_corrections)))

    def answer(
        self,
        query: str,
        candidates: list[EvidenceCandidate],
    ) -> AnswerPlanResult:
        calls = 0
        corrections = 0
        trace: list[str] = []
        plan = self._planner.create(query, candidates)
        calls += 1
        checked = validate_answer_plan(plan, candidates)
        trace.append(f"plan_validation={checked.status}")
        while (
            checked.status != "pass"
            and corrections < self._max_corrections
        ):
            corrections += 1
            plan = self._planner.repair(
                query, candidates, plan, checked.issues
            )
            calls += 1
            checked = validate_answer_plan(plan, candidates)
            trace.append(
                f"repair={corrections} plan_validation={checked.status}"
            )
        rendered = render_with_candidates(checked, candidates)
        return AnswerPlanResult(
            plan=plan,
            validation=checked,
            rendered=rendered,
            correction_attempts=corrections,
            planner_calls=calls,
            trace=trace,
        )
