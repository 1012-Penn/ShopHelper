"""Explicit, budgeted factory for the optional answer-plan graph path."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.answer_plan.budget import AnswerPlanBudgetLedger, BudgetedModel
from app.answer_plan.graph_node import make_answer_plan_node
from app.answer_plan.models import AnswerPlan
from app.answer_plan.pipeline import AnswerPlanPipeline
from app.answer_plan.planner import StructuredAnswerPlanner


@dataclass(frozen=True)
class AnswerPlanRuntime:
    ledger: AnswerPlanBudgetLedger
    model: BudgetedModel
    pipeline: AnswerPlanPipeline

    def make_node(self):
        return make_answer_plan_node(self.pipeline)


def build_optional_answer_plan_runtime(settings, *, base_model=None, pricing):
    if not bool(getattr(settings, "answer_plan_enabled", False)):
        return None
    ledger_path = str(getattr(settings, "answer_plan_ledger_path", "") or "").strip()
    if not ledger_path:
        raise ValueError(
            "answer_plan_ledger_path is required when answer plan is enabled"
        )
    limit = float(getattr(settings, "answer_plan_max_cost_usd", 2.0))
    if not 0 < limit <= 2.0:
        raise ValueError("answer_plan_max_cost_usd must be in (0, 2.0]")
    Path(ledger_path).parent.mkdir(parents=True, exist_ok=True)
    ledger = AnswerPlanBudgetLedger(ledger_path, limit_usd=limit)
    ledger.assert_no_pending()

    from app.llm import make_extract_model, make_structured_extractor

    raw = base_model or make_extract_model(settings)
    model = BudgetedModel(
        raw,
        model=settings.openai_model,
        pricing=pricing,
        ledger=ledger,
        max_tokens=int(getattr(settings, "max_output_tokens", 1200)),
        max_retries=2,
        backoff_seconds=0.5,
    )
    extractor = make_structured_extractor(
        settings,
        AnswerPlan,
        base_model=model,
    )
    pipeline = AnswerPlanPipeline(
        StructuredAnswerPlanner(extractor),
        max_corrections=1,
    )
    return AnswerPlanRuntime(ledger=ledger, model=model, pipeline=pipeline)
