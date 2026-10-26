from __future__ import annotations

import asyncio

from app.answer_plan.graph_node import make_answer_plan_node


class _Pipeline:
    def answer(self, query, candidates):
        from app.answer_plan import (
            AnswerPlan,
            AnswerPlanResult,
            PlanClaim,
            EvidenceBinding,
            EvidenceCandidate,
        )
        from app.answer_plan.verification import validate_answer_plan
        from app.answer_plan.renderer import render_with_candidates

        candidate = candidates[0]
        plan = AnswerPlan(
            claims=[
                PlanClaim(
                    claim_id="C1",
                    text="SH-K68 售价 149 元",
                    subject="SH-K68",
                    value="149",
                    unit="元",
                    evidence=[
                        EvidenceBinding(
                            evidence_id=candidate.evidence_id,
                            quote="售价 149 元",
                        )
                    ],
                )
            ]
        )
        validation = validate_answer_plan(plan, candidates)
        return AnswerPlanResult(
            plan=plan,
            validation=validation,
            rendered=render_with_candidates(validation, candidates),
            planner_calls=1,
        )


def test_graph_node_emits_progress_then_final_answer_once():
    frames = []
    node = make_answer_plan_node(_Pipeline())
    state = {
        "resolved_message": "SH-K68 多少钱？",
        "evidence": [{
            "n": 3,
            "chunk_id": 31,
            "section_path": "商品目录 > SH-K68",
            "question": "SH-K68",
            "answer": "SH-K68 售价 149 元。",
        }],
        "trace": ["node=intent"],
    }

    result = asyncio.run(node(state, writer=frames.append))

    assert [frame["type"] for frame in frames] == [
        "answer_plan_status",
        "answer_plan_status",
        "citations",
        "token",
    ]
    assert frames[3]["content"] == result["final_reply"]
    assert "售价 149 元 [1]" in result["final_reply"]
    assert result["answer_plan"]["status"] == "pass"
    assert result["answer_plan"]["correction_attempts"] == 0
