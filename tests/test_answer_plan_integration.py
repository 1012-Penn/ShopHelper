"""Integration contracts for the optional evidence-bound answer-plan graph path."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from app.answer_plan.graph_node import make_answer_plan_node
from app.answer_plan.models import (
    AnswerPlan,
    AnswerPlanResult,
    PlanValidation,
    RenderedAnswer,
)
from app.config import Settings


class _Pipeline:
    def answer(self, query, candidates):
        assert query == "K104 多少钱"
        assert len(candidates) == 1
        return AnswerPlanResult(
            plan=AnswerPlan(),
            validation=PlanValidation(
                plan=AnswerPlan(),
                status="pass",
                evidence=candidates,
            ),
            rendered=RenderedAnswer(
                text="K104 售价 149 元 [1]。",
                citations={
                    1: {
                        "evidence_id": "chunk-1",
                        "source_id": "kb:1",
                        "source_title": "商品参数",
                    }
                },
                used_evidence_ids=["chunk-1"],
            ),
        )


@pytest.mark.asyncio
async def test_answer_plan_node_returns_assistant_message_for_history_persistence():
    frames = []
    node = make_answer_plan_node(_Pipeline())

    result = await node(
        {
            "user_message": "K104 多少钱",
            "resolved_message": "K104 多少钱",
            "evidence": [
                {
                    "n": 1,
                    "chunk_id": "chunk-1",
                    "question": "K104 售价",
                    "answer": "K104 售价 149 元。",
                    "section_path": "商品参数 > K104",
                }
            ],
            "trace": [],
        },
        writer=frames.append,
    )

    assistant = [item for item in result.get("messages", [])
                 if isinstance(item, AIMessage) and item.content]
    assert len(assistant) == 1
    assert assistant[0].content == result["final_reply"]
    assert result["final_reply"] == "K104 售价 149 元 [1]。"


def test_answer_plan_settings_default_off_and_require_ledger_when_enabled(tmp_path):
    disabled = Settings(
        openai_api_key="sk-test",
        embedding_api_key="sk-test",
        _env_file=None,
    )
    assert disabled.answer_plan_enabled is False
    assert disabled.answer_plan_ledger_path == ""

    enabled = Settings(
        openai_api_key="sk-test",
        embedding_api_key="sk-test",
        _env_file=None,
        answer_plan_enabled=True,
    )
    from app.answer_plan.runtime import build_optional_answer_plan_runtime
    from scripts.eval_cost import PricingTable

    pricing = PricingTable.load(
        "tests/eval/rag_ablation/pricing-snapshot-2026-10-26.yaml"
    )
    with pytest.raises(ValueError, match="answer_plan_ledger_path"):
        build_optional_answer_plan_runtime(enabled, pricing=pricing)


def test_app_factory_mounts_answer_plan_node_only_when_explicitly_enabled(tmp_path):
    from app.main import create_app

    settings = Settings(
        openai_api_key="sk-test",
        embedding_api_key="sk-test",
        admin_session_secret="integration-secret",
        admin_password_iterations=2_000,
        database_url=f"sqlite:///{tmp_path / 'app.db'}",
        milvus_db_path=str(tmp_path / "milvus.db"),
        answer_plan_enabled=True,
        answer_plan_ledger_path=str(tmp_path / "answer-plan-budget.sqlite3"),
        answer_plan_pricing_path="tests/eval/rag_ablation/pricing-snapshot-2026-10-26.yaml",
        _env_file=None,
    )

    app = create_app(settings)
    nodes = app.state.graph.get_graph().nodes

    assert "answer_plan" in nodes
    assert app.state.answer_plan.ledger.limit_usd == 2.0
