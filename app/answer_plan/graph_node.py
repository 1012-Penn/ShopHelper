"""Optional graph node for the evidence-bound answer-plan path.

The shared graph builder can adopt this node without changing the legacy agent
path. Progress is emitted before the final answer so streamed output never
exposes an unvalidated draft.
"""
from __future__ import annotations

import asyncio
import re

from app.answer_plan.models import EvidenceCandidate
from app.answer_plan.pipeline import AnswerPlanPipeline

_ENTITY_RE = re.compile(r"\b(?:SH|ORD)-[A-Za-z0-9]+\b", re.IGNORECASE)


def _emit(writer, frame: dict) -> None:
    if writer is not None:
        writer(frame)


def candidates_from_state_evidence(rows: list[dict]) -> list[EvidenceCandidate]:
    candidates: list[EvidenceCandidate] = []
    for index, row in enumerate(rows or [], 1):
        evidence_id = str(
            row.get("evidence_id")
            or row.get("chunk_id")
            or row.get("n")
            or f"E{index}"
        )
        question = str(row.get("question") or "")
        answer = str(row.get("answer") or row.get("text") or "")
        text = answer if not question or question in answer else f"{question}\n{answer}"
        if not text.strip():
            text = str(row.get("section_path") or f"evidence {evidence_id}")
        entities = row.get("entities") or _ENTITY_RE.findall(text)
        policy_scope = row.get("policy_scope")
        candidates.append(
            EvidenceCandidate(
                evidence_id=evidence_id,
                text=text.strip(),
                source_id=str(
                    row.get("source_id")
                    or row.get("chunk_id")
                    or evidence_id
                ),
                source_title=str(
                    row.get("source_title")
                    or row.get("section_path")
                    or row.get("category")
                    or ""
                ),
                entities=list(entities or []),
                required_conditions=list(row.get("required_conditions") or []),
                condition_aliases=dict(row.get("condition_aliases") or {}),
                policy_scope=policy_scope,
            )
        )
    return candidates


def make_answer_plan_node(pipeline: AnswerPlanPipeline):
    async def answer_plan_node(state, writer=None, config=None) -> dict:
        query = str(state.get("resolved_message") or state.get("user_message") or "")
        candidates = candidates_from_state_evidence(state.get("evidence") or [])
        _emit(
            writer,
            {
                "type": "answer_plan_status",
                "stage": "planning",
                "candidate_count": len(candidates),
            },
        )
        result = await asyncio.to_thread(pipeline.answer, query, candidates)
        _emit(
            writer,
            {
                "type": "answer_plan_status",
                "stage": "validated",
                "status": result.validation.status,
                "correction_attempts": result.correction_attempts,
                "planner_calls": result.planner_calls,
            },
        )
        by_id = {
            str(row.get("evidence_id") or row.get("chunk_id") or row.get("n")): row
            for row in state.get("evidence") or []
        }
        citations = []
        for number, citation in result.rendered.citations.items():
            source = dict(by_id.get(citation["evidence_id"], {}))
            source.update(
                {
                    "n": number,
                    "evidence_id": citation["evidence_id"],
                    "source_id": citation["source_id"],
                    "source_title": citation["source_title"],
                }
            )
            citations.append(source)
        if citations:
            _emit(writer, {"type": "citations", "items": citations})
        _emit(writer, {"type": "token", "content": result.rendered.text})
        return {
            "final_reply": result.rendered.text,
            "answer_plan": {
                "status": result.validation.status,
                "disposition": result.plan.disposition,
                "correction_attempts": result.correction_attempts,
                "planner_calls": result.planner_calls,
                "used_evidence_ids": result.rendered.used_evidence_ids,
            },
            "trace": [
                *(state.get("trace") or []),
                f"node=answer_plan status={result.validation.status} "
                f"corrections={result.correction_attempts} "
                f"calls={result.planner_calls}",
            ],
        }

    return answer_plan_node
