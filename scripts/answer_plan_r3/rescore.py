"""Recompute deterministic validation/rendering/scoring from saved trajectories."""
from __future__ import annotations

import json
from pathlib import Path

from app.answer_plan.models import AnswerPlan, EvidenceCandidate
from app.answer_plan.renderer import render_with_candidates
from app.answer_plan.verification import validate_answer_plan
from scripts.answer_plan_r3.data import load_evidence, load_frozen_dataset
from scripts.answer_plan_r3.runner import summarize_results
from scripts.answer_plan_r3.scoring import score_case
from scripts.answer_plan_r3.scoring_v2 import score_case_v2


def rescore_row(
    stored: dict,
    case: dict,
    evidence: list[dict],
    *,
    scoring_version: str = "v1",
) -> dict:
    row = {**stored}
    candidates = [EvidenceCandidate.model_validate(item) for item in evidence]
    if row.get("arm") == "answer_plan" and row.get("plan") is not None:
        plan = AnswerPlan.model_validate(row["plan"])
        checked = validate_answer_plan(plan, candidates)
        rendered = render_with_candidates(checked, candidates)
        answer = rendered.text
        citation_items = {
            int(number): item["evidence_id"]
            for number, item in rendered.citations.items()
        }
        row["answer"] = answer
        row["citation_items"] = citation_items
        row["plan_validation"] = checked.model_dump(mode="json")
    else:
        answer = str(row.get("answer") or "")
    scorer = score_case_v2 if scoring_version == "v2" else score_case
    row["score"] = scorer(
        case,
        {
            "answer": answer,
            "citation_items": row.get("citation_items") or {},
            "technical_failure": bool(row.get("technical_failure")),
        },
        evidence,
    )
    row["rescored"] = True
    row["rescoring_version"] = f"answer-plan-r3-deterministic-rescore-{scoring_version}"
    return row


def rescore_run(
    *,
    result_path: Path,
    data_root: Path,
    output_path: Path,
    scoring_version: str = "v1",
) -> dict:
    dataset = load_frozen_dataset(data_root)
    evidence = load_evidence(data_root)
    cases = {
        row["id"]: row
        for rows in dataset.values()
        for row in rows
    }
    rows = []
    for line in Path(result_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        stored = json.loads(line)
        case = cases[stored["case_id"]]
        case_evidence = [evidence[item] for item in case.get("evidence_ids") or []]
        rows.append(
            rescore_row(
                stored,
                case,
                case_evidence,
                scoring_version=scoring_version,
            )
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = summarize_results(rows)
    summary["rescored"] = True
    summary["scoring_version"] = (
        "answer-plan-r3-score-v2-audit"
        if scoring_version == "v2"
        else "answer-plan-r3-score-v1"
    )
    summary["source_results"] = str(result_path)
    (output_path.parent / "aggregate.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
