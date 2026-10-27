from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path

from scripts.answer_quality_depth.data import load_depth_cases

ROOT = Path(__file__).resolve().parents[3]
RUN_ROOT = ROOT / "tests" / "eval" / "answer_quality_depth"
FINAL_PATHS = (
    RUN_ROOT / "runs" / "dev-final3" / "results-rescored.jsonl",
    RUN_ROOT / "runs" / "frozen-v1-final3" / "results-rescored.jsonl",
    RUN_ROOT / "runs" / "retrieval-slate-holdout-v1-final3" / "results-rescored.jsonl",
)


def _final_rows() -> list[dict]:
    rows = []
    for path in FINAL_PATHS:
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return rows


def test_final_artifact_has_every_question_twice_and_no_truncated_answers() -> None:
    rows = _final_rows()
    cases = [case for split in load_depth_cases().values() for case in split]

    assert len(rows) == 340
    assert len({row["case_id"] for row in rows}) == 170
    assert all(str(row.get("answer") or "").strip() for row in rows)
    assert not [
        row
        for row in rows
        if (((row.get("generation") or {}).get("raw") or {}).get("choices") or [{}])[
            0
        ].get("finish_reason")
        == "length"
    ]
    assert not [row for row in rows if row.get("technical_failure")]
    assert Counter(row["arm"] for row in rows) == Counter(
        {"baseline": 170, "slate": 170}
    )
    assert {row["case_id"] for row in rows} == {case["id"] for case in cases}


def test_evidence_artifact_matches_frozen_top5_and_top10_denominators() -> None:
    groups = []
    for name in ("dev", "frozen-v1", "retrieval-slate-holdout-v1"):
        payload = json.loads(
            (RUN_ROOT / "evidence" / f"{name}.json").read_text(encoding="utf-8")
        )
        groups.extend(row for row in payload["rows"] if row["baseline_hit"] is not None)

    assert len(groups) == 128
    assert sum(row["baseline_top5_hit"] for row in groups) == 88
    assert sum(row["slate_top5_hit"] for row in groups) == 125
    assert sum(row["baseline_hit"] for row in groups) == 111
    assert sum(row["slate_hit"] for row in groups) == 128


def test_final_artifact_reports_separate_denominators_and_cost_cap() -> None:
    rows = _final_rows()
    aggregate = json.loads((RUN_ROOT / "aggregate.json").read_text(encoding="utf-8"))

    assert (
        sum(
            arm["n"]
            for arm in aggregate["overall"]["by_denominator"][
                "evidence_answer"
            ].values()
        )
        == 252
    )
    assert len(rows) == 340
    assert aggregate["overall"]["pairwise"]["slate_vs_baseline"]["wins"] == 9
    assert aggregate["overall"]["pairwise"]["slate_vs_baseline"]["losses"] == 1

    connection = sqlite3.connect(
        f"file:{RUN_ROOT / 'experiment-ledger.sqlite3'}?mode=ro", uri=True
    )
    try:
        total = connection.execute(
            "SELECT SUM(CASE WHEN state='pending' THEN reserved_usd ELSE actual_usd END) "
            "FROM reservations WHERE state IN ('pending','settled')"
        ).fetchone()[0]
    finally:
        connection.close()
    assert total <= 2.0


def test_manual_review_covers_uncertainty_regression_and_refusal() -> None:
    rows = [
        json.loads(line)
        for line in (RUN_ROOT / "manual-review.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    decisions = {row["manual_review"] for row in rows}

    assert "confirmed_regression" in decisions
    assert "kept_uncertain" in decisions
    assert "refusal_spot_check" in decisions
    assert any(row["case_id"] == "FRZ-ENTITY-02" for row in rows)
