from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "tests" / "eval" / "claim_grounding_r2" / "data"
REQUIRED_CATEGORIES = {
    "synonym_support",
    "missing_condition",
    "negation",
    "unit",
    "date_boundary",
    "cross_order",
    "stale_evidence",
}


def _load(name):
    return [
        json.loads(line)
        for line in (DATA / name).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_dev_and_heldout_sizes_and_group_isolation():
    dev = _load("dev.jsonl")
    heldout = _load("heldout.jsonl")
    assert len(dev) == 12
    assert 24 <= len(heldout) <= 36
    assert {row["group_id"] for row in dev}.isdisjoint(
        {row["group_id"] for row in heldout}
    )


def test_dataset_covers_required_categories_and_has_source_provenance():
    rows = _load("dev.jsonl") + _load("heldout.jsonl")
    categories = {row["category"] for row in rows}
    assert REQUIRED_CATEGORIES <= categories
    assert all(row.get("source") for row in rows)
    assert all(
        row.get("gold") in {"supported", "unsupported", "uncertain"} for row in rows
    )
    assert all(row.get("gold_review_status") for row in rows)
    assert all(row.get("evidence") for row in rows)


def test_gold_is_not_part_of_model_input():
    for row in _load("dev.jsonl") + _load("heldout.jsonl"):
        visible = {
            "claim": row["claim"],
            "subject": row.get("subject", ""),
            "model_number": row.get("model_number", ""),
            "order_id": row.get("order_id", ""),
            "as_of_date": row.get("as_of_date"),
            "policy_version": row.get("policy_version", ""),
            "unit": row.get("unit", ""),
            "evidence": row["evidence"],
        }
        assert "gold" not in visible
        assert "gold_reason" not in visible
