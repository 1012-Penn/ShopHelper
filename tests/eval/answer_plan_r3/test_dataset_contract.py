from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from scripts.answer_plan_r3.data import (
    AnswerPlanDatasetError,
    dataset_hashes,
    load_frozen_dataset,
    model_projection,
)

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "tests" / "eval" / "answer_plan_r3"


def test_dataset_has_frozen_counts_and_disjoint_source_templates():
    dataset = load_frozen_dataset(DATA)

    assert len(dataset["dev"]) == 8
    assert len(dataset["heldout"]) == 20
    assert {
        row["source_group"] for row in dataset["dev"]
    }.isdisjoint({row["source_group"] for row in dataset["heldout"]})
    assert {
        row["evidence_template"] for row in dataset["dev"]
    }.isdisjoint({row["evidence_template"] for row in dataset["heldout"]})


def test_manifest_hash_mismatch_fails_closed(tmp_path):
    source = DATA
    for name in ("cases-dev.jsonl", "cases-heldout.jsonl", "evidence.json", "manifest.json"):
        (tmp_path / name).write_bytes((source / name).read_bytes())
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"]["cases-dev.jsonl"] = "0" * 64
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )

    with pytest.raises(AnswerPlanDatasetError, match="hash"):
        load_frozen_dataset(tmp_path)


def test_model_projection_contains_question_and_visible_evidence_only():
    case = {
        "id": "D1",
        "turns": [{"role": "user", "content": "SH-K68 多少钱？"}],
        "evidence_ids": ["EV-K68"],
        "expected_behavior": "answer",
        "facts": [{"value": 149}],
    }
    evidence = {
        "EV-K68": {
            "evidence_id": "EV-K68",
            "text": "SH-K68 售价 149 元。",
            "source_id": "catalog",
            "source_title": "商品目录",
            "entities": ["SH-K68"],
        }
    }

    projection = model_projection(case, evidence)
    serialized = json.dumps(projection, ensure_ascii=False).lower()

    assert projection["query"] == "SH-K68 多少钱？"
    assert projection["evidence"][0]["evidence_id"] == "EV-K68"
    for forbidden in ("expected_behavior", "facts", "gold", "scoring"):
        assert forbidden not in serialized


def test_frozen_config_and_runs_live_under_independent_ledger_path():
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_policy"]["heldout_runs"] == 1
    assert manifest["run_policy"]["max_new_api_usd"] == 2.0
    assert manifest["run_policy"]["ledger"] == "runs/answer-plan-r3-paid.sqlite3"
    assert manifest["run_policy"]["correction_limit"] == 1
    assert manifest["scoring_version"] == "answer-plan-r3-score-v1"
    assert manifest["config_sha256"] in dataset_hashes(DATA).values() or re.fullmatch(
        r"[0-9a-f]{64}", manifest["config_sha256"]
    )
