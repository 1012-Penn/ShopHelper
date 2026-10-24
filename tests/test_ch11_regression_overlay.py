"""regression-v1 标签覆盖层:C04/C05 审计、旧文件不变、默认字段完整。"""
from __future__ import annotations

import json
from pathlib import Path

from scripts.eval_dataset import dataset_sha256, load_split

ROOT = Path(__file__).resolve().parent.parent
CH04 = ROOT / "tests" / "eval" / "ch04_eval_set.jsonl"
CH09 = ROOT / "tests" / "eval" / "ch09_junk_set.jsonl"
CHANGELOG = ROOT / "tests" / "eval" / "CHANGELOG.md"

OLD_SHA = {
    CH04: "5852485baf591d52ceef794950ba47071c0e98ca005a6b0ced8dfa1571b3a74b",
    CH09: "d4bb3e03321eace9eace7e7bf1269fa98aefd0cdc9140fbf669d46aa41c95846",
}

AUDITED = {
    "C01": "answer_with_conditions",
    "C02": "answer_with_conditions",
    "C04": "answer_with_conditions",
    "C05": "verify_then_answer",
    "E01": "answer_with_conditions",
    "E03": "answer_with_conditions",
    "E04": "verify_then_answer",
    "E06": "answer_with_conditions",
    "E08": "answer_with_conditions",
    "F04": "answer_with_conditions",
}


def _old_cases() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for path in (CH04, CH09):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                case = json.loads(line)
                rows[case["id"]] = case
    return rows


def test_regression_has_60_unique_cases():
    cases = load_split("regression-v1")
    assert len(cases) == 60
    assert len({case["id"] for case in cases}) == 60


def test_regression_preserves_required_evidence_and_answerable():
    old = _old_cases()
    for case in load_split("regression-v1"):
        assert case["answerable"] == old[case["id"]]["answerable"]
        assert case.get("required_evidence") == old[case["id"]].get("required_evidence")


def test_c04_and_c05_new_labels_exact():
    by_id = {case["id"]: case for case in load_split("regression-v1")}
    c04 = by_id["C04"]
    assert c04["expected_behavior"] == "answer_with_conditions"
    assert c04["evidence_state"] == "partial"
    assert c04["required_slots"] == ["sku", "member_status"]
    assert "可以便宜" in c04["forbidden_claims"]

    c05 = by_id["C05"]
    assert c05["expected_behavior"] == "verify_then_answer"
    assert c05["evidence_state"] == "partial"
    assert c05["required_slots"] == ["order_id", "damage_photo"]
    assert "算质量问题" in c05["forbidden_claims"]
    assert "检测" in c05["verification_markers"]


def test_ten_audited_cases_have_expected_behavior():
    by_id = {case["id"]: case for case in load_split("regression-v1")}
    for case_id, behavior in AUDITED.items():
        assert by_id[case_id]["expected_behavior"] == behavior, case_id


def test_all_regression_cases_have_v2_defaults():
    for case in load_split("regression-v1"):
        assert case["schema_version"] == 2
        assert case["split"] == "regression-v1"
        assert case["label_version"] == "v2-overlay"
        assert case["expected_behavior"] in {
            "answer",
            "answer_with_conditions",
            "verify_then_answer",
            "clarify",
            "refuse",
        }
        assert case["evidence_state"] in {
            "sufficient",
            "partial",
            "insufficient",
            "conflicting",
        }


def test_changelog_covers_every_overlay_id():
    text = CHANGELOG.read_text(encoding="utf-8")
    for case_id in AUDITED:
        assert case_id in text


def test_source_eval_files_not_modified():
    for path, expected in OLD_SHA.items():
        assert dataset_sha256(path) == expected
