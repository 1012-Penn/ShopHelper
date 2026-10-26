from __future__ import annotations

import json

import pytest

from scripts.e2e_quality_r2.data import (
    DatasetContractError,
    dataset_hashes,
    load_frozen_cases,
)


def _write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def _fixture_root(tmp_path):
    dev = [
        {
            "id": "DEV-1",
            "split": "dev",
            "source_group": "source-a",
            "turns": [{"role": "user", "content": "SH-E300 多少钱？"}],
            "expected_behavior": "answer",
            "facts": [
                {
                    "fact_id": "F-1",
                    "entity": "SH-E300",
                    "value": 299,
                    "unit": "元",
                }
            ],
        }
    ]
    heldout = [
        {
            "id": "HELD-1",
            "split": "heldout",
            "source_group": "source-b",
            "turns": [{"role": "user", "content": "SH-E120 续航多久？"}],
            "expected_behavior": "answer",
            "facts": [
                {
                    "fact_id": "F-2",
                    "entity": "SH-E120",
                    "value": 24,
                    "unit": "小时",
                }
            ],
        }
    ]
    dev_path = tmp_path / "cases-dev.jsonl"
    held_path = tmp_path / "cases-heldout.jsonl"
    dev_path.write_text(json.dumps(dev[0], ensure_ascii=False) + "\n", encoding="utf-8")
    held_path.write_text(json.dumps(heldout[0], ensure_ascii=False) + "\n", encoding="utf-8")
    files = dataset_hashes(tmp_path)
    _write_json(
        tmp_path / "manifest.json",
        {
            "schema_version": 1,
            "case_counts": {"dev": 1, "heldout": 1},
            "files": files,
            "source_groups": {"dev": ["source-a"], "heldout": ["source-b"]},
        },
    )


def test_loader_rejects_case_file_changed_after_freeze(tmp_path):
    _fixture_root(tmp_path)
    path = tmp_path / "cases-dev.jsonl"
    row = json.loads(path.read_text(encoding="utf-8"))
    row["turns"][0]["content"] += " "
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    with pytest.raises(DatasetContractError, match="hash"):
        load_frozen_cases(tmp_path)


def test_loader_rejects_source_group_leak_between_splits(tmp_path):
    _fixture_root(tmp_path)
    path = tmp_path / "cases-heldout.jsonl"
    row = json.loads(path.read_text(encoding="utf-8"))
    row["source_group"] = "source-a"
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = dataset_hashes(tmp_path)
    _write_json(manifest_path, manifest)

    with pytest.raises(DatasetContractError, match="source"):
        load_frozen_cases(tmp_path)


def test_frozen_dataset_has_required_split_sizes_and_groups():
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    cases = load_frozen_cases(root / "tests" / "eval" / "e2e_quality_r2")

    assert len(cases["dev"]) == 8
    assert len(cases["heldout"]) == 20
    dev_groups = {case["source_group"] for case in cases["dev"]}
    heldout_groups = {case["source_group"] for case in cases["heldout"]}
    assert dev_groups.isdisjoint(heldout_groups)
    assert {case["id"] for case in cases["dev"]}.isdisjoint(
        {case["id"] for case in cases["heldout"]}
    )


def test_declared_order_fixture_matches_runtime_catalog():
    from pathlib import Path

    from app.orders import ORDER_CATALOG

    root = Path(__file__).resolve().parents[3]
    fixture = json.loads(
        (root / "tests" / "eval" / "e2e_quality_r2" / "fixtures" / "orders.json")
        .read_text(encoding="utf-8")
    )
    assert fixture["catalog"] == ORDER_CATALOG
    assert fixture["source"] == "app/orders.py:ORDER_CATALOG"
