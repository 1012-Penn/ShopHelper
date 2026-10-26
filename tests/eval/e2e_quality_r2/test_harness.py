from __future__ import annotations

import sqlite3

import pytest

from scripts.e2e_quality_r2.harness import (
    RuntimeContractError,
    prepare_private_runtime,
    summarize_results,
    verify_snapshot_identity,
)
from scripts.eval_snapshot import tree_sha256


def _make_snapshot(root):
    snapshot = root / "source-snapshot"
    (snapshot / "collections" / "knowledge").mkdir(parents=True)
    (snapshot / "collections" / "knowledge" / "manifest.json").write_text(
        '{"ok": true}\n', encoding="utf-8"
    )
    return snapshot


def test_snapshot_identity_rejects_wrong_tree_hash(tmp_path):
    snapshot = _make_snapshot(tmp_path)
    expected = tree_sha256(snapshot)

    assert verify_snapshot_identity(snapshot, expected) == expected
    with pytest.raises(RuntimeContractError, match="snapshot"):
        verify_snapshot_identity(snapshot, "0" * 64)


def test_private_runtime_copies_and_leaves_sources_unchanged(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    snapshot = _make_snapshot(source_root)
    source_db = source_root / "metadata.db"
    with sqlite3.connect(source_db) as connection:
        connection.execute("create table marker(value text)")
        connection.execute("insert into marker values ('source')")
    expected = tree_sha256(snapshot)

    paths = prepare_private_runtime(
        source_snapshot=snapshot,
        source_metadata_db=source_db,
        expected_snapshot_sha256=expected,
        destination=tmp_path / "private",
    )

    assert paths.snapshot.exists()
    assert paths.metadata_db.exists()
    assert tree_sha256(paths.snapshot) == expected
    with sqlite3.connect(paths.metadata_db) as connection:
        assert connection.execute("select value from marker").fetchone() == ("source",)
    with sqlite3.connect(source_db) as connection:
        assert connection.execute("select value from marker").fetchone() == ("source",)


def test_summary_keeps_generation_and_judge_costs_separate():
    results = [
        {
            "generation_cost_usd": 0.01,
            "judge_cost_usd": 0.002,
            "embedding_cost_usd": 0.0,
            "rerank_cost_usd": 0.0,
            "input_tokens": 100,
            "output_tokens": 20,
            "latency_ms": 100.0,
            "task_success": True,
            "score": {"task_success": True, "review_required": False,
                      "technical_failure": False, "failure_labels": []},
        },
        {
            "generation_cost_usd": 0.02,
            "judge_cost_usd": 0.003,
            "embedding_cost_usd": 0.0,
            "rerank_cost_usd": 0.0,
            "input_tokens": 200,
            "output_tokens": 30,
            "latency_ms": 300.0,
            "task_success": False,
            "score": {"task_success": False, "review_required": True,
                      "technical_failure": False, "failure_labels": ["citation_uncertain"]},
        },
    ]

    summary = summarize_results(results)

    assert summary["success"] == 1
    assert summary["review_required"] == 1
    assert summary["generation_cost_usd"] == pytest.approx(0.03)
    assert summary["judge_cost_usd"] == pytest.approx(0.005)
    assert summary["p95_latency_ms"] == 300.0
