import json
from pathlib import Path

from tests.eval.fact_scope_r3.harness import ARMS, load_frozen_manifest, summarize

RUNS = Path(__file__).resolve().parent / "runs"


def _load(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_final_runs_use_frozen_config_without_technical_failures():
    frozen = load_frozen_manifest()
    for name, cases in (
        ("live-20261027-dev", 8),
        ("live-20261027-heldout", 20),
    ):
        run_dir = RUNS / name
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        rows = _load(run_dir / "results.jsonl")
        assert summary["config_hash"] == frozen["config_hash"]
        assert len(rows) == cases * len(ARMS)
        assert {row["arm"] for row in rows} == set(ARMS)
        assert all(row["technical_error"] is None for row in rows)
        assert summary["summary"] == summarize(rows)


def test_heldout_failures_are_preserved_rather_than_selectively_removed():
    heldout = _load(RUNS / "live-20261027-heldout" / "results.jsonl")
    failed = {
        (row["case_id"], row["arm"])
        for row in heldout
        if not row["score"]["task_success"]
    }

    assert ("FSR3-HLD-011", "identity") in failed
    assert ("FSR3-HLD-011", "scope") in failed
    assert ("FSR3-HLD-017", "identity") in failed
    assert len(heldout) == 60


def test_initial_scorer_run_is_separate_from_final_dev_run():
    initial = RUNS / "live-20261027-dev-initial-scorer"
    final = RUNS / "live-20261027-dev"

    assert initial.is_dir() and final.is_dir()
    assert initial != final
    assert len(_load(initial / "results.jsonl")) == 24
    assert len(_load(final / "results.jsonl")) == 24
