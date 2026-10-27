import json
from pathlib import Path

from scripts.eval_dataset import dataset_sha256, split_sha256
from scripts.query_rewrite_ablation.runner import load_cases

BASE = Path("tests/eval/query_rewrite_ablation")


def _load(name, split):
    return load_cases(BASE / name, split)


def test_query_rewrite_splits_have_gap_driven_size_and_distinct_templates():
    dev = _load("dev.seeds.jsonl", "dev")
    frozen = _load("frozen-v1.seeds.jsonl", "frozen-v1")

    assert len(dev) == 20
    assert len(frozen) == 20
    assert len({case["template_id"] for case in dev}) == 20
    assert len({case["template_id"] for case in frozen}) == 20
    assert all(
        case["gap_type"] in {
            "history_resolution",
            "lexical_gap",
            "negation_correction",
            "explicit_control",
            "entity_date_protection",
        }
        for case in [*dev, *frozen]
    )


def test_dev_and_frozen_do_not_share_gold_facts_or_templates():
    dev = _load("dev.seeds.jsonl", "dev")
    frozen = _load("frozen-v1.seeds.jsonl", "frozen-v1")
    dev_facts = {fact["fact_id"] for case in dev for fact in case["gold_facts"]}
    frozen_facts = {fact["fact_id"] for case in frozen for fact in case["gold_facts"]}

    assert dev_facts.isdisjoint(frozen_facts)
    assert {case["template_id"] for case in dev}.isdisjoint(
        {case["template_id"] for case in frozen}
    )


def test_frozen_hash_is_locked_before_run():
    frozen = _load("frozen-v1.seeds.jsonl", "frozen-v1")
    lock = json.loads((BASE / "frozen-v1.lock.json").read_text(encoding="utf-8"))

    assert split_sha256(frozen) == lock["split_sha256"]
    assert dataset_sha256(BASE / "frozen-v1.seeds.jsonl") == lock["seed_sha256"]


def test_ablation_reaches_all_eight_registered_documents():
    dev = _load("dev.seeds.jsonl", "dev")
    frozen = _load("frozen-v1.seeds.jsonl", "frozen-v1")
    docs = {
        evidence["doc"]
        for case in [*dev, *frozen]
        for evidence in case["required_evidence"]
    }

    assert len(docs) == 8
