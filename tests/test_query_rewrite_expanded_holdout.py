import json
from collections import Counter
from pathlib import Path

from scripts.corpus_facts import load_factbook
from scripts.eval_dataset import dataset_sha256, split_sha256
from scripts.query_rewrite_ablation.dataset import build_cases
from scripts.query_rewrite_ablation.metrics import summarize_variant_traces
from scripts.query_rewrite_ablation.queries import VariantQuery
from scripts.query_rewrite_ablation.runner import _variant_trace, load_cases

BASE = Path("tests/eval/query_rewrite_ablation")
OLD_SPLITS = ("dev.seeds.jsonl", "frozen-v1.seeds.jsonl")
NEW_SEEDS = "frozen-v2.seeds.jsonl"
NEW_LOCK = "frozen-v2.lock.json"
RUBRIC = Path("docs/eval/query-rewrite-expanded-rubric.md")


def _load(name: str, split: str) -> list[dict]:
    return load_cases(BASE / name, split)


def _seed_rows(name: str) -> list[dict]:
    return [
        json.loads(line)
        for line in (BASE / name).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_builder_supports_intent_only_clarification_without_gold_evidence():
    seed = {
        "id": "TEST-QR-CLARIFY",
        "bucket": "clarify",
        "gap_type": "ambiguity",
        "semantic_family": "multiple-valid-product-answers",
        "template_id": "clarify-product-model",
        "evaluation_mode": "intent_only",
        "expected_behavior": "clarify",
        "clarifying_slots": ["product_model"],
        "query": "耳机多少钱？",
        "conversation_context": [],
        "forbidden_anchors": ["SH-E120", "SH-E300"],
        "source": "curated_from_corpus",
        "source_ref": "knowledge/商品目录.md#商品条目",
        "source_docs": ["商品目录.md"],
    }

    cases = build_cases([seed], load_factbook(), split="frozen-v2")

    assert len(cases) == 1
    assert cases[0]["evaluation_mode"] == "intent_only"
    assert cases[0]["expected_behavior"] == "clarify"
    assert cases[0]["gold_facts"] == []
    assert cases[0]["required_evidence"] == []
    assert cases[0]["clarifying_slots"] == ["product_model"]
    assert cases[0]["source_docs"] == ["商品目录.md"]


def test_expanded_holdout_has_gap_driven_scale_and_unique_semantic_families():
    cases = _load(NEW_SEEDS, "frozen-v2")

    expected_gaps = {
        "history_resolution": 10,
        "topic_switch": 8,
        "entity_order_numeric": 8,
        "ambiguity": 8,
        "negation_correction": 8,
        "explicit_no_rewrite": 10,
        "multi_constraint": 8,
    }
    assert len(cases) == 60
    assert Counter(case["gap_type"] for case in cases) == expected_gaps
    assert len({case["semantic_family"] for case in cases}) == 60
    assert len({case["template_id"] for case in cases}) == 60
    assert len({case["query"] for case in cases}) == 60
    assert sum(
        case["expected_behavior"] == "clarify" for case in cases
    ) == 8
    assert sum(
        case["gap_type"] == "explicit_no_rewrite" for case in cases
    ) == 10


def test_expanded_holdout_is_isolated_from_old_dev_and_frozen_sets():
    old = [*_load(OLD_SPLITS[0], "dev"), *_load(OLD_SPLITS[1], "frozen-v1")]
    new = _load(NEW_SEEDS, "frozen-v2")

    old_facts = {fact["fact_id"] for case in old for fact in case["gold_facts"]}
    new_facts = {fact["fact_id"] for case in new for fact in case["gold_facts"]}
    old_queries = {case["query"] for case in old}
    new_queries = {case["query"] for case in new}
    old_templates = {case["template_id"] for case in old}
    new_templates = {case["template_id"] for case in new}
    old_ids = {case["id"] for case in old}
    new_ids = {case["id"] for case in new}

    assert old_facts.isdisjoint(new_facts)
    assert old_queries.isdisjoint(new_queries)
    assert old_templates.isdisjoint(new_templates)
    assert old_ids.isdisjoint(new_ids)

    docs = {doc for case in new for doc in case["source_docs"]}
    assert len(docs) == 8
    assert all(case["source_ref"] for case in new)
    assert all(case["review_status"] == "frozen" for case in new)


def test_expanded_holdout_hash_and_rubric_are_locked_before_run():
    cases = _load(NEW_SEEDS, "frozen-v2")
    lock = json.loads((BASE / NEW_LOCK).read_text(encoding="utf-8"))

    assert lock["case_count"] == 60
    assert lock["dataset_sha256"] == dataset_sha256(BASE / NEW_SEEDS)
    assert lock["split_sha256"] == split_sha256(cases)
    assert lock["rubric_id"] == "query-rewrite-expanded-v1"
    assert lock["rubric_sha256"] == dataset_sha256(RUBRIC)
    assert lock["isolation"] == {
        "gold_fact_overlap_with_dev_or_frozen_v1": 0,
        "query_overlap_with_dev_or_frozen_v1": 0,
        "template_overlap_with_dev_or_frozen_v1": 0,
        "semantic_family_count": 60,
        "synonym_variants_counted_as_cases": False,
    }


def test_summary_excludes_intent_only_cases_from_coverage_denominator():
    traces = [
        _trace("R", "raw", coverage=True, applicable=True),
        _trace("R", "safe", coverage=True, applicable=True),
        _trace("Q", "raw", coverage=False, applicable=False),
        _trace("Q", "safe", coverage=False, applicable=False),
    ]

    summary = summarize_variant_traces(traces)

    assert summary["variants"]["raw"]["coverage_cases"] == 1
    assert summary["variants"]["raw"]["complete_at_5"] == {
        "numerator": 1,
        "denominator": 1,
    }
    assert summary["variants"]["safe"]["cases"] == 2
    assert summary["variants"]["safe"]["intent_preserved"] == {
        "numerator": 2,
        "denominator": 2,
    }
    assert summary["paired"]["safe_vs_raw"]["complete_at_5"] == {
        "pairs": 1,
        "gains": 0,
        "losses": 0,
        "ties": 1,
    }


def test_summary_keeps_failed_trace_in_denominator():
    traces = [
        _trace("A", "safe", coverage=True, applicable=True),
        {
            **_trace("B", "safe", coverage=False, applicable=True),
            "error": "retrieval timeout",
        },
    ]

    summary = summarize_variant_traces(traces)
    variant = summary["variants"]["safe"]

    assert variant["cases"] == 2
    assert variant["failures"] == 1
    assert variant["failure_rate"] == {"numerator": 1, "denominator": 2}
    assert variant["complete_at_5"] == {"numerator": 1, "denominator": 2}


class _Service:
    def __init__(self, error=None):
        self.error = error
        self.calls = 0

    def retrieve(self, *_args, **_kwargs):
        self.calls += 1
        if self.error:
            raise self.error
        raise AssertionError("test does not need a successful retrieval payload")


class _Context:
    def __init__(self, service):
        self.service = service
        self.all_rows = {}


def test_intent_only_trace_excludes_retrieval_coverage():
    case = {
        "id": "A",
        "split": "frozen-v2",
        "bucket": "clarify",
        "gap_type": "ambiguity",
        "query": "耳机多少钱？",
    }
    service = _Service(error=RuntimeError("must not be called"))
    trace = _variant_trace(
        case,
        VariantQuery("raw", "耳机多少钱？", "耳机多少钱？", "耳机多少钱？"),
        ctx=_Context(service),
        gold=[],
        latency_ms=0.0,
        retrieval_cost_usd=0.0,
        query_cost_usd=0.0,
        stage_stats={},
    )

    assert trace["metrics"]["coverage_applicable"] is False
    assert trace["error"] is None
    assert service.calls == 0


def test_retrieval_failure_keeps_applicable_trace_with_error():
    case = {
        "id": "B",
        "split": "frozen-v2",
        "bucket": "single_hop",
        "gap_type": "explicit_no_rewrite",
        "query": "ORD-1001 状态？",
    }
    trace = _variant_trace(
        case,
        VariantQuery("safe", "ORD-1001 状态？", "ORD-1001 状态？", "ORD-1001 状态？"),
        ctx=_Context(_Service(error=TimeoutError("index timeout"))),
        gold=[frozenset({1})],
        latency_ms=0.0,
        retrieval_cost_usd=0.0,
        query_cost_usd=0.0,
        stage_stats={},
    )

    assert trace["metrics"]["coverage_applicable"] is True
    assert trace["metrics"]["complete_at_10"] is False
    assert trace["error"] == "TimeoutError: index timeout"
    assert trace["gate"]["degraded"] is True


def _trace(case_id: str, variant: str, *, coverage: bool, applicable: bool) -> dict:
    return {
        "case_id": case_id,
        "variant": variant,
        "gap_type": "history_resolution" if applicable else "ambiguity",
        "metrics": {
            "coverage_applicable": applicable,
            "complete_at_5": coverage,
            "complete_at_10": coverage,
            "mrr_at_10": 1.0 if coverage else 0.0,
            "intent_preserved": True,
            "anchors_complete": True,
            "wrong_entity": False,
        },
        "latency_ms": 1.0,
        "cost_usd": 0.0,
    }
