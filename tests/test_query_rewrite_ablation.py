import json

from scripts.corpus_facts import load_factbook
from scripts.query_rewrite_ablation.dataset import build_cases
from scripts.query_rewrite_ablation.metrics import summarize_variant_traces
from scripts.query_rewrite_ablation.queries import build_query_variants
from scripts.query_rewrite_ablation.runner import query_intent_metrics
from scripts.query_rewrite_ablation.verify import compare_run_files


def test_build_cases_resolves_fact_and_evidence_gold_from_seed():
    seed = {
        "id": "DEV-QR-CTX-01",
        "bucket": "history_resolution",
        "query": "它续航多久？",
        "conversation_context": [
            {"role": "user", "content": "SH-E300 多少钱？"},
            {"role": "assistant", "content": "SH-E300 售价 299 元。"},
        ],
        "gold_fact_ids": ["CATALOG-SH-E300-BATTERY-HOURS"],
        "evidence_ids": ["EV-CATALOG-SH-E300-ENTRY"],
        "required_anchor_groups": [["SH-E300"]],
        "forbidden_anchors": [],
        "preserve_negation": False,
        "notes": "history ellipsis",
    }

    cases = build_cases([seed], load_factbook(), split="dev")

    assert len(cases) == 1
    case = cases[0]
    assert case["split"] == "dev"
    assert case["gold_facts"][0]["fact_id"] == "CATALOG-SH-E300-BATTERY-HOURS"
    assert case["gold_facts"][0]["value"] == 30
    assert case["required_evidence"] == [{
        "doc": "商品目录.md",
        "section": "SH-E300 头戴式无线降噪耳机",
        "quotes": [
            "SH-E300 头戴式无线降噪耳机,售价 299 元,支持主动降噪,蓝牙 5.3,整机续航 30 小时,整机一年质保。"
        ],
        "fact_ids": [
            "CATALOG-SH-E300-PRICE",
            "CATALOG-SH-E300-PRODUCT-TYPE",
            "CATALOG-SH-E300-ANC",
            "CATALOG-SH-E300-BLUETOOTH-VERSION",
            "CATALOG-SH-E300-BATTERY-HOURS",
            "CATALOG-SH-E300-WARRANTY-MONTHS",
        ],
    }]
    assert case["required_anchor_groups"] == [["SH-E300"]]


def test_build_cases_rejects_fact_evidence_mismatch():
    seed = {
        "id": "BAD-01",
        "bucket": "history_resolution",
        "query": "续航多久？",
        "conversation_context": [],
        "gold_fact_ids": ["CATALOG-SH-E300-BATTERY-HOURS"],
        "evidence_ids": ["EV-CATALOG-SH-E120-ENTRY"],
        "required_anchor_groups": [["SH-E300"]],
        "forbidden_anchors": [],
        "preserve_negation": False,
    }

    try:
        build_cases([seed], load_factbook(), split="dev")
    except ValueError as exc:
        assert "evidence" in str(exc)
    else:
        raise AssertionError("mismatched evidence must fail closed")


def _variant(
    case_id,
    variant,
    *,
    coverage5,
    coverage10,
    intent=True,
    wrong=False,
    latency=1.0,
):
    return {
        "case_id": case_id,
        "variant": variant,
        "metrics": {
            "complete_at_5": coverage5,
            "complete_at_10": coverage10,
            "mrr_at_10": 1.0 if coverage10 else 0.0,
            "intent_preserved": intent,
            "wrong_entity": wrong,
        },
        "latency_ms": latency,
        "cost_usd": 0.001,
    }


def test_summarize_variant_traces_reports_denominators_and_paired_changes():
    traces = [
        _variant("A", "raw", coverage5=False, coverage10=True),
        _variant("A", "legacy", coverage5=True, coverage10=True),
        _variant("A", "safe", coverage5=True, coverage10=True),
        _variant("B", "raw", coverage5=True, coverage10=True),
        _variant("B", "legacy", coverage5=False, coverage10=True),
        _variant("B", "safe", coverage5=True, coverage10=True),
        _variant("C", "raw", coverage5=False, coverage10=False, intent=False, wrong=True),
        _variant("C", "legacy", coverage5=False, coverage10=False, intent=False, wrong=True),
        _variant("C", "safe", coverage5=False, coverage10=True, intent=True),
    ]

    summary = summarize_variant_traces(traces)

    assert summary["variants"]["raw"]["complete_at_5"] == {"numerator": 1, "denominator": 3}
    assert summary["variants"]["legacy"]["complete_at_5"] == {"numerator": 1, "denominator": 3}
    assert summary["variants"]["safe"]["complete_at_5"] == {"numerator": 2, "denominator": 3}
    assert summary["variants"]["safe"]["intent_preserved"] == {"numerator": 3, "denominator": 3}
    assert summary["variants"]["legacy"]["wrong_entity"] == 1
    assert summary["paired"]["safe_vs_raw"]["complete_at_5"] == {
        "pairs": 3,
        "gains": 1,
        "losses": 0,
        "ties": 2,
    }
    assert summary["paired"]["safe_vs_legacy"]["complete_at_5"] == {
        "pairs": 3,
        "gains": 1,
        "losses": 0,
        "ties": 2,
    }


def _case_for_queries(query):
    return {
        "id": "DEV-QR-CTX-02",
        "query": query,
        "conversation_context": [
            {"role": "user", "content": "SH-E300 多少钱？"},
            {"role": "assistant", "content": "SH-E300 售价 299 元。"},
        ],
    }


def test_build_query_variants_keeps_unsafe_legacy_output_but_guards_safe_output():
    case = _case_for_queries("它续航多久？")

    def legacy_resolve(_prompt):
        return {"resolved": "SH-E120 续航多久？", "changed": True}

    def safe_resolve(_prompt):
        return {"resolved": "SH-E300 续航多久？", "changed": True}

    def rewrite(prompt):
        query = "SH-E120 续航多久？" if "SH-E120" in prompt else "SH-E300 续航多久？"
        return {"normalized": query, "synonyms": []}

    variants = build_query_variants(
        case,
        legacy_resolve=legacy_resolve,
        legacy_rewrite=rewrite,
        safe_resolve=safe_resolve,
        safe_rewrite=rewrite,
    )

    assert variants["raw"].text == "它续航多久？"
    assert variants["legacy"].text == "SH-E120 续航多久？"
    assert variants["safe"].text == "SH-E300 续航多久？"
    assert "new_anchor:SH-E120" in variants["legacy"].guard_violations
    assert variants["safe"].guard_violations == ()


def test_build_query_variants_falls_back_when_safe_rewrite_flips_polarity():
    case = {
        "id": "DEV-QR-NEG-01",
        "query": "ORD-1003 未发货，能取消吗？",
        "conversation_context": [],
    }

    variants = build_query_variants(
        case,
        legacy_resolve=lambda _prompt: {"resolved": case["query"], "changed": False},
        legacy_rewrite=lambda _prompt: {"normalized": "ORD-1003 已发货，能取消吗？"},
        safe_resolve=lambda _prompt: {"resolved": case["query"], "changed": False},
        safe_rewrite=lambda _prompt: {
            "normalized": "ORD-1003 已发货，能取消吗？",
            "synonyms": [],
        },
    )

    assert variants["legacy"].text == "ORD-1003 已发货，能取消吗？"
    assert variants["safe"].text == case["query"]
    assert "polarity_changed" in variants["safe"].guard_violations


def test_query_intent_metrics_require_anchor_and_preserve_negation():
    case = {
        "required_anchor_groups": [["SH-E300"]],
        "forbidden_anchors": ["SH-E120"],
        "preserve_negation": True,
    }

    raw = query_intent_metrics(case, "它是不是不支持主动降噪？")
    safe = query_intent_metrics(case, "SH-E300 是不是不支持主动降噪？")
    wrong = query_intent_metrics(case, "SH-E120 是不是不支持主动降噪？")

    assert raw["intent_preserved"] is False
    assert safe["intent_preserved"] is True
    assert wrong["wrong_entity"] is True
    assert wrong["intent_preserved"] is False


def test_query_intent_metrics_do_not_flag_explicitly_rejected_entity_as_wrong():
    case = {
        "required_anchor_groups": [["SH-E200"]],
        "forbidden_anchors": ["SH-E120"],
        "preserve_negation": True,
    }

    metrics = query_intent_metrics(
        case, "不要按 SH-E120 回答，请问 SH-E200 续航多久？"
    )

    assert metrics["wrong_entity"] is False
    assert metrics["intent_preserved"] is True


def test_compare_run_files_detects_constructed_query_drift(tmp_path):
    baseline = tmp_path / "baseline.jsonl"
    replay = tmp_path / "replay.jsonl"
    row = _variant("A", "safe", coverage5=True, coverage10=True)
    row["constructed_query"] = {"text": "SH-E300 续航多久？"}
    baseline.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    replay.write_text(json.dumps({
        **row,
        "constructed_query": {"text": "SH-E120 续航多久？"},
    }, ensure_ascii=False) + "\n", encoding="utf-8")

    result = compare_run_files(baseline, replay)

    assert result["ok"] is False
    assert result["mismatches"] == [{
        "case_id": "A",
        "variant": "safe",
        "fields": ["constructed_query"],
    }]


def test_compare_run_files_ignores_cache_latency_and_cost_changes(tmp_path):
    baseline = tmp_path / "baseline.jsonl"
    replay = tmp_path / "replay.jsonl"
    row = _variant("A", "safe", coverage5=True, coverage10=True, latency=1000.0)
    row["constructed_query"] = {"text": "SH-E300 续航多久？"}
    baseline.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    replay.write_text(json.dumps({
        **row,
        "latency_ms": 5.0,
        "cost_usd": 0.0,
    }, ensure_ascii=False) + "\n", encoding="utf-8")

    result = compare_run_files(baseline, replay)

    assert result["ok"] is True
    assert result["aggregate_changed"] == []
