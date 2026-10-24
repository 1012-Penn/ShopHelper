"""分层聚合、阻断项与报告渲染。

不合成单一总分;阻断项单独给分子/分母,避免被召回提升抵消。
"""
from __future__ import annotations

import json
from pathlib import Path


def _avg(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _rate(hits: int, total: int):
    return None if total == 0 else hits / total


def _ranks(traces: list[dict], key: str) -> dict:
    positives = [trace for trace in traces if trace.get("l1", {}).get("n")]
    hits = sum(1 for trace in positives if trace["l1"].get(key) is not None)
    return {"hits": hits, "n": len(positives), "rate": _rate(hits, len(positives))}


def _layer_l1(traces: list[dict]) -> dict:
    positives = [trace for trace in traces if trace.get("l1", {}).get("n")]
    by_bucket = {}
    for trace in traces:
        if trace.get("l1", {}).get("n"):
            by_bucket.setdefault(trace["bucket"], []).append(trace)

    def bucket_summary(rows):
        return {
            "n": len(rows),
            "r3": _rate(sum(1 for t in rows if t["l1"].get("r3") is not None), len(rows)),
            "r5": _rate(sum(1 for t in rows if t["l1"].get("r5") is not None), len(rows)),
            "r10": _rate(sum(1 for t in rows if t["l1"].get("r10") is not None), len(rows)),
            "mrr": _avg([
                (1.0 / t["l1"]["coverage_rank"]) if t["l1"].get("coverage_rank") else 0.0
                for t in rows
            ]),
            "group_recall": _avg([t["l1"].get("group_recall") for t in rows]),
            "single1": _rate(sum(1 for t in rows if t["l1"].get("single1") == 1), len(rows)),
            "single3": _rate(sum(1 for t in rows if t["l1"].get("single3") is not None), len(rows)),
            "evidence_precision": _avg([t["l1"].get("evidence_precision") for t in rows]),
        }

    negatives = [trace for trace in traces if trace.get("l1", {}).get("n") == 0]
    hard_negative_hits = [
        trace["l1"].get("hard_negative_hit_rate")
        for trace in negatives
        if trace.get("bucket") in {"hard_negative", "ood"}
    ]
    return {
        "n": len(positives),
        "r3": _ranks(traces, "r3"),
        "r5": _ranks(traces, "r5"),
        "r10": _ranks(traces, "r10"),
        "mrr": _avg([
            (1.0 / t["l1"]["coverage_rank"]) if t["l1"].get("coverage_rank") else 0.0
            for t in positives
        ]),
        "group_recall": _avg([t["l1"].get("group_recall") for t in positives]),
        "single1": _rate(sum(1 for t in positives if t["l1"].get("single1") == 1), len(positives)),
        "single3": _rate(sum(1 for t in positives if t["l1"].get("single3") is not None), len(positives)),
        "evidence_precision": _avg([t["l1"].get("evidence_precision") for t in positives]),
        "hard_negative_hit_rate": _avg(hard_negative_hits),
        "by_bucket": {bucket: bucket_summary(rows) for bucket, rows in sorted(by_bucket.items())},
    }


def _layer_l2(traces: list[dict]) -> dict:
    refuse_cases = [t for t in traces if t.get("expected_behavior") == "refuse"]
    answer_cases = [t for t in traces if t.get("expected_behavior") != "refuse"]
    false_accept = [t for t in refuse_cases if t["gate"]["passed"]]
    false_block = [t for t in answer_cases if not t["gate"]["passed"]]
    conflicts = [t for t in traces if t.get("bucket") == "policy_conflict"]
    conflict_ok = [
        t for t in conflicts
        if (t.get("deterministic_checks") or {}).get("gold_fact_accuracy", {}).get("passed")
    ]
    partial = [
        t for t in traces
        if (t.get("deterministic_checks") is not None)
        and t.get("expected_behavior") in {"verify_then_answer", "clarify"}
    ]
    partial_ok = [
        t for t in partial
        if t.get("deterministic_checks", {}).get("behavior_pred") != "answer"
        or t.get("deterministic_checks", {}).get("condition_coverage", {}).get("passed")
    ]
    return {
        "gate_false_accept": len(false_accept),
        "gate_false_block": len(false_block),
        "false_accept_ids": [t["case_id"] for t in false_accept],
        "false_block_ids": [t["case_id"] for t in false_block],
        "conflict_detection_rate": _rate(len(conflict_ok), len(conflicts)),
        "partial_handling_rate": _rate(len(partial_ok), len(partial)),
    }


def _layer_l3(traces: list[dict]) -> dict:
    judged = [t for t in traces if t.get("deterministic_checks") is not None]
    strict = []
    for trace in judged:
        checks = trace["deterministic_checks"]
        if checks.get("gold_facts_checked", 0) == 0:
            continue
        strict.append(
            checks["gold_fact_accuracy"]["passed"]
            and checks["condition_coverage"]["passed"]
            and not checks["forbidden_claims"]
            and checks["citation_support"]["passed"]
        )
    forbidden_cases = [t for t in judged if (t.get("forbidden_claims") or [])]
    forbidden_hits = [t for t in forbidden_cases if t["deterministic_checks"]["forbidden_claims"]]
    clarify = [t for t in judged if t.get("expected_behavior") == "clarify"]
    return {
        "answer_accuracy_strict": _rate(sum(strict), len(strict)),
        "strict_n": len(strict),
        "answer_accuracy_partial": _avg([
            t["deterministic_checks"]["gold_fact_accuracy"]["score"]
            for t in judged if t["deterministic_checks"].get("gold_facts_checked", 0)
        ]),
        "condition_coverage": _avg([
            t["deterministic_checks"]["condition_coverage"]["score"] for t in judged
        ]),
        "forbidden_claim_rate": _rate(len(forbidden_hits), len(forbidden_cases)),
        "forbidden_claim_ids": [t["case_id"] for t in forbidden_hits],
        "clarify_slot_recall": _avg([
            t["deterministic_checks"]["clarify_slot_recall"]["score"] for t in clarify
        ]),
        "clarify_precision": _avg([
            t["deterministic_checks"]["clarify_precision"]["score"] for t in clarify
        ]),
        "unsupported_claim_rate": _avg([
            t["deterministic_checks"]["unsupported_claim_rate"] for t in judged
        ]),
        "failure_labels": {
            label: sum(1 for t in traces for item in (t.get("failure_labels") or []) if item == label)
            for label in sorted({item for t in traces for item in (t.get("failure_labels") or [])})
        },
    }


def _layer_l4(traces: list[dict]) -> dict:
    judged = [t for t in traces if t.get("l4_applicable")]
    refuse_cases = [t for t in judged if t.get("expected_behavior") == "refuse"]
    answer_cases = [t for t in judged if t.get("expected_behavior") != "refuse"]
    conditional = [t for t in judged if t.get("expected_behavior") == "answer_with_conditions"]
    verify = [t for t in judged if t.get("expected_behavior") == "verify_then_answer"]

    def pred(trace):
        return (trace.get("deterministic_checks") or {}).get("behavior_pred")

    under = [t for t in refuse_cases if pred(t) != "refuse"]
    over = [t for t in answer_cases if pred(t) == "refuse"]
    conditional_ok = [
        t for t in conditional
        if t["deterministic_checks"]["condition_coverage"]["passed"]
    ]
    verify_ok = [
        t for t in verify
        if t["deterministic_checks"].get("verification_compliant")
    ]
    return {
        "under_refusal_rate": _rate(len(under), len(refuse_cases)),
        "under_refusal_ids": [t["case_id"] for t in under],
        "over_refusal_rate": _rate(len(over), len(answer_cases)),
        "over_refusal_ids": [t["case_id"] for t in over],
        "correct_refusal_rate": _rate(len(refuse_cases) - len(under), len(refuse_cases)),
        "conditional_compliance": _rate(len(conditional_ok), len(conditional)),
        "verification_compliance": _rate(len(verify_ok), len(verify)),
        "n": len(judged),
    }


def _layer_l5(traces: list[dict]) -> dict:
    latencies = sorted(
        trace["latency_ms"] for trace in traces if trace.get("latency_ms") is not None
    )

    def percentile(ratio):
        if not latencies:
            return None
        return latencies[min(len(latencies) - 1, int(ratio * (len(latencies) - 1)))]

    return {
        "llm_calls": sum(trace.get("llm_calls", 0) for trace in traces),
        "tokens_in": sum(trace.get("tokens_in", 0) for trace in traces),
        "tokens_out": sum(trace.get("tokens_out", 0) for trace in traces),
        "cost_estimate": sum(trace.get("cost_estimate", 0.0) for trace in traces),
        "cached_cases": sum(1 for trace in traces if trace.get("cached")),
        "latency_p50_ms": percentile(0.50),
        "latency_p95_ms": percentile(0.95),
    }


def _blocking(traces: list[dict]) -> dict:
    hard = [t for t in traces if t.get("bucket") == "hard_negative"]
    ood = [t for t in traces if t.get("bucket") == "ood"]
    forbidden = [t for t in traces if t.get("forbidden_claims")]
    under = [t for t in traces if "under_refusal" in (t.get("failure_labels") or [])]
    unsupported = [
        t for t in traces
        if (t.get("deterministic_checks") or {}).get("unsupported_claim_rate", 0) > 0
    ]
    return {
        "hard_negative_false_accept": {
            "count": sum(1 for t in hard if t["gate"]["passed"]),
            "denominator": len(hard),
        },
        "ood_false_accept": {
            "count": sum(1 for t in ood if t["gate"]["passed"]),
            "denominator": len(ood),
        },
        "forbidden_claim": {
            "count": len([
                t for t in forbidden
                if (t.get("deterministic_checks") or {}).get("forbidden_claims")
            ]),
            "denominator": len(forbidden),
        },
        "unsupported_claim": {"count": len(unsupported), "denominator": len(traces)},
        "under_refusal": {"count": len(under), "denominator": len(
            [t for t in traces if t.get("expected_behavior") == "refuse"]
        )},
    }


def aggregate_traces(traces: list[dict]) -> dict:
    return {
        "n": len(traces),
        "l1": _layer_l1(traces),
        "l2": _layer_l2(traces),
        "l3": _layer_l3(traces),
        "l4": _layer_l4(traces),
        "l5": _layer_l5(traces),
        "blocking": _blocking(traces),
    }


def blocking_metrics(agg: dict) -> dict:
    return agg["blocking"]


def render_report(agg: dict, contract: dict, *, historical_md: str = "") -> str:
    lines = [
        "# ch11 分层评测报告",
        "",
        "## 契约",
        f"- contract_sha256: `{contract.get('_sha', '')}`",
        f"- dataset: {contract['dataset']['split']} / n={contract['dataset']['n']} / sha={contract['dataset']['sha256'][:16]}",
        f"- build_id: `{contract['index']['build_id']}`",
        f"- calibration: `{contract['retrieval'].get('calibration_status')}`",
        f"- pricing verified: {contract['pricing']['verified']}",
        "",
        "## L1 检索",
        "| 指标 | 值 | 分母/说明 |",
        "|---|---:|---|",
        f"| R@3 | {agg['l1']['r3']['rate']} | {agg['l1']['r3']['hits']}/{agg['l1']['r3']['n']} |",
        f"| R@5 | {agg['l1']['r5']['rate']} | {agg['l1']['r5']['hits']}/{agg['l1']['r5']['n']} |",
        f"| R@10 | {agg['l1']['r10']['rate']} | {agg['l1']['r10']['hits']}/{agg['l1']['r10']['n']} |",
        f"| MRR@10 | {agg['l1']['mrr']} | n={agg['l1']['n']} |",
        f"| group_recall | {agg['l1']['group_recall']} | n={agg['l1']['n']} |",
        f"| single@1 | {agg['l1']['single1']} | n={agg['l1']['n']} |",
        f"| evidence_precision | {agg['l1']['evidence_precision']} | n={agg['l1']['n']} |",
        f"| hard_negative_hit_rate | {agg['l1']['hard_negative_hit_rate']} | 负例题 |",
        "",
        "## L2 证据",
        f"- gate_false_accept: {agg['l2']['gate_false_accept']} {agg['l2']['false_accept_ids']}",
        f"- gate_false_block: {agg['l2']['gate_false_block']} {agg['l2']['false_block_ids']}",
        f"- conflict_detection_rate: {agg['l2']['conflict_detection_rate']}",
        f"- partial_handling_rate: {agg['l2']['partial_handling_rate']}",
        "",
        "## L3 回答",
        f"- strict accuracy: {agg['l3']['answer_accuracy_strict']} (n={agg['l3']['strict_n']})",
        f"- partial accuracy: {agg['l3']['answer_accuracy_partial']}",
        f"- condition_coverage: {agg['l3']['condition_coverage']}",
        f"- forbidden_claim_rate: {agg['l3']['forbidden_claim_rate']} {agg['l3']['forbidden_claim_ids']}",
        f"- clarify_slot_recall: {agg['l3']['clarify_slot_recall']}",
        f"- clarify_precision: {agg['l3']['clarify_precision']}",
        f"- unsupported_claim_rate: {agg['l3']['unsupported_claim_rate']}",
        f"- failure_labels: {agg['l3']['failure_labels']}",
        "",
        "## L4 行为",
        f"- under_refusal_rate: {agg['l4']['under_refusal_rate']} {agg['l4']['under_refusal_ids']}",
        f"- over_refusal_rate: {agg['l4']['over_refusal_rate']} {agg['l4']['over_refusal_ids']}",
        f"- correct_refusal_rate: {agg['l4']['correct_refusal_rate']}",
        f"- conditional_compliance: {agg['l4']['conditional_compliance']}",
        f"- verification_compliance: {agg['l4']['verification_compliance']}",
        f"- L4 分母(剔除工具题): {agg['l4']['n']}",
        "",
        "## L5 过程",
        f"- llm_calls: {agg['l5']['llm_calls']}",
        f"- tokens_in/out: {agg['l5']['tokens_in']}/{agg['l5']['tokens_out']}",
        f"- cost_estimate: ${agg['l5']['cost_estimate']:.4f}",
        f"- cached_cases: {agg['l5']['cached_cases']}",
        f"- latency_p50/p95_ms: {agg['l5']['latency_p50_ms']} / {agg['l5']['latency_p95_ms']}",
        "",
        "## 阻断项（不可被召回提升抵消）",
        "| 项 | 计数/分母 |",
        "|---|---|",
        f"| 硬负例误放行 | {agg['blocking']['hard_negative_false_accept']['count']}/{agg['blocking']['hard_negative_false_accept']['denominator']} |",
        f"| OOD 误放行 | {agg['blocking']['ood_false_accept']['count']}/{agg['blocking']['ood_false_accept']['denominator']} |",
        f"| 禁用断言 | {agg['blocking']['forbidden_claim']['count']}/{agg['blocking']['forbidden_claim']['denominator']} |",
        f"| 无证据断言 | {agg['blocking']['unsupported_claim']['count']}/{agg['blocking']['unsupported_claim']['denominator']} |",
        f"| 应拒未拒 | {agg['blocking']['under_refusal']['count']}/{agg['blocking']['under_refusal']['denominator']} |",
    ]
    if historical_md:
        lines += ["", "## 历史基线（只作参照，不与本次拼提升）", "", historical_md.strip()]
    return "\n".join(lines) + "\n"


def write_report(out_dir: Path, agg: dict, contract: dict, traces: list[dict]) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = render_report(agg, contract)
    path = out_dir / "report.md"
    path.write_text(report, encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps(agg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (out_dir / "traces.jsonl").open("w", encoding="utf-8") as fh:
        for trace in traces:
            fh.write(json.dumps(trace, ensure_ascii=False, sort_keys=True) + "\n")
    return path
