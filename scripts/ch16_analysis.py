"""CH16 配对评测指标。

检索覆盖和最终回答质量分层汇报。检索 miss 是 R@/MRR 的失败；它不会被当成
答案正确率。严格回答分母包括被证据闸拦下的可答题，缓存缺失则单独记作未测。
"""
from __future__ import annotations

import math
import hashlib
import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Iterable


ANSWER_BEHAVIORS = frozenset({"answer", "answer_with_conditions", "verify_then_answer"})
NON_REFUSAL_BEHAVIORS = ANSWER_BEHAVIORS | {"clarify"}
RETRIEVAL_METRICS = ("r3", "r5", "mrr")
EXPERIMENT_GROUPS = {
    "retrieval": ("dense", "bm25", "hybrid", "hybrid_rerank"),
    "rag": ("baseline", "decompose_once", "controlled_loop"),
}
PROTOCOL_CODE_PATHS = (
    "scripts/ch16_analysis.py", "scripts/eval_rag_ablation.py",
    "scripts/eval_cost.py", "scripts/eval_ch11.py", "scripts/ch11_harness.py",
    "scripts/eval_cache.py", "scripts/eval_prompts.py", "scripts/eval_checks.py",
    "scripts/eval_metrics.py", "scripts/eval_contract.py", "scripts/eval_snapshot.py",
    "app/config.py", "app/llm.py", "app/embedding.py", "app/rerank.py",
    "app/retrieval.py", "app/rag_loop.py", "app/evidence.py", "app/prompts.py",
)


class ContractError(ValueError):
    """组间契约漂移或运行预算不符合 CH16 限制。"""


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _case_map(cases) -> dict[str, dict]:
    if cases is None:
        return {}
    if isinstance(cases, dict):
        return {str(case_id): case for case_id, case in cases.items()}
    return {str(case["id"]): case for case in cases}


def _trace_case(trace: dict, cases_by_id: dict[str, dict]) -> dict:
    case_id = str(trace.get("case_id", ""))
    case = cases_by_id.get(case_id, {})
    return {
        **case,
        "id": case_id,
        "expected_behavior": case.get("expected_behavior", trace.get("expected_behavior", "")),
        "bucket": case.get("bucket", trace.get("bucket", "")),
        "required_evidence": case.get("required_evidence", trace.get("required_evidence", [])),
        "required_conditions": case.get("required_conditions", trace.get("required_conditions", [])),
        "gold_facts": case.get("gold_facts") or [None] * int(
            (trace.get("deterministic_checks") or {}).get("gold_facts_checked", 0)
        ),
    }


def _answer_eligible(case: dict) -> bool:
    return case.get("expected_behavior") in ANSWER_BEHAVIORS and bool(case.get("gold_facts"))


def _blocked(trace: dict) -> bool:
    gate = trace.get("gate") or {}
    return bool(trace.get("gate_blocked", gate.get("low_confidence", False)))


def strict_answer_outcome(case: dict, trace: dict) -> bool | None:
    """返回严格答对/答错；没有可用答案轨迹或不属于有事实金标的答题时返回 None。

    可答题分母由 case 的金标决定。闸拦、误拒、事实错、漏条件、禁用断言、无引用或
    引用不支持均为严格失败；不能因 gate 阻断而从分母中消失。
    """
    expected = case.get("expected_behavior", trace.get("expected_behavior"))
    checks = trace.get("deterministic_checks")
    if expected not in ANSWER_BEHAVIORS:
        return None
    if trace.get("answer_status") == "unmeasured":
        return None
    has_gold = bool(case.get("gold_facts")) or bool(
        (checks or {}).get("gold_facts_checked", 0)
    )
    if not has_gold:
        return None
    if _blocked(trace):
        return False
    if not isinstance(checks, dict):
        return None
    return bool(
        checks.get("gold_facts_checked", 0) > 0
        and checks.get("gold_fact_accuracy", {}).get("passed")
        and checks.get("condition_coverage", {}).get("passed")
        and not checks.get("forbidden_claims")
        and checks.get("citation_support", {}).get("passed")
        and checks.get("verification_compliant", True)
        and checks.get("behavior_pred") != "refuse"
    )


def _rank_score(trace: dict, metric: str) -> float | None:
    l1 = trace.get("l1") or {}
    if l1.get("n", 0) != 1:
        return None
    if metric in {"r3", "r5"}:
        return 1.0 if l1.get(metric) is not None else 0.0
    value = l1.get("mrr")
    return 0.0 if value is None else float(value)


def _retrieval_population(traces: list[dict], cases_by_id: dict[str, dict]):
    by_id = {str(trace["case_id"]): trace for trace in traces if trace.get("case_id")}
    if cases_by_id:
        positives = [case_id for case_id, case in cases_by_id.items()
                     if case.get("required_evidence")]
    else:
        positives = [case_id for case_id, trace in by_id.items()
                     if (trace.get("l1") or {}).get("n", 0) == 1]
    return by_id, positives


def _retrieval_metric(traces_by_id: dict[str, dict], case_ids: list[str], metric: str) -> dict:
    measured = []
    for case_id in case_ids:
        trace = traces_by_id.get(case_id)
        if trace is None or _rank_score(trace, metric) is None:
            continue
        measured.append(_rank_score(trace, metric))
    if metric in {"r3", "r5"}:
        hits = sum(measured)
        return {
            "hits": int(hits),
            "measured": len(measured),
            "eligible": len(case_ids),
            "unmeasured": len(case_ids) - len(measured),
            "rate": _rate(int(hits), len(measured)),
        }
    return {
        "mean": sum(measured) / len(measured) if measured else None,
        "n": len(measured),
        "eligible": len(case_ids),
        "unmeasured": len(case_ids) - len(measured),
    }


def _bucket_retrieval(traces_by_id: dict[str, dict], case_ids: list[str],
                      cases_by_id: dict[str, dict]) -> dict:
    buckets: dict[str, list[str]] = defaultdict(list)
    for case_id in case_ids:
        trace = traces_by_id.get(case_id, {})
        bucket = cases_by_id.get(case_id, {}).get("bucket", trace.get("bucket", ""))
        buckets[str(bucket)].append(case_id)
    return {
        bucket: {
            "n": len(ids),
            "r3": _retrieval_metric(traces_by_id, ids, "r3"),
            "r5": _retrieval_metric(traces_by_id, ids, "r5"),
            "mrr": _retrieval_metric(traces_by_id, ids, "mrr"),
        }
        for bucket, ids in sorted(buckets.items())
    }


def _answer_population(traces: list[dict], cases_by_id: dict[str, dict]):
    by_id = {str(trace["case_id"]): trace for trace in traces if trace.get("case_id")}
    if cases_by_id:
        eligible_ids = [case_id for case_id, case in cases_by_id.items() if _answer_eligible(case)]
        condition_ids = [case_id for case_id, case in cases_by_id.items()
                         if case.get("required_conditions")]
        non_refusal_ids = [case_id for case_id, case in cases_by_id.items()
                           if case.get("expected_behavior") in NON_REFUSAL_BEHAVIORS]
    else:
        eligible_ids = [case_id for case_id, trace in by_id.items()
                        if _answer_eligible(_trace_case(trace, {}))]
        condition_ids = [case_id for case_id, trace in by_id.items()
                         if trace.get("required_conditions")]
        non_refusal_ids = [case_id for case_id, trace in by_id.items()
                           if trace.get("expected_behavior") in NON_REFUSAL_BEHAVIORS]
    return by_id, eligible_ids, condition_ids, non_refusal_ids


def _nearest_rank(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def summarize_group(traces: list[dict], cases=None) -> dict:
    """按题目元数据聚合检索、答案质量与运行成本，保留每项分子/分母。"""
    cases_by_id = _case_map(cases)
    retrieval_by_id, retrieval_ids = _retrieval_population(traces, cases_by_id)
    answer_by_id, strict_ids, condition_ids, non_refusal_ids = _answer_population(
        traces, cases_by_id
    )
    evidence_group_recall_metrics = {}
    for k in (3, 5):
        values = [
            float((retrieval_by_id.get(case_id, {}).get("l1") or {})
                  [f"evidence_group_recall_at_{k}"])
            for case_id in retrieval_ids
            if (retrieval_by_id.get(case_id, {}).get("l1") or {}).get(
                f"evidence_group_recall_at_{k}") is not None
        ]
        evidence_group_recall_metrics[f"at_{k}"] = {
            "mean": sum(values) / len(values) if values else None,
            "measured": len(values),
            "eligible": len(retrieval_ids),
            "unmeasured": len(retrieval_ids) - len(values),
        }

    strict_values = []
    for case_id in strict_ids:
        trace = answer_by_id.get(case_id)
        if trace is None:
            continue
        outcome = strict_answer_outcome(_trace_case(trace, cases_by_id), trace)
        if outcome is not None:
            strict_values.append((case_id, outcome))
    strict_correct = sum(outcome for _, outcome in strict_values)

    ch11_legacy_strict = []
    for trace in answer_by_id.values():
        checks = trace.get("deterministic_checks") or {}
        if not checks or _blocked(trace) or not checks.get("gold_facts_checked", 0):
            continue
        ch11_legacy_strict.append(bool(
            checks.get("gold_fact_accuracy", {}).get("passed")
            and checks.get("condition_coverage", {}).get("passed")
            and not checks.get("forbidden_claims")
            and checks.get("citation_support", {}).get("passed")
        ))

    condition_scores = []
    for case_id in condition_ids:
        trace = answer_by_id.get(case_id)
        checks = (trace or {}).get("deterministic_checks") or {}
        coverage = checks.get("condition_coverage")
        if isinstance(coverage, dict) and coverage.get("score") is not None:
            condition_scores.append(float(coverage["score"]))

    citation_eligible = strict_ids
    citation_support = []
    unsupported = substantive = 0
    for case_id in citation_eligible:
        trace = answer_by_id.get(case_id)
        checks = (trace or {}).get("deterministic_checks") or {}
        if not checks:
            continue
        citation = checks.get("citation_support") or {}
        citation_support.append(
            bool(not _blocked(trace) and checks.get("behavior_pred") != "refuse"
                 and citation.get("passed"))
        )

    # 与 CH11 一致：无证据断言是所有实际作答（非闸拦）的 claim 级微平均，
    # 包括澄清/拒答行为轨迹；正常拒答的 substantive_claims 应为 0。
    unsupported_case_ids = set()
    for case_id, trace in answer_by_id.items():
        checks = trace.get("deterministic_checks") or {}
        if not checks or _blocked(trace):
            continue
        count = int(checks.get("unsupported_claims", 0) or 0)
        unsupported += count
        substantive += int(checks.get("substantive_claims", 0) or 0)
        if count > 0:
            unsupported_case_ids.add(case_id)

    failure_labels: dict[str, set[str]] = defaultdict(set)
    for trace in traces:
        for label in trace.get("failure_labels") or []:
            failure_labels[str(label)].add(str(trace.get("case_id", "")))
    unmeasured_by_layer = {}
    for layer, status_key, reason_key in (
        ("retrieval", "retrieval_status", "retrieval_unmeasured_reason"),
        ("answer", "answer_status", "answer_unmeasured_reason"),
        ("rag", "rag_status", "rag_unmeasured_reason"),
    ):
        misses = [trace for trace in traces if trace.get(status_key) == "unmeasured"]
        if misses:
            reason_counts: dict[str, int] = defaultdict(int)
            for trace in misses:
                reason = str(trace.get(reason_key) or "reason_not_recorded")
                reason_counts[reason] += 1
            unmeasured_by_layer[layer] = {
                "count": len(misses),
                "case_ids": sorted(str(trace.get("case_id", "")) for trace in misses),
                "reasons": dict(sorted(reason_counts.items())),
            }

    over_refusal_ids = []
    over_refusal_measured = 0
    over_refusal_eligible = 0
    for case_id in non_refusal_ids:
        trace = answer_by_id.get(case_id)
        if trace is not None and trace.get("l4_applicable", True) is False:
            continue
        over_refusal_eligible += 1
        checks = (trace or {}).get("deterministic_checks") or {}
        if not checks:
            continue
        over_refusal_measured += 1
        if checks.get("behavior_pred") == "refuse":
            over_refusal_ids.append(case_id)

    latencies = [float(trace["latency_ms"]) for trace in traces
                 if trace.get("latency_ms") is not None]
    latency_by_source = defaultdict(list)
    for trace in traces:
        if trace.get("latency_ms") is not None:
            latency_by_source[str(trace.get("latency_source") or "unspecified")].append(
                float(trace["latency_ms"])
            )
    api_calls = sum(int(trace.get("external_api_calls", 0) or 0) for trace in traces)
    if not api_calls:
        api_calls = sum(
            int(trace.get("llm_calls", 0) or 0)
            + int(trace.get("evidence_judge_calls", 0) or 0)
            + int(trace.get("rag_analyzer_api_calls", 0) or 0)
            for trace in traces
        )
    paid_api_calls = sum(int(trace.get("paid_api_calls", 0) or 0) for trace in traces)
    return {
        "n": len(traces),
        "retrieval": {
            "positive_cases": len(retrieval_ids),
            "r3": _retrieval_metric(retrieval_by_id, retrieval_ids, "r3"),
            "r5": _retrieval_metric(retrieval_by_id, retrieval_ids, "r5"),
            "mrr": _retrieval_metric(retrieval_by_id, retrieval_ids, "mrr"),
            "all_required_evidence_group_recall": evidence_group_recall_metrics,
            "by_bucket": _bucket_retrieval(retrieval_by_id, retrieval_ids, cases_by_id),
        },
        "answers": {
            "strict_accuracy": {
                "correct": int(strict_correct),
                "measured": len(strict_values),
                "eligible": len(strict_ids),
                "unmeasured": len(strict_ids) - len(strict_values),
                "rate": _rate(int(strict_correct), len(strict_values)),
            },
            "ch11_compatible_attempted_strict_accuracy": {
                "correct": sum(ch11_legacy_strict),
                "measured": len(ch11_legacy_strict),
                "excludes_gate_blocked": True,
                "includes_any_expected_behavior_with_gold_fact_check": True,
                "rate": _rate(sum(ch11_legacy_strict), len(ch11_legacy_strict)),
            },
            "condition_coverage": {
                "mean": sum(condition_scores) / len(condition_scores) if condition_scores else None,
                "measured": len(condition_scores),
                "eligible": len(condition_ids),
                "unmeasured": len(condition_ids) - len(condition_scores),
            },
            "citation_supported_cases": {
                "supported": sum(citation_support),
                "measured": len(citation_support),
                "rate": _rate(sum(citation_support), len(citation_support)),
            },
            "unsupported_claims": {
                "unsupported": unsupported,
                "claims": substantive,
                "rate": _rate(unsupported, substantive),
                "cases_with_unsupported_claims": len(unsupported_case_ids),
                "case_ids": sorted(unsupported_case_ids),
            },
            "over_refusal": {
                "refused": len(over_refusal_ids),
                "measured": over_refusal_measured,
                "eligible": over_refusal_eligible,
                "unmeasured": over_refusal_eligible - over_refusal_measured,
                "rate": _rate(len(over_refusal_ids), over_refusal_measured),
                "case_ids": sorted(over_refusal_ids),
            },
        },
        "operations": {
            "latency_ms": {
                "method": "nearest_rank",
                "n": len(latencies),
                "p50": _nearest_rank(latencies, 0.50),
                "p95": _nearest_rank(latencies, 0.95),
                "by_source": {
                    source: {
                        "n": len(values),
                        "p50": _nearest_rank(values, 0.50),
                        "p95": _nearest_rank(values, 0.95),
                    }
                    for source, values in sorted(latency_by_source.items())
                },
            },
            "calls": {
                "external_api_calls": api_calls,
                "paid_api_calls": paid_api_calls,
                "answer_api_calls": sum(int(t.get("llm_calls", 0) or 0) for t in traces),
                "evidence_judge_api_calls": sum(
                    int(t.get("evidence_judge_calls", 0) or 0) for t in traces
                ),
                "evidence_judge_cache_hits": sum(
                    int(t.get("evidence_judge_cached", 0) or 0) for t in traces
                ),
                "gap_analyzer_api_calls": sum(
                    int(t.get("rag_analyzer_api_calls", 0) or 0) for t in traces
                ),
                "gap_analyzer_cache_hits": sum(
                    int(t.get("rag_analyzer_cached", 0) or 0) for t in traces
                ),
                "gap_analyzer_logical_calls": sum(
                    int(t.get("rag_analyzer_calls", 0) or 0) for t in traces
                ),
                "rag_judge_logical_calls": sum(
                    int(t.get("rag_loop_judge_calls", 0) or 0) for t in traces
                ),
            },
            "cost_actual_usd": round(sum(float(t.get("cost_actual", 0.0) or 0.0)
                                          for t in traces), 8),
            "cost_estimate_usd": round(sum(float(t.get("cost_estimate", 0.0) or 0.0)
                                            for t in traces), 8),
        },
        "failure_attribution": {
            "failure_labels": {
                label: {
                    "count": len(case_ids),
                    "case_ids": sorted(case_id for case_id in case_ids if case_id),
                }
                for label, case_ids in sorted(failure_labels.items())
            },
            "unmeasured": unmeasured_by_layer,
        },
    }


def _trace_map(traces: Iterable[dict]) -> dict[str, dict]:
    mapped = {}
    for trace in traces:
        case_id = str(trace.get("case_id", ""))
        if not case_id:
            continue
        if case_id in mapped:
            raise ValueError(f"重复 case_id 轨迹: {case_id}")
        mapped[case_id] = trace
    return mapped


def _paired_change(reference: dict[str, dict], candidate: dict[str, dict],
                   case_ids: list[str], metric: str) -> dict:
    pairs = []
    for case_id in case_ids:
        before, after = reference.get(case_id), candidate.get(case_id)
        if before is None or after is None:
            continue
        before_score = _rank_score(before, metric)
        after_score = _rank_score(after, metric)
        if before_score is None or after_score is None:
            continue
        pairs.append((case_id, before_score, after_score))
    gained = [case_id for case_id, before, after in pairs if after > before]
    lost = [case_id for case_id, before, after in pairs if after < before]
    ties = [case_id for case_id, before, after in pairs if after == before]
    deltas = [after - before for _, before, after in pairs]
    return {
        "pairs": len(pairs),
        "gained_ids": gained,
        "lost_ids": lost,
        "ties": len(ties),
        "tie_ids": ties,
        "mean_delta": sum(deltas) / len(deltas) if deltas else None,
    }


def compare_groups(reference_traces: list[dict], candidate_traces: list[dict], cases=None) -> dict:
    """按 case_id 对齐两组，分别比较检索覆盖与严格回答结果。"""
    reference, candidate = _trace_map(reference_traces), _trace_map(candidate_traces)
    cases_by_id = _case_map(cases)
    common = sorted(set(reference) & set(candidate))
    if cases_by_id:
        positives = [case_id for case_id, case in cases_by_id.items()
                     if case.get("required_evidence") and case_id in common]
    else:
        positives = [case_id for case_id in common
                     if (reference[case_id].get("l1") or {}).get("n", 0) == 1
                     and (candidate[case_id].get("l1") or {}).get("n", 0) == 1]
    retrieval = {metric: _paired_change(reference, candidate, positives, metric)
                 for metric in RETRIEVAL_METRICS}

    strict_pairs = []
    for case_id in common:
        case = cases_by_id.get(case_id) or _trace_case(reference[case_id], {})
        before = strict_answer_outcome(case, reference[case_id])
        after = strict_answer_outcome(case, candidate[case_id])
        if before is not None and after is not None:
            strict_pairs.append((case_id, before, after))
    gained = [case_id for case_id, before, after in strict_pairs if after and not before]
    lost = [case_id for case_id, before, after in strict_pairs if before and not after]
    ties = [case_id for case_id, before, after in strict_pairs if before == after]
    before_correct = sum(before for _, before, _ in strict_pairs)
    after_correct = sum(after for _, _, after in strict_pairs)
    return {
        "common_case_ids": len(common),
        "retrieval": retrieval,
        "answers": {
            "strict": {
                "pairs": len(strict_pairs),
                "gained_ids": gained,
                "lost_ids": lost,
                "ties": len(ties),
                "tie_ids": ties,
                "reference_correct": before_correct,
                "candidate_correct": after_correct,
                "candidate_minus_reference": after_correct - before_correct,
            }
        },
    }


def paired_bucket_outcomes(groups: dict[str, list[dict]], buckets: tuple[str, ...]) -> dict:
    """以 groups 插入顺序中的首组为参照，输出指定桶内每组逐题得失。"""
    if not groups:
        return {}
    names = list(groups)
    reference_name = names[0]

    def select(rows: list[dict], bucket: str) -> list[dict]:
        return [trace for trace in rows if trace.get("bucket") == bucket]

    result = {}
    for candidate_name in names[1:]:
        result[candidate_name] = {}
        for bucket in buckets:
            result[candidate_name][bucket] = compare_groups(
                select(groups[reference_name], bucket),
                select(groups[candidate_name], bucket),
            )
    return result


def protocol_sha256(source_contract: dict) -> str:
    """哈希共同实验设置；忽略题集、运行时间、检索/循环组别本身。"""
    retrieval = deepcopy(source_contract.get("retrieval") or {})
    strategy = retrieval.pop("strategy", None)
    if retrieval.get("candidates") != 50 or retrieval.get("top_k") != 10:
        raise ContractError(
            "候选上限不匹配:CH16 固定 candidates=50、top_k=10 "
            f"(got {retrieval.get('candidates')}/{retrieval.get('top_k')})"
        )
    index_source = source_contract.get("index") or {}
    index = {key: index_source.get(key) for key in (
        "backend", "build_id", "snapshot_sha256", "embedding_model", "dim", "collection"
    ) if key in index_source}
    rewrite = deepcopy(source_contract.get("rewrite") or {})
    # 每个 split 有独立的冻结改写文件；缓存 SHA 保留在逐组契约供审计，
    # 但不能让 dev -> frozen 的阶段锁因不同题集缓存文件而失效。
    rewrite.pop("cache_sha256", None)
    harness_source = source_contract.get("harness") or {}
    run_source = source_contract.get("run") or {}
    repo_root = Path(__file__).resolve().parent.parent
    code_hashes = {
        name: hashlib.sha256((repo_root / name).read_bytes()).hexdigest()
        for name in PROTOCOL_CODE_PATHS
    }
    payload = {
        "corpus": source_contract.get("corpus"),
        "index": index,
        "retrieval": retrieval,
        "rewrite": rewrite,
        "llm": source_contract.get("llm"),
        "judge": source_contract.get("judge"),
        "pricing": source_contract.get("pricing"),
        "harness": {key: harness_source.get(key) for key in
                    ("mode", "allow_paid", "use_judge") if key in harness_source},
        "fixed": {
            "analysis_code_sha256": code_hashes,
            "candidate_cap": 50,
            "answer_context_top_k": 10,
            "retrieval_metrics": ["R@3", "R@5", "MRR@10"],
            "strict_accuracy": "gold_answer_behaviors;gate_blocked_is_failure",
            "unsupported_claims": "all_non_gate_blocked_attempts_micro_average",
            "latency": "full_case_wall_clock_includes_live_calls",
            "requested_allow_paid": bool(run_source.get("ch16_allow_paid", False)),
            "budget_mode": "unbounded" if run_source.get("ch16_unbounded_budget") else "bounded",
            "retry_policy": run_source.get("ch16_retry_policy", "default"),
            "answer_prompt_is_shared": True,
        },
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_experiment_contract(*, split: str, experiment: str, strategy: str,
                              rag_mode: str, source_contract: dict) -> dict:
    """将 CH11 contract 固定为一组 CH16 可比较配置。"""
    if experiment not in EXPERIMENT_GROUPS:
        raise ContractError(f"未知实验类型: {experiment}")
    if strategy not in {"dense", "bm25", "hybrid", "hybrid_rerank"}:
        raise ContractError(f"未知检索策略: {strategy}")
    if rag_mode not in EXPERIMENT_GROUPS["rag"]:
        raise ContractError(f"未知 CH12 RAG 模式: {rag_mode}")
    source = deepcopy(source_contract)
    source.pop("_sha", None)
    source_split = (source.get("dataset") or {}).get("split")
    if source_split != split:
        raise ContractError(f"数据集 split 不一致: {source_split} != {split}")
    if experiment == "retrieval" and (source.get("retrieval") or {}).get("strategy") != strategy:
        raise ContractError("source contract 的 retrieval.strategy 与组名不一致")
    if experiment == "rag" and (source.get("rag") or {}).get("mode") != rag_mode:
        raise ContractError("source contract 的 rag.mode 与组名不一致")
    protocol_hash = protocol_sha256(source)
    group = {"strategy": strategy} if experiment == "retrieval" else {"rag_mode": rag_mode}
    return {
        "schema_version": 1,
        "experiment": experiment,
        "split": split,
        "dataset": deepcopy(source.get("dataset") or {}),
        "protocol_sha256": protocol_hash,
        "analysis_code_sha256": {
            name: hashlib.sha256(
                (Path(__file__).resolve().parent.parent / name).read_bytes()
            ).hexdigest()
            for name in PROTOCOL_CODE_PATHS
        },
        "group": group,
        "source_contract": source,
    }


def _contract_differences(left: dict, right: dict) -> list[str]:
    from scripts.eval_contract import contract_diff

    left_copy, right_copy = deepcopy(left), deepcopy(right)
    left_copy.pop("_sha", None)
    right_copy.pop("_sha", None)
    return contract_diff(left_copy, right_copy)


def validate_group_contracts(contracts: list[dict]) -> None:
    """校验组间只变更对应实验变量，且冻结使用相同协议、题集和候选上限。"""
    if not contracts:
        raise ContractError("实验组为空")
    experiment = contracts[0].get("experiment")
    if experiment not in EXPERIMENT_GROUPS:
        raise ContractError(f"未知实验类型: {experiment}")
    group_keys = [tuple(sorted((contract.get("group") or {}).items())) for contract in contracts]
    if len(set(group_keys)) != len(group_keys):
        raise ContractError("实验组重复")
    reference = contracts[0]
    for candidate in contracts:
        if candidate.get("experiment") != experiment:
            raise ContractError("混合了不同实验类型")
        if candidate.get("split") != reference.get("split"):
            raise ContractError("split 不一致")
        if candidate.get("dataset") != reference.get("dataset"):
            raise ContractError("题集 hash/题数不一致")
        if candidate.get("protocol_sha256") != reference.get("protocol_sha256"):
            raise ContractError("共同实验协议 hash 不一致")
        source = candidate.get("source_contract") or {}
        if source:
            protocol_sha256(source)
    reference_source = reference.get("source_contract") or {}
    for candidate in contracts[1:]:
        candidate_source = candidate.get("source_contract") or {}
        differences = _contract_differences(reference_source, candidate_source)
        if experiment == "retrieval":
            allowed = {"retrieval.strategy"}
        else:
            allowed = {"harness.rag_mode", "harness.version"}
            allowed.update(key for key in differences if key.startswith("rag."))
        unexpected = [key for key in differences if key not in allowed]
        if unexpected:
            raise ContractError("组间契约漂移: " + ", ".join(unexpected))


def check_phase_lock(phase: str, lock_path: str | Path, expected_protocol_sha256: str) -> None:
    """formal frozen-v1 只在 dev 协议锁定后运行。"""
    if phase == "dev":
        return
    if phase != "frozen-v1":
        raise ContractError(f"未知阶段: {phase}")
    path = Path(lock_path)
    try:
        lock = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"frozen-v1 运行前必须有 dev lock: {path}") from exc
    if lock.get("protocol_sha256") != expected_protocol_sha256:
        raise ContractError("dev lock 与本次协议不一致，拒绝冻结集对照")


def check_cross_experiment_locks(
    phase: str,
    lock_root: str | Path,
    expected_protocol_sha256: str,
) -> None:
    """Keep retrieval and RAG on one dev protocol; frozen requires both dev locks."""
    if phase not in {"dev", "frozen-v1"}:
        raise ContractError(f"未知阶段: {phase}")
    # Dev can deliberately refresh locks after a protocol/code change. Frozen is
    # blocked until both experiments have been rerun and agree on the new hash.
    if phase == "dev":
        return
    root = Path(lock_root)
    for experiment in ("retrieval", "rag"):
        path = root / f"{experiment}.json"
        try:
            lock = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ContractError(
                "冻结集运行前必须先锁定 retrieval 与 rag 两组 dev 协议"
            ) from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise ContractError(f"跨实验 dev lock 无效:{path}") from exc
        if lock.get("protocol_sha256") != expected_protocol_sha256:
            raise ContractError(f"跨实验 dev lock 与本次协议不一致:{path}")


def validate_budget_request(*, allow_paid: bool, max_cost_usd: float | None,
                            ledger, unbounded: bool = False) -> None:
    """Require explicit paid mode and a durable ledger for live requests."""
    if unbounded:
        if not allow_paid:
            raise ContractError("无限预算模式需要显式 --allow-paid")
        if ledger is None:
            raise ContractError("付费运行必须显式指定共享 --ledger")
        if max_cost_usd is not None:
            raise ContractError("无限预算模式不能同时设置 max_cost_usd")
        return
    if max_cost_usd is None or not math.isfinite(max_cost_usd):
        raise ContractError("max_cost_usd 必须是有限数字；无限模式需显式 --unbounded-budget")
    if max_cost_usd < 0:
        raise ContractError("max_cost_usd 不能为负")
    if max_cost_usd > 0 and not allow_paid:
        raise ContractError("正费用预算需要显式 --allow-paid")
    if max_cost_usd > 0 and ledger is None:
        raise ContractError("付费运行必须显式指定共享 --ledger")
