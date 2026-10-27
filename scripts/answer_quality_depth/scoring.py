"""Versioned deterministic ruler and paired aggregate for answer depth."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

from scripts.eval_checks import check_case_deterministic

ROOT = Path(__file__).resolve().parents[2]
OVERLAY_PATH = (
    ROOT / "tests" / "eval" / "answer_quality_depth" / "scoring-overlay-v1.json"
)
SCORING_VERSION = "answer-quality-depth-score-v7"
ANSWER_BEHAVIORS = {"answer", "answer_with_conditions", "verify_then_answer"}


def load_scoring_overlay(path: Path = OVERLAY_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _apply_metric_overrides(case: dict, overlay: dict) -> dict:
    result = deepcopy(case)
    override = (overlay.get("fact_overrides") or {}).get(str(case.get("id")))
    if override:
        if "gold_facts" in override:
            result["gold_facts"] = deepcopy(override["gold_facts"])
        if "required_conditions" in override:
            result["required_conditions"] = list(override["required_conditions"])
        if "condition_aliases" in override:
            result["condition_aliases"] = dict(override["condition_aliases"])
    return result


def _denominator(case: dict) -> str:
    expected = str(case.get("expected_behavior") or "")
    if expected in ANSWER_BEHAVIORS:
        return (
            "evidence_answer"
            if case.get("required_evidence")
            else "answer_without_evidence"
        )
    if expected == "clarify":
        return "clarify"
    return "refusal"


def _natural_clarification(checks: dict, answer: str) -> bool:
    asks = (
        "?" in answer
        or "？" in answer
        or any(
            phrase in answer
            for phrase in (
                "请提供",
                "请先提供",
                "请告知",
                "请补充",
                "麻烦提供",
                "请把",
                "请回复",
                "请核对",
                "请麻烦",
                "请您核对",
            )
        )
    )
    return asks and bool((checks.get("clarify_slot_recall") or {}).get("passed"))


def _natural_refusal(answer: str) -> bool:
    markers = (
        "无法确认",
        "无法确定",
        "无法判断",
        "无法给出",
        "无法直接给出",
        "无法直接判断",
        "无法可靠回答",
        "无法可靠报价",
        "无法据此",
        "无法告知",
        "无法核实",
        "不能确定",
        "不能确认",
        "不能直接判断",
        "不能猜测",
        "不能据此猜测",
        "查不到",
        "不在本客服",
        "没有包含",
        "没有说明",
        "没有明确说明",
        "没有列出",
        "未明确说明",
        "未包含",
        "未说明",
        "未提及",
        "未提供",
        "资料不足",
        "证据不足",
    )
    return any(marker in answer for marker in markers)


def _clean_answer_for_scoring(answer: str) -> str:
    """Markdown emphasis and code markers are presentation, not business facts."""
    text = str(answer or "")
    for marker in ("**", "__", "`"):
        text = text.replace(marker, "")
    return text


def _substantive_answer(answer: str) -> bool:
    text = _clean_answer_for_scoring(answer)
    if len(text) < 50 or not re.search(r"\[\d+\]", text):
        return False
    return any(
        term in text
        for term in ("不能", "可以", "支持", "适用", "价格", "金额", "为", "是")
    )


def score_response(
    case: dict,
    answer: str,
    evidence: list[dict],
    *,
    overlay: dict | None = None,
    technical_failure: str | None = None,
) -> dict:
    if technical_failure:
        return {
            "scoring_version": SCORING_VERSION,
            "case_id": case["id"],
            "bucket": case.get("bucket"),
            "denominator": _denominator(case),
            "status": "technical_failure",
            "technical_failure": technical_failure,
            "technical_failures": [technical_failure],
            "behavior_passed": False,
            "facts_passed": False,
            "conditions_passed": False,
            "citation_status": "not_run",
            "facts_checked": 0,
            "failure_labels": ["technical_failure"],
        }
    scoring_answer = _clean_answer_for_scoring(str(answer or ""))
    scoring_overlay = overlay or load_scoring_overlay()
    scoring_case = _apply_metric_overrides(case, scoring_overlay)
    evidence_by_n = {
        index: str(item.get("text") or item.get("answer") or "")
        for index, item in enumerate(evidence, start=1)
    }
    checks = check_case_deterministic(
        scoring_case,
        scoring_answer,
        evidence_by_n,
        scoring_overlay=scoring_overlay,
    )
    expected = str(scoring_case.get("expected_behavior") or "answer")
    behavior = str(checks.get("behavior_pred") or "answer")
    facts_passed = bool((checks.get("gold_fact_accuracy") or {}).get("passed"))
    conditions_passed = bool((checks.get("condition_coverage") or {}).get("passed"))
    forbidden = list(checks.get("forbidden_claims") or [])
    verification_ok = bool(checks.get("verification_compliant"))
    citation = checks.get("citation_support") or {}
    citation_status = str(citation.get("status") or "unknown")

    if expected in ANSWER_BEHAVIORS:
        behavior_passed = behavior == "answer" or _substantive_answer(scoring_answer)
        hard_passed = (
            behavior_passed
            and facts_passed
            and conditions_passed
            and not forbidden
            and verification_ok
        )
        if not hard_passed or citation_status == "unsupported":
            status = "incorrect"
        elif citation_status == "uncertain":
            status = "uncertain"
        elif citation_status == "supported":
            status = "correct"
        else:
            status = "incorrect"
    elif expected == "clarify":
        behavior_passed = behavior == "clarify" or _natural_clarification(
            checks, str(answer or "")
        )
        recall = bool((checks.get("clarify_slot_recall") or {}).get("passed"))
        behavior_passed = behavior_passed and recall
        status = "correct" if behavior_passed and not forbidden else "incorrect"
        citation_status = "not_applicable"
    elif expected == "refuse":
        behavior_passed = behavior == "refuse" or _natural_refusal(str(answer or ""))
        status = "correct" if behavior_passed and not forbidden else "incorrect"
        citation_status = "not_applicable"
    else:
        behavior_passed = False
        status = "incorrect"

    failure_labels = list(checks.get("failure_labels") or [])
    if expected in ANSWER_BEHAVIORS and citation_status in {"uncertain", "unsupported"}:
        failure_labels = [
            label for label in failure_labels if not label.startswith("citation_")
        ]
        failure_labels.append(f"citation_{citation_status}")
    result = {
        "scoring_version": SCORING_VERSION,
        "case_id": case["id"],
        "bucket": case.get("bucket"),
        "denominator": _denominator(case),
        "status": status,
        "technical_failure": False,
        "technical_failures": [],
        "behavior_expected": expected,
        "behavior_observed": behavior,
        "behavior_passed": behavior_passed,
        "facts_checked": int(checks.get("gold_facts_checked") or 0),
        "facts_passed": facts_passed,
        "fact_score": float(
            (checks.get("gold_fact_accuracy") or {}).get("score") or 0.0
        ),
        "conditions_passed": conditions_passed,
        "condition_score": float(
            (checks.get("condition_coverage") or {}).get("score") or 0.0
        ),
        "citation_status": citation_status,
        "forbidden_claims": forbidden,
        "verification_compliant": verification_ok,
        "failure_labels": list(dict.fromkeys(failure_labels)),
        "deterministic_checks": checks,
    }
    semantic_override = (scoring_overlay.get("semantic_overrides") or {}).get(
        str(case["id"])
    )
    if semantic_override:
        result["semantic_override"] = semantic_override
    return result


def apply_semantic_status(score: dict, judge: dict | None) -> dict:
    """Resolve citation uncertainty with a blind judge, never a deterministic hard error."""
    result = dict(score)
    technical = bool(score.get("technical_failure"))
    expected = str(score.get("behavior_expected") or "")
    hard_passed = (
        bool(score.get("behavior_passed"))
        and bool(score.get("facts_passed"))
        and bool(score.get("conditions_passed"))
        and not bool(score.get("forbidden_claims"))
        and bool(score.get("verification_compliant"))
    )
    if technical:
        result.update(
            final_status="technical_failure",
            semantic_correct=None,
            grounded_correct=None,
        )
        return result
    semantic_override = score.get("semantic_override")
    if semantic_override:
        final = str(semantic_override.get("final_status") or "uncertain")
        if final not in {"correct", "incorrect", "uncertain"}:
            raise ValueError(f"invalid semantic override: {final}")
        result.update(
            final_status=final,
            semantic_correct=(
                True if final == "correct" else False if final == "incorrect" else None
            ),
            grounded_correct=(
                True if final == "correct" else False if final == "incorrect" else None
            ),
        )
        return result
    if expected not in ANSWER_BEHAVIORS:
        final = "correct" if score.get("status") == "correct" else "incorrect"
        result.update(
            final_status=final,
            semantic_correct=None,
            grounded_correct=None,
        )
        return result
    if not hard_passed:
        result.update(
            final_status="incorrect",
            semantic_correct=False,
            grounded_correct=False,
        )
        return result
    if judge is None:
        if score.get("citation_status") == "supported":
            result.update(
                final_status="correct",
                semantic_correct=True,
                grounded_correct=True,
            )
        else:
            result.update(
                final_status="uncertain",
                semantic_correct=None,
                grounded_correct=False,
            )
        return result
    judge_correct = (
        judge.get("answer_status") == "correct"
        and judge.get("grounding") == "supported"
    )
    unresolved_requirements = bool(judge.get("missing_requirements"))
    if judge_correct and unresolved_requirements:
        result.update(
            final_status="uncertain",
            semantic_correct=None,
            grounded_correct=False,
        )
        return result
    judge_incorrect = judge.get("answer_status") in {"incorrect", "partial"}
    final = (
        "correct"
        if judge_correct
        else ("incorrect" if judge_incorrect else "uncertain")
    )
    result.update(
        final_status=final,
        semantic_correct=judge_correct,
        grounded_correct=(
            judge_correct and judge.get("citation_support") == "supported"
        ),
    )
    return result


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)], 3)


def _empty_counts() -> dict:
    return {
        "n": 0,
        "correct": 0,
        "incorrect": 0,
        "uncertain": 0,
        "technical_failure": 0,
        "semantic_correct": 0,
        "semantic_incorrect": 0,
        "grounded_correct": 0,
        "grounded_incorrect": 0,
        "determinate": 0,
        "strict_accuracy": None,
        "accuracy_among_determinate": None,
        "uncertainty_rate": None,
        "technical_failure_rate": None,
        "determinate_accuracy": None,
        "cost_usd": 0.0,
        "latency_ms": [],
    }


def _finalize_counts(counts: dict) -> dict:
    result = dict(counts)
    result["determinate"] = result["correct"] + result["incorrect"]
    result["strict_accuracy"] = result["correct"] / result["n"] if result["n"] else None
    result["accuracy_among_determinate"] = (
        result["correct"] / result["determinate"] if result["determinate"] else None
    )
    result["uncertainty_rate"] = (
        result["uncertain"] / result["n"] if result["n"] else None
    )
    result["technical_failure_rate"] = (
        result["technical_failure"] / result["n"] if result["n"] else None
    )
    result["cost_usd"] = round(float(result.get("cost_usd") or 0.0), 8)
    result["latency_ms"] = {
        "p50": (
            round(sorted(result["latency_ms"])[len(result["latency_ms"]) // 2], 3)
            if result["latency_ms"]
            else None
        ),
        "p95": _p95(result["latency_ms"]),
    }
    return result


def summarize_scores(rows: list[dict]) -> dict:
    splits = sorted({str(row["split"]) for row in rows})
    summary: dict = {
        "scoring_version": SCORING_VERSION,
        "splits": {},
        "total_cost_usd": round(
            sum(float(row.get("cost_usd") or 0.0) for row in rows), 8
        ),
    }
    for split in splits:
        split_rows = [row for row in rows if str(row["split"]) == split]
        by_denominator: dict[str, dict[str, dict]] = {}
        for denominator in sorted({str(row["denominator"]) for row in split_rows}):
            by_arm: dict[str, dict] = {}
            for arm in sorted({str(row["arm"]) for row in split_rows}):
                selected = [
                    row
                    for row in split_rows
                    if row["denominator"] == denominator and row["arm"] == arm
                ]
                counts = _empty_counts()
                for row in selected:
                    status = str(row.get("final_status") or row["status"])
                    counts["n"] += 1
                    if status in {
                        "correct",
                        "incorrect",
                        "uncertain",
                        "technical_failure",
                    }:
                        counts[status] += 1
                    if row.get("semantic_correct") is True:
                        counts["semantic_correct"] += 1
                    elif row.get("semantic_correct") is False:
                        counts["semantic_incorrect"] += 1
                    if row.get("grounded_correct") is True:
                        counts["grounded_correct"] += 1
                    elif row.get("grounded_correct") is False:
                        counts["grounded_incorrect"] += 1
                    counts["cost_usd"] += float(row.get("cost_usd") or 0.0)
                    if row.get("latency_ms") is not None:
                        counts["latency_ms"].append(float(row["latency_ms"]))
                by_arm[arm] = _finalize_counts(counts)
            by_denominator[denominator] = by_arm

        by_case: dict[str, dict[str, str]] = defaultdict(dict)
        for row in split_rows:
            by_case[str(row["case_id"])][str(row["arm"])] = str(
                row.get("final_status") or row["status"]
            )
        arms = sorted({str(row["arm"]) for row in split_rows})
        pairwise = {}
        if len(arms) == 2:
            left, right = arms
            wins = losses = ties = 0
            for values in by_case.values():
                if left not in values or right not in values:
                    continue
                if values[right] == "correct" and values[left] != "correct":
                    wins += 1
                elif values[left] == "correct" and values[right] != "correct":
                    losses += 1
                else:
                    ties += 1
            pairwise[f"{right}_vs_{left}"] = {
                "wins": wins,
                "losses": losses,
                "ties": ties,
            }

        by_bucket: dict[str, dict[str, dict]] = {}
        for bucket in sorted({str(row.get("bucket") or "") for row in split_rows}):
            bucket_arms = {}
            for arm in arms:
                selected = [
                    row
                    for row in split_rows
                    if row.get("bucket") == bucket and row["arm"] == arm
                ]
                counter = Counter(
                    str(row.get("final_status") or row["status"]) for row in selected
                )
                bucket_arms[arm] = {
                    "n": len(selected),
                    **{
                        key: counter[key]
                        for key in (
                            "correct",
                            "incorrect",
                            "uncertain",
                            "technical_failure",
                        )
                    },
                }
            by_bucket[bucket] = bucket_arms
        summary["splits"][split] = {
            "by_denominator": by_denominator,
            "pairwise": pairwise,
            "by_bucket": by_bucket,
        }
    return summary
