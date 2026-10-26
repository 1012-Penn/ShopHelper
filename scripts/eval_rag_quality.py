"""Unified answer-quality rescoring for the CH11/CH16 comparison groups.

The historical traces remain read-only. This runner writes a separate
versioned directory and keeps citations as supported/unsupported/uncertain.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

from app.answer_verifier import AnswerVerdict, AnswerVerifier
from app.prompts import ANSWER_REVISION_PROMPT, ANSWER_VERIFIER_PROMPT, build_evidence_block
from app.history import estimate_tokens
from scripts.eval_checks import check_case_deterministic
from scripts.eval_dataset import load_split
from scripts.eval_cache import ResponseCache
from scripts.eval_cost import (
    AttemptLedger,
    BudgetExceeded,
    BudgetLedger,
    CostTracker,
    PricingTable,
    attempt_context,
)
from scripts.eval_prompts import parse_json_object, prompt_sha256

ROOT = Path(__file__).resolve().parent.parent
SOURCE_RUNS = ROOT / "tests" / "eval" / "rag_ablation" / "runs"
OUTPUT_ROOT = ROOT / "tests" / "eval" / "rag_quality"
OVERLAY = ROOT / "tests" / "eval" / "overlays" / "rag-quality-v3.json"
SNAPSHOT = (
    ROOT / "data" / "eval-snapshots"
    / "ebfd49478b37ef387c8b124319a450ed73a86ae2a4b9f4fc28d6a52b2ccaa0ba.db"
)
ANSWER_BEHAVIORS = {"answer", "answer_with_conditions", "verify_then_answer"}
DEV_SUBSET = ROOT / "tests" / "eval" / "rag_quality" / "dev-case-set-v1.json"
PRICING = ROOT / "tests" / "eval" / "rag_ablation" / "pricing-snapshot-2026-10-26.yaml"
GROUPS = {
    "retrieval": ("vector", "bm25", "rrf", "rrf_rerank"),
    "rag": ("baseline", "decompose_once", "controlled_loop"),
}


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def load_evidence_texts(snapshot: Path = SNAPSHOT) -> dict[int, str]:
    import pyarrow.parquet as pq

    files = sorted(Path(snapshot).glob(
        "collections/*/partitions/*/data/*.parquet"
    ))
    if not files:
        raise RuntimeError(f"快照中未找到数据 parquet:{snapshot}")
    rows: dict[int, str] = {}
    for path in files:
        for row in pq.read_table(path, columns=["id", "text"]).to_pylist():
            rows[int(row["id"])] = str(row["text"])
    return rows


def _failure_labels(checks: dict) -> list[str]:
    return list(dict.fromkeys(str(item) for item in checks.get("failure_labels") or []))


def classify_answer_outcome(case: dict, trace: dict) -> str:
    if trace.get("answer_status") == "unmeasured":
        return "unmeasured"
    if case.get("expected_behavior") not in ANSWER_BEHAVIORS:
        return "not_applicable"
    checks = trace.get("deterministic_checks") or {}
    if int(checks.get("gold_facts_checked", 0) or 0) <= 0:
        return "not_applicable"
    if trace.get("gate_blocked") or checks.get("behavior_pred") == "refuse":
        return "incorrect"
    hard_failures = {
        label for label in _failure_labels(checks)
        if label != "citation_uncertain"
    }
    if hard_failures:
        return "incorrect"
    citation = checks.get("citation_support") or {}
    if citation.get("status") == "uncertain":
        return "uncertain"
    if not citation.get("passed"):
        return "incorrect"
    return "correct"


def classify_factual_answer(case: dict, trace: dict) -> str:
    """Answer-layer correctness independent from citation-support uncertainty."""
    if trace.get("answer_status") == "unmeasured":
        return "unmeasured"
    if case.get("expected_behavior") not in ANSWER_BEHAVIORS:
        return "not_applicable"
    checks = trace.get("deterministic_checks") or {}
    if int(checks.get("gold_facts_checked", 0) or 0) <= 0:
        return "not_applicable"
    if trace.get("gate_blocked") or checks.get("behavior_pred") == "refuse":
        return "incorrect"
    hard_failures = {
        label for label in _failure_labels(checks)
        if label not in {"citation_supported", "citation_unsupported", "citation_uncertain"}
    }
    return "incorrect" if hard_failures else "correct"


def _nearest_rank(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def summarize_quality(cases: list[dict], traces: list[dict]) -> dict:
    cases_by_id = {str(case["id"]): case for case in cases}
    outcomes = Counter()
    factual_outcomes = Counter()
    factual_uncertain_ids: list[str] = []
    uncertain_ids: list[str] = []
    citation_counts = Counter()
    citation_ids: dict[str, list[str]] = defaultdict(list)
    condition_scores: list[float] = []
    gold_scores: list[float] = []
    condition_ids: list[str] = []
    retrieval = Counter()
    retrieval_measured = 0
    stop_reasons = Counter()
    latencies: list[float] = []

    for trace in traces:
        case_id = str(trace["case_id"])
        case = cases_by_id[case_id]
        checks = trace.get("deterministic_checks") or {}
        outcome = classify_answer_outcome(case, trace)
        outcomes[outcome] += 1
        if outcome == "uncertain":
            uncertain_ids.append(case_id)
        factual_outcome = classify_factual_answer(case, trace)
        factual_outcomes[factual_outcome] += 1
        if factual_outcome == "uncertain":
            factual_uncertain_ids.append(case_id)
        if checks.get("gold_fact_accuracy"):
            gold_scores.append(float(checks["gold_fact_accuracy"].get("score") or 0.0))
        status = str((checks.get("citation_support") or {}).get("status") or "")
        if status in {"supported", "unsupported", "uncertain"}:
            citation_counts[status] += 1
            citation_ids[status].append(case_id)
        if case.get("required_conditions") and checks.get("condition_coverage"):
            condition_ids.append(case_id)
            condition_scores.append(float(checks["condition_coverage"].get("score") or 0.0))
        l1 = trace.get("l1") or {}
        if l1.get("n") == 1:
            retrieval_measured += 1
            for key in ("r3", "r5"):
                if l1.get(key) is not None:
                    retrieval[key] += 1
        if trace.get("rag_stop_reason"):
            stop_reasons[str(trace["rag_stop_reason"])] += 1
        if trace.get("latency_ms") is not None:
            latencies.append(float(trace["latency_ms"]))

    determinate = outcomes["correct"] + outcomes["incorrect"]
    factual_determinate = factual_outcomes["correct"] + factual_outcomes["incorrect"]
    over_refusal_ids = [
        str(trace["case_id"]) for trace in traces
        if (cases_by_id[str(trace["case_id"])].get("expected_behavior") in ANSWER_BEHAVIORS)
        and ((trace.get("deterministic_checks") or {}).get("behavior_pred") == "refuse")
    ]
    verification_rows = [
        trace.get("answer_verification") for trace in traces
        if trace.get("answer_verification")
    ]
    return {
        "n": len(traces),
        "retrieval": {
            "measured": retrieval_measured,
            "r3": retrieval["r3"],
            "r5": retrieval["r5"],
            "r3_rate": retrieval["r3"] / retrieval_measured if retrieval_measured else None,
            "r5_rate": retrieval["r5"] / retrieval_measured if retrieval_measured else None,
        },
        "answers": {
            "correct": outcomes["correct"],
            "incorrect": outcomes["incorrect"],
            "uncertain": outcomes["uncertain"],
            "unmeasured": outcomes["unmeasured"],
            "not_applicable": outcomes["not_applicable"],
            "determinate": determinate,
            "accuracy_among_determinate": (
                outcomes["correct"] / determinate if determinate else None
            ),
            "uncertain_case_ids": sorted(uncertain_ids),
        },
        "factual_answers": {
            "correct": factual_outcomes["correct"],
            "incorrect": factual_outcomes["incorrect"],
            "uncertain": factual_outcomes["uncertain"],
            "unmeasured": factual_outcomes["unmeasured"],
            "not_applicable": factual_outcomes["not_applicable"],
            "determinate": factual_determinate,
            "accuracy_among_determinate": (
                factual_outcomes["correct"] / factual_determinate
                if factual_determinate else None
            ),
            "uncertain_case_ids": sorted(factual_uncertain_ids),
            "mean_gold_fact_coverage": (
                sum(gold_scores) / len(gold_scores) if gold_scores else None
            ),
            "gold_fact_measured": len(gold_scores),
        },
        "conditions": {
            "mean": sum(condition_scores) / len(condition_scores)
            if condition_scores else None,
            "measured": len(condition_scores),
            "measured_case_ids": sorted(condition_ids),
        },
        "citations": {
            "supported": citation_counts["supported"],
            "unsupported": citation_counts["unsupported"],
            "uncertain": citation_counts["uncertain"],
            "measurable": sum(citation_counts.values()),
            "supported_case_ids": sorted(citation_ids["supported"]),
            "unsupported_case_ids": sorted(citation_ids["unsupported"]),
            "uncertain_case_ids": sorted(citation_ids["uncertain"]),
        },
        "over_refusal": {
            "count": len(over_refusal_ids),
            "case_ids": sorted(over_refusal_ids),
        },
        "answer_verification": {
            "measured": len(verification_rows),
            "pass": sum(row["status"] == "pass" for row in verification_rows),
            "refuse": sum(row["status"] == "refuse" for row in verification_rows),
            "uncertain": sum(row["status"] == "uncertain" for row in verification_rows),
            "revision_calls": sum(int(row.get("revision_calls") or 0)
                                  for row in verification_rows),
            "changed": sum(bool(row.get("changed")) for row in verification_rows),
            "entity_issue_case_ids": sorted(
                str(trace["case_id"]) for trace in traces
                if trace.get("answer_verification")
                and any("订单" in issue or "实体" in issue
                        for issue in trace["answer_verification"].get("issues") or [])
            ),
        },
        "stop_reasons": dict(sorted(stop_reasons.items())),
        "latency_ms": {
            "n": len(latencies),
            "p50": _nearest_rank(latencies, 0.50),
            "p95": _nearest_rank(latencies, 0.95),
        },
    }


def rescore_trace(
    case: dict,
    trace: dict,
    evidence_texts: dict[int, str],
    overlay: dict,
) -> dict:
    evidence_by_n = {
        index: evidence_texts.get(int(item["chunk_id"]), "")
        for index, item in enumerate(trace.get("retrieved") or [], start=1)
        if item.get("chunk_id") is not None
    }
    checks = check_case_deterministic(
        case, str(trace.get("answer") or ""), evidence_by_n,
        scoring_overlay=overlay,
    )
    rescored = dict(trace)
    rescored.update({
        "legacy_failure_labels": list(trace.get("failure_labels") or []),
        "scoring_version": str(overlay.get("version") or "rag-quality-v3"),
        "scoring_overlay": checks.get("scoring_overlay"),
        "citations": checks["citation_support"],
        "deterministic_checks": checks,
        "behavior_pred": checks["behavior_pred"],
        "failure_labels": list(checks["failure_labels"]),
        "answer_preserved": True,
    })
    return rescored


def apply_answer_verifier(
    case: dict,
    trace: dict,
    evidence_texts: dict[int, str],
    verifier,
    overlay: dict,
    *,
    revise=None,
) -> dict:
    evidence = []
    for index, item in enumerate(trace.get("retrieved") or [], start=1):
        chunk_id = item.get("chunk_id")
        if chunk_id is None:
            continue
        evidence.append({
            "n": index,
            "chunk_id": int(chunk_id),
            "question": "",
            "answer": evidence_texts.get(int(chunk_id), ""),
        })
    evidence_block = build_evidence_block([
        {**item, "section_path": ""} for item in evidence
    ])

    def revise_current(answer: str, issues: list[str]) -> str:
        if revise is None:
            return ""
        prompt = ANSWER_REVISION_PROMPT.format(
            query=str(case.get("query") or ""),
            evidence=evidence_block,
            answer=answer,
            issues="\n".join(issues),
        )
        return str(revise(prompt) or "")

    checked = verifier.review(
        str(trace.get("answer") or ""),
        evidence,
        query=str(case.get("query") or ""),
        revise=revise_current if revise is not None else None,
    )
    candidate = dict(trace)
    candidate["answer"] = checked.final_answer or str(trace.get("answer") or "")
    rescored = rescore_trace(case, candidate, evidence_texts, overlay)
    rescored["original_answer"] = str(trace.get("answer") or "")
    rescored["answer_verification"] = {
        "status": checked.status,
        "reason": checked.reason,
        "issues": list(checked.issues),
        "revision_calls": checked.revision_calls,
        "changed": bool(checked.final_answer and checked.final_answer != trace.get("answer")),
    }
    return rescored


def _usage(response) -> tuple[int, int, int]:
    usage = getattr(response, "usage_metadata", None)
    if not usage:
        usage = (getattr(response, "response_metadata", None) or {}).get("token_usage") or {}
    input_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
    details = usage.get("input_token_details") or {}
    cached = int(
        details.get("cache_read") or details.get("cached_tokens")
        or usage.get("prompt_cache_hit_tokens") or 0
    )
    return input_tokens, output_tokens, cached


class GuardedAnswerSemantic:
    """Structured semantic answer checker with pre-call budget and exact cache."""

    def __init__(self, llm, cost: CostTracker, cache: ResponseCache, model: str) -> None:
        self.llm = llm
        self.cost = cost
        self.cache = cache
        self.model = model
        self.calls = 0
        self.hits = 0
        self.kind = "answer_verifier"
        self.prompt_sha = prompt_sha256(ANSWER_VERIFIER_PROMPT.template)

    def invoke(self, prompt: str):
        key = self.cache.response_key(
            model=self.model, prompt_sha256=self.prompt_sha,
            temperature=0.0, query=prompt, evidence_ids=[],
        )
        payload = self.cache.get(key)
        if payload is not None:
            self.hits += 1
            return AnswerVerdict(**payload)
        tokens = estimate_tokens(prompt)
        self.cost.guard(tokens, self.model, kind=self.kind, request_text=prompt)
        self.calls += 1
        try:
            response = self.llm.invoke(
                prompt + '\n只输出 JSON: {"status":"pass|fail|uncertain",'
                '"reason":"...","issues":["..."]}'
            )
        except BaseException as exc:
            self.cost.fail_attempt(exc)
            raise
        raw = str(getattr(response, "content", response))
        payload = parse_json_object(raw)
        result = AnswerVerdict(**payload)
        self.cost.note_provider_usage(*_usage(response))
        self.cost.add(self.model, tokens, estimate_tokens(raw), kind=self.kind)
        self.cache.put(key, {
            "status": result.status,
            "reason": result.reason,
            "issues": list(result.issues),
        })
        return result


def _load_credential_env(path: Path | None) -> None:
    if path is None:
        return
    from dotenv import dotenv_values

    values = dotenv_values(path)
    for name in ("OPENAI_API_KEY", "EMBEDDING_API_KEY", "RERANK_API_KEY"):
        if values.get(name):
            os.environ[name] = str(values[name])


def estimate_verification_cost(case_count: int, *, pricing: PricingTable,
                               model: str = "deepseek-flash") -> dict:
    # Conservative bound: three existing groups, one checker call plus 25%
    # correction headroom, 2.5k input and 384 output tokens per call.
    calls = math.ceil(case_count * 3 * 1.25)
    amount = pricing.cost(model, calls * 2_500, calls * 384)
    return {
        "cases_per_group": case_count,
        "groups": 3,
        "estimated_calls_with_revision_headroom": calls,
        "basis": "2.5k input + 384 output tokens per call at verified peak prices",
        "estimated_usd": round(amount, 6),
    }


def _revision_invoker(llm, cost: CostTracker, cache: ResponseCache, model: str):
    kind = "answer_revision"
    prompt_sha = prompt_sha256(ANSWER_REVISION_PROMPT.template)

    def invoke(prompt: str) -> str:
        key = cache.response_key(
            model=model, prompt_sha256=prompt_sha, temperature=0.0,
            query=prompt, evidence_ids=[],
        )
        payload = cache.get(key)
        if payload is not None:
            return str(payload["answer"])
        tokens = estimate_tokens(prompt)
        cost.guard(tokens, model, kind=kind, request_text=prompt)
        try:
            response = llm.invoke(prompt)
        except BaseException as exc:
            cost.fail_attempt(exc)
            raise
        answer = str(getattr(response, "content", response)).strip()
        cost.note_provider_usage(*_usage(response))
        cost.add(model, tokens, estimate_tokens(answer), kind=kind)
        cache.put(key, {"answer": answer})
        return answer

    return invoke


def _verify_dev_subset(
    output_root: Path,
    snapshot: Path,
    *,
    credential_env_file: Path | None,
    max_cost_usd: float,
    allow_paid: bool,
    cache_dir: Path,
) -> dict:
    if not allow_paid:
        raise BudgetExceeded("答案复核真实调用需要显式 --allow-paid")
    if max_cost_usd > 2.0:
        raise BudgetExceeded("答案复核实验保护上限不得超过 $2.00")
    _load_credential_env(credential_env_file)
    os.environ["OPENAI_MODEL"] = "deepseek-flash"
    os.environ["LLM_MAX_RETRIES"] = "0"
    if not os.environ.get("OPENAI_API_KEY"):
        raise BudgetExceeded("缺少 OPENAI_API_KEY")

    from app.config import Settings
    from app.llm import make_chat_model

    pricing = PricingTable.load(PRICING)
    if not pricing.verified:
        raise BudgetExceeded("价格表未核实")
    run_root = output_root / "verified" / "dev"
    ledger = BudgetLedger(output_root / "experiment-cost-ledger.json",
                          total_limit_usd=max_cost_usd)
    attempts = AttemptLedger(run_root / "attempts.jsonl")
    cost = CostTracker(
        pricing, ledger, max_cost_usd,
        allow_paid=True, attempt_ledger=attempts,
    )
    responses = ResponseCache(cache_dir / "responses.json", "responses")
    base_llm = make_chat_model(Settings()).bind(temperature=0)
    semantic = GuardedAnswerSemantic(base_llm, cost, responses, "deepseek-flash")
    verifier = AnswerVerifier(semantic, max_revisions=1)
    revise = _revision_invoker(base_llm, cost, responses, "deepseek-flash")

    subset = json.loads(DEV_SUBSET.read_text(encoding="utf-8"))
    case_ids = set(subset["case_ids"])
    cases = [case for case in load_split("dev") if case["id"] in case_ids]
    if len(cases) != len(case_ids):
        raise RuntimeError("dev 固定小集包含未知 case_id")
    overlay = json.loads(OVERLAY.read_text(encoding="utf-8"))
    evidence_texts = load_evidence_texts(snapshot)
    estimate = estimate_verification_cost(len(cases), pricing=pricing)
    report = {
        "schema_version": 1,
        "scoring_version": overlay["version"],
        "case_set_version": subset["version"],
        "case_ids": sorted(case_ids),
        "max_cost_usd": max_cost_usd,
        "estimate": estimate,
        "groups": {},
    }
    for mode in GROUPS["rag"]:
        source = SOURCE_RUNS / "dev" / "rag" / mode / "traces.jsonl"
        by_id = {str(trace["case_id"]): trace for trace in _read_jsonl(source)}
        verified = []
        for case in cases:
            before_calls = int(cost.api_attempts)
            before_paid = int(cost.paid_api_attempts)
            before_spent = float(cost.spent_usd)
            started = time.perf_counter()
            with attempt_context(split="dev", experiment="rag_quality", group=mode,
                                 case_id=case["id"]):
                result = apply_answer_verifier(
                    case, by_id[case["id"]], evidence_texts, verifier, overlay,
                    revise=revise,
                )
            result["verification_latency_ms"] = round(
                (time.perf_counter() - started) * 1000, 3
            )
            result["verification_api_calls"] = int(cost.api_attempts) - before_calls
            result["verification_paid_calls"] = int(cost.paid_api_attempts) - before_paid
            result["verification_cost_actual"] = round(
                float(cost.spent_usd) - before_spent, 8
            )
            verified.append(result)
        group_root = run_root / mode
        _write_jsonl(group_root / "traces.jsonl", verified)
        original = [
            rescore_trace(
                {**next(c for c in cases if c["id"] == trace["case_id"])},
                by_id[str(trace["case_id"])], evidence_texts, overlay,
            )
            for trace in verified
        ]
        summary = summarize_quality(cases, verified)
        summary["before"] = summarize_quality(cases, original)
        summary["answer_verification"] = {
            "pass": sum(item["answer_verification"]["status"] == "pass" for item in verified),
            "refuse": sum(item["answer_verification"]["status"] == "refuse" for item in verified),
            "revisions": sum(item["answer_verification"]["revision_calls"] for item in verified),
            "changed": sum(item["answer_verification"]["changed"] for item in verified),
        }
        _write_json(group_root / "aggregate.json", summary)
        report["groups"][mode] = summary
    report["cost"] = {
        "actual_usd": round(cost.spent_usd, 8),
        "ledger_total_usd": round(ledger.total_usd(), 8),
        "limit_usd": max_cost_usd,
        "api_calls": cost.api_attempts,
        "paid_calls": cost.paid_api_attempts,
        "semantic_calls": semantic.calls,
        "semantic_cache_hits": semantic.hits,
    }
    from scripts.eval_rag_ablation import _summarize_attempt_ledger

    report["attempt_ledger"] = _summarize_attempt_ledger(
        output_root / "verified" / "dev" / "attempts.jsonl",
        split="dev", experiment="rag_quality",
    )
    _write_json(run_root / "aggregate.json", report)
    return report


def _run_controlled_trigger(
    output_root: Path,
    snapshot: Path,
    *,
    credential_env_file: Path | None,
    max_cost_usd: float,
    allow_paid: bool,
    cache_dir: Path,
) -> dict:
    if not allow_paid:
        raise BudgetExceeded("受控触发运行需要显式 --allow-paid")
    if max_cost_usd > 2.0:
        raise BudgetExceeded("保护上限不得超过 $2.00")
    _load_credential_env(credential_env_file)
    os.environ["OPENAI_MODEL"] = "deepseek-flash"
    os.environ["LLM_MAX_RETRIES"] = "0"
    os.environ["DATABASE_URL"] = f"sqlite:///{(ROOT / 'data' / 'ch12_eval.db').resolve()}"
    if not os.environ.get("OPENAI_API_KEY"):
        raise BudgetExceeded("缺少 OPENAI_API_KEY")

    from scripts.ch11_harness import HarnessConfig, run_case
    from scripts.eval_ch11 import _build_context

    run_root = output_root / "gap-trigger" / "dev"
    args = argparse.Namespace(
        snapshot=snapshot,
        cache_dir=cache_dir,
        pricing=PRICING,
        ledger=output_root / "experiment-cost-ledger.json",
        allow_paid=True,
        no_rewrite=False,
        rewrite_cache=None,
        rewrite_cache_mode="read",
    )
    config = HarnessConfig(
        split="dev", mode="full", strategy="hybrid_rerank",
        max_cost_usd=max_cost_usd, allow_paid=True,
        rag_mode="controlled_loop", rag_gap_trigger_enabled=True,
        temperature=0.0,
    )
    subset = json.loads(DEV_SUBSET.read_text(encoding="utf-8"))
    case_ids = set(subset["case_ids"])
    cases = [case for case in load_split("dev") if case["id"] in case_ids]
    ctx, config, _ = _build_context(config, args, cases)
    ctx.cost.ledger.total_limit_usd = max_cost_usd
    ctx.cost.attempt_ledger = AttemptLedger(run_root / "attempts.jsonl")
    overlay = json.loads(OVERLAY.read_text(encoding="utf-8"))
    evidence_texts = load_evidence_texts(snapshot)
    traces = []
    for case in cases:
        before_calls = int(ctx.cost.api_attempts)
        before_paid = int(ctx.cost.paid_api_attempts)
        before_spent = float(ctx.cost.spent_usd)
        with attempt_context(split="dev", experiment="rag_quality",
                             group="controlled_loop_gap_trigger",
                             case_id=case["id"]):
            raw = run_case(case, ctx, config)
        trace = rescore_trace(case, raw, evidence_texts, overlay)
        trace["raw_run_contract_sha"] = raw.get("trace_id")
        trace["trigger_experiment_cost_actual"] = round(
            float(ctx.cost.spent_usd) - before_spent, 8
        )
        trace["trigger_experiment_api_calls"] = (
            int(ctx.cost.api_attempts) - before_calls
        )
        trace["trigger_experiment_paid_calls"] = (
            int(ctx.cost.paid_api_attempts) - before_paid
        )
        traces.append(trace)
    _write_jsonl(run_root / "traces.jsonl", traces)
    summary = summarize_quality(cases, traces)
    source = {
        str(trace["case_id"]): trace
        for trace in _read_jsonl(SOURCE_RUNS / "dev" / "rag" / "controlled_loop" / "traces.jsonl")
        if str(trace["case_id"]) in case_ids
    }
    source_rescored = [
        rescore_trace(case, source[case["id"]], evidence_texts, overlay)
        for case in cases
    ]
    summary["before"] = summarize_quality(cases, source_rescored)
    summary["triggered_case_ids"] = sorted(
        str(trace["case_id"]) for trace in traces
        if any(step.get("gap_trigger") for step in trace.get("rag_steps") or [])
    )
    summary["cost"] = {
        "actual_usd": round(ctx.cost.spent_usd, 8),
        "ledger_total_usd": round(ctx.cost.ledger.total_usd(), 8),
        "limit_usd": max_cost_usd,
        "api_calls": ctx.cost.api_attempts,
        "paid_calls": ctx.cost.paid_api_attempts,
    }
    _write_json(run_root / "aggregate.json", summary)
    return {
        "model_calls": summary["cost"]["api_calls"],
        "scoring_version": overlay["version"],
        "group": summary,
        "output": str(run_root / "aggregate.json"),
    }


def _offline_rescore(output_root: Path, snapshot: Path) -> dict:
    from scripts.eval_snapshot import tree_sha256

    overlay = json.loads(OVERLAY.read_text(encoding="utf-8"))
    evidence_texts = load_evidence_texts(snapshot)
    report = {
        "schema_version": 1,
        "scoring_version": overlay["version"],
        "source": str(SOURCE_RUNS.relative_to(ROOT)),
        "output": str(output_root.relative_to(ROOT)),
        "raw_answers_preserved": True,
        "model_calls": 0,
        "provenance": {
            "snapshot_tree_sha256": tree_sha256(snapshot),
            "overlay_sha256": hashlib.sha256(OVERLAY.read_bytes()).hexdigest(),
            "dev_case_set_sha256": hashlib.sha256(DEV_SUBSET.read_bytes()).hexdigest(),
        },
        "splits": {},
    }
    for split in ("dev", "frozen-v1"):
        cases = load_split(split)
        cases_by_id = {str(case["id"]): case for case in cases}
        split_report = {}
        for experiment, groups in GROUPS.items():
            for group in groups:
                source = SOURCE_RUNS / split / experiment / group / "traces.jsonl"
                traces = _read_jsonl(source)
                rescored = [
                    rescore_trace(cases_by_id[str(trace["case_id"])], trace,
                                  evidence_texts, overlay)
                    for trace in traces
                ]
                summary = summarize_quality(cases, rescored)
                group_root = output_root / "runs" / split / experiment / group
                _write_jsonl(group_root / "rescore.jsonl", rescored)
                _write_json(group_root / "aggregate.json", summary)
                split_report[f"{experiment}:{group}"] = summary
        report["splits"][split] = split_report
    _write_json(output_root / "aggregate.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="受控 RAG 评分修订与质量复核")
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument("--offline-rescore", action="store_true",
                        help="零模型调用,统一重评历史响应")
    parser.add_argument("--verify-dev-subset", action="store_true")
    parser.add_argument("--run-controlled-trigger", action="store_true")
    parser.add_argument("--cache-dir", type=Path,
                        default=ROOT / "data" / "eval-caches-agentic-quality")
    parser.add_argument("--credential-env-file", type=Path)
    parser.add_argument("--allow-paid", action="store_true")
    parser.add_argument("--max-cost-usd", type=float, default=0.0)
    parser.add_argument("--estimate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.estimate_only:
        subset = json.loads(DEV_SUBSET.read_text(encoding="utf-8"))
        pricing = PricingTable.load(PRICING)
        estimate = estimate_verification_cost(len(subset["case_ids"]), pricing=pricing)
        print(json.dumps({
            **estimate,
            "pricing_version": pricing.version,
            "pricing_verified": pricing.verified,
            "recommended_hard_limit_usd": 2.0,
        }, ensure_ascii=False, indent=2))
        return 0
    if not any((args.offline_rescore, args.verify_dev_subset,
                args.run_controlled_trigger)):
        parser.error("至少需要一个运行模式")
    report = {"scoring_version": "rag-quality-v3", "model_calls": 0}
    output_path = args.output / "aggregate.json"
    if args.offline_rescore:
        report = _offline_rescore(args.output, args.snapshot)
    if args.verify_dev_subset:
        report = _verify_dev_subset(
            args.output, args.snapshot,
            credential_env_file=args.credential_env_file,
            max_cost_usd=args.max_cost_usd,
            allow_paid=args.allow_paid,
            cache_dir=args.cache_dir,
        )
        output_path = args.output / "verified" / "dev" / "aggregate.json"
    if args.run_controlled_trigger:
        report = _run_controlled_trigger(
            args.output, args.snapshot,
            credential_env_file=args.credential_env_file,
            max_cost_usd=args.max_cost_usd,
            allow_paid=args.allow_paid,
            cache_dir=args.cache_dir,
        )
        output_path = Path(report["output"])
    print(json.dumps({
        "scoring_version": report["scoring_version"],
        "model_calls": report.get("model_calls", report.get("cost", {}).get("api_calls")),
        "output": str(output_path),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
