"""ch11 分层评测 harness:同一套代码跑 L1–L5。

L1/L2 可用离线缓存复现;L3/L4 用评测回答 prompt + 确定性检查;L5 记录调用/token/
延迟/费用。不跑完整 LangGraph 意图/工具链,工具题在 L4 标 not_applicable。
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from app.guard import REFUSAL_MARKER
from app.history import estimate_tokens
from app.kb import vector_text
from app.prompts import FAITHFULNESS_JUDGE_PROMPT
from scripts.eval_checks import check_case_deterministic
from scripts.eval_metrics import (
    coverage_rank,
    evidence_group_recall,
    evidence_precision,
    hard_negative_hit_rate,
    mrr_at,
    single_chunk_rank,
)
from scripts.eval_prompts import EVAL_ANSWER_PROMPT, prompt_sha256

TOOL_BUCKETS = {"hard_negative", "ood"}


@dataclass(frozen=True)
class HarnessConfig:
    split: str
    mode: Literal["offline", "retrieval", "evidence", "full"]
    strategy: str = "hybrid_rerank"
    limit: int = 0
    max_cost_usd: float = 0.0
    allow_paid: bool = False
    use_judge: bool = False
    include_tool_fixtures: bool = True
    contract_sha: str = ""
    model: str = ""
    temperature: float = 0.0
    seed: int = 20261024


@dataclass
class HarnessContext:
    service: Any
    all_rows: dict[int, Any]
    gold_by_case: dict[str, list[frozenset[int]]]
    prepared: dict[str, Any] = field(default_factory=dict)
    chat: Any = None
    judge: Any = None
    cache: Any = None
    cost: Any = None


def _evidence_block(items: list[dict]) -> str:
    return "\n".join(
        f"[{item['n']}] ({item['section_path']}) 问:{item['question']} 答:{item['answer']}"
        for item in items
    )


def _evidence_items(result, all_rows: dict[int, Any], top_n: int = 10) -> list[dict]:
    items = []
    for rank, item in enumerate(result.items[:top_n], start=1):
        row = all_rows.get(item.chunk_id)
        if row is None:
            continue
        items.append({
            "n": rank,
            "chunk_id": item.chunk_id,
            "score": item.score,
            "section_path": row.section_path or "",
            "question": row.questions,
            "answer": row.answer,
            "category": row.category,
        })
    return items


def _evidence_by_n(items: list[dict]) -> dict[int, str]:
    return {item["n"]: vector_text(item["category"], item["question"], item["answer"]) for item in items}


def _query_with_context(case: dict) -> str:
    context = case.get("conversation_context") or []
    if not context:
        return str(case["query"])
    turns = "\n".join(f"{turn['role']}: {turn['content']}" for turn in context)
    return f"前文:\n{turns}\n当前问题:{case['query']}"


def _usage(response) -> tuple[int, int]:
    usage = getattr(response, "usage_metadata", None)
    if not usage:
        usage = (getattr(response, "response_metadata", None) or {}).get("token_usage") or {}
    input_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
    return input_tokens, output_tokens


def run_retrieval_case(
    case: dict,
    service,
    all_rows: dict[int, Any],
    gold_by_case: dict[str, list[frozenset[int]]],
    prepared: dict[str, Any],
    *,
    strategy: str,
    cap: int = 10,
) -> dict:
    judge = getattr(service, "_evidence_judge", None)
    judge_calls_before = int(getattr(judge, "calls", 0))
    judge_hits_before = int(getattr(judge, "hits", 0))
    started = time.perf_counter()
    result = service.retrieve(
        case["query"], strategy=strategy, prepared=prepared.get(case["id"])
    )
    latency_ms = (time.perf_counter() - started) * 1000
    evidence_judge_calls = int(getattr(judge, "calls", 0)) - judge_calls_before
    evidence_judge_cached = int(getattr(judge, "hits", 0)) - judge_hits_before
    ranked = [item.chunk_id for item in result.items]
    gold = gold_by_case.get(case["id"], [])
    support = set().union(*gold) if gold else set()
    items = _evidence_items(result, all_rows)
    return {
        "case_id": case["id"],
        "split": case.get("split", ""),
        "bucket": case.get("bucket", ""),
        "expected_behavior": case.get("expected_behavior", ""),
        "evidence_state": case.get("evidence_state", ""),
        "query": case["query"],
        "retrieved": [
            {"rank": rank, "chunk_id": item.chunk_id, "score": item.score}
            for rank, item in enumerate(result.items, start=1)
        ],
        "l1": {
            "n": 1 if gold else 0,
            "coverage_rank": coverage_rank(ranked, gold, cap=cap) if gold else None,
            "r3": coverage_rank(ranked, gold, cap=3) if gold else None,
            "r5": coverage_rank(ranked, gold, cap=5) if gold else None,
            "r10": coverage_rank(ranked, gold, cap=10) if gold else None,
            "group_recall": evidence_group_recall(ranked, gold, cap=cap) if gold else None,
            "single_chunk_rank": single_chunk_rank(ranked, case, all_rows, cap=cap) if gold else None,
            "single1": single_chunk_rank(ranked, case, all_rows, cap=1) if gold else None,
            "single3": single_chunk_rank(ranked, case, all_rows, cap=3) if gold else None,
            "mrr": mrr_at(ranked, gold, k=cap) if gold else None,
            "evidence_precision": evidence_precision(ranked, support, cap=cap) if gold else None,
            "hard_negative_hit_rate": (
                hard_negative_hit_rate(ranked, k=cap)
                if case.get("bucket") in TOOL_BUCKETS else None
            ),
        },
        "gate": {
            "passed": not result.low_confidence,
            "low_confidence": result.low_confidence,
            "reason": result.reason,
            "evidence_entailment": result.evidence_entailment,
            "degraded": result.degraded,
        },
        "evidence_items": items,
        "evidence_by_n": _evidence_by_n(items),
        "evidence_judge_calls": evidence_judge_calls,
        "evidence_judge_cached": evidence_judge_cached,
        "latency_ms": round(latency_ms, 3),
    }


def run_answer_case(
    case: dict,
    trace: dict,
    ctx: HarnessContext,
    config: HarnessConfig,
) -> dict:
    expected = case.get("expected_behavior", "answer")
    items = trace.pop("evidence_items")
    evidence_by_n = trace.pop("evidence_by_n")
    answer = ""
    cached = False
    input_tokens = output_tokens = 0
    llm_calls = 0

    if trace["gate"]["low_confidence"]:
        answer = f"{REFUSAL_MARKER} {trace['gate']['reason'] or '证据不足,建议转人工。'}"
    elif ctx.chat is None:
        answer = ""
    else:
        query = _query_with_context(case)
        prompt = EVAL_ANSWER_PROMPT.format(
            behavior=expected, evidence=_evidence_block(items), query=query
        )
        key = None
        if ctx.cache is not None:
            key = ctx.cache.response_key(
                model=config.model,
                prompt_sha256=prompt_sha256(),
                temperature=config.temperature,
                query=f"{expected}\0{query}\0{_evidence_block(items)}",
                evidence_ids=[item["chunk_id"] for item in items],
            )
            payload = ctx.cache.get(key)
            if payload is not None:
                answer = payload["answer"]
                input_tokens = int(payload.get("input_tokens") or 0)
                output_tokens = int(payload.get("output_tokens") or 0)
                cached = True
                pricing = getattr(ctx.cost, "pricing", None) if ctx.cost is not None else None
                if pricing is not None:
                    trace["cost_estimate_cached"] = pricing.cost(
                        config.model, input_tokens, output_tokens
                    )
        if not cached:
            if ctx.cost is not None:
                ctx.cost.guard(estimate_tokens(prompt), config.model)
            started = time.perf_counter()
            response = ctx.chat.invoke(prompt)
            latency_ms = (time.perf_counter() - started) * 1000
            trace["latency_ms"] = round(trace.get("latency_ms", 0.0) + latency_ms, 3)
            answer = str(getattr(response, "content", response))
            input_tokens, output_tokens = _usage(response)
            if not input_tokens:
                input_tokens = estimate_tokens(prompt)
            if not output_tokens:
                output_tokens = estimate_tokens(answer)
            llm_calls = 1
            if ctx.cost is not None:
                ctx.cost.add(config.model, input_tokens, output_tokens, kind="answer")
            if ctx.cache is not None and key is not None:
                ctx.cache.put(key, {
                    "answer": answer,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                })

    deterministic = check_case_deterministic(case, answer, evidence_by_n)
    trace.update({
        "answer": answer,
        "citations": deterministic["citation_support"],
        "deterministic_checks": deterministic,
        "behavior_pred": deterministic["behavior_pred"],
        "l4_applicable": not bool(case.get("requires_tools")),
        "llm_calls": llm_calls,
        "tokens_in": input_tokens,
        "tokens_out": output_tokens,
        "cached": cached,
    })
    if ctx.judge is not None and config.use_judge and answer:
        try:
            verdict = ctx.judge.invoke(
                FAITHFULNESS_JUDGE_PROMPT.format(
                    query=_query_with_context(case),
                    evidence=_evidence_block(items),
                    answer=answer,
                )
            )
            trace["judge"] = {
                "fabricated": bool(getattr(verdict, "fabricated", False)),
                "reason": str(getattr(verdict, "reason", "")),
            }
        except Exception as exc:  # 裁判失败不阻断样本
            trace["judge"] = {"error": str(exc)}
    else:
        trace["judge"] = None
    return trace


def run_case(case: dict, ctx: HarnessContext, config: HarnessConfig) -> dict:
    cost_before = float(getattr(ctx.cost, "spent_usd", 0.0)) if ctx.cost is not None else 0.0
    trace = run_retrieval_case(
        case,
        ctx.service,
        ctx.all_rows,
        ctx.gold_by_case,
        ctx.prepared,
        strategy=config.strategy,
    )
    trace["trace_id"] = hashlib.sha256(
        f"{config.contract_sha}\0{case['id']}".encode("utf-8")
    ).hexdigest()[:16]
    trace["expected_behavior"] = case.get("expected_behavior", "")
    trace["as_of_date"] = case.get("as_of_date")
    trace["requires_tools"] = list(case.get("requires_tools") or [])
    trace["forbidden_claims"] = list(case.get("forbidden_claims") or [])
    trace["required_conditions"] = list(case.get("required_conditions") or [])
    trace["gate_blocked"] = bool(trace["gate"]["low_confidence"])
    trace["failure_labels"]: list[str] = []
    if config.mode in {"full"}:
        trace = run_answer_case(case, trace, ctx, config)
        trace["failure_labels"] = trace["deterministic_checks"]["failure_labels"]
        if ctx.cost is not None:
            trace["cost_actual"] = round(
                float(getattr(ctx.cost, "spent_usd", 0.0)) - cost_before, 6
            )
            trace["cost_estimate"] = round(
                trace["cost_actual"]
                + float(trace.pop("cost_estimate_cached", 0.0) or 0.0),
                6,
            )
    else:
        trace.pop("evidence_items", None)
        trace.pop("evidence_by_n", None)
        trace["answer"] = ""
        trace["behavior_pred"] = None
        trace["l4_applicable"] = False
        trace["llm_calls"] = 0
        trace["tokens_in"] = trace["tokens_out"] = 0
        trace["cached"] = False
        trace["judge"] = None
        trace["cost_estimate"] = 0.0
        trace["cost_actual"] = 0.0
    return trace


def run_split(cases: list[dict], ctx: HarnessContext, config: HarnessConfig) -> list[dict]:
    traces = []
    for case in cases[: config.limit or None]:
        traces.append(run_case(case, ctx, config))
    return traces


def dump_traces(traces: list[dict], path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for trace in traces:
            fh.write(json.dumps(trace, ensure_ascii=False, sort_keys=True) + "\n")
