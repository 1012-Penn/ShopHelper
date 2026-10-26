"""Paired dev/held-out runner for free-text vs structured answer plans."""
from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from app.answer_plan.models import EvidenceCandidate
from app.answer_plan.pipeline import AnswerPlanPipeline
from app.answer_plan.planner import StructuredAnswerPlanner
from scripts.answer_plan_r3.budget import (
    AnswerPlanBudgetLedger,
    BudgetedModel,
)
from scripts.answer_plan_r3.data import load_evidence, load_frozen_dataset, model_projection
from scripts.answer_plan_r3.scoring import score_case
from scripts.eval_cost import PricingTable

MODEL = "deepseek-flash"
PRICING = Path("tests/eval/rag_ablation/pricing-snapshot-2026-10-26.yaml")

FREE_TEXT_PROMPT = """你是电商客服回答器。只使用用户当前问题和候选证据回答；不得使用未给出的业务事实。

【用户问题】
{query}

【候选证据】
{evidence}

要求：
1. 每个实质事实使用候选编号 [n] 引用对应证据。
2. 保留适用条件、例外、单位、订单主体和政策范围。
3. 证据不足时明确要求补充信息或拒绝猜测。
4. 不要机械粘贴整段证据。
5. 只输出给用户的最终答案。
"""


class HeldoutAlreadyRun(RuntimeError):
    pass


def claim_heldout_once(
    marker: Path,
    *,
    split: str,
    config_sha256: str,
    case_ids: list[str],
) -> None:
    marker = Path(marker)
    marker.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "split": split,
        "config_sha256": config_sha256,
        "case_ids": list(case_ids),
        "status": "started",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        with marker.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    except FileExistsError as exc:
        raise HeldoutAlreadyRun(
            f"held-out was already claimed at {marker}; refusing a second run"
        ) from exc


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return round(ordered[index], 3)


def summarize_results(rows: list[dict]) -> dict:
    summary = {"dev": {}, "heldout": {}, "pairwise": {"dev": {}, "heldout": {}}}
    for split in ("dev", "heldout"):
        split_rows = [row for row in rows if row.get("split") == split]
        by_arm = {
            arm: [row for row in split_rows if row.get("arm") == arm]
            for arm in ("free_text", "answer_plan")
        }
        for arm, arm_rows in by_arm.items():
            successful = sum(
                bool(row.get("score", {}).get("strict_success")) for row in arm_rows
            )
            summary[split][arm] = {
                "success": successful,
                "n": len(arm_rows),
                "strict_success": successful,
                "facts_passed": sum(
                    bool(row.get("score", {}).get("facts", {}).get("passed"))
                    for row in arm_rows
                ),
                "conditions_passed": sum(
                    bool(row.get("score", {}).get("conditions", {}).get("passed"))
                    for row in arm_rows
                ),
                "review_required": sum(
                    bool(row.get("score", {}).get("review_required"))
                    for row in arm_rows
                ),
                "misrefusal": sum(
                    bool(row.get("score", {}).get("misrefusal"))
                    for row in arm_rows
                ),
                "behavior_passed": sum(
                    bool(row.get("score", {}).get("behavior", {}).get("passed"))
                    for row in arm_rows
                ),
                "citation_supported": sum(
                    row.get("score", {}).get("citations", {}).get("status")
                    == "supported"
                    for row in arm_rows
                ),
                "citation_unsupported": sum(
                    row.get("score", {}).get("citations", {}).get("status")
                    == "unsupported"
                    for row in arm_rows
                ),
                "citation_uncertain": sum(
                    row.get("score", {}).get("citations", {}).get("status")
                    == "uncertain"
                    for row in arm_rows
                ),
                "correction_attempts": sum(
                    int(row.get("correction_attempts") or 0) for row in arm_rows
                ),
                "provider_calls": sum(
                    int(row.get("provider_calls") or 0) for row in arm_rows
                ),
                "input_tokens": sum(
                    int(row.get("input_tokens") or 0) for row in arm_rows
                ),
                "output_tokens": sum(
                    int(row.get("output_tokens") or 0) for row in arm_rows
                ),
                "cached_input_tokens": sum(
                    int(row.get("cached_input_tokens") or 0) for row in arm_rows
                ),
                "cost_usd": round(
                    sum(float(row.get("cost_usd") or 0.0) for row in arm_rows), 8
                ),
                "p95_latency_ms": _p95([
                    float(row.get("latency_ms") or 0.0) for row in arm_rows
                ]),
            }
        by_case = {
            case_id: {
                row.get("arm"): bool(row.get("score", {}).get("strict_success"))
                for row in split_rows
                if str(row.get("case_id")) == case_id
            }
            for case_id in sorted({str(row.get("case_id")) for row in split_rows})
        }
        wins = losses = ties = 0
        for pair in by_case.values():
            if "answer_plan" not in pair or "free_text" not in pair:
                continue
            if pair["answer_plan"] and not pair["free_text"]:
                wins += 1
            elif pair["free_text"] and not pair["answer_plan"]:
                losses += 1
            else:
                ties += 1
        summary["pairwise"][split]["answer_plan_vs_free_text"] = {
            "wins": wins,
            "losses": losses,
            "ties": ties,
        }
    summary["total_cost_usd"] = round(
        sum(float(row.get("cost_usd") or 0.0) for row in rows), 8
    )
    summary["technical_failures"] = sum(
        bool(row.get("technical_failure")) for row in rows
    )
    return summary


def _evidence_candidates(evidence_rows: list[dict]) -> list[EvidenceCandidate]:
    return [EvidenceCandidate.model_validate(row) for row in evidence_rows]


def _evidence_block(candidates: list[EvidenceCandidate]) -> str:
    rows = []
    for number, candidate in enumerate(candidates, 1):
        rows.append(
            json.dumps(
                {
                    "n": number,
                    "evidence_id": candidate.evidence_id,
                    "text": candidate.text,
                    "source_id": candidate.source_id,
                    "source_title": candidate.source_title,
                    "entities": candidate.entities,
                    "required_conditions": candidate.required_conditions,
                    "condition_aliases": candidate.condition_aliases,
                    "policy_scope": (
                        candidate.policy_scope.model_dump(mode="json")
                        if candidate.policy_scope
                        else None
                    ),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    return "\n".join(rows)


def _load_credentials(path: Path) -> None:
    from dotenv import dotenv_values

    values = dotenv_values(Path(path))
    for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL"):
        if values.get(key):
            os.environ[key] = str(values[key])
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing; refusing paid run")
    os.environ.update({
        "OPENAI_MODEL": MODEL,
        "STRUCTURED_OUTPUT_MODE": "auto",
        "EXTRACT_TEMPERATURE": "0",
        "MAX_OUTPUT_TOKENS": "1200",
        "LLM_MAX_RETRIES": "0",
        "LANGFUSE_ENABLED": "false",
    })


def _call_usage(calls: list[dict]) -> dict:
    return {
        "calls": len(calls),
        "input_tokens": sum(
            int((call.get("usage") or {}).get("input_tokens") or 0)
            for call in calls
        ),
        "output_tokens": sum(
            int((call.get("usage") or {}).get("output_tokens") or 0)
            for call in calls
        ),
        "cached_input_tokens": sum(
            int((call.get("usage") or {}).get("cached_input_tokens") or 0)
            for call in calls
        ),
        "cost_usd": round(
            sum(float(call.get("cost_usd") or 0.0) for call in calls), 8
        ),
    }


def run(
    *,
    split: str,
    allow_paid: bool,
    max_cost_usd: float,
    data_root: Path,
    output_root: Path,
    ledger_path: Path,
    credentials_path: Path,
) -> dict:
    if split not in {"dev", "heldout"}:
        raise ValueError("split must be dev or heldout")
    if not allow_paid:
        raise RuntimeError("refusing paid provider calls without --allow-paid")
    if not 0 < float(max_cost_usd) <= 2.0:
        raise RuntimeError("max_cost_usd must be in (0, 2.0]")
    dataset = load_frozen_dataset(data_root)
    manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
    evidence = load_evidence(data_root)
    cases = dataset[split]
    if split == "heldout":
        claim_heldout_once(
            output_root / "heldout-once.json",
            split=split,
            config_sha256=str(manifest["config_sha256"]),
            case_ids=[row["id"] for row in cases],
        )
    _load_credentials(credentials_path)
    from app.config import Settings
    from app.llm import make_extract_model, make_structured_extractor

    settings = Settings(_env_file=None)
    pricing = PricingTable.load(PRICING)
    if not pricing.verified:
        raise RuntimeError("frozen pricing is not verified")
    ledger = AnswerPlanBudgetLedger(ledger_path, limit_usd=max_cost_usd)
    ledger.assert_no_pending()
    raw = make_extract_model(settings, model=MODEL)
    budgeted = BudgetedModel(
        raw,
        model=MODEL,
        pricing=pricing,
        ledger=ledger,
        max_tokens=1200,
        max_retries=2,
        backoff_seconds=0.5,
    )
    plan_extractor = make_structured_extractor(
        settings,
        __import__("app.answer_plan.models", fromlist=["AnswerPlan"]).AnswerPlan,
        base_model=budgeted,
    )
    pipeline = AnswerPlanPipeline(
        StructuredAnswerPlanner(plan_extractor),
        max_corrections=1,
    )

    run_dir = output_root / f"{split}-{datetime.now().strftime('%Y%m%dT%H%M%SZ')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    result_path = run_dir / "results.jsonl"
    rows: list[dict] = []
    with result_path.open("w", encoding="utf-8") as output:
        for case in cases:
            projection = model_projection(case, evidence)
            candidates = _evidence_candidates(projection["evidence"])
            query = projection["query"]
            mapping = {
                number: candidate.evidence_id
                for number, candidate in enumerate(candidates, 1)
            }
            for arm in ("free_text", "answer_plan"):
                calls_before = len(budgeted.calls)
                started = time.perf_counter()
                error = None
                correction_attempts = 0
                answer = ""
                plan_result = None
                citation_items = mapping if arm == "free_text" else {}
                try:
                    if arm == "free_text":
                        with budgeted.context(
                            case_id=case["id"],
                            split=split,
                            arm=arm,
                            kind="free_text",
                        ):
                            prompt = FREE_TEXT_PROMPT.format(
                                query=query,
                                evidence=_evidence_block(candidates),
                            )
                            response = budgeted.invoke(prompt)
                            answer = str(getattr(response, "content", response)).strip()
                    else:
                        with budgeted.context(
                            case_id=case["id"],
                            split=split,
                            arm=arm,
                            kind="answer_plan",
                        ):
                            plan_result = pipeline.answer(query, candidates)
                        answer = plan_result.rendered.text
                        citation_items = {
                            int(number): item["evidence_id"]
                            for number, item in plan_result.rendered.citations.items()
                        }
                        correction_attempts = plan_result.correction_attempts
                except BaseException as exc:
                    error = f"{type(exc).__name__}: {exc}"[:500]
                latency = (time.perf_counter() - started) * 1000
                calls = budgeted.calls[calls_before:]
                usage = _call_usage(calls)
                score = score_case(
                    case,
                    {
                        "answer": answer,
                        "citation_items": citation_items,
                        "technical_failure": bool(error),
                    },
                    [evidence[item] for item in case.get("evidence_ids") or []],
                )
                row = {
                    "case_id": case["id"],
                    "source_group": case["source_group"],
                    "split": split,
                    "arm": arm,
                    "query": query,
                    "answer": answer,
                    "citation_items": citation_items,
                    "evidence_ids": list(case.get("evidence_ids") or []),
                    "score": score,
                    "technical_failure": bool(error),
                    "error": error,
                    "correction_attempts": correction_attempts,
                    "provider_calls": usage["calls"],
                    "input_tokens": usage["input_tokens"],
                    "output_tokens": usage["output_tokens"],
                    "cached_input_tokens": usage["cached_input_tokens"],
                    "cost_usd": usage["cost_usd"],
                    "latency_ms": round(latency, 3),
                    "plan": (
                        plan_result.plan.model_dump(mode="json")
                        if plan_result is not None
                        else None
                    ),
                    "plan_validation": (
                        plan_result.validation.model_dump(mode="json")
                        if plan_result is not None
                        else None
                    ),
                }
                rows.append(row)
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
    aggregate = summarize_results(rows)
    contract = {
        "schema_version": 1,
        "split": split,
        "model": MODEL,
        "data_manifest": manifest,
        "pricing": {
            "version": pricing.version,
            "verified": pricing.verified,
            "path": str(PRICING),
        },
        "ledger": {
            "path": str(ledger_path),
            "committed_usd": round(ledger.committed_usd(), 8),
            "limit_usd": ledger.limit_usd,
        },
        "summary": aggregate,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "aggregate.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    latest = output_root / f"latest-{split}.json"
    latest.write_text(
        json.dumps(
            {"run_dir": str(run_dir), **aggregate},
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return {"run_dir": str(run_dir), **contract}
