"""Bounded real-app smoke for the integrated third-round optimizations."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import time
from pathlib import Path

from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select

from scripts.e2e_quality_r2.budget import BudgetExceeded, ExactCallCache, R2Budget
from scripts.e2e_quality_r2.data import load_frozen_cases
from scripts.e2e_quality_r2.harness import (
    TraceCapture,
    build_instrumented_app,
    prepare_private_runtime,
    run_case,
)
from scripts.eval_cost import PricingTable

ROOT = Path(__file__).resolve().parent.parent
SMOKE_ROOT = ROOT / "tests" / "eval" / "optimization_r3_smoke"
MANIFEST_PATH = SMOKE_ROOT / "manifest.json"
DEFAULT_OUTPUT = SMOKE_ROOT / "results" / "current-app-dev-20261027"
PRICING_PATH = (
    ROOT / "tests" / "eval" / "rag_ablation" / "pricing-snapshot-2026-10-26.yaml"
)
E2E_CASES = ROOT / "tests" / "eval" / "e2e_quality_r2"
SOURCE_SNAPSHOT = Path(
    "/root/.codex/worktrees/ch16-rag-ablation/ShopHelper/data/eval-snapshots/"
    "ebfd49478b37ef387c8b124319a450ed73a86ae2a4b9f4fc28d6a52b2ccaa0ba.db"
)
SOURCE_METADATA_DB = Path(
    "/root/.codex/worktrees/ch16-rag-ablation/ShopHelper/data/ch12_eval.db"
)
SOURCE_CACHE = Path(
    "/root/.codex/worktrees/ch16-rag-ablation/ShopHelper/data/"
    "eval-caches-ch16-live-2026-10-26"
)
EXPECTED_SNAPSHOT_SHA256 = (
    "b6b1fa237d03d2c0f8ddcfe8e003f13c4cb584f0764ac04c2d57ae834bbdca73"
)


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)], 3)


def _route_nodes(row: dict) -> list[str]:
    route = ((row.get("score") or {}).get("route") or {})
    return list(route.get("nodes") or [])


def _ticket_count(app) -> int:
    from app.models import Ticket

    with app.state.session_factory() as session:
        return int(session.scalar(select(func.count()).select_from(Ticket)) or 0)


class _CallCappedBudget(R2Budget):
    def __init__(self, path: Path, *, limit_usd: float, max_calls: int) -> None:
        super().__init__(path, limit_usd=limit_usd)
        self.max_calls = int(max_calls)

    def reserve(self, context: dict, estimated_usd: float) -> str:
        if len(self.records()) >= self.max_calls:
            raise BudgetExceeded(
                f"provider-call stop rule reached: {self.max_calls}"
            )
        return super().reserve(context, estimated_usd)


async def _run_multi_agent_case(app, client, case: dict, *, capture, output: Path) -> dict:
    case_id = str(case["id"])
    capture.start_case(case={"id": case_id, "facts": []}, split="dev_smoke", turn=0)
    before_tickets = _ticket_count(app)
    started = time.perf_counter()
    response = await client.post(
        "/api/multi-agent/chat",
        json={"message": case["message"], "session_id": None},
        timeout=180.0,
    )
    latency_ms = (time.perf_counter() - started) * 1000
    body = response.json() if response.headers.get("content-type", "").startswith(
        "application/json"
    ) else {"text": response.text}
    previews = list(body.get("write_previews") or [])
    artifacts = list(body.get("action_artifacts") or [])
    after_preview_tickets = _ticket_count(app)
    confirmation = None
    replay = None
    if case.get("confirm_preview") and previews:
        preview = previews[0]
        confirm_payload = {
            "conversation_id": body["conversation_id"],
            "tool_name": "create_ticket",
            "description": preview.get("description", ""),
            "ticket_type": preview.get("ticket_type", ""),
            "confirm_key": preview.get("confirm_key"),
        }
        first = await client.post("/api/tickets/confirm", json=confirm_payload)
        retry = await client.post("/api/tickets/confirm", json=confirm_payload)
        confirmation = {
            "status": first.status_code,
            "body": first.json(),
        }
        replay = {
            "status": retry.status_code,
            "body": retry.json(),
        }
    after_tickets = _ticket_count(app)
    reply = str(body.get("final_reply") or "")
    result = {
        "case_id": case_id,
        "message": case["message"],
        "http_status": response.status_code,
        "status": body.get("status"),
        "final_reply": reply,
        "conversation_id": body.get("conversation_id"),
        "write_previews": previews,
        "action_artifacts": artifacts,
        "tickets_before": before_tickets,
        "tickets_after_preview": after_preview_tickets,
        "tickets_after": after_tickets,
        "preview_without_write": bool(previews) and after_preview_tickets == before_tickets,
        "text_preview_without_artifact": (
            "预览" in reply
            and not any(
                item.get("kind") == "ticket_preview"
                and item.get("state") == "awaiting_confirmation"
                for item in artifacts
            )
        ),
        "confirmation": confirmation,
        "confirmation_replay": replay,
        "confirmation_idempotent": bool(
            confirmation
            and replay
            and confirmation["status"] == 200
            and replay["status"] == 200
            and confirmation["body"].get("ticket_no")
            == replay["body"].get("ticket_no")
            and replay["body"].get("deduplicated") is True
        ),
        "latency_ms": round(latency_ms, 3),
        "technical_failures": capture.case_failures(case_id),
    }
    _write_json(output / f"{case_id}.json", result)
    return result


async def _run(args) -> dict:
    if not args.allow_paid:
        raise RuntimeError("--run requires --allow-paid")
    if not 0 < float(args.max_cost_usd) <= 2.0:
        raise RuntimeError("integration smoke cap must be in (0, 2.0] USD")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    output = Path(args.output)
    attempt = output / f"attempt-{int(args.attempt):02d}"
    if attempt.exists() and any(attempt.iterdir()):
        raise RuntimeError(f"smoke output already exists: {attempt}")
    attempt.mkdir(parents=True, exist_ok=True)
    _write_json(attempt / "manifest-snapshot.json", manifest)

    e2e_cases = load_frozen_cases(E2E_CASES)
    lookup = {str(case["id"]): case for case in e2e_cases["dev"]}
    selected = [lookup[case_id] for case_id in manifest["chat_case_ids"]]
    selected.extend(manifest["chat_cases"])

    runtime = prepare_private_runtime(
        source_snapshot=SOURCE_SNAPSHOT,
        source_metadata_db=SOURCE_METADATA_DB,
        expected_snapshot_sha256=EXPECTED_SNAPSHOT_SHA256,
        destination=ROOT / "data" / "optimization-r3-smoke" / "private-runtime",
        source_cache_dir=SOURCE_CACHE,
    )
    budget = _CallCappedBudget(
        attempt / "budget-ledger.json",
        limit_usd=args.max_cost_usd,
        max_calls=int(manifest["max_provider_calls"]),
    )
    budget.assert_no_pending()
    exact_cache = ExactCallCache(attempt / "exact-calls.json")
    capture = TraceCapture()
    capture.arm = "r3-integration-smoke"
    pricing = PricingTable.load(PRICING_PATH)
    if not pricing.verified:
        raise RuntimeError("pricing snapshot is not verified")

    os.environ["ANSWER_PLAN_ENABLED"] = "true"
    os.environ["ANSWER_PLAN_LEDGER_PATH"] = str(attempt / "answer-plan-budget.sqlite3")
    os.environ["ANSWER_PLAN_MAX_COST_USD"] = str(args.max_cost_usd)
    os.environ["ANSWER_PLAN_PRICING_PATH"] = str(PRICING_PATH)
    os.environ["MULTI_AGENT_ACTION_CONTRACTS_ENABLED"] = "true"

    namespace = "|".join((
        str(manifest["run_version"]),
        _sha(MANIFEST_PATH),
        _sha(PRICING_PATH),
        str(manifest["model"]),
    ))
    app, settings = build_instrumented_app(
        paths=runtime,
        credential_env_file=Path(args.credential_env_file),
        pricing=pricing,
        budget=budget,
        capture=capture,
        exact_cache=exact_cache,
        namespace=namespace,
        answer_verification_enabled=False,
        structured_mode="json-fallback",
    )
    if not settings.answer_plan_enabled:
        raise RuntimeError("application factory did not enable the answer-plan path")
    if not settings.multi_agent_action_contracts_enabled:
        raise RuntimeError("application factory did not enable action contracts")

    contract = {
        "schema_version": 1,
        "run_version": manifest["run_version"],
        "manifest": str(MANIFEST_PATH.relative_to(ROOT)),
        "manifest_sha256": _sha(MANIFEST_PATH),
        "pricing_sha256": _sha(PRICING_PATH),
        "snapshot_expected_sha256": EXPECTED_SNAPSHOT_SHA256,
        "model": settings.openai_model,
        "structured_output_mode": settings.structured_output_mode,
        "flags": {
            "answer_plan_enabled": settings.answer_plan_enabled,
            "multi_agent_action_contracts_enabled": (
                settings.multi_agent_action_contracts_enabled
            ),
            "multi_agent_handoff_mode": settings.multi_agent_handoff_mode,
            "adaptive_routing_enabled": settings.adaptive_routing_enabled,
            "answer_verification_enabled": settings.answer_verification_enabled,
            "claim_grounding_enabled": settings.claim_grounding_enabled,
            "rag_gap_trigger_enabled": settings.rag_gap_trigger_enabled,
        },
        "stop_rules": manifest["stop_rules"],
        "max_provider_calls": manifest["max_provider_calls"],
        "max_new_api_usd": args.max_cost_usd,
    }
    _write_json(attempt / "run-contract.json", contract)

    chat_results = []
    multi_results = []
    infrastructure_error = None
    run_started = time.monotonic()
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://r3-smoke",
        timeout=180.0,
    ) as client:
        for index, case in enumerate(selected):
            if time.monotonic() - run_started > float(manifest["max_wall_seconds"]):
                raise RuntimeError("fixed wall-clock stop rule reached")
            if len(budget.records()) >= int(manifest["max_provider_calls"]):
                raise RuntimeError("fixed provider-call stop rule reached")
            try:
                result = await run_case(
                    app,
                    client,
                    case,
                    capture=capture,
                    budget=budget,
                    output_root=attempt / "chat",
                )
            except Exception as exc:  # noqa: BLE001 - failure remains evidence
                infrastructure_error = (
                    f"{case['id']}:{type(exc).__name__}:{exc}"[:500]
                )
                break
            chat_results.append(result)
            _write_json(attempt / "chat-progress.json", {
                "completed": index + 1,
                "total": len(selected),
                "case_ids": [row["case_id"] for row in chat_results],
            })

        if infrastructure_error is None:
            for case in manifest["multi_agent_cases"]:
                if time.monotonic() - run_started > float(manifest["max_wall_seconds"]):
                    raise RuntimeError("fixed wall-clock stop rule reached")
                if len(budget.records()) >= int(manifest["max_provider_calls"]):
                    raise RuntimeError("fixed provider-call stop rule reached")
                try:
                    result = await _run_multi_agent_case(
                        app, client, case, capture=capture, output=attempt / "multi"
                    )
                except Exception as exc:  # noqa: BLE001 - failure remains evidence
                    infrastructure_error = (
                        f"{case['id']}:{type(exc).__name__}:{exc}"[:500]
                    )
                    break
                multi_results.append(result)

    latency = [float(row.get("latency_ms") or 0.0) for row in chat_results]
    chat_success = sum(bool(row.get("task_success")) for row in chat_results)
    summary = {
        "schema_version": 1,
        "contract": contract,
        "chat": {
            "completed": len(chat_results),
            "planned": len(selected),
            "success": chat_success,
            "technical_failures": sum(
                bool(row.get("technical_failures")) for row in chat_results
            ),
            "answer_plan_nodes": sum(
                "answer_plan" in _route_nodes(row)
                for row in chat_results
            ),
            "p95_latency_ms": _p95(latency),
            "cases": [
                {
                    "case_id": row["case_id"],
                    "task_success": row["task_success"],
                    "failure_labels": row["failure_labels"],
                    "route_nodes": _route_nodes(row),
                    "latency_ms": row["latency_ms"],
                    "technical_failures": row["technical_failures"],
                }
                for row in chat_results
            ],
        },
        "multi_agent": {
            "completed": len(multi_results),
            "planned": len(manifest["multi_agent_cases"]),
            "preview_without_write": sum(
                bool(row["preview_without_write"]) for row in multi_results
            ),
            "text_preview_without_artifact": sum(
                bool(row["text_preview_without_artifact"]) for row in multi_results
            ),
            "confirmation_idempotent": sum(
                bool(row["confirmation_idempotent"]) for row in multi_results
            ),
            "cases": multi_results,
        },
        "budget": budget.summary(),
        "infrastructure_error": infrastructure_error,
    }
    _write_json(attempt / "summary.json", summary)
    if infrastructure_error is not None:
        raise RuntimeError(infrastructure_error)
    return summary


def _rescore_summary(output: Path, attempt_number: int) -> dict:
    attempt = Path(output) / f"attempt-{int(attempt_number):02d}"
    source = attempt / "summary.json"
    summary = json.loads(source.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (attempt / "chat" / "cases.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    corrected = {
        **summary,
        "source_summary_sha256": _sha(source),
        "rescored_without_provider_calls": True,
        "chat": {
            **summary["chat"],
            "answer_plan_nodes": sum(
                "answer_plan" in _route_nodes(row) for row in rows
            ),
        },
    }
    _write_json(attempt / "summary-rescored.json", corrected)
    return corrected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the fixed real-app R3 optimization integration smoke"
    )
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--rescore", action="store_true")
    parser.add_argument("--allow-paid", action="store_true")
    parser.add_argument("--max-cost-usd", type=float, default=2.0)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--credential-env-file",
        type=Path,
        default=Path("/root/workplace/ShopHelper/.env"),
    )
    args = parser.parse_args(argv)
    if args.rescore:
        summary = _rescore_summary(Path(args.output), args.attempt)
        print(json.dumps({
            "mode": "optimization-r3-integration-smoke-rescore",
            "provider_calls": 0,
            "chat": summary["chat"],
            "multi_agent": summary["multi_agent"],
            "budget": summary["budget"],
        }, ensure_ascii=False))
        return 0
    if not args.run:
        parser.error("choose --run or --rescore")
    summary = asyncio.run(_run(args))
    print(json.dumps({
        "mode": "optimization-r3-integration-smoke",
        "chat": summary["chat"],
        "multi_agent": {
            "completed": summary["multi_agent"]["completed"],
            "preview_without_write": summary["multi_agent"]["preview_without_write"],
            "text_preview_without_artifact": summary["multi_agent"][
                "text_preview_without_artifact"
            ],
            "confirmation_idempotent": summary["multi_agent"][
                "confirmation_idempotent"
            ],
        },
        "budget": summary["budget"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
