"""Frozen claim-grounding experiment: v3.1 rules vs selective grounding."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.claim_grounding.backends import (
    CLAIM_GROUNDING_PROMPT,
    ProviderResponse,
    StructuredGroundingBackend,
)
from app.claim_grounding.budget import ClaimBudgetLedger
from app.claim_grounding.grounder import ClaimGrounder
from app.claim_grounding.models import ClaimInput, EvidenceCandidate
from app.claim_grounding.v31 import v31_rule_status

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "tests" / "eval" / "claim_grounding_r2"
RESULT_ROOT = DATA_ROOT / "results"
FROZEN_CONFIG = DATA_ROOT / "frozen-config.json"
PRICING = ROOT / "tests" / "eval" / "rag_ablation" / "pricing-snapshot-2026-10-26.yaml"
MODEL = "deepseek-flash"
SCORING_VERSION = "claim-grounding-r2-v1"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    names = ("dev.jsonl", "heldout.jsonl") if split == "all" else (f"{split}.jsonl",)
    for name in names:
        path = DATA_ROOT / "data" / name
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return rows


def to_claim(row: dict[str, Any]) -> ClaimInput:
    return ClaimInput(
        claim=row["claim"],
        subject=row.get("subject", ""),
        model_number=row.get("model_number", ""),
        order_id=row.get("order_id", ""),
        as_of_date=row.get("as_of_date"),
        policy_version=row.get("policy_version", ""),
        unit=row.get("unit", ""),
        evidence=[EvidenceCandidate(**item) for item in row["evidence"]],
    )


def freeze_config() -> dict[str, Any]:
    dev = load_rows("dev")
    heldout = load_rows("heldout")
    rows = dev + heldout
    payload = {
        "schema_version": 1,
        "freeze_id": "claim-grounding-r2-20261027",
        "scoring_version": SCORING_VERSION,
        "prompt_version": "claim-grounding-api-v1",
        "prompt_sha256": hashlib.sha256(
            CLAIM_GROUNDING_PROMPT.encode("utf-8")
        ).hexdigest(),
        "data_sha256": {
            "dev": _sha256(DATA_ROOT / "data" / "dev.jsonl"),
            "heldout": _sha256(DATA_ROOT / "data" / "heldout.jsonl"),
        },
        "pricing_sha256": _sha256(PRICING),
        "sizes": {"dev": len(dev), "heldout": len(heldout)},
        "categories": dict(sorted(Counter(row["category"] for row in rows).items())),
        "group_counts": {
            "dev": len({row["group_id"] for row in dev}),
            "heldout": len({row["group_id"] for row in heldout}),
        },
        "backend": {
            "selected": "structured_api",
            "model": MODEL,
            "local_nli_implemented": False,
            "local_candidate_documented": (
                "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"
            ),
        },
        "selection": "only v3.1 uncertain claims",
        "stop_rule": {
            "dev_tuning_allowed": True,
            "heldout_tuning_after_freeze": False,
            "official_heldout_runs": 1,
            "provider_retries_max": 2,
            "infrastructure_retry_must_use_new_run_id": True,
        },
        "budget": {
            "hard_limit_usd": 2.0,
            "ledger": "independent SQLite; atomic reservation and provider-usage settlement",
        },
    }
    _write_json(FROZEN_CONFIG, payload)
    return payload


def _verify_freeze() -> dict[str, Any]:
    if not FROZEN_CONFIG.exists():
        raise RuntimeError("请先运行 freeze 子命令")
    frozen = _read_json(FROZEN_CONFIG)
    expected = {
        "dev": _sha256(DATA_ROOT / "data" / "dev.jsonl"),
        "heldout": _sha256(DATA_ROOT / "data" / "heldout.jsonl"),
    }
    if frozen.get("data_sha256") != expected:
        raise RuntimeError("数据哈希与冻结配置不一致")
    if frozen.get("pricing_sha256") != _sha256(PRICING):
        raise RuntimeError("价格表哈希与冻结配置不一致")
    prompt_sha = hashlib.sha256(CLAIM_GROUNDING_PROMPT.encode("utf-8")).hexdigest()
    if frozen.get("prompt_sha256") != prompt_sha:
        raise RuntimeError("prompt 哈希与冻结配置不一致")
    return frozen


class SQLiteResponseCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            str(path), timeout=5, check_same_thread=False
        )
        self._connection.execute("PRAGMA busy_timeout = 5000")
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS responses "
            "(cache_key TEXT PRIMARY KEY, payload_json TEXT NOT NULL)"
        )
        self._connection.commit()
        self._lock = threading.RLock()

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload_json FROM responses WHERE cache_key = ?", (key,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self._connection.execute(
                "INSERT OR REPLACE INTO responses(cache_key, payload_json) VALUES (?, ?)",
                (key, json.dumps(payload, ensure_ascii=False, sort_keys=True)),
            )
            self._connection.commit()


def _load_credentials(path: Path) -> tuple[str, str]:
    from dotenv import dotenv_values

    values = dotenv_values(path)
    key = str(values.get("OPENAI_API_KEY") or "")
    base_url = str(values.get("OPENAI_BASE_URL") or "https://api.deepseek.com/v1")
    if not key:
        raise RuntimeError("缺少 OPENAI_API_KEY")
    return key, base_url


def make_api_caller(credential_env: Path):
    from openai import OpenAI

    api_key, base_url = _load_credentials(credential_env)
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=30.0,
        max_retries=0,
    )

    def call(prompt: str) -> ProviderResponse:
        started = time.perf_counter()
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "只依据用户提供的候选证据做 claim 判断，严格输出 JSON。",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        latency_ms = (time.perf_counter() - started) * 1000
        text = str(response.choices[0].message.content or "")
        usage = response.usage
        details = getattr(usage, "prompt_tokens_details", None)
        usage_payload = {
            "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "cached_input_tokens": int(getattr(details, "cached_tokens", 0) or 0),
        }
        raw = (
            response.model_dump(mode="json") if hasattr(response, "model_dump") else {}
        )
        return ProviderResponse(
            text=text,
            usage=usage_payload,
            model_version=str(getattr(response, "model", "") or ""),
            latency_ms=latency_ms,
            raw=raw,
        )

    return call


def _nearest_rank(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[max(0, math.ceil(len(values) * percentile) - 1)]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = ("supported", "unsupported", "uncertain")
    confusion = {gold: Counter() for gold in labels}
    for row in rows:
        confusion[row["gold"]][row["final_status"]] += 1
    per_label: dict[str, Any] = {}
    f1_values = []
    for label in labels:
        tp = confusion[label][label]
        predicted = sum(confusion[gold][label] for gold in labels)
        actual = sum(confusion[label].values())
        precision = tp / predicted if predicted else None
        recall = tp / actual if actual else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall
            else None
        )
        if f1 is not None:
            f1_values.append(f1)
        per_label[label] = {
            "tp": tp,
            "predicted_n": predicted,
            "gold_n": actual,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    n = len(rows)
    unsupported_rows = [row for row in rows if row["gold"] == "unsupported"]
    supported_rows = [row for row in rows if row["gold"] == "supported"]
    called = [row for row in rows if row.get("backend_called")]
    online = [
        row
        for row in called
        if not row.get("cache_hit") and row.get("latency_ms") is not None
    ]
    category_errors = Counter(
        row["category"] for row in rows if row["final_status"] != row["gold"]
    )
    entity_condition = {
        category: category_errors[category]
        for category in (
            "missing_condition",
            "cross_order",
            "date_boundary",
            "unit",
            "negation",
        )
    }
    total_input_tokens = sum(
        int((row.get("usage") or {}).get("input_tokens") or 0) for row in rows
    )
    total_output_tokens = sum(
        int((row.get("usage") or {}).get("output_tokens") or 0) for row in rows
    )
    total_cost = sum(float(row.get("cost_usd") or 0.0) for row in rows)
    return {
        "scoring_version": SCORING_VERSION,
        "n": n,
        "accuracy": sum(row["final_status"] == row["gold"] for row in rows) / n
        if n
        else None,
        "confusion": {gold: dict(confusion[gold]) for gold in labels},
        "per_label": per_label,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else None,
        "supported_precision": {
            **per_label["supported"],
            "denominator": per_label["supported"]["predicted_n"],
        },
        "uncertain": {
            "predicted": confusion["supported"]["uncertain"]
            + confusion["unsupported"]["uncertain"]
            + confusion["uncertain"]["uncertain"],
            "rate": sum(row["final_status"] == "uncertain" for row in rows) / n
            if n
            else None,
            "determinate_coverage": sum(
                row["final_status"] != "uncertain" for row in rows
            )
            / n
            if n
            else None,
        },
        "contradiction_misses": {
            "gold_unsupported_n": len(unsupported_rows),
            "unsupported_to_supported": sum(
                row["final_status"] == "supported" for row in unsupported_rows
            ),
            "unsupported_to_uncertain": sum(
                row["final_status"] == "uncertain" for row in unsupported_rows
            ),
            "deterministic_conflict_recall": (
                sum(row.get("hard_status") == "unsupported" for row in unsupported_rows)
                / len(unsupported_rows)
                if unsupported_rows
                else None
            ),
        },
        "supported": {
            "gold_n": len(supported_rows),
            "correct": sum(
                row["final_status"] == "supported" for row in supported_rows
            ),
        },
        "category_errors": dict(sorted(category_errors.items())),
        "entity_condition_errors": entity_condition,
        "calls": {
            "backend_called": len(called),
            "call_ratio": len(called) / n if n else None,
            "cache_hits": sum(bool(row.get("cache_hit")) for row in rows),
            "online_timed_calls": len(online),
            "hidden_calls": sum(
                int(row.get("attempts") or 0)
                for row in rows
                if not row.get("cache_hit")
            ),
        },
        "tokens": {
            "input": total_input_tokens,
            "output": total_output_tokens,
            "total": total_input_tokens + total_output_tokens,
        },
        "cost_usd": total_cost,
        "latency_ms_online": {
            "n": len(online),
            "p50": _nearest_rank([float(row["latency_ms"]) for row in online], 0.50),
            "p95": _nearest_rank([float(row["latency_ms"]) for row in online], 0.95),
        },
        "latency_scope": (
            "provider request wall time only for cache-miss calls; "
            "not end-to-end chat latency"
        ),
    }


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def derive_comparison(results: list[dict[str, Any]]) -> dict[str, Any]:
    def non_semantic(row: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in row.items()
            if key
            not in {
                "backend_called",
                "cache_hit",
                "latency_ms",
                "attempts",
                "usage",
                "cost_usd",
            }
        }

    baseline_rows = [
        {**non_semantic(row), "final_status": row["v31_status"]} for row in results
    ]
    hard_rows = [
        {**non_semantic(row), "final_status": row["hard_status"]} for row in results
    ]
    return {
        "v3_1_rules": summarize(baseline_rows),
        "hard_guard_only": summarize(hard_rows),
        "selective_grounding": summarize(results),
    }


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    frozen = _verify_freeze()
    if args.split == "heldout" and args.mode == "rules":
        raise RuntimeError("held-out 只进入冻结后的 selective 完整对照")

    run_dir = RESULT_ROOT / args.run_id
    if run_dir.exists() and not args.resume:
        raise RuntimeError(f"结果目录已存在，拒绝覆盖:{run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.split)

    backend = None
    ledger = None
    if args.mode == "selective":
        if not args.allow_paid:
            raise RuntimeError("selective 真模型运行需要显式 --allow-paid")
        if not args.credential_env:
            raise RuntimeError("缺少 --credential-env")
        max_cost = min(float(args.max_cost_usd), 2.0)
        if max_cost <= 0:
            raise RuntimeError("max_cost_usd 必须大于 0")
        import yaml

        pricing_payload = yaml.safe_load(PRICING.read_text(encoding="utf-8"))
        model_pricing = pricing_payload["models"][MODEL]
        if not pricing_payload.get("verified"):
            raise RuntimeError("价格表未核实")
        ledger = ClaimBudgetLedger(
            run_dir / "budget.sqlite",
            total_limit_usd=max_cost,
        )
        cache = SQLiteResponseCache(DATA_ROOT / "cache" / "exact-responses.sqlite")
        backend = StructuredGroundingBackend(
            caller=make_api_caller(Path(args.credential_env)),
            ledger=ledger,
            pricing=model_pricing,
            model=MODEL,
            reserve_usd=0.005,
            max_retries=2,
            retry_sleep_s=0.5,
            cache=cache,
            event_sink=lambda payload: _append_jsonl(
                run_dir / "attempts.jsonl", payload
            ),
        )
    grounder = ClaimGrounder(backend=backend, only_uncertain=True)
    results: list[dict[str, Any]] = []
    started_at = datetime.now(timezone.utc).isoformat()
    for index, row in enumerate(rows, 1):
        claim = to_claim(row)
        baseline = v31_rule_status(claim)
        hard = ClaimGrounder(backend=None).ground(claim)
        if args.mode == "rules":
            grounded = hard.model_copy(
                update={
                    "status": baseline,
                    "reason": "rag-quality-v3.1 rule baseline",
                    "backend": "v3.1",
                    "baseline_status": baseline,
                }
            )
        else:
            grounded = grounder.ground(claim)
        result = {
            "id": row["id"],
            "split": row["split"],
            "group_id": row["group_id"],
            "category": row["category"],
            "source": row["source"],
            "claim": row["claim"],
            "claim_input": claim.model_dump(mode="json"),
            "gold": row["gold"],
            "gold_reason": row["gold_reason"],
            "gold_review_status": row["gold_review_status"],
            "v31_status": baseline,
            "hard_status": hard.status,
            "hard_reason": hard.reason,
            "final_status": grounded.status,
            "final_reason": grounded.reason,
            "final_evidence_ids": grounded.evidence_ids,
            "backend": grounded.backend,
            "model_version": grounded.model_version,
            "backend_called": grounded.backend_called,
            "cache_hit": grounded.cache_hit,
            "latency_ms": grounded.latency_ms,
            "attempts": grounded.attempts,
            "usage": grounded.usage.model_dump(mode="json"),
            "cost_usd": (
                sum(
                    float(item.get("actual_usd") or 0.0)
                    for item in ledger.records()
                    if str(item.get("note", "")).find(
                        hashlib.sha256(
                            backend.build_prompt(claim).encode("utf-8")
                        ).hexdigest()
                    )
                    >= 0
                )
                if ledger is not None
                else 0.0
            ),
        }
        results.append(result)
        print(
            f"[{index}/{len(rows)}] {row['id']} "
            f"gold={row['gold']} v31={baseline} final={grounded.status} "
            f"backend={grounded.backend_called}",
            flush=True,
        )

    aggregate = summarize(results)
    aggregate["mode"] = args.mode
    aggregate["split"] = args.split
    aggregate["freeze_id"] = frozen["freeze_id"]
    aggregate["data_sha256"] = frozen["data_sha256"]
    aggregate["started_at"] = started_at
    aggregate["finished_at"] = datetime.now(timezone.utc).isoformat()
    aggregate["cost_usd"] = ledger.total_usd() if ledger else 0.0
    aggregate["budget"] = {
        "limit_usd": ledger.total_limit_usd if ledger else 0.0,
        "total_usd": ledger.total_usd() if ledger else 0.0,
        "records": len(ledger.records()) if ledger else 0,
    }
    _write_jsonl(run_dir / "results.jsonl", results)
    _write_json(run_dir / "aggregate.json", aggregate)
    _write_json(run_dir / "comparison.json", derive_comparison(results))
    _write_json(
        run_dir / "manifest.json",
        {
            "freeze": frozen,
            "mode": args.mode,
            "split": args.split,
            "credentials_persisted": False,
            "backend": MODEL if args.mode == "selective" else "rules-only",
        },
    )
    if ledger is not None:
        _write_json(
            run_dir / "budget-records.json",
            {"records": ledger.records(), "total_usd": ledger.total_usd()},
        )
        if args.measure_cache_replay and backend is not None:
            replay_times = []
            replay_calls = 0
            for row in rows:
                claim = to_claim(row)
                started = time.perf_counter()
                replay = grounder.ground(claim)
                elapsed = (time.perf_counter() - started) * 1000
                if replay.cache_hit:
                    replay_times.append(elapsed)
                    replay_calls += 1
            _write_json(
                run_dir / "cache-replay.json",
                {
                    "hits": replay_calls,
                    "latency_ms": {
                        "p50": _nearest_rank(replay_times, 0.50),
                        "p95": _nearest_rank(replay_times, 0.95),
                    },
                    "scope": (
                        "exact-cache replay of previously eligible claims; "
                        "not online provider latency"
                    ),
                },
            )
    return aggregate


def rescore(run_id: str) -> dict[str, Any]:
    run_dir = RESULT_ROOT / run_id
    results = [
        json.loads(line)
        for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    comparison = derive_comparison(results)
    _write_json(run_dir / "derived-comparison.json", comparison)
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze")
    run = sub.add_parser("run")
    run.add_argument("--mode", choices=("rules", "selective"), required=True)
    run.add_argument("--split", choices=("dev", "heldout", "all"), required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument("--allow-paid", action="store_true")
    run.add_argument("--credential-env", type=Path)
    run.add_argument("--max-cost-usd", type=float, default=2.0)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--measure-cache-replay", action="store_true")
    score = sub.add_parser("rescore")
    score.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        payload = freeze_config()
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "run":
        aggregate = run_evaluation(args)
        print(json.dumps(aggregate, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        comparison = rescore(args.run_id)
        print(json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
