"""Reusable contracts for the answer-depth runner and future CLI."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from scripts.answer_quality_depth.budget import (
    AnswerQualityLedger,
    CappedProvider,
)
from scripts.answer_quality_depth.data import (
    ARM_BASELINE,
    ARM_SLATE,
    GENERATION_SYSTEM_PROMPT,
    GENERATION_USER_TEMPLATE,
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_TEMPLATE,
    build_generation_messages,
    build_judge_prompt,
    case_projection,
    dataset_hashes,
    load_depth_cases,
    prompt_sha256,
)
from scripts.answer_quality_depth.evidence import (
    configure_retrieval_environment,
    evidence_rows_from_ids,
    load_chunk_rows,
    load_saved_dev_reranks,
    select_from_saved_rerank,
)
from scripts.answer_quality_depth.scoring import (
    OVERLAY_PATH,
    apply_semantic_status,
    load_scoring_overlay,
    score_response,
    summarize_scores,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data" / "answer-quality-depth"
DEFAULT_SQLITE = DATA_ROOT / "ch12_eval.db"
DEFAULT_SNAPSHOT = (
    DATA_ROOT
    / "eval-snapshots"
    / "ebfd49478b37ef387c8b124319a450ed73a86ae2a4b9f4fc28d6a52b2ccaa0ba.db"
)
DEFAULT_CACHE_DIR = DATA_ROOT / "eval-caches"
DEFAULT_DEV_RERANK = DATA_ROOT / "retrieval-analysis" / "rerank50-dev.json"
DEFAULT_PRICING = (
    ROOT / "tests" / "eval" / "rag_ablation" / "pricing-snapshot-2026-10-26.yaml"
)
DEFAULT_RUNS = ROOT / "tests" / "eval" / "answer_quality_depth" / "runs"
CODE_FILES = (
    "scripts/answer_quality_depth/budget.py",
    "scripts/answer_quality_depth/data.py",
    "scripts/answer_quality_depth/evidence.py",
    "scripts/answer_quality_depth/runner.py",
    "scripts/answer_quality_depth/scoring.py",
    "scripts/eval_answer_quality_depth.py",
    "app/evidence_slate.py",
    "app/retrieval.py",
    "app/config.py",
)


class ResponseJournal:
    """Durable attempt journal owned by this experiment only."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self.path, timeout=5, check_same_thread=False
        )
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS responses "
            "(kind TEXT NOT NULL, request_key TEXT NOT NULL, payload_json TEXT NOT NULL, "
            "PRIMARY KEY(kind, request_key))"
        )
        self._connection.commit()

    def get(self, kind: str, request_key: str) -> dict | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload_json FROM responses WHERE kind = ? AND request_key = ?",
                (kind, request_key),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, kind: str, request_key: str, payload: dict) -> None:
        with self._lock:
            self._connection.execute(
                "INSERT OR REPLACE INTO responses(kind, request_key, payload_json) "
                "VALUES (?, ?, ?)",
                (
                    kind,
                    request_key,
                    json.dumps(payload, ensure_ascii=False, sort_keys=True),
                ),
            )
            self._connection.commit()

    def count(self) -> int:
        with self._lock:
            return int(
                self._connection.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
            )


def evidence_projection(
    *, case: dict, baseline: list[dict], slate: list[dict], top_k: int
) -> dict:
    projection = case_projection(case)
    return {
        "case": projection,
        "arms": {
            "baseline": [dict(item) for item in baseline[:top_k]],
            "slate": [dict(item) for item in slate[:top_k]],
        },
    }


def run_config_hash(config: dict) -> str:
    return hashlib.sha256(
        json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def request_key(
    *,
    phase: str,
    model: str,
    prompt_sha256: str,
    case_id: str,
    arm: str,
    evidence: list[dict],
    variant: str = "",
) -> str:
    payload = {
        "phase": phase,
        "model": model,
        "prompt_sha256": prompt_sha256,
        "case_id": case_id,
        "arm": arm,
        "variant": variant,
        "evidence": [
            {
                key: item.get(key)
                for key in ("chunk_id", "section_path", "text", "score")
                if key in item
            }
            for item in evidence
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


_JUDGE_KEYS = {
    "disposition",
    "answer_status",
    "grounding",
    "completeness",
    "citation_support",
    "unsupported_claims",
    "missing_requirements",
    "rationale",
}


def parse_judge_payload(raw: str) -> dict:
    text = str(raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("judge response has no JSON object")
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise TypeError("judge response must be a JSON object")
    extra = set(payload) - _JUDGE_KEYS - {"type"}
    missing = _JUDGE_KEYS - set(payload)
    if extra or missing:
        raise ValueError(
            f"judge response schema mismatch: extra={sorted(extra)}, "
            f"missing={sorted(missing)}"
        )
    payload.pop("type", None)
    choices = {
        "disposition": {"answer", "clarify", "refuse"},
        "answer_status": {"correct", "partial", "incorrect", "not_answerable"},
        "grounding": {"supported", "partly_supported", "unsupported", "not_applicable"},
        "citation_support": {"supported", "partial", "unsupported", "not_applicable"},
    }
    for key, allowed in choices.items():
        if payload[key] not in allowed:
            raise ValueError(f"invalid judge {key}: {payload[key]!r}")
    completeness = payload["completeness"]
    if (
        isinstance(completeness, bool)
        or not isinstance(completeness, int)
        or not 0 <= completeness <= 100
    ):
        raise ValueError("judge completeness must be an integer in [0, 100]")
    for key in ("unsupported_claims", "missing_requirements"):
        value = payload[key]
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            raise ValueError(f"judge {key} must be a string list")
    if not isinstance(payload["rationale"], str):
        raise TypeError("judge rationale must be a string")
    return payload


def judge_response_text(content: str, raw: dict) -> str:
    """Recover a terminal JSON object when a reasoning model leaves content empty."""
    text = str(content or "").strip()
    if text:
        return text
    choices = (raw or {}).get("choices") or []
    if choices:
        message = (choices[0] or {}).get("message") or {}
        reasoning = str(message.get("reasoning_content") or "").strip()
        if reasoning:
            matches = re.findall(r"\{.*?\}", reasoning, flags=re.DOTALL)
            return matches[-1] if matches else reasoning
    return ""


def generation_was_truncated(row: dict) -> bool:
    choices = ((row.get("generation") or {}).get("raw") or {}).get("choices") or []
    return bool(choices and choices[0].get("finish_reason") == "length")


def _write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _read_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _file_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _evidence_hit(case: dict, evidence: list[dict]) -> bool | None:
    groups = case.get("required_evidence") or []
    if not groups:
        return None
    texts = "\n".join(str(item.get("text") or "") for item in evidence)
    return all(
        all(str(quote) in texts for quote in group.get("quotes") or [])
        for group in groups
    )


def _row_projection(row: Any) -> dict:
    return {
        "chunk_id": int(row.id),
        "section_path": str(row.section_path or ""),
        "text": f"{row.questions or ''!s}\n{row.answer or ''!s}".strip(),
    }


def _retrieval_payload(result, case: dict, rows: dict[int, dict], top_k: int) -> dict:
    baseline = evidence_rows_from_ids(
        [item.chunk_id for item in result.rerank_candidates],
        [item.score for item in result.rerank_candidates],
        rows,
        top_k=top_k,
    )
    slate = evidence_rows_from_ids(
        [item.chunk_id for item in result.items],
        [item.score for item in result.items],
        rows,
        top_k=top_k,
    )
    # Keep context length identical even if an implementation unexpectedly
    # returns fewer than top_k items.
    common = min(len(baseline), len(slate), top_k)
    baseline = baseline[:common]
    slate = slate[:common]
    return {
        "case_id": str(case["id"]),
        "split": str(case["split"]),
        "query": str(case["query"]),
        "baseline": baseline,
        "slate": slate,
        "context_k": common,
        "selection_facets": list(getattr(result, "selection_facets", []) or []),
        "rerank_pool_k": len(result.rerank_candidates),
        "degraded": bool(getattr(result, "degraded", False)),
        "low_confidence": bool(getattr(result, "low_confidence", False)),
        "baseline_hit": _evidence_hit(case, baseline),
        "slate_hit": _evidence_hit(case, slate),
        "baseline_top5_hit": _evidence_hit(case, baseline[:5]),
        "slate_top5_hit": _evidence_hit(case, slate[:5]),
    }


def prepare_evidence(
    split: str,
    *,
    output_path: Path,
    top_k: int = 10,
    credential_env_file: Path,
    sqlite_path: Path = DEFAULT_SQLITE,
    snapshot_path: Path = DEFAULT_SNAPSHOT,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    dev_rerank_path: Path = DEFAULT_DEV_RERANK,
    pricing_path: Path = DEFAULT_PRICING,
    retrieval_ledger_path: Path | None = None,
) -> dict:
    from scripts.ch11_harness import HarnessConfig
    from scripts.eval_ch11 import _build_context

    cases = load_depth_cases()[split]
    rows = load_chunk_rows(sqlite_path)
    if split == "dev" and Path(dev_rerank_path).exists():
        artifacts = load_saved_dev_reranks(dev_rerank_path)
        output = []
        for case in cases:
            artifact = artifacts.get(str(case["id"]))
            if artifact is None:
                raise RuntimeError(f"missing saved dev rerank: {case['id']}")
            selected = select_from_saved_rerank(
                case_id=str(case["id"]),
                query=str(case["query"]),
                artifact=artifact,
                rows=rows,
                top_k=top_k,
            )
            output.append(
                {
                    "case_id": str(case["id"]),
                    "split": split,
                    "query": str(case["query"]),
                    **selected,
                    "context_k": min(len(selected["baseline"]), len(selected["slate"])),
                    "baseline_hit": _evidence_hit(case, selected["baseline"]),
                    "slate_hit": _evidence_hit(case, selected["slate"]),
                    "baseline_top5_hit": _evidence_hit(case, selected["baseline"][:5]),
                    "slate_top5_hit": _evidence_hit(case, selected["slate"][:5]),
                    "degraded": False,
                    "low_confidence": False,
                }
            )
        source = "saved-50-candidate-rerank"
    else:
        configure_retrieval_environment(
            credential_env_file=credential_env_file,
            sqlite_db=sqlite_path,
            snapshot=snapshot_path,
        )
        config = HarnessConfig(
            split=split,
            mode="retrieval",
            strategy="hybrid_rerank",
            max_cost_usd=2.0,
            allow_paid=True,
            rag_mode="baseline",
            temperature=0.0,
        )
        ledger_path = retrieval_ledger_path or (
            Path(output_path).parent / "retrieval-budget.json"
        )
        context_args = SimpleNamespace(
            snapshot=snapshot_path,
            cache_dir=cache_dir,
            pricing=pricing_path,
            ledger=ledger_path,
            allow_paid=True,
            no_rewrite=False,
            rewrite_cache=None,
            rewrite_cache_mode="read",
        )
        ctx, _cfg, _contract = _build_context(config, context_args, cases)
        db_rows = {
            chunk_id: _row_projection(row) for chunk_id, row in ctx.all_rows.items()
        }
        output = []
        for case in cases:
            result = ctx.service.retrieve(
                str(case["query"]),
                strategy="hybrid_rerank",
                prepared=ctx.prepared.get(str(case["id"])),
            )
            output.append(_retrieval_payload(result, case, db_rows, top_k))
        source = "replayed-unchanged-hybrid-rerank"
    payload = {
        "schema_version": 1,
        "experiment": "answer_quality_depth",
        "selection_version": "evidence-slate-v1",
        "split": split,
        "top_k": top_k,
        "source": source,
        "dataset_sha256": dataset_hashes()[split],
        "code_sha256": {
            path: _file_sha(ROOT / path)
            for path in (
                "app/evidence_slate.py",
                "app/retrieval.py",
                "app/config.py",
            )
        },
        "rows": output,
    }
    _write_json(output_path, payload)
    return {
        "split": split,
        "n": len(output),
        "top_k": top_k,
        "source": source,
        "baseline_hits": sum(bool(row["baseline_hit"]) for row in output),
        "slate_hits": sum(bool(row["slate_hit"]) for row in output),
        "eligible": sum(row["baseline_hit"] is not None for row in output),
        "output": str(output_path),
    }


def _load_credentials(path: Path) -> tuple[str, str]:
    from dotenv import dotenv_values

    values = dotenv_values(path)
    api_key = str(
        values.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    )
    base_url = str(
        values.get("OPENAI_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or "https://api.deepseek.com/v1"
    )
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing; refusing paid run")
    return api_key, base_url


def _openai_client(credentials_path: Path):
    from openai import OpenAI

    api_key, base_url = _load_credentials(credentials_path)
    return OpenAI(api_key=api_key, base_url=base_url, timeout=45.0, max_retries=0)


def _pricing(path: Path) -> tuple[dict[str, Any], dict[str, float]]:
    import yaml

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not payload.get("verified"):
        raise RuntimeError("pricing snapshot is not verified")
    model = payload.get("models", {}).get("deepseek-flash")
    if not isinstance(model, dict):
        raise TypeError("pricing snapshot has no deepseek-flash entry")
    return payload, model


def _estimate_prompts(
    cases: list[dict], evidence_rows: dict[str, dict], top_k: int
) -> dict:
    from app.history import estimate_tokens

    generation = 0
    judge = 0
    per_case = []
    for case in cases:
        evidence = evidence_rows.get(str(case["id"]))
        if evidence is None:
            continue
        arm_tokens = []
        for arm_name, arm, arm_constant in (
            ("baseline", "baseline", ARM_BASELINE),
            ("slate", "slate", ARM_SLATE),
        ):
            messages = build_generation_messages(
                case, evidence[arm][:top_k], arm=arm_constant
            )
            generation += sum(
                estimate_tokens(message["content"]) for message in messages
            )
            judge_prompt = build_judge_prompt(case, evidence[arm][:top_k], "答" * 600)
            judge += estimate_tokens(JUDGE_SYSTEM_PROMPT + judge_prompt)
            arm_tokens.append(
                {
                    "arm": arm,
                    "generation_input_tokens": sum(
                        estimate_tokens(message["content"]) for message in messages
                    ),
                    "judge_input_tokens": estimate_tokens(
                        JUDGE_SYSTEM_PROMPT + judge_prompt
                    ),
                }
            )
        per_case.append({"case_id": case["id"], "arms": arm_tokens})
    return {
        "cases": len(per_case),
        "calls": len(per_case) * 4,
        "generation_input_tokens": generation,
        "judge_input_tokens": judge,
        "per_case": per_case,
    }


def estimate_run(
    split: str,
    *,
    evidence_path: Path,
    top_k: int = 10,
    pricing_path: Path = DEFAULT_PRICING,
) -> dict:
    cases = load_depth_cases()[split]
    evidence_payload = json.loads(Path(evidence_path).read_text(encoding="utf-8"))
    evidence_rows = {
        str(item["case_id"]): item for item in evidence_payload.get("rows") or []
    }
    estimates = _estimate_prompts(cases, evidence_rows, top_k)
    _pricing_payload, model_price = _pricing(pricing_path)

    def cost(input_tokens: int, output_tokens: int) -> float:
        return input_tokens / 1_000_000 * float(
            model_price.get("input_per_million") or 0
        ) + output_tokens / 1_000_000 * float(
            model_price.get("output_per_million") or 0
        )

    generation = cost(
        estimates["generation_input_tokens"], 2400 * estimates["cases"] * 2
    )
    judge = cost(estimates["judge_input_tokens"], 1600 * estimates["cases"] * 2)
    return {
        "split": split,
        "model": "deepseek-flash",
        "calls": estimates["calls"],
        "max_output_tokens": {"answer": 2400, "judge": 1600},
        "estimated_input_tokens": {
            "answer": estimates["generation_input_tokens"],
            "judge": estimates["judge_input_tokens"],
        },
        "estimated_cost_usd": round(generation + judge, 6),
        "hard_limit_usd": 2.0,
        "note": "Conservative no-cache estimate; actual provider usage is settled per call.",
    }


def _prompt_hashes() -> dict[str, str]:
    return {
        "generation_system": prompt_sha256(
            GENERATION_SYSTEM_PROMPT + GENERATION_USER_TEMPLATE
        ),
        "judge_system": prompt_sha256(JUDGE_SYSTEM_PROMPT + JUDGE_USER_TEMPLATE),
    }


def code_hashes() -> dict[str, str]:
    return {path: _file_sha(ROOT / path) for path in CODE_FILES}


def _run_config(
    split: str,
    *,
    evidence_path: Path,
    top_k: int,
    pricing_path: Path,
    max_cost_usd: float,
) -> dict:
    return {
        "experiment": "answer_quality_depth_v1",
        "split": split,
        "model": "deepseek-flash",
        "temperature": 0,
        "generation_thinking": "enabled",
        "judge_thinking": "disabled",
        "answer_max_tokens": 2400,
        "judge_max_tokens": 1600,
        "scoring_version": "answer-quality-depth-score-v7",
        "top_k": int(top_k),
        "generation_prompt_sha256": _prompt_hashes()["generation_system"],
        "judge_prompt_sha256": _prompt_hashes()["judge_system"],
        "scoring_overlay_sha256": _file_sha(OVERLAY_PATH),
        "code_sha256": code_hashes(),
        "evidence_sha256": _file_sha(evidence_path),
        "pricing_sha256": _file_sha(pricing_path),
        "max_cost_usd": float(max_cost_usd),
        "max_retries": 2,
    }


def _call_with_json_validation(
    provider: CappedProvider,
    *,
    messages: list[dict[str, str]],
    kind: str,
    max_tokens: int,
    note: str,
    validator,
    extra_body: dict | None = None,
):
    last_error: Exception | None = None
    for validation_attempt in range(2):
        result = provider.call(
            messages,
            kind=kind,
            max_tokens=max_tokens,
            temperature=0,
            response_format={"type": "json_object"},
            extra_body=extra_body,
            note=f"{note}; validation_attempt={validation_attempt + 1}",
        )
        candidate_text = (
            judge_response_text(result.text, result.raw)
            if kind == "judge"
            else result.text
        )
        try:
            payload = validator(candidate_text)
            if candidate_text != result.text:
                from dataclasses import replace

                result = replace(result, text=candidate_text)
            return result, payload
        except (TypeError, ValueError) as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def run_experiment(
    split: str,
    *,
    evidence_path: Path,
    run_dir: Path,
    credentials_path: Path,
    ledger_path: Path,
    pricing_path: Path = DEFAULT_PRICING,
    top_k: int = 10,
    max_cost_usd: float = 2.0,
    allow_paid: bool = False,
    frozen_config: dict | None = None,
) -> dict:
    if not allow_paid:
        raise RuntimeError("refusing model calls without --allow-paid")
    if not 0 < float(max_cost_usd) <= 2.0:
        raise RuntimeError("max_cost_usd must be in (0, 2.0]")
    cases = load_depth_cases()[split]
    evidence_payload = json.loads(Path(evidence_path).read_text(encoding="utf-8"))
    evidence_rows = {
        str(item["case_id"]): item for item in evidence_payload.get("rows") or []
    }
    config = _run_config(
        split,
        evidence_path=evidence_path,
        top_k=top_k,
        pricing_path=pricing_path,
        max_cost_usd=max_cost_usd,
    )
    if frozen_config is not None:
        expected = frozen_config.get("run_config") or {}
        for key in (
            "model",
            "temperature",
            "generation_thinking",
            "judge_thinking",
            "answer_max_tokens",
            "judge_max_tokens",
            "top_k",
            "generation_prompt_sha256",
            "judge_prompt_sha256",
            "scoring_overlay_sha256",
            "scoring_version",
            "code_sha256",
            "pricing_sha256",
            "max_cost_usd",
            "max_retries",
        ):
            if config.get(key) != expected.get(key):
                raise RuntimeError(f"frozen config mismatch: {key}")

    run_dir = Path(run_dir)
    state_path = run_dir / "run-state.json"
    if run_dir.exists():
        if not state_path.exists():
            raise RuntimeError(f"run directory exists without state: {run_dir}")
        existing_state = json.loads(state_path.read_text(encoding="utf-8"))
        if existing_state.get("config_sha256") != run_config_hash(config):
            raise RuntimeError("existing run directory belongs to a different config")
        if existing_state.get("status") == "completed":
            raise RuntimeError(
                "run directory is already complete; refusing duplicate calls"
            )
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
    journal = ResponseJournal(run_dir / "responses.sqlite3")
    _pricing_payload, model_price = _pricing(pricing_path)
    ledger = AnswerQualityLedger(ledger_path, total_limit_usd=max_cost_usd)
    client = _openai_client(credentials_path)
    provider = CappedProvider(
        client,
        ledger=ledger,
        pricing=model_price,
        model="deepseek-flash",
        max_retries=2,
        retry_backoff_seconds=0.5,
    )
    overlay = load_scoring_overlay()
    results_path = run_dir / "results.jsonl"
    results: list[dict] = _read_jsonl(results_path)
    completed_rows = {(str(row.get("case_id")), str(row.get("arm"))) for row in results}
    run_state = {
        **config,
        "config_sha256": run_config_hash(config),
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_cases": 0,
    }
    if not state_path.exists():
        _write_json(state_path, run_state)
    else:
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        run_state["started_at"] = previous.get("started_at") or run_state["started_at"]
        _write_json(state_path, run_state)

    for case in cases:
        evidence = evidence_rows.get(str(case["id"]))
        if evidence is None:
            raise RuntimeError(f"missing evidence row for {case['id']}")
        for arm_label, arm_key, arm_constant in (
            ("baseline", "baseline", ARM_BASELINE),
            ("slate", "slate", ARM_SLATE),
        ):
            if (str(case["id"]), arm_label) in completed_rows:
                continue
            arm_evidence = evidence[arm_key][:top_k]
            messages = build_generation_messages(case, arm_evidence, arm=arm_constant)
            answer_key = request_key(
                phase="answer",
                model="deepseek-flash",
                prompt_sha256=_prompt_hashes()["generation_system"],
                case_id=str(case["id"]),
                arm=arm_label,
                evidence=arm_evidence,
            )
            answer_record = journal.get("answer", answer_key)
            generation_cost = 0.0
            generation_latency = 0.0
            generation_attempts = 0
            technical_failure = None
            if answer_record is None:
                try:
                    generated = provider.call(
                        messages,
                        kind="answer",
                        max_tokens=2400,
                        temperature=0,
                        note=f"{case['id']}:{arm_label}",
                    )
                    answer_record = {
                        "answer": generated.text.strip(),
                        "usage": generated.usage,
                        "raw": generated.raw,
                        "model_version": generated.model_version,
                        "latency_ms": generated.latency_ms,
                        "attempts": generated.attempts,
                        "cost_usd": generated.cost_usd,
                    }
                    journal.put("answer", answer_key, answer_record)
                except Exception as exc:  # noqa: BLE001 - preserve provider failure
                    technical_failure = f"{type(exc).__name__}: {exc}"[:500]
                    answer_record = {
                        "answer": "",
                        "usage": {},
                        "raw": {},
                        "model_version": "",
                        "latency_ms": 0.0,
                        "attempts": 0,
                        "cost_usd": 0.0,
                        "error": technical_failure,
                    }
            generation_cost = float(answer_record.get("cost_usd") or 0.0)
            generation_latency = float(answer_record.get("latency_ms") or 0.0)
            generation_attempts = int(answer_record.get("attempts") or 0)

            judge_prompt = build_judge_prompt(
                case, arm_evidence, answer_record.get("answer") or ""
            )
            judge_key = request_key(
                phase="judge",
                model="deepseek-flash",
                prompt_sha256=_prompt_hashes()["judge_system"],
                case_id=str(case["id"]),
                arm=arm_label,
                evidence=arm_evidence,
                variant="non_thinking",
            )
            judge_record = journal.get("judge", judge_key)
            judge_cost = 0.0
            judge_latency = 0.0
            judge_attempts = 0
            if technical_failure is None and judge_record is None:
                try:
                    judged, judged_payload = _call_with_json_validation(
                        provider,
                        messages=[
                            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                            {"role": "user", "content": judge_prompt},
                        ],
                        kind="judge",
                        max_tokens=1600,
                        note=f"{case['id']}:{arm_label}",
                        validator=parse_judge_payload,
                        extra_body={"thinking": {"type": "disabled"}},
                    )
                    judge_record = {
                        "verdict": judged_payload,
                        "raw_text": judged.text,
                        "raw": judged.raw,
                        "usage": judged.usage,
                        "model_version": judged.model_version,
                        "latency_ms": judged.latency_ms,
                        "attempts": judged.attempts,
                        "cost_usd": judged.cost_usd,
                    }
                    journal.put("judge", judge_key, judge_record)
                except Exception as exc:  # noqa: BLE001 - preserve provider failure
                    technical_failure = (
                        f"{technical_failure + '; ' if technical_failure else ''}"
                        f"judge {type(exc).__name__}: {exc}"
                    )[:500]
            if judge_record is not None:
                judge_cost = float(judge_record.get("cost_usd") or 0.0)
                judge_latency = float(judge_record.get("latency_ms") or 0.0)
                judge_attempts = int(judge_record.get("attempts") or 0)

            score = score_response(
                case,
                answer_record.get("answer") or "",
                arm_evidence,
                overlay=overlay,
                technical_failure=technical_failure,
            )
            score = apply_semantic_status(
                score,
                (judge_record or {}).get("verdict"),
            )
            row = {
                "schema_version": 1,
                "split": split,
                "case_id": str(case["id"]),
                "bucket": case.get("bucket"),
                "expected_behavior": case.get("expected_behavior"),
                "arm": arm_label,
                "query": case["query"],
                "evidence": arm_evidence,
                "answer": answer_record.get("answer") or "",
                "generation": {
                    "usage": answer_record.get("usage") or {},
                    "model_version": answer_record.get("model_version") or "",
                    "latency_ms": generation_latency,
                    "attempts": generation_attempts,
                    "cost_usd": generation_cost,
                    "raw": answer_record.get("raw") or {},
                },
                "judge": {
                    "verdict": (judge_record or {}).get("verdict"),
                    "raw_text": (judge_record or {}).get("raw_text", ""),
                    "usage": (judge_record or {}).get("usage") or {},
                    "model_version": (judge_record or {}).get("model_version", ""),
                    "latency_ms": judge_latency,
                    "attempts": judge_attempts,
                    "cost_usd": judge_cost,
                    "raw": (judge_record or {}).get("raw") or {},
                },
                "score": score,
                **score,
                "cost_usd": round(generation_cost + judge_cost, 8),
                "latency_ms": round(generation_latency + judge_latency, 3),
                "technical_failure": technical_failure,
            }
            results.append(row)
            completed_rows.add((str(case["id"]), arm_label))
            with results_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            print(
                f"[{len(results):03d}/{len(cases) * 2:03d}] "
                f"{case['id']} {arm_label} deterministic={score['status']} "
                f"judge={(judge_record or {}).get('verdict', {}).get('answer_status', 'n/a')} "
                f"cost=${generation_cost + judge_cost:.5f}",
                flush=True,
            )
        run_state["completed_cases"] += 1
        _write_json(state_path, run_state)

    _write_jsonl(results_path, results)
    summary = summarize_scores(results)
    summary.update(
        {
            "split": split,
            "config_sha256": run_config_hash(config),
            "journal_count": journal.count(),
            "ledger_total_usd": round(ledger.total_usd(), 8),
            "technical_failures": sum(
                bool(row.get("technical_failure")) for row in results
            ),
            "judge_disposition": {
                status: sum(
                    ((row.get("judge") or {}).get("verdict") or {}).get("disposition")
                    == status
                    for row in results
                )
                for status in ("answer", "clarify", "refuse")
            },
            "judge_answer_status": {
                status: sum(
                    ((row.get("judge") or {}).get("verdict") or {}).get("answer_status")
                    == status
                    for row in results
                )
                for status in ("correct", "partial", "incorrect", "not_answerable")
            },
        }
    )
    _write_json(run_dir / "summary.json", summary)
    _write_json(
        run_dir / "ledger.json",
        {
            "limit_usd": ledger.total_limit_usd,
            "total_usd": round(ledger.total_usd(), 8),
            "records": ledger.records(),
        },
    )
    run_state.update(
        {
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "ledger_total_usd": round(ledger.total_usd(), 8),
        }
    )
    _write_json(state_path, run_state)
    ledger.close()
    return summary


def freeze_dev_config(
    *,
    dev_evidence_path: Path,
    output_path: Path,
    pricing_path: Path = DEFAULT_PRICING,
    top_k: int = 10,
    max_cost_usd: float = 2.0,
) -> dict:
    run_config = _run_config(
        "all",
        evidence_path=dev_evidence_path,
        top_k=top_k,
        pricing_path=pricing_path,
        max_cost_usd=max_cost_usd,
    )
    run_config.pop("evidence_sha256", None)
    run_config.pop("split", None)
    payload = {
        "schema_version": 1,
        "experiment": "answer_quality_depth_20261027",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "run_config": run_config,
        "prompt_hashes": _prompt_hashes(),
        "dataset_sha256": dataset_hashes(),
        "scoring": {
            "version": "answer-quality-depth-score-v7",
            "overlay_path": str(OVERLAY_PATH.relative_to(ROOT)),
            "overlay_sha256": _file_sha(OVERLAY_PATH),
            "policy": "uncertain remains a separate denominator; no label enters generation or judge",
        },
        "budget": {
            "hard_limit_usd": float(max_cost_usd),
            "ledger_scope": "one independent SQLite ledger for this experiment",
            "provider_retries_max": 2,
        },
    }
    _write_json(output_path, payload)
    return payload


def repair_judge_failures(
    *,
    run_dir: Path,
    credentials_path: Path,
    ledger_path: Path,
    pricing_path: Path = DEFAULT_PRICING,
    max_cost_usd: float = 2.0,
    allow_paid: bool = False,
) -> dict:
    """Retry only missing judge calls; generated answers remain byte-identical."""
    if not allow_paid:
        raise RuntimeError("refusing judge repair without --allow-paid")
    run_dir = Path(run_dir)
    results_path = run_dir / "results.jsonl"
    initial_path = run_dir / "results-initial.jsonl"
    rows = _read_jsonl(results_path)
    if not initial_path.exists():
        initial_path.write_bytes(results_path.read_bytes())
    cases = {
        str(case["id"]): case
        for dataset in load_depth_cases().values()
        for case in dataset
    }
    _pricing_payload, model_price = _pricing(pricing_path)
    ledger = AnswerQualityLedger(ledger_path, total_limit_usd=max_cost_usd)
    provider = CappedProvider(
        _openai_client(credentials_path),
        ledger=ledger,
        pricing=model_price,
        model="deepseek-flash",
        max_retries=2,
        retry_backoff_seconds=0.5,
    )
    journal = ResponseJournal(run_dir / "responses.sqlite3")
    repaired = []
    for row in rows:
        candidate = dict(row)
        has_generation = bool(str(candidate.get("answer") or "").strip())
        judge = (candidate.get("judge") or {}).get("verdict")
        if candidate.get("technical_failure") and has_generation and judge is None:
            case = cases[str(candidate["case_id"])]
            evidence = list(candidate.get("evidence") or [])
            prompt = build_judge_prompt(
                case, evidence, str(candidate.get("answer") or "")
            )
            attempts = []
            parsed = None
            record = None
            for attempt_index, response_format in enumerate(
                (None, {"type": "json_object"}), start=1
            ):
                try:
                    result = provider.call(
                        [
                            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        kind="judge_repair",
                        max_tokens=1800,
                        temperature=0,
                        response_format=response_format,
                        extra_body={"thinking": {"type": "disabled"}},
                        note=f"{candidate['case_id']}:{candidate['arm']}:repair",
                    )
                    text = judge_response_text(result.text, result.raw)
                    parsed = parse_judge_payload(text)
                    attempts.append(
                        {
                            "attempt": attempt_index,
                            "response_format": bool(response_format),
                            "usage": result.usage,
                            "latency_ms": result.latency_ms,
                            "cost_usd": result.cost_usd,
                        }
                    )
                    record = {
                        "verdict": parsed,
                        "raw_text": text,
                        "raw": result.raw,
                        "usage": result.usage,
                        "model_version": result.model_version,
                        "latency_ms": sum(
                            float(item["latency_ms"]) for item in attempts
                        ),
                        "attempts": len(attempts),
                        "cost_usd": sum(float(item["cost_usd"]) for item in attempts),
                        "repair_attempts": attempts,
                    }
                    journal.put(
                        "judge",
                        request_key(
                            phase="judge",
                            model="deepseek-flash",
                            prompt_sha256=_prompt_hashes()["judge_system"],
                            case_id=str(candidate["case_id"]),
                            arm=str(candidate["arm"]),
                            evidence=evidence,
                        ),
                        record,
                    )
                    break
                except Exception as exc:  # noqa: BLE001 - preserve provider failure
                    attempts.append(
                        {
                            "attempt": attempt_index,
                            "response_format": bool(response_format),
                            "error": f"{type(exc).__name__}: {exc}"[:300],
                        }
                    )
            candidate["technical_history"] = [
                *(candidate.get("technical_history") or []),
                candidate.get("technical_failure"),
            ]
            if record is not None:
                candidate["judge"] = record
                candidate["technical_failure"] = None
                candidate["technical_failures"] = []
                score = score_response(
                    case,
                    str(candidate.get("answer") or ""),
                    evidence,
                    overlay=load_scoring_overlay(),
                )
                candidate["score"] = apply_semantic_status(score, parsed)
                candidate.update(candidate["score"])
                candidate["cost_usd"] = round(
                    float((candidate.get("generation") or {}).get("cost_usd") or 0)
                    + float(record.get("cost_usd") or 0),
                    8,
                )
                candidate["latency_ms"] = round(
                    float((candidate.get("generation") or {}).get("latency_ms") or 0)
                    + float(record.get("latency_ms") or 0),
                    3,
                )
            else:
                candidate["judge_repair_attempts"] = attempts
                candidate["technical_failure"] = (
                    f"judge repair failed after {len(attempts)} attempts"
                )
                candidate["status"] = "technical_failure"
                candidate["final_status"] = "technical_failure"
        elif judge is not None:
            case = cases[str(candidate["case_id"])]
            score = score_response(
                case,
                str(candidate.get("answer") or ""),
                list(candidate.get("evidence") or []),
                overlay=load_scoring_overlay(),
                technical_failure=candidate.get("technical_failure"),
            )
            candidate["score"] = apply_semantic_status(score, judge)
            candidate.update(candidate["score"])
        repaired.append(candidate)
    if (run_dir / "summary.json").exists() and not (
        run_dir / "summary-initial.json"
    ).exists():
        (run_dir / "summary-initial.json").write_bytes(
            (run_dir / "summary.json").read_bytes()
        )
    _write_jsonl(results_path, repaired)
    summary = summarize_scores(repaired)
    summary.update(
        {
            "split": str(repaired[0]["split"]) if repaired else "",
            "technical_failures": sum(
                bool(row.get("technical_failure")) for row in repaired
            ),
            "judge_repaired": sum(
                bool(row.get("technical_history")) and not row.get("technical_failure")
                for row in repaired
            ),
            "ledger_total_usd": round(ledger.total_usd(), 8),
        }
    )
    _write_json(run_dir / "summary.json", summary)
    _write_json(
        run_dir / "ledger-after-judge-repair.json",
        {
            "limit_usd": ledger.total_limit_usd,
            "total_usd": round(ledger.total_usd(), 8),
            "records": ledger.records(),
        },
    )
    ledger.close()
    return summary


def rejudge_existing(
    *,
    source_run_dir: Path,
    output_run_dir: Path,
    evidence_path: Path,
    credentials_path: Path,
    ledger_path: Path,
    pricing_path: Path = DEFAULT_PRICING,
    top_k: int = 10,
    max_cost_usd: float = 2.0,
    allow_paid: bool = False,
) -> dict:
    """Rejudge saved answers in non-thinking mode without regenerating them."""
    if not allow_paid:
        raise RuntimeError("refusing judge calls without --allow-paid")
    source = Path(source_run_dir)
    source_path = source / "results-initial.jsonl"
    if not source_path.exists():
        source_path = source / "results.jsonl"
    rows = _read_jsonl(source_path)
    if not rows:
        raise RuntimeError(f"source run has no results: {source}")
    split = str(rows[0]["split"])
    output_run_dir = Path(output_run_dir)
    output_run_dir.mkdir(parents=True, exist_ok=False)
    config = _run_config(
        split,
        evidence_path=evidence_path,
        top_k=top_k,
        pricing_path=pricing_path,
        max_cost_usd=max_cost_usd,
    )
    config["source_generation_sha256"] = hashlib.sha256(
        json.dumps(
            [
                {
                    "case_id": row["case_id"],
                    "arm": row["arm"],
                    "answer": row.get("answer") or "",
                    "generation": row.get("generation") or {},
                }
                for row in rows
            ],
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    config["judge_input"] = "anonymous_answer_and_evidence_only"
    state = {
        **config,
        "config_sha256": run_config_hash(config),
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(output_run_dir / "run-state.json", state)

    _pricing_payload, model_price = _pricing(pricing_path)
    ledger = AnswerQualityLedger(ledger_path, total_limit_usd=max_cost_usd)
    provider = CappedProvider(
        _openai_client(credentials_path),
        ledger=ledger,
        pricing=model_price,
        model="deepseek-flash",
        max_retries=2,
        retry_backoff_seconds=0.5,
    )
    journal = ResponseJournal(output_run_dir / "responses.sqlite3")
    cases = {
        str(case["id"]): case
        for dataset in load_depth_cases().values()
        for case in dataset
    }
    output_rows = []
    results_path = output_run_dir / "results.jsonl"
    for index, source_row in enumerate(rows, start=1):
        row = dict(source_row)
        case = cases[str(row["case_id"])]
        evidence = list(row.get("evidence") or [])[:top_k]
        prompt = build_judge_prompt(case, evidence, str(row.get("answer") or ""))
        key = request_key(
            phase="judge",
            model="deepseek-flash",
            prompt_sha256=_prompt_hashes()["judge_system"],
            case_id=str(row["case_id"]),
            arm=str(row["arm"]),
            evidence=evidence,
            variant="non_thinking",
        )
        record = journal.get("judge", key)
        failure = None
        if record is None:
            try:
                judged, payload = _call_with_json_validation(
                    provider,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    kind="judge",
                    max_tokens=1600,
                    note=f"{row['case_id']}:{row['arm']}",
                    validator=parse_judge_payload,
                    extra_body={"thinking": {"type": "disabled"}},
                )
                record = {
                    "verdict": payload,
                    "raw_text": judged.text,
                    "raw": judged.raw,
                    "usage": judged.usage,
                    "model_version": judged.model_version,
                    "latency_ms": judged.latency_ms,
                    "attempts": judged.attempts,
                    "cost_usd": judged.cost_usd,
                }
                journal.put("judge", key, record)
            except Exception as exc:  # noqa: BLE001 - preserve provider failure
                failure = f"judge {type(exc).__name__}: {exc}"[:500]
                record = {
                    "verdict": None,
                    "raw_text": "",
                    "raw": {},
                    "usage": {},
                    "model_version": "",
                    "latency_ms": 0.0,
                    "attempts": 0,
                    "cost_usd": 0.0,
                }
        row["judge"] = record
        row["technical_history"] = [
            *(row.get("technical_history") or []),
            row.get("technical_failure"),
            failure,
        ]
        row["technical_history"] = [item for item in row["technical_history"] if item]
        row["technical_failure"] = failure
        score = score_response(
            case,
            str(row.get("answer") or ""),
            evidence,
            overlay=load_scoring_overlay(),
            technical_failure=failure,
        )
        row["score"] = apply_semantic_status(score, record.get("verdict"))
        row.update(row["score"])
        row["cost_usd"] = round(
            float((row.get("generation") or {}).get("cost_usd") or 0.0)
            + float(record.get("cost_usd") or 0.0),
            8,
        )
        row["latency_ms"] = round(
            float((row.get("generation") or {}).get("latency_ms") or 0.0)
            + float(record.get("latency_ms") or 0.0),
            3,
        )
        output_rows.append(row)
        with results_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(
            f"[{index}/{len(rows)}] {row['case_id']} {row['arm']} "
            f"judge={(record.get('verdict') or {}).get('answer_status', 'failed')} "
            f"cost=${float(record.get('cost_usd') or 0):.5f}",
            flush=True,
        )
    summary = summarize_scores(output_rows)
    summary.update(
        {
            "split": split,
            "config_sha256": run_config_hash(config),
            "technical_failures": sum(
                bool(row.get("technical_failure")) for row in output_rows
            ),
            "ledger_total_usd": round(ledger.total_usd(), 8),
            "judge_answer_status": {
                status: sum(
                    ((row.get("judge") or {}).get("verdict") or {}).get("answer_status")
                    == status
                    for row in output_rows
                )
                for status in ("correct", "partial", "incorrect", "not_answerable")
            },
        }
    )
    _write_json(output_run_dir / "summary.json", summary)
    _write_json(
        output_run_dir / "ledger.json",
        {
            "limit_usd": ledger.total_limit_usd,
            "total_usd": round(ledger.total_usd(), 8),
            "records": ledger.records(),
        },
    )
    state.update(
        {
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "technical_failures": summary["technical_failures"],
        }
    )
    _write_json(output_run_dir / "run-state.json", state)
    ledger.close()
    return summary


def repair_generation_truncations(
    *,
    source_run_dir: Path,
    output_run_dir: Path,
    evidence_path: Path,
    credentials_path: Path,
    ledger_path: Path,
    pricing_path: Path = DEFAULT_PRICING,
    top_k: int = 10,
    max_cost_usd: float = 2.0,
    allow_paid: bool = False,
) -> dict:
    """Retry only generations with finish_reason=length.

    Non-truncated answers are copied byte-for-byte. The failed generation remains
    in ``generation_history`` so the artifact does not hide the infrastructure
    failure.
    """
    if not allow_paid:
        raise RuntimeError("refusing generation repair without --allow-paid")
    source = Path(source_run_dir)
    for name in ("results-rescored.jsonl", "results.jsonl"):
        source_path = source / name
        if source_path.exists():
            break
    else:
        raise RuntimeError(f"source run has no results: {source}")
    rows = _read_jsonl(source_path)
    if not rows:
        raise RuntimeError(f"source run is empty: {source}")
    split = str(rows[0]["split"])
    output_run_dir = Path(output_run_dir)
    output_run_dir.mkdir(parents=True, exist_ok=False)
    config = _run_config(
        split,
        evidence_path=evidence_path,
        top_k=top_k,
        pricing_path=pricing_path,
        max_cost_usd=max_cost_usd,
    )
    config.update(
        {
            "repair": {
                "trigger": "finish_reason=length, including non-empty partial text",
                "max_output_tokens": 2400,
                "max_attempts": 2,
                "nonempty_answers_preserved": True,
            },
            "source_results_sha256": _file_sha(source_path),
        }
    )
    state = {
        **config,
        "config_sha256": run_config_hash(config),
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(output_run_dir / "run-state.json", state)

    _pricing_payload, model_price = _pricing(pricing_path)
    ledger = AnswerQualityLedger(ledger_path, total_limit_usd=max_cost_usd)
    provider = CappedProvider(
        _openai_client(credentials_path),
        ledger=ledger,
        pricing=model_price,
        model="deepseek-flash",
        max_retries=2,
        retry_backoff_seconds=0.5,
    )
    cases = {
        str(case["id"]): case
        for dataset in load_depth_cases().values()
        for case in dataset
    }
    output_rows = []
    repaired_count = 0
    results_path = output_run_dir / "results.jsonl"
    for index, source_row in enumerate(rows, start=1):
        row = dict(source_row)
        case = cases[str(row["case_id"])]
        evidence = list(row.get("evidence") or [])[:top_k]
        if generation_was_truncated(row):
            original_generation = dict(row.get("generation") or {})
            arm_constant = ARM_BASELINE if row["arm"] == "baseline" else ARM_SLATE
            messages = build_generation_messages(case, evidence, arm=arm_constant)
            generation = None
            last_error = None
            for retry_index, extra_body in enumerate(
                (
                    {"thinking": {"type": "enabled"}},
                    {"thinking": {"type": "disabled"}},
                ),
                start=1,
            ):
                try:
                    generated = provider.call(
                        messages,
                        kind="answer_repair",
                        max_tokens=2400,
                        temperature=0,
                        extra_body=extra_body,
                        note=f"{row['case_id']}:{row['arm']}:empty-retry",
                    )
                    if generated.text.strip():
                        generation = {
                            "usage": generated.usage,
                            "raw": generated.raw,
                            "model_version": generated.model_version,
                            "latency_ms": generated.latency_ms,
                            "attempts": generated.attempts,
                            "cost_usd": generated.cost_usd,
                            "repair_attempt": retry_index,
                            "thinking": extra_body["thinking"]["type"],
                        }
                        row["answer"] = generated.text.strip()
                        break
                    last_error = "empty content after repair"
                except Exception as exc:  # noqa: BLE001 - preserve provider failure
                    last_error = f"{type(exc).__name__}: {exc}"[:300]
            row["generation_history"] = [
                *(row.get("generation_history") or []),
                original_generation,
            ]
            if generation is not None:
                repaired_count += 1
                row["generation"] = generation
                judge_prompt = build_judge_prompt(case, evidence, str(row["answer"]))
                judged, payload = _call_with_json_validation(
                    provider,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": judge_prompt},
                    ],
                    kind="judge",
                    max_tokens=1600,
                    note=f"{row['case_id']}:{row['arm']}",
                    validator=parse_judge_payload,
                    extra_body={"thinking": {"type": "disabled"}},
                )
                judge_record = {
                    "verdict": payload,
                    "raw_text": judged.text,
                    "raw": judged.raw,
                    "usage": judged.usage,
                    "model_version": judged.model_version,
                    "latency_ms": judged.latency_ms,
                    "attempts": judged.attempts,
                    "cost_usd": judged.cost_usd,
                }
                row["judge"] = judge_record
                row["technical_failure"] = None
                row["technical_failures"] = []
            else:
                row["technical_failure"] = f"generation repair failed: {last_error}"
        row.setdefault("technical_history", [])
        if generation_was_truncated(source_row):
            row["technical_history"] = [
                *row["technical_history"],
                "empty_content_finish_length",
            ]
        score = score_response(
            case,
            str(row.get("answer") or ""),
            evidence,
            overlay=load_scoring_overlay(),
            technical_failure=row.get("technical_failure"),
        )
        row["score"] = apply_semantic_status(
            score,
            (row.get("judge") or {}).get("verdict"),
        )
        row.update(row["score"])
        row["cost_usd"] = round(
            float((row.get("generation") or {}).get("cost_usd") or 0)
            + float((row.get("judge") or {}).get("cost_usd") or 0)
            + sum(
                float(item.get("cost_usd") or 0)
                for item in row.get("generation_history") or []
            ),
            8,
        )
        row["latency_ms"] = round(
            float((row.get("generation") or {}).get("latency_ms") or 0)
            + float((row.get("judge") or {}).get("latency_ms") or 0)
            + sum(
                float(item.get("latency_ms") or 0)
                for item in row.get("generation_history") or []
            ),
            3,
        )
        output_rows.append(row)
        with results_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(
            f"[{index}/{len(rows)}] {row['case_id']} {row['arm']} "
            f"repaired={generation is not None if generation_was_truncated(source_row) else 'copied'} "
            f"status={row.get('final_status')}",
            flush=True,
        )
    summary = summarize_scores(output_rows)
    summary.update(
        {
            "split": split,
            "config_sha256": run_config_hash(config),
            "generation_repaired": repaired_count,
            "technical_failures": sum(
                bool(row.get("technical_failure")) for row in output_rows
            ),
            "ledger_total_usd": round(ledger.total_usd(), 8),
        }
    )
    _write_json(output_run_dir / "summary.json", summary)
    _write_json(
        output_run_dir / "ledger.json",
        {
            "limit_usd": ledger.total_limit_usd,
            "total_usd": round(ledger.total_usd(), 8),
            "records": ledger.records(),
        },
    )
    state.update(
        {
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "generation_repaired": repaired_count,
            "technical_failures": summary["technical_failures"],
        }
    )
    _write_json(output_run_dir / "run-state.json", state)
    ledger.close()
    return summary


__all__ = [
    "ARM_BASELINE",
    "ARM_SLATE",
    "ResponseJournal",
    "build_generation_messages",
    "evidence_projection",
    "judge_response_text",
    "parse_judge_payload",
    "request_key",
    "run_config_hash",
]
