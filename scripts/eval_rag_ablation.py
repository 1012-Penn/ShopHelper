"""CH16 分阶段、可续跑的 RAG 消融运行器。

底层检索、答案 prompt、确定性判分和 CH12 循环全部复用 eval_ch11。
普通策略分开记录检索层与可缓存的答案重放；任何零预算缓存未命中均标为未测。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import subprocess
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from scripts.ch16_analysis import (
    ContractError,
    build_experiment_contract,
    check_cross_experiment_locks,
    check_phase_lock,
    compare_groups,
    paired_bucket_outcomes,
    protocol_sha256,
    summarize_group,
    validate_budget_request,
    validate_group_contracts,
)
from scripts.eval_cost import (
    AttemptLedger,
    AuditedChat,
    BudgetExceeded,
    CallResultJournal,
    attempt_context,
)

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_BUILD_ID = "ebfd49478b37ef387c8b124319a450ed73a86ae2a4b9f4fc28d6a52b2ccaa0ba"
SNAPSHOT_SHA256 = "b6b1fa237d03d2c0f8ddcfe8e003f13c4cb584f0764ac04c2d57ae834bbdca73"
REFERENCE_MODEL = "deepseek-flash"
RETRIEVAL_GROUPS = {
    "vector": "dense",
    "bm25": "bm25",
    "rrf": "hybrid",
    "rrf_rerank": "hybrid_rerank",
}
RAG_GROUPS = {
    "baseline": "baseline",
    "decompose_once": "decompose_once",
    "controlled_loop": "controlled_loop",
}
RAG_BUCKETS = ("multi_hop", "policy_conflict")


def validate_snapshot_identity(
    snapshot: Path,
    *,
    expected_sha256: str | None = None,
) -> str:
    """Require the exact CH11 frozen index snapshot, including explicit local copies."""
    path = Path(snapshot)
    if not path.exists() or not (path.is_file() or path.is_dir()):
        raise ContractError(f"评测快照不存在或类型无效:{path}")
    from scripts.eval_snapshot import tree_sha256

    actual = tree_sha256(path)
    expected = expected_sha256 or SNAPSHOT_SHA256
    if actual != expected:
        raise ContractError(
            f"快照 SHA-256 与 CH11 canonical 不一致:{actual} != {expected}"
        )
    return actual


def estimate_group_costs(split: str) -> dict:
    """按 CH11 实测成本与 CH12 已记录估算给出逐组预估及单组预算上限。

    这是缓存未命中情况下的规划值，不是账单；模型调用按冻结语料中 CH11 首跑
    $0.1019/90 题缩放，CH12 复杂组沿用 CH12 报告的 $0.04–0.08/$0.07–0.15。
    """
    if split not in {"dev", "frozen-v1"}:
        raise ContractError(f"费用估算只支持 dev/frozen-v1: {split}")
    n = 60 if split == "dev" else 90
    scale = n / 90
    answer_and_judge = 0.1019 * scale
    retrieval_cap = 0.12 if split == "dev" else 0.18
    rag_decompose_cap = 0.06 if split == "dev" else 0.09
    rag_controlled_cap = 0.11 if split == "dev" else 0.17
    table = []
    for name, strategy in RETRIEVAL_GROUPS.items():
        # Current SiliconFlow standard BGE-M3 and BGE reranker endpoints are listed free.
        extra = 0.0
        low = max(0.0, answer_and_judge * 0.70 + extra)
        high = answer_and_judge * 1.50 + extra
        table.append({
            "experiment": "retrieval",
            "group": name,
            "strategy": strategy,
            "estimated_usd": [round(low, 4), round(high, 4)],
            "max_cost_usd_per_group": retrieval_cap,
            "basis": "CH11 首跑回答+证据判断实测 $0.1019/90 题；DeepSeek official peak cache-miss rate, BGE standard endpoints free",
        })
    table.extend([
        {
            "experiment": "rag", "group": "baseline", "strategy": "hybrid_rerank",
            "estimated_usd": [0.0, 0.0], "max_cost_usd_per_group": 0.0,
            "basis": "CH12 baseline 完整缓存重放为 $0；缓存未命中需重新估算",
        },
        {
            "experiment": "rag", "group": "decompose_once", "strategy": "hybrid_rerank",
            "estimated_usd": [round(0.04 * scale, 4), round(0.08 * scale, 4)],
            "max_cost_usd_per_group": rag_decompose_cap,
            "basis": "CH12 报告 90 题预估 $0.04–0.08，dev 按 60/90 缩放",
        },
        {
            "experiment": "rag", "group": "controlled_loop", "strategy": "hybrid_rerank",
            "estimated_usd": [round(0.07 * scale, 4), round(0.15 * scale, 4)],
            "max_cost_usd_per_group": rag_controlled_cap,
            "basis": "CH12 报告 90 题预估 $0.07–0.15，dev 按 60/90 缩放",
        },
    ])
    return {
        "split": split,
        "case_count": n,
        "currency": "USD",
        "paid_calls_executed": 0,
        "groups": table,
        "total_estimated_usd": [
            round(sum(row["estimated_usd"][0] for row in table), 4),
            round(sum(row["estimated_usd"][1] for row in table), 4),
        ],
        "total_max_cost_usd_if_all_groups_run": round(
            sum(row["max_cost_usd_per_group"] for row in table), 4
        ),
        "note": "规划范围使用 DeepSeek 官方峰值 cache-miss 单价；缓存输入及实际时段可能降低费用。BGE 免费端点仍记录外部请求。",
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _write_jsonl(path: Path, traces: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        for trace in traces:
            stream.write(json.dumps(trace, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _append_retry_history(group_root: Path, attempt_index: int, trace: dict) -> None:
    path = group_root / "retry-history.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({
            "attempt_index": attempt_index,
            "case_id": trace.get("case_id"),
            "rag_status": trace.get("rag_status"),
            "reason": trace.get("rag_unmeasured_reason"),
            "trace": trace,
        }, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _wait_for_pending_attempt(cost, timeout_s: float = 120.0) -> bool:
    """Drain a timed-out worker's provider request before starting another case."""
    deadline = time.monotonic() + timeout_s
    while getattr(cost, "_pending_attempt_id", None) and time.monotonic() < deadline:
        time.sleep(0.05)
    return getattr(cost, "_pending_attempt_id", None) is None


def _database_url(sqlite_db: Path) -> str:
    absolute = sqlite_db.resolve()
    return f"sqlite:///{absolute}"


def _configure_environment(args) -> None:
    """固定评测模型与私有 SQLite，不把 CH16 路由到共享业务库。"""
    from dotenv import dotenv_values

    local_env = dotenv_values(ROOT / ".env")
    credential_env = dotenv_values(args.credential_env_file) if args.credential_env_file else {}
    for name in ("OPENAI_API_KEY", "EMBEDDING_API_KEY", "RERANK_API_KEY"):
        value = credential_env.get(name) or local_env.get(name)
        if value:
            os.environ[name] = str(value)
    database_url = local_env.get("DATABASE_URL")
    if database_url:
        expected = _database_url(args.sqlite_db)
        if database_url != expected:
            raise ContractError("CH16 必须使用本 worktree 的私有 SQLite 评测库")
    if local_env.get("HYBRID_CANDIDATES") and local_env["HYBRID_CANDIDATES"] != "50":
        raise ContractError("CH16 必须固定 HYBRID_CANDIDATES=50")
    if local_env.get("RERANK_TOP_K") and local_env["RERANK_TOP_K"] != "10":
        raise ContractError("CH16 必须固定 RERANK_TOP_K=10")
    os.environ["DATABASE_URL"] = _database_url(args.sqlite_db)
    os.environ["OPENAI_MODEL"] = REFERENCE_MODEL
    # Disable implicit SDK retries: each real request attempt is recorded exactly once.
    os.environ["LLM_MAX_RETRIES"] = "0"
    os.environ["HYBRID_CANDIDATES"] = "50"
    os.environ["RERANK_TOP_K"] = "10"
    if not os.environ.get("OPENAI_API_KEY"):
        if args.allow_paid:
            raise ContractError("付费运行缺少 OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "offline-cache-replay-only"
    if args.allow_paid and not os.environ.get("EMBEDDING_API_KEY"):
        raise ContractError("付费运行缺少 EMBEDDING_API_KEY")


def _context(config, args, cases):
    from scripts.eval_ch11 import _build_context
    from app.config import Settings

    namespace = SimpleNamespace(
        snapshot=args.snapshot,
        cache_dir=args.cache_dir,
        pricing=args.pricing,
        ledger=args.ledger,
        allow_paid=bool(args.allow_paid),
        no_rewrite=False,
        rewrite_cache=None,
        rewrite_cache_mode="read",
    )
    ctx, config, source_contract = _build_context(config, namespace, cases)
    settings = Settings()
    if (ctx.service._candidates, ctx.service._final_top_k) != (50, 10):
        raise ContractError("运行时检索上下文上限漂移:CH16 固定 50 个候选 / top 10 上下文")
    # CH11 only assembles its reranker when credentials are available. CH16 requires
    # the same real reranker for every route; missing credentials fail before running.
    if ctx.service._reranker is None:
        raise ContractError("CH16 缺少 reranker；拒绝把 RRF 降级结果标成融合加重排")
    journal = CallResultJournal(Path(args.cache_dir) / "ch16-call-results.json")
    # The common CH11 context keeps its default finite safeguards. CH16 widens them
    # only when the explicit unbounded-budget switch is present, and uses a separate ledger.
    if args.unbounded_budget:
        ctx.cost.max_cost_usd = None
        ctx.cost.ledger.total_limit_usd = None
    ctx.cost.attempt_ledger = AttemptLedger(args.attempt_ledger)
    ctx.service._embedder.inner.journal = journal
    ctx.service._reranker.inner.journal = journal
    if ctx.chat is not None:
        ctx.chat = AuditedChat(ctx.chat, ctx.cost, journal, REFERENCE_MODEL, "answer")
    for guarded in (
        getattr(ctx.service, "_evidence_judge", None), ctx.gap_analyzer, ctx.judge,
    ):
        inner = getattr(guarded, "inner", None)
        llm = getattr(inner, "_llm", None)
        if llm is not None:
            inner._llm = AuditedChat(llm, ctx.cost, journal, REFERENCE_MODEL,
                                     "gap_analyzer" if guarded is ctx.gap_analyzer
                                     else "evidence_judge" if guarded is getattr(ctx.service, "_evidence_judge", None)
                                     else "faithfulness_judge")
    source_contract["retrieval"]["rerank_model"] = settings.rerank_model
    source_contract["retrieval"]["rerank_cache_policy"] = "exact-request durable cache, then live API"
    source_contract["run"]["ch16_allow_paid"] = bool(args.allow_paid)
    source_contract["run"]["ch16_max_cost_usd"] = args.max_cost_usd
    source_contract["run"]["ch16_unbounded_budget"] = bool(args.unbounded_budget)
    source_contract["run"]["ch16_retry_policy"] = "zero implicit retries; failed requests are checkpointed"
    return ctx, config, source_contract


def _preflight_contract(args, cases, *, strategy: str, rag_mode: str):
    """在首题执行前从同一快照构造共同协议，冻结阶段先比对 dev lock。"""
    from scripts.ch11_harness import HarnessConfig

    config = HarnessConfig(
        split=args.split,
        mode="full",
        strategy=strategy,
        max_cost_usd=args.max_cost_usd,
        allow_paid=bool(args.allow_paid),
        rag_mode=rag_mode,
        temperature=0.0,
    )
    _, _, source_contract = _context(config, args, cases)
    contract = build_experiment_contract(
        split=args.split,
        experiment=args.experiment,
        strategy=strategy,
        rag_mode=rag_mode,
        source_contract=source_contract,
    )
    return contract


def _base_trace(case: dict) -> dict:
    return {
        "case_id": case["id"],
        "split": case.get("split", ""),
        "bucket": case.get("bucket", ""),
        "expected_behavior": case.get("expected_behavior", ""),
        "evidence_state": case.get("evidence_state", ""),
        "query": case.get("query", ""),
        "required_conditions": list(case.get("required_conditions") or []),
        "requires_tools": list(case.get("requires_tools") or []),
        "required_evidence": list(case.get("required_evidence") or []),
        "deterministic_checks": None,
        "answer_status": "unmeasured",
        "retrieval_status": "unmeasured",
    }


def _cost_delta(ctx, before_calls: int, before_paid_calls: int,
                before_spent: float) -> tuple[int, int, float]:
    return (
        int(getattr(ctx.cost, "api_attempts", 0)) - before_calls,
        int(getattr(ctx.cost, "paid_api_attempts", 0)) - before_paid_calls,
        round(float(getattr(ctx.cost, "spent_usd", 0.0)) - before_spent, 8),
    )


def _safe_reason(exc: BaseException) -> str:
    message = f"{type(exc).__name__}: {exc}"
    for name in ("OPENAI_API_KEY", "EMBEDDING_API_KEY", "RERANK_API_KEY"):
        secret = os.environ.get(name)
        if secret:
            message = message.replace(secret, "[redacted]")
    return message[:300]


def _enrich_retrieval_coverage(trace: dict, case: dict, ctx) -> None:
    """Add partial evidence-group recall alongside all-groups-complete R@K."""
    from scripts.eval_metrics import coverage_rank, evidence_group_recall

    gold = ctx.gold_by_case.get(str(case["id"]), [])
    if not gold:
        return
    ranked = [int(item["chunk_id"]) for item in trace.get("retrieved", [])]
    l1 = trace.setdefault("l1", {})
    for k in (3, 5):
        l1[f"evidence_group_recall_at_{k}"] = evidence_group_recall(ranked, gold, cap=k)
        rank = coverage_rank(ranked, gold, cap=k)
        l1[f"all_required_groups_at_{k}"] = rank is not None


def _summarize_attempt_ledger(path: Path, *, split: str, experiment: str) -> dict:
    if not path.exists():
        return {"attempts": 0, "groups": {}}
    events = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ContractError(f"逐次调用账本损坏:{path}:{line_no}") from exc
    started = {}
    terminal = {}
    for event in events:
        attempt_id = str(event.get("attempt_id", ""))
        if event.get("event") == "started":
            started[attempt_id] = event
        elif event.get("event") in {"succeeded", "failed", "cache_hit"}:
            terminal[attempt_id] = event
    by_group: dict[str, dict] = {}
    for attempt_id, start in started.items():
        context = start
        if context.get("split") != split or context.get("experiment") != experiment:
            continue
        group_name = str(context.get("group") or "unknown")
        group = by_group.setdefault(group_name, {
            "external_api_calls": 0, "paid_api_calls": 0, "cache_hits": 0,
            "successes": 0, "failures": 0, "input_tokens": 0,
            "output_tokens": 0, "cost_actual_usd": 0.0, "retry_count": 0,
            "latencies_ms": [], "by_kind": {}, "errors": [],
        })
        result = terminal.get(attempt_id, {})
        status = result.get("event", "pending")
        kind = str(context.get("kind") or "api")
        kind_row = group["by_kind"].setdefault(kind, {
            "external_api_calls": 0, "cache_hits": 0, "successes": 0,
            "failures": 0, "input_tokens": 0, "output_tokens": 0,
            "cost_actual_usd": 0.0, "retry_count": 0,
        })
        if status == "cache_hit":
            group["cache_hits"] += 1
            kind_row["cache_hits"] += 1
            continue
        if status in {"succeeded", "failed", "pending"}:
            group["external_api_calls"] += 1
            kind_row["external_api_calls"] += 1
        amount = float(result.get("amount_usd") or 0.0)
        group["cost_actual_usd"] += amount
        kind_row["cost_actual_usd"] += amount
        if amount > 0:
            group["paid_api_calls"] += 1
        if status == "succeeded":
            group["successes"] += 1
            kind_row["successes"] += 1
        elif status == "failed":
            group["failures"] += 1
            kind_row["failures"] += 1
            if result.get("error"):
                group["errors"].append({
                    "case_id": context.get("case_id"), "kind": kind,
                    "error": result["error"],
                })
        input_tokens = result.get("input_tokens")
        if input_tokens is None:
            input_tokens = context.get("estimated_input_tokens") or 0
        output_tokens = result.get("output_tokens") or 0
        group["input_tokens"] += int(input_tokens)
        group["output_tokens"] += int(output_tokens)
        kind_row["input_tokens"] += int(input_tokens)
        kind_row["output_tokens"] += int(output_tokens)
        retries = int(result.get("retry_count") or 0)
        group["retry_count"] += retries
        kind_row["retry_count"] += retries
        elapsed = result.get("elapsed_ms")
        if elapsed is not None:
            group["latencies_ms"].append(float(elapsed))
    for group in by_group.values():
        values = group.pop("latencies_ms")
        ordered = sorted(values)
        group["api_latency_ms"] = {
            "n": len(ordered),
            "p50": ordered[max(0, (len(ordered) + 1) // 2 - 1)] if ordered else None,
            "p95": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
            if ordered else None,
        }
        group["cost_actual_usd"] = round(group["cost_actual_usd"], 8)
    return {"attempts": sum(g["external_api_calls"] for g in by_group.values()),
            "groups": by_group}


def _checkpoint_state(group_root: Path, contract: dict, cases: list[dict]) -> list[dict]:
    """Load an exact-contract partial run; retry unmeasured rows after safe replay."""
    group_root.mkdir(parents=True, exist_ok=True)
    metadata_path = group_root / "checkpoint-contract.json"
    expected = {
        "protocol_sha256": contract["protocol_sha256"],
        "experiment": contract["experiment"],
        "split": contract["split"],
        "group": contract["group"],
    }
    if metadata_path.exists():
        actual = json.loads(metadata_path.read_text(encoding="utf-8"))
        if actual != expected:
            raise ContractError(f"检查点契约不匹配，拒绝续跑:{metadata_path}")
    else:
        _write_json(metadata_path, expected)
    checkpoint = group_root / "case-checkpoint.jsonl"
    if not checkpoint.exists():
        saved_contract_path = group_root / "contract.json"
        saved_traces_path = group_root / "traces.jsonl"
        if saved_contract_path.exists() and saved_traces_path.exists():
            saved_contract = json.loads(saved_contract_path.read_text(encoding="utf-8"))
            same_run = all(saved_contract.get(key) == value for key, value in expected.items())
            if same_run:
                saved_traces = [json.loads(line) for line in
                                saved_traces_path.read_text(encoding="utf-8").splitlines()
                                if line.strip()]
                allowed = {str(case["id"]) for case in cases}
                seen = set()
                measured_traces = []
                for trace in saved_traces:
                    case_id = str(trace.get("case_id", ""))
                    if case_id not in allowed or case_id in seen:
                        raise ContractError(f"已完成组轨迹出现未知或重复 case_id:{case_id}")
                    seen.add(case_id)
                    measured = (
                        trace.get("rag_status") == "measured"
                        if contract.get("experiment") == "rag"
                        else trace.get("retrieval_status") == "measured"
                        and trace.get("answer_status") == "measured"
                    )
                    if measured:
                        measured_traces.append(trace)
                    else:
                        _append_retry_history(group_root, 0, trace)
                if len(measured_traces) != len(saved_traces):
                    _write_jsonl(checkpoint, measured_traces)
                else:
                    return measured_traces
        else:
            return []
        if not checkpoint.exists():
            return []
    allowed = {str(case["id"]) for case in cases}
    traces = []
    seen = set()
    for line_no, line in enumerate(checkpoint.read_text(encoding="utf-8").splitlines(), 1):
        try:
            trace = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractError(f"逐题检查点损坏:{checkpoint}:{line_no}") from exc
        case_id = str(trace.get("case_id", ""))
        if case_id not in allowed or case_id in seen:
            raise ContractError(f"检查点出现未知或重复 case_id:{case_id}")
        seen.add(case_id)
        experiment = str(contract.get("experiment", ""))
        if experiment == "rag":
            measured = trace.get("rag_status") == "measured"
        else:
            measured = (trace.get("retrieval_status") == "measured"
                        and trace.get("answer_status") == "measured")
        if measured:
            traces.append(trace)
        else:
            _append_retry_history(group_root, 0, trace)
    if len(traces) != len(seen):
        # Drop stale unmeasured rows before appending their retry, avoiding duplicate
        # case IDs while keeping completed per-case work resumable.
        _write_jsonl(checkpoint, traces)
    return traces


def _checkpoint_append(group_root: Path, trace: dict) -> None:
    path = group_root / "case-checkpoint.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(trace, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _complete_group_checkpoint(group_root: Path, contract: dict,
                               traces: list[dict]) -> None:
    _write_json(group_root / "contract.json", contract)
    _write_jsonl(group_root / "traces.jsonl", traces)
    (group_root / "case-checkpoint.jsonl").unlink(missing_ok=True)
    (group_root / "checkpoint-contract.json").unlink(missing_ok=True)


def cache_only_reranker(path: Path, model: str):
    """只读既有 rerank 缓存；miss 必须穿透为预算失败，不能静默降级。"""
    from scripts.eval_cache import RerankCache

    class _CacheMissBlockedReranker:
        def rerank(self, query, documents, top_n=None):
            raise BudgetExceeded("CH16 零预算：rerank 缓存未命中，未发起外部调用")

    return RerankCache(_CacheMissBlockedReranker(), Path(path), model)


def _run_retrieval_group(name: str, strategy: str, cases: list[dict], args) -> tuple[list[dict], dict]:
    from scripts.ch11_harness import HarnessConfig, run_case
    from scripts.eval_snapshot import acquire_eval_lock

    config = HarnessConfig(
        split=args.split, mode="full", strategy=strategy,
        max_cost_usd=args.max_cost_usd, allow_paid=bool(args.allow_paid),
        rag_mode="baseline", temperature=0.0,
    )
    ctx, config, source_contract = _context(config, args, cases)
    if source_contract.get("llm", {}).get("model") != REFERENCE_MODEL:
        raise ContractError("回答模型与 CH11 不一致")
    if source_contract.get("rewrite", {}).get("mode") != "frozen":
        raise ContractError("改写缓存未能按 CH11 contract 复用")
    contract = build_experiment_contract(
        split=args.split, experiment="retrieval", strategy=strategy,
        rag_mode="baseline", source_contract=source_contract,
    )
    group_root = args.out / "runs" / args.split / "retrieval" / name
    traces = _checkpoint_state(group_root, contract, cases)
    completed = {str(trace["case_id"]) for trace in traces}
    with acquire_eval_lock():
        for case in cases:
            if str(case["id"]) in completed:
                continue
            before_calls = int(ctx.cost.api_attempts)
            before_paid_calls = int(ctx.cost.paid_api_attempts)
            before_spent = float(ctx.cost.spent_usd)
            with attempt_context(split=args.split, experiment="retrieval", group=name,
                                 case_id=str(case["id"])):
                try:
                    trace = run_case(case, ctx, config)
                    trace["retrieval_status"] = "measured"
                    trace["answer_status"] = "measured"
                    _enrich_retrieval_coverage(trace, case, ctx)
                    if getattr(ctx.chat, "last_cache_hit", False):
                        trace["cached"] = True
                        trace["llm_calls"] = 0
                except BudgetExceeded as exc:
                    trace = _base_trace(case)
                    trace["answer_status"] = trace["retrieval_status"] = "unmeasured"
                    trace["answer_unmeasured_reason"] = str(exc)[:240]
                    trace["retrieval_unmeasured_reason"] = str(exc)[:240]
                except Exception as exc:
                    trace = _base_trace(case)
                    trace["answer_status"] = trace["retrieval_status"] = "unmeasured"
                    trace["failure_labels"] = ["api_error"]
                    trace["answer_unmeasured_reason"] = _safe_reason(exc)
            (trace["external_api_calls"], trace["paid_api_calls"],
             trace["cost_actual"]) = _cost_delta(
                ctx, before_calls, before_paid_calls, before_spent
            )
            trace["latency_source"] = (
                "live_api" if trace["external_api_calls"] else "cache_or_local"
            )
            traces.append(trace)
            _checkpoint_append(group_root, trace)
    _complete_group_checkpoint(group_root, contract, traces)
    return traces, contract


def _run_rag_group(mode: str, cases: list[dict], args) -> tuple[list[dict], dict]:
    from scripts.ch11_harness import HarnessConfig, run_case
    from scripts.eval_snapshot import acquire_eval_lock

    config = HarnessConfig(
        split=args.split,
        mode="full",
        strategy="hybrid_rerank",
        max_cost_usd=args.max_cost_usd,
        allow_paid=bool(args.allow_paid),
        rag_mode=mode,
        temperature=0.0,
    )
    ctx, config, source_contract = _context(config, args, cases)
    if source_contract.get("llm", {}).get("model") != REFERENCE_MODEL:
        raise ContractError("回答模型与 CH11 不一致")
    if source_contract.get("rewrite", {}).get("mode") != "frozen":
        raise ContractError("改写缓存未能按 CH11 contract 复用")
    contract = build_experiment_contract(
        split=args.split, experiment="rag", strategy="hybrid_rerank",
        rag_mode=mode, source_contract=source_contract,
    )
    group_root = args.out / "runs" / args.split / "rag" / mode
    traces = _checkpoint_state(group_root, contract, cases)
    completed = {str(trace["case_id"]) for trace in traces}
    gap = ctx.gap_analyzer
    judge = getattr(ctx.service, "_evidence_judge", None)
    with acquire_eval_lock():
        for case in cases:
            if str(case["id"]) in completed:
                continue
            before_calls = int(getattr(ctx.cost, "api_attempts", 0))
            before_paid_calls = int(getattr(ctx.cost, "paid_api_attempts", 0))
            before_spent = float(getattr(ctx.cost, "spent_usd", 0.0))
            analyzer_calls = int(getattr(gap, "calls", 0))
            analyzer_hits = int(getattr(gap, "hits", 0))
            retry_index = 0
            while True:
                retryable_pending = False
                with attempt_context(split=args.split, experiment="rag", group=mode,
                                     case_id=str(case["id"])):
                    try:
                        trace = run_case(case, ctx, config)
                        trace["retrieval_status"] = "measured"
                        trace["answer_status"] = "measured"
                        trace["rag_status"] = "measured"
                        _enrich_retrieval_coverage(trace, case, ctx)
                        if getattr(ctx.chat, "last_cache_hit", False):
                            trace["cached"] = True
                            trace["llm_calls"] = 0
                    except BudgetExceeded as exc:
                        trace = _base_trace(case)
                        trace["rag_status"] = "unmeasured"
                        reason = str(exc)[:240]
                        trace["rag_unmeasured_reason"] = reason
                        trace["retrieval_unmeasured_reason"] = reason
                        trace["answer_unmeasured_reason"] = reason
                        retryable_pending = "上一条 API 请求尚未闭合" in reason
                    except Exception as exc:
                        trace = _base_trace(case)
                        trace["rag_status"] = trace["retrieval_status"] = "unmeasured"
                        trace["answer_status"] = "unmeasured"
                        trace["failure_labels"] = ["api_error"]
                        reason = _safe_reason(exc)
                        trace["rag_unmeasured_reason"] = reason
                        trace["retrieval_unmeasured_reason"] = reason
                        trace["answer_unmeasured_reason"] = reason
                settled = _wait_for_pending_attempt(ctx.cost)
                if retryable_pending and settled and retry_index < 8:
                    _append_retry_history(group_root, retry_index + 1, trace)
                    retry_index += 1
                    continue
                if retry_index:
                    trace["replayed_after_pending_attempts"] = retry_index
                if not settled:
                    trace["rag_status"] = trace["retrieval_status"] = "unmeasured"
                    trace["answer_status"] = "unmeasured"
                    trace["rag_unmeasured_reason"] = (
                        "后台 API 请求 120 秒内未闭合；为避免重复计费已停止本组"
                    )
                    trace["retrieval_unmeasured_reason"] = trace["rag_unmeasured_reason"]
                    trace["answer_unmeasured_reason"] = trace["rag_unmeasured_reason"]
                    _append_retry_history(group_root, retry_index + 1, trace)
                    break
                if retryable_pending and retry_index >= 8:
                    _append_retry_history(group_root, retry_index + 1, trace)
                break
            api_calls, paid_calls, cost = _cost_delta(
                ctx, before_calls, before_paid_calls, before_spent
            )
            trace["external_api_calls"] = api_calls
            trace["paid_api_calls"] = paid_calls
            trace["cost_actual"] = cost
            trace["latency_source"] = "live_api" if api_calls else "cache_or_local"
            trace["rag_analyzer_api_calls"] = int(getattr(gap, "calls", 0)) - analyzer_calls
            trace["rag_analyzer_cached"] = int(getattr(gap, "hits", 0)) - analyzer_hits
            traces.append(trace)
            _checkpoint_append(group_root, trace)
            if not settled:
                raise BudgetExceeded(
                    "后台 API 请求未闭合；已保存逐题检查点，待账本闭合后安全续跑"
                )
    _complete_group_checkpoint(group_root, contract, traces)
    return traces, contract


def _answer_comparability(experiment: str, outputs: dict[str, list[dict]]) -> dict:
    """Document whether answer metrics share the same evidence-gate policy."""
    measured = {
        name: sum(
            (trace.get("gate") or {}).get("evidence_entailment") is not None
            for trace in traces
        )
        for name, traces in outputs.items()
    }
    policies = {
        frozenset(
            str(trace.get("case_id", index))
            for index, trace in enumerate(traces)
            if (trace.get("gate") or {}).get("evidence_entailment") is not None
        )
        for traces in outputs.values()
    }
    same_policy = len(policies) <= 1
    return {
        "retrieval_ranking_comparable": experiment in {"retrieval", "rag"},
        "answer_gate_policy_comparable": same_policy,
        "gate_entailment_measured_cases": measured,
        "answer_metric_scope": (
            "strategy_default_end_to_end" if experiment == "retrieval"
            else "same_hybrid_rerank_gate_across_rag_modes"
        ),
        "interpretation": (
            "检索排序指标可直接比较；答案指标包含各组默认闸门策略差异。"
            if experiment == "retrieval" and not same_policy
            else "答案指标未检测到组间闸门策略差异。"
        ),
    }


def _budget_audit(contracts: list[dict]) -> dict:
    """Distinguish a planning estimate from a hard limit approved before spending."""
    run = ((contracts[0].get("source_contract") or {}).get("run") or {})
    unbounded = bool(run.get("ch16_unbounded_budget"))
    max_cost = run.get("ch16_max_cost_usd")
    approval_artifact = bool(
        run.get("ch16_budget_approval")
        or run.get("budget_approval")
        or run.get("approval_reference")
    )
    return {
        "contract_mode": "unbounded" if unbounded else "bounded",
        "per_group_hard_limit_usd": None if unbounded else max_cost,
        "enforced_group_caps": bool(
            not unbounded and isinstance(max_cost, (int, float))
            and math.isfinite(float(max_cost)) and float(max_cost) >= 0
        ),
        "approval_artifact_recorded": approval_artifact,
        "cost_is_posthoc_estimate": True,
        "note": (
            "budget-estimate.json 是执行前规划值；只有契约中的有限逐组上限"
            "才表示运行时硬约束，实际金额仍须按调用账本事后对账。"
        ),
    }


def _build_experiment_report(
    args,
    cases: list[dict],
    outputs: dict[str, list[dict]],
    contracts: list[dict],
    *,
    validate_current_code: bool = True,
    attempt_ledger_summary: dict | None = None,
) -> dict:
    validate_group_contracts(
        contracts, verify_current_code=validate_current_code
    )
    protocol = contracts[0]["protocol_sha256"]
    all_cases = {case["id"]: case for case in cases}
    summaries = {name: summarize_group(traces, all_cases) for name, traces in outputs.items()}
    reference = "vector" if args.experiment == "retrieval" else "baseline"
    paired = {
        name: compare_groups(outputs[reference], traces, all_cases)
        for name, traces in outputs.items() if name != reference
    }
    pairwise = {}
    if args.experiment == "retrieval":
        names = list(outputs)
        for left_index, left in enumerate(names):
            for right in names[left_index + 1:]:
                pairwise[f"{right}_vs_{left}"] = compare_groups(
                    outputs[left], outputs[right], all_cases
                )
    buckets = paired_bucket_outcomes(
        outputs, RAG_BUCKETS if args.experiment == "rag"
        else tuple(sorted({case["bucket"] for case in cases}))
    )
    operations_by_bucket = {}
    if args.experiment == "rag":
        for name, traces in outputs.items():
            operations_by_bucket[name] = {}
            for bucket in RAG_BUCKETS:
                subset = [trace for trace in traces if trace.get("bucket") == bucket]
                operations_by_bucket[name][bucket] = summarize_group(subset)["operations"]
    report = {
        "schema_version": 1,
        "experiment": args.experiment,
        "split": args.split,
        "dataset_sha256": contracts[0]["dataset"].get("sha256"),
        "dataset_n": len(cases),
        "protocol_sha256": protocol,
        "paid_calls_executed": sum(
            row["operations"]["calls"]["paid_api_calls"] for row in summaries.values()
        ),
        "external_api_calls_executed": sum(
            row["operations"]["calls"]["external_api_calls"] for row in summaries.values()
        ),
        "budget_mode": "unbounded" if args.unbounded_budget else "bounded",
        "max_cost_usd_per_group": args.max_cost_usd,
        "reference_group": reference,
        "groups": summaries,
        "paired_vs_reference": paired,
        "pairwise_comparisons": pairwise,
        "paired_by_bucket": buckets,
        "operations_by_bucket": operations_by_bucket,
        "answer_comparability": _answer_comparability(args.experiment, outputs),
        "budget_audit": _budget_audit(contracts),
        "latency_semantics": (
            "每题端到端 wall-clock；operations.latency_ms.by_source 分别列出实际在线 API "
            "与精确响应缓存/本地路径。缓存重放延迟不能解释为线上请求延迟；API 请求耗时另记入 attempts ledger。"
        ),
        "measurement_status": {
            name: {
                "retrieval_measured": sum(
                    trace.get("retrieval_status") == "measured" for trace in traces
                ),
                "answers_measured": sum(
                    trace.get("answer_status") == "measured" for trace in traces
                ),
                "rag_measured": sum(
                    trace.get("rag_status") != "unmeasured" for trace in traces
                ) if args.experiment == "rag" else None,
            }
            for name, traces in outputs.items()
        },
    }
    if attempt_ledger_summary is None:
        attempt_ledger_summary = _summarize_attempt_ledger(
            args.attempt_ledger, split=args.split, experiment=args.experiment
        )
    report["api_attempt_ledger"] = attempt_ledger_summary
    report["cost_actual_usd"] = round(sum(
        row["operations"]["cost_actual_usd"] for row in summaries.values()
    ), 8)
    estimate = estimate_group_costs(args.split)
    selected_estimates = [
        row for row in estimate["groups"] if row["experiment"] == args.experiment
    ]
    report["cost_estimate"] = {
        "currency": "USD",
        "paid_calls_executed": report["paid_calls_executed"],
        "groups": selected_estimates,
        "total_estimated_usd": [
            round(sum(row["estimated_usd"][0] for row in selected_estimates), 4),
            round(sum(row["estimated_usd"][1] for row in selected_estimates), 4),
        ],
        "total_max_cost_usd_if_all_groups_run": round(
            sum(row["max_cost_usd_per_group"] for row in selected_estimates), 4
        ),
    }
    return report


def _run(args) -> dict:
    from scripts.eval_dataset import load_split

    if args.split not in {"dev", "frozen-v1"}:
        raise ContractError("CH16 只允许 dev 与 frozen-v1")
    validate_snapshot_identity(args.snapshot)
    validate_budget_request(
        allow_paid=args.allow_paid,
        max_cost_usd=args.max_cost_usd,
        ledger=args.ledger,
        unbounded=bool(args.unbounded_budget),
    )
    if args.ledger is None:
        args.ledger = ROOT / "tests" / "eval" / "rag_ablation" / "runs" / "live-cost-ledger.json"
    if args.attempt_ledger is None:
        args.attempt_ledger = ROOT / "tests" / "eval" / "rag_ablation" / "runs" / "live-attempts.jsonl"
    protected_ledger = (ROOT / "data" / "ch11_budget.json").resolve()
    if args.ledger is not None and Path(args.ledger).resolve() == protected_ledger:
        raise ContractError("CH16 不能写入 CH11 预算台账")
    if args.unbounded_budget and args.attempt_ledger is None:
        raise ContractError("无限付费运行必须显式指定 --attempt-ledger")
    if args.attempt_ledger is not None and Path(args.attempt_ledger).resolve() == protected_ledger:
        raise ContractError("调用账本不能覆盖 CH11 预算台账")
    _configure_environment(args)
    cases = load_split(args.split)
    if args.experiment == "retrieval":
        groups = [(name, strategy, "baseline") for name, strategy in RETRIEVAL_GROUPS.items()]
    else:
        groups = [(name, "hybrid_rerank", mode) for name, mode in RAG_GROUPS.items()]
    preflight_contracts = [
        _preflight_contract(args, cases, strategy=strategy, rag_mode=rag_mode)
        for _, strategy, rag_mode in groups
    ]
    validate_group_contracts(preflight_contracts)
    preflight_protocol = preflight_contracts[0]["protocol_sha256"]
    check_phase_lock(args.split, args.out / "dev-locks" / f"{args.experiment}.json",
                     preflight_protocol)
    check_cross_experiment_locks(
        args.split, args.out / "dev-locks", preflight_protocol
    )
    outputs = {}
    contracts = []
    for name, strategy, rag_mode in groups:
        if args.experiment == "retrieval":
            traces, contract = _run_retrieval_group(name, strategy, cases, args)
        else:
            traces, contract = _run_rag_group(rag_mode, cases, args)
        outputs[name] = traces
        contracts.append(contract)

    validate_group_contracts(contracts)
    report = _build_experiment_report(args, cases, outputs, contracts)
    protocol = contracts[0]["protocol_sha256"]
    check_phase_lock(args.split, args.out / "dev-locks" / f"{args.experiment}.json", protocol)
    phase_root = args.out / "runs" / args.split / args.experiment
    for (name, _, _), contract, traces in zip(groups, contracts,
                                               [outputs[name] for name, _, _ in groups]):
        group_root = phase_root / name
        _complete_group_checkpoint(group_root, contract, traces)
    _write_json(phase_root / "aggregate.json", report)
    _write_json(phase_root / "budget-estimate.json", report["cost_estimate"])
    if args.split == "dev":
        lock_root = args.out / "dev-locks"
        _write_json(lock_root / f"{args.experiment}.json", {
            "experiment": args.experiment,
            "protocol_sha256": protocol,
            "group_contracts": [contract["protocol_sha256"] for contract in contracts],
            "dev_dataset_sha256": contracts[0]["dataset"].get("sha256"),
            "analysis_code_sha256": contracts[0].get("analysis_code_sha256", {}),
            "locked_at_commit": subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "working_tree_dirty": bool(subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()),
            "threshold_tuning_after_lock": False,
        })
    return report


def _load_kb_rows(args) -> dict[int, object]:
    """Load only corpus rows needed by deterministic answer rescoring."""
    from app.config import Settings
    from app.db import make_engine, make_session_factory
    from app.kb import KnowledgeBaseStore
    from scripts.eval_snapshot import is_runtime_chunk

    validate_snapshot_identity(args.snapshot)
    _configure_environment(args)
    settings = Settings()
    factory = make_session_factory(make_engine(settings))
    kb = KnowledgeBaseStore(factory)
    return {
        row.id: row
        for row in kb.get_chunks(kb.doc_chunk_ids_all())
        if not is_runtime_chunk(row)
    }


def _scoring_code_hashes() -> dict[str, str]:
    return {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in (
            "scripts/eval_checks.py", "scripts/ch11_harness.py",
            "scripts/ch16_analysis.py", "scripts/eval_rag_ablation.py",
        )
    }


def _rescore_existing(args) -> dict:
    """Recompute deterministic answer checks over saved traces without model calls."""
    from scripts.ch11_harness import rescore_answer_trace
    from scripts.eval_dataset import load_split

    if args.split not in {"dev", "frozen-v1"}:
        raise ContractError("CH16 只允许 dev 与 frozen-v1")
    cases = load_split(args.split)
    cases_by_id = {case["id"]: case for case in cases}
    groups = (
        list(RETRIEVAL_GROUPS)
        if args.experiment == "retrieval"
        else list(RAG_GROUPS)
    )
    phase_root = args.out / "runs" / args.split / args.experiment
    aggregate_path = phase_root / "aggregate.json"
    if not aggregate_path.exists():
        raise ContractError(f"缺少待复算聚合:{aggregate_path}")
    previous = json.loads(aggregate_path.read_text(encoding="utf-8"))
    all_rows = _load_kb_rows(args)
    outputs: dict[str, list[dict]] = {}
    contracts: list[dict] = []
    for name in groups:
        group_root = phase_root / name
        contract_path = group_root / "contract.json"
        traces_path = group_root / "traces.jsonl"
        if not contract_path.exists() or not traces_path.exists():
            raise ContractError(f"缺少待复算轨迹:{group_root}")
        contracts.append(json.loads(contract_path.read_text(encoding="utf-8")))
        traces = [
            json.loads(line)
            for line in traces_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        ids = [str(trace.get("case_id", "")) for trace in traces]
        if len(ids) != len(set(ids)) or set(ids) != set(cases_by_id):
            raise ContractError(f"复算轨迹未完整覆盖 {args.split}:{group_root}")
        outputs[name] = [
            rescore_answer_trace(cases_by_id[str(trace["case_id"])], trace, all_rows)
            if trace.get("answer_status") == "measured" and trace.get("deterministic_checks")
            else trace
            for trace in traces
        ]

    report = {
        **previous,
        **_build_experiment_report(
            args,
            cases,
            outputs,
            contracts,
            validate_current_code=False,
            attempt_ledger_summary=previous.get("api_attempt_ledger", {}),
        ),
    }
    # Rescoring changes no request, response, latency or payment facts.
    for key in (
        "budget_mode", "max_cost_usd_per_group", "latency_semantics",
        "protocol_sha256", "dataset_sha256", "dataset_n",
    ):
        if key in previous:
            report[key] = previous[key]
    current_commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
    ).strip()
    working_tree_dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip())
    report["posthoc_rescore"] = {
        "scoring_version": "ch16-rescore-1",
        "model_calls": 0,
        "paid_calls": 0,
        "raw_answers_preserved": True,
        "recomputed": [
            "behavior_pred", "citation_support", "unsupported_claims",
            "refusal_rules", "failure_labels",
        ],
        "reason": "修复自然拒答与带标记澄清的确定性行为分类；模型答案原样保留",
        "code_basis_commit": current_commit,
        "working_tree_dirty": working_tree_dirty,
        "scoring_code_sha256": _scoring_code_hashes(),
    }
    for name in groups:
        _write_jsonl(phase_root / name / "traces.jsonl", outputs[name])
    _write_json(aggregate_path, report)
    _write_json(phase_root / "budget-estimate.json", report["cost_estimate"])
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CH16 同口径 RAG 消融与答案可信度对照")
    parser.add_argument("--experiment", choices=("retrieval", "rag"), required=True)
    parser.add_argument("--split", choices=("dev", "frozen-v1"), required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "tests" / "eval" / "rag_ablation")
    parser.add_argument("--snapshot", type=Path,
                        default=ROOT / "data" / "eval-snapshots" / f"{SNAPSHOT_BUILD_ID}.db")
    parser.add_argument("--sqlite-db", type=Path, default=ROOT / "data" / "ch12_eval.db")
    parser.add_argument("--cache-dir", type=Path,
                        default=ROOT / "data" / "eval-caches-ch16-live-2026-10-26")
    parser.add_argument("--pricing", type=Path,
                        default=ROOT / "tests" / "eval" / "rag_ablation" / "pricing-snapshot-2026-10-26.yaml")
    parser.add_argument("--ledger", type=Path, default=None,
                        help="CH16 专用费用台账；严禁指向 CH11 台账")
    parser.add_argument("--attempt-ledger", type=Path, default=None,
                        help="逐次 API 尝试追加日志，用于审计和安全续跑")
    parser.add_argument("--credential-env-file", type=Path, default=None,
                        help="可选只读凭据来源；仅读取三种 API key，不导入其他配置")
    parser.add_argument("--allow-paid", action="store_true")
    parser.add_argument("--max-cost-usd", type=float, default=0.0,
                        help="单组硬上限；默认 $0")
    parser.add_argument("--unbounded-budget", action="store_true",
                        help="显式启用无限累计预算；仍持久记录每次费用")
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument(
        "--rescore-only", action="store_true",
        help="只重算已保存答案的确定性检查与聚合；不调用模型、不改原始回答",
    )
    args = parser.parse_args(argv)
    if args.unbounded_budget:
        args.max_cost_usd = None
    if args.estimate_only:
        split = args.split
        print(json.dumps(estimate_group_costs(split), ensure_ascii=False, indent=2))
        return 0
    try:
        report = _rescore_existing(args) if args.rescore_only else _run(args)
    except BudgetExceeded as exc:
        print(f"预算守卫阻止运行: {exc}", file=sys.stderr)
        return 2
    except ContractError as exc:
        print(f"实验契约阻止运行: {exc}", file=sys.stderr)
        return 2
    output = {
        "experiment": report["experiment"],
        "split": report["split"],
        "aggregate": str(args.out / "runs" / args.split / args.experiment / "aggregate.json"),
    }
    if args.rescore_only:
        output.update({
            "model_calls_executed": 0,
            "historical_paid_calls": report["paid_calls_executed"],
        })
    else:
        output["paid_calls_executed"] = report["paid_calls_executed"]
    print(json.dumps(output, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
