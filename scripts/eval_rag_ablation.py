"""CH16 分阶段、零预算默认的 RAG 消融运行器。

底层检索、答案 prompt、确定性判分和 CH12 循环全部复用 eval_ch11。
普通策略分开记录检索层与可缓存的答案重放；任何零预算缓存未命中均标为未测。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import subprocess
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
from scripts.eval_cost import BudgetExceeded

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
        extra = 0.00002 * scale if strategy in {"dense", "hybrid", "hybrid_rerank"} else 0.0
        if strategy == "hybrid_rerank":
            extra += 0.005 * scale
        low = max(0.0, answer_and_judge * 0.70 + extra)
        high = answer_and_judge * 1.50 + extra
        table.append({
            "experiment": "retrieval",
            "group": name,
            "strategy": strategy,
            "estimated_usd": [round(low, 4), round(high, 4)],
            "max_cost_usd_per_group": retrieval_cap,
            "basis": "CH11 首跑回答+证据判断 $0.1019/90 题，含嵌入/重排占位价",
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
        "note": "估算依赖缓存命中与实际 token；每组分别限额、串行执行，不是新增调用授权。",
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, traces: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for trace in traces:
            stream.write(json.dumps(trace, ensure_ascii=False, sort_keys=True) + "\n")


def _database_url(sqlite_db: Path) -> str:
    absolute = sqlite_db.resolve()
    return f"sqlite:///{absolute}"


def _configure_environment(args) -> None:
    """固定评测模型与私有 SQLite，不把 CH16 路由到共享业务库。"""
    from dotenv import dotenv_values

    local_env = dotenv_values(ROOT / ".env")
    database_url = local_env.get("DATABASE_URL")
    if database_url:
        expected = _database_url(args.sqlite_db)
        if database_url != expected:
            raise ContractError("CH16 必须使用本 worktree 的私有 SQLite 评测库")
    model = local_env.get("OPENAI_MODEL")
    if model and model != REFERENCE_MODEL:
        raise ContractError("CH16 模型必须与 CH11 contract 的 deepseek-flash 一致")
    if local_env.get("HYBRID_CANDIDATES") and local_env["HYBRID_CANDIDATES"] != "50":
        raise ContractError("CH16 必须固定 HYBRID_CANDIDATES=50")
    if local_env.get("RERANK_TOP_K") and local_env["RERANK_TOP_K"] != "10":
        raise ContractError("CH16 必须固定 RERANK_TOP_K=10")
    os.environ["DATABASE_URL"] = _database_url(args.sqlite_db)
    os.environ["OPENAI_MODEL"] = REFERENCE_MODEL
    os.environ["HYBRID_CANDIDATES"] = "50"
    os.environ["RERANK_TOP_K"] = "10"
    if not os.environ.get("OPENAI_API_KEY"):
        if args.allow_paid and args.max_cost_usd > 0:
            raise ContractError("付费运行缺少 API key")
        os.environ["OPENAI_API_KEY"] = "offline-cache-replay-only"


def _context(config, args, cases):
    from scripts.eval_ch11 import _build_context
    from app.config import Settings

    namespace = SimpleNamespace(
        snapshot=args.snapshot,
        cache_dir=args.cache_dir,
        pricing=args.pricing,
        ledger=args.ledger,
        allow_paid=True,  # 缓存读取先于 CostTracker.guard；max_cost=0 会阻断所有非缓存调用。
        no_rewrite=False,
        rewrite_cache=None,
        rewrite_cache_mode="read",
    )
    ctx, config, source_contract = _build_context(config, namespace, cases)
    settings = Settings()
    if (ctx.service._candidates, ctx.service._final_top_k) != (50, 10):
        raise ContractError("运行时检索上下文上限漂移:CH16 固定 50 个候选 / top 10 上下文")
    # CH11 只在存在线上 key 时装配 RerankCache。CH16 允许复用既有 rerank
    # 结果；缓存 miss 通过 BudgetExceeded 中止并记未测，绝不降级伪装成 rerank。
    if ctx.service._reranker is None:
        ctx.service._reranker = cache_only_reranker(
            Path(args.cache_dir) / "rerank.json",
            settings.rerank_model,
        )
    source_contract["retrieval"]["rerank_model"] = settings.rerank_model
    source_contract["retrieval"]["rerank_cache_policy"] = "cache_then_zero_budget_block"
    source_contract["run"]["ch16_allow_paid"] = bool(args.allow_paid)
    source_contract["run"]["ch16_max_cost_usd"] = float(args.max_cost_usd)
    return ctx, config, source_contract


def _preflight_protocol(args, cases) -> str:
    """在首题执行前从同一快照构造共同协议，冻结阶段先比对 dev lock。"""
    from scripts.ch11_harness import HarnessConfig

    if args.experiment == "retrieval":
        strategy, rag_mode = "dense", "baseline"
    else:
        strategy, rag_mode = "hybrid_rerank", "baseline"
    config = HarnessConfig(
        split=args.split,
        mode="full",
        strategy=strategy,
        max_cost_usd=args.max_cost_usd,
        allow_paid=True,
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
    return contract["protocol_sha256"]


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


def _cost_delta(ctx, before_calls: int, before_spent: float) -> tuple[int, float]:
    return (
        int(getattr(ctx.cost, "calls", 0)) - before_calls,
        round(float(getattr(ctx.cost, "spent_usd", 0.0)) - before_spent, 8),
    )


def cache_only_reranker(path: Path, model: str):
    """只读既有 rerank 缓存；miss 必须穿透为预算失败，不能静默降级。"""
    from scripts.eval_cache import RerankCache

    class _CacheMissBlockedReranker:
        def rerank(self, query, documents, top_n=None):
            raise BudgetExceeded("CH16 零预算：rerank 缓存未命中，未发起外部调用")

    return RerankCache(_CacheMissBlockedReranker(), Path(path), model)


def _run_retrieval_group(name: str, strategy: str, cases: list[dict], args) -> tuple[list[dict], dict]:
    from scripts.ch11_harness import HarnessConfig, run_case, run_retrieval_case
    from scripts.eval_snapshot import acquire_eval_lock

    config = HarnessConfig(
        split=args.split,
        mode="full",
        strategy=strategy,
        max_cost_usd=args.max_cost_usd,
        allow_paid=True,
        rag_mode="baseline",
        temperature=0.0,
    )
    ctx, config, source_contract = _context(config, args, cases)
    judge = getattr(ctx.service, "_evidence_judge", None)
    retrieval_traces = []
    full_traces = []
    with acquire_eval_lock():
        for case in cases:
            retrieval = _base_trace(case)
            before_calls = int(getattr(ctx.cost, "calls", 0))
            before_spent = float(getattr(ctx.cost, "spent_usd", 0.0))
            ctx.service._evidence_judge = None
            try:
                retrieval = run_retrieval_case(
                    case, ctx.service, ctx.all_rows, ctx.gold_by_case, ctx.prepared,
                    strategy=strategy,
                )
                retrieval.pop("evidence_items", None)
                retrieval.pop("evidence_by_n", None)
                retrieval["retrieval_status"] = "measured"
            except BudgetExceeded as exc:
                retrieval["retrieval_status"] = "unmeasured"
                retrieval["retrieval_unmeasured_reason"] = str(exc)[:240]
            finally:
                ctx.service._evidence_judge = judge
            paid_calls, cost = _cost_delta(ctx, before_calls, before_spent)
            retrieval["paid_api_calls"] = paid_calls
            retrieval["cost_actual"] = cost
            retrieval_traces.append(retrieval)

            before_calls = int(getattr(ctx.cost, "calls", 0))
            before_spent = float(getattr(ctx.cost, "spent_usd", 0.0))
            try:
                full = run_case(case, ctx, config)
                full["retrieval_status"] = "measured"
                full["answer_status"] = "measured"
                full_traces.append(full)
            except BudgetExceeded as exc:
                failed = deepcopy(retrieval)
                failed["answer_status"] = "unmeasured"
                failed["answer_unmeasured_reason"] = str(exc)[:240]
                failed["deterministic_checks"] = None
                full_traces.append(failed)
            paid_calls, cost = _cost_delta(ctx, before_calls, before_spent)
            full_traces[-1]["paid_api_calls"] = int(full_traces[-1].get("paid_api_calls", 0)) + paid_calls
            full_traces[-1]["cost_actual"] = round(
                float(retrieval.get("cost_actual", 0.0)) + cost, 8
            )
    if source_contract.get("llm", {}).get("model") != REFERENCE_MODEL:
        raise ContractError("回答模型与 CH11 不一致")
    if source_contract.get("rewrite", {}).get("mode") != "frozen":
        raise ContractError("改写缓存未能按 CH11 contract 复用")
    contract = build_experiment_contract(
        split=args.split,
        experiment="retrieval",
        strategy=strategy,
        rag_mode="baseline",
        source_contract=source_contract,
    )
    return full_traces, contract


def _run_rag_group(mode: str, cases: list[dict], args) -> tuple[list[dict], dict]:
    from scripts.ch11_harness import HarnessConfig, run_case
    from scripts.eval_snapshot import acquire_eval_lock

    config = HarnessConfig(
        split=args.split,
        mode="full",
        strategy="hybrid_rerank",
        max_cost_usd=args.max_cost_usd,
        allow_paid=True,
        rag_mode=mode,
        temperature=0.0,
    )
    ctx, config, source_contract = _context(config, args, cases)
    gap = ctx.gap_analyzer
    judge = getattr(ctx.service, "_evidence_judge", None)
    traces = []
    with acquire_eval_lock():
        for case in cases:
            before_calls = int(getattr(ctx.cost, "calls", 0))
            before_spent = float(getattr(ctx.cost, "spent_usd", 0.0))
            analyzer_calls = int(getattr(gap, "calls", 0))
            analyzer_hits = int(getattr(gap, "hits", 0))
            try:
                trace = run_case(case, ctx, config)
                trace["retrieval_status"] = "measured"
                trace["answer_status"] = "measured"
            except BudgetExceeded as exc:
                trace = _base_trace(case)
                trace["rag_status"] = "unmeasured"
                reason = str(exc)[:240]
                trace["rag_unmeasured_reason"] = reason
                trace["retrieval_unmeasured_reason"] = reason
                trace["answer_unmeasured_reason"] = reason
            paid_calls, cost = _cost_delta(ctx, before_calls, before_spent)
            trace["paid_api_calls"] = paid_calls
            trace["cost_actual"] = cost
            trace["rag_analyzer_api_calls"] = int(getattr(gap, "calls", 0)) - analyzer_calls
            trace["rag_analyzer_cached"] = int(getattr(gap, "hits", 0)) - analyzer_hits
            traces.append(trace)
    if source_contract.get("llm", {}).get("model") != REFERENCE_MODEL:
        raise ContractError("回答模型与 CH11 不一致")
    if source_contract.get("rewrite", {}).get("mode") != "frozen":
        raise ContractError("改写缓存未能按 CH11 contract 复用")
    contract = build_experiment_contract(
        split=args.split,
        experiment="rag",
        strategy="hybrid_rerank",
        rag_mode=mode,
        source_contract=source_contract,
    )
    return traces, contract


def _run(args) -> dict:
    from scripts.eval_dataset import load_split

    if args.split not in {"dev", "frozen-v1"}:
        raise ContractError("CH16 只允许 dev 与 frozen-v1")
    validate_snapshot_identity(args.snapshot)
    validate_budget_request(
        allow_paid=args.allow_paid,
        max_cost_usd=args.max_cost_usd,
        ledger=args.ledger,
    )
    if args.max_cost_usd > 0 and args.split == "frozen-v1":
        raise ContractError("冻结集只允许在单组逐项预算完成统一安排后另行运行")
    _configure_environment(args)
    cases = load_split(args.split)
    preflight_protocol = _preflight_protocol(args, cases)
    check_phase_lock(args.split, args.out / "dev-locks" / f"{args.experiment}.json",
                     preflight_protocol)
    check_cross_experiment_locks(
        args.split, args.out / "dev-locks", preflight_protocol
    )
    if args.experiment == "retrieval":
        groups = [(name, strategy, None) for name, strategy in RETRIEVAL_GROUPS.items()]
    else:
        groups = [(name, "hybrid_rerank", mode) for name, mode in RAG_GROUPS.items()]

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
    protocol = contracts[0]["protocol_sha256"]
    check_phase_lock(args.split, args.out / "dev-locks" / f"{args.experiment}.json", protocol)
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
    buckets = paired_bucket_outcomes(outputs, RAG_BUCKETS if args.experiment == "rag"
                                     else tuple(sorted({case["bucket"] for case in cases})))
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
        "max_cost_usd_per_group": args.max_cost_usd,
        "reference_group": reference,
        "groups": summaries,
        "paired_vs_reference": paired,
        "pairwise_comparisons": pairwise,
        "paired_by_bucket": buckets,
        "operations_by_bucket": operations_by_bucket,
        "latency_semantics": (
            "本地产生的检索/缓存回放 wall-clock；不是新模型调用的线上延迟。"
            "付费调用为 0，故不可据此推断线上 P50/P95。"
        ),
        "measurement_status": {
            name: {
                "retrieval_measured": sum(t.get("retrieval_status") == "measured" for t in traces),
                "answers_measured": sum(t.get("answer_status") == "measured" for t in traces),
                "rag_measured": sum(t.get("rag_status") != "unmeasured" for t in traces)
                if args.experiment == "rag" else None,
            }
            for name, traces in outputs.items()
        },
    }
    phase_root = args.out / "runs" / args.split / args.experiment
    estimate = estimate_group_costs(args.split)
    selected_estimates = [
        row for row in estimate["groups"] if row["experiment"] == args.experiment
    ]
    report["cost_estimate"] = {
        "currency": "USD",
        "paid_calls_executed": 0,
        "groups": selected_estimates,
        "total_estimated_usd": [
            round(sum(row["estimated_usd"][0] for row in selected_estimates), 4),
            round(sum(row["estimated_usd"][1] for row in selected_estimates), 4),
        ],
        "total_max_cost_usd_if_all_groups_run": round(
            sum(row["max_cost_usd_per_group"] for row in selected_estimates), 4
        ),
    }
    for (name, _, _), contract, traces in zip(groups, contracts,
                                               [outputs[name] for name, _, _ in groups]):
        group_root = phase_root / name
        _write_json(group_root / "contract.json", contract)
        _write_jsonl(group_root / "traces.jsonl", traces)
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CH16 同口径 RAG 消融与答案可信度对照")
    parser.add_argument("--experiment", choices=("retrieval", "rag"), required=True)
    parser.add_argument("--split", choices=("dev", "frozen-v1"), required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "tests" / "eval" / "rag_ablation")
    parser.add_argument("--snapshot", type=Path,
                        default=ROOT / "data" / "eval-snapshots" / f"{SNAPSHOT_BUILD_ID}.db")
    parser.add_argument("--sqlite-db", type=Path, default=ROOT / "data" / "ch12_eval.db")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "eval-caches")
    parser.add_argument("--pricing", type=Path, default=ROOT / "config" / "pricing.yaml")
    parser.add_argument("--ledger", type=Path, default=None,
                        help="付费运行必须指向统一预算台账；零预算缓存回放不写台账")
    parser.add_argument("--allow-paid", action="store_true")
    parser.add_argument("--max-cost-usd", type=float, default=0.0,
                        help="单组硬上限；默认 $0，只复用缓存")
    parser.add_argument("--estimate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.estimate_only:
        split = args.split
        print(json.dumps(estimate_group_costs(split), ensure_ascii=False, indent=2))
        return 0
    try:
        report = _run(args)
    except BudgetExceeded as exc:
        print(f"预算守卫阻止运行: {exc}", file=sys.stderr)
        return 2
    except ContractError as exc:
        print(f"实验契约阻止运行: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "experiment": report["experiment"],
        "split": report["split"],
        "paid_calls_executed": report["paid_calls_executed"],
        "aggregate": str(args.out / "runs" / args.split / args.experiment / "aggregate.json"),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
