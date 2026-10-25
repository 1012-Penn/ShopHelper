"""ch12 离线验收:冻结不动、默认路由零变化、循环界限、审计表、幂等、三组契约、
缓存重放零新增、台账零扣减、证据闸未放宽。对应设计稿 §6 验收标准 1-9。"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MANIFEST = ROOT / "tests" / "eval" / "manifests" / "frozen-v1.json"
GLOBAL_LEDGER = Path("/root/workplace/ShopHelper/.worktrees/ch11-eval-baseline/data/ch11_budget.json")
REPLAY_TRACES = ROOT / "reports" / "ch12" / "frozen-v1" / "traces.jsonl"


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)


def _check_frozen_unchanged() -> tuple[bool, str]:
    from scripts.acceptance_ch11 import OLD_SHA, _check_frozen, _check_regression

    ok_frozen, detail_frozen = _check_frozen()
    ok_reg, detail_reg = _check_regression()
    old_ok = all(
        path.exists() and _sha256(path) == sha
        for path, sha in ((ROOT / rel, sha) for rel, sha in OLD_SHA.items())
        for _ in [None]
    )
    return ok_frozen and ok_reg and old_ok, f"frozen[{detail_frozen}] regression[{detail_reg}] old-sha={old_ok}"


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_default_route_zero_drift() -> tuple[bool, str]:
    from app.config import Settings

    settings = Settings(openai_api_key="sk-test", _env_file=None)
    if settings.retrieval_route != "hybrid_rerank":
        return False, f"默认 retrieval_route={settings.retrieval_route}(必须 hybrid_rerank)"
    try:
        from tests.conftest import make_session_factory
        from tests.helpers import EchoResolver, GraphChatModel, StubExpand, builtin_engine

        from app.graph.builder import build_graph
        from app.guard import LowConfidencePool
        from app.kb import KnowledgeBaseStore

        factory = make_session_factory()
        graph = build_graph(
            GraphChatModel(), builtin_engine(), settings,
            _NoopStore(), LowConfidencePool(factory),
            _StubService(), KnowledgeBaseStore(factory),
            resolver=EchoResolver({}), expander=StubExpand({}))
        if "agentic_retrieve" in graph.nodes:
            return False, "默认路由下不应注册 agentic_retrieve"
        return True, "默认 hybrid_rerank:图无 agentic_retrieve,既有全量测试另由 tests 检查"
    except Exception as exc:  # noqa: BLE001
        return False, f"图构建异常:{exc}"


class _NoopStore:
    async def append(self, session_id, rows):
        return None


class _StubService:
    def retrieve(self, *args, **kwargs):
        raise AssertionError("默认路由不应触发 agentic 检索")


def _check_loop_bounds() -> tuple[bool, str]:
    result = _run([sys.executable, "-m", "pytest", "tests/test_ch12_rag_loop.py", "-q"])
    return result.returncode == 0, (result.stdout or "").strip().splitlines()[-1]


def _check_steps_table_audit_only() -> tuple[bool, str]:
    from app.models import RagLoopStep

    ddl = (ROOT / "db" / "init.sql").read_text(encoding="utf-8")
    model_doc = RagLoopStep.__doc__ or ""
    ok = (
        "rag_loop_steps" in ddl
        and "uk_rag_step_round" in ddl
        and hasattr(RagLoopStep, "turn_user_msg_id")
        and "仅可审计" in model_doc
        and "断点续跑" in model_doc
    )
    return ok, f"DDL+ORM 就绪;模型声明仅可审计(不含恢复/回放承诺):{ok}"


def _check_idempotency() -> tuple[bool, str]:
    result = _run([sys.executable, "-m", "pytest", "tests/test_ch12_idempotency.py", "-q"])
    return result.returncode == 0, (result.stdout or "").strip().splitlines()[-1]


def _check_contract_three_modes() -> tuple[bool, str]:
    from scripts.eval_contract import build_contract, contract_diff

    def contract(mode: str) -> dict:
        same = {
            "dataset": {"split": "frozen-v1", "sha256": "d", "n": 90},
            "corpus": {"docs_sha256": "c", "facts_sha256": "f", "doc_count": 8, "chunk_count": 130},
            "index": {"backend": "milvus-lite", "build_id": "b", "embedding_model": "e", "dim": 1024},
            "retrieval": {"strategy": "hybrid_rerank", "candidates": 50, "top_k": 10},
            "rewrite": {"enabled": True, "model": "m", "mode": "frozen", "cache_sha256": "r"},
            "llm": {"model": "m", "temperature": 0.0, "prompt_sha256": "p"},
            "judge": {"model": "m", "temperature": 0, "prompt_sha256": "j"},
            "pricing": {"version": "v", "currency": "USD", "verified": True},
            "run": {"git_commit": "x", "seed": 1},
        }
        return build_contract(
            **same,
            harness={"version": "ch11-harness-1" if mode == "baseline" else "ch11-harness-1-rag",
                     "mode": "full", "allow_paid": True, "use_judge": False, "rag_mode": mode},
            rag={"mode": mode, "max_rounds": 1 if mode == "decompose_once" else 3,
                 "token_budget": 3000, "time_budget_s": 45.0, "max_sub_queries": 3,
                 "analyzer_enabled": mode != "baseline", "analyzer_model": "m",
                 "analyzer_temperature": 0, "analyzer_prompt_sha256": "g"})

    base = contract("baseline")
    for mode in ("decompose_once", "controlled_loop"):
        changed = contract_diff(base, contract(mode))
        if not all(k.startswith("rag.") or k.startswith("harness.") for k in changed):
            return False, f"{mode} 契约差异越界:{changed}"
    return True, "三组契约差异仅 rag.*/harness.*(rag_mode、version)"


def _check_replay_zero_new_calls() -> tuple[bool, str]:
    if not REPLAY_TRACES.exists():
        return False, f"缺少缓存重放产物:{REPLAY_TRACES}"
    traces = [json.loads(line) for line in REPLAY_TRACES.read_text(encoding="utf-8").splitlines()]
    real_calls = sum(t.get("llm_calls", 0) for t in traces)
    cost = sum(t.get("cost_actual", 0.0) or 0.0 for t in traces)
    cached = sum(1 for t in traces if t.get("cached"))
    ok = len(traces) == 90 and real_calls == 0 and abs(cost) < 1e-9 and cached == 67
    return ok, (f"traces={len(traces)} 真实调用={real_calls} cost_actual=${cost:.6f} "
                f"缓存命中回答={cached}(与 CH11 报告 67 一致)")


def _check_ledger_zero_ch12_spend(ledger: Path) -> tuple[bool, str]:
    if not ledger.exists():
        return False, f"台账不存在:{ledger}"
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    records = payload.get("records") or []
    ch12_records = [r for r in records if "gap_analyzer" in str(r.get("note", ""))]
    total = sum(float(r.get("amount_usd") or 0.0) for r in records)
    ok = not ch12_records
    return ok, (f"记录数={len(records)} 累计=${total:.4f}/{payload.get('limit_usd')} "
                f"ch12(gap_analyzer)扣减={len(ch12_records)} 条")


def _check_gate_not_loosened() -> tuple[bool, str]:
    """证据闸与蕴含判定源码相对 origin/main 零改动(阈值/语义未放宽,ADR-5)。"""
    diff = _run(["git", "diff", "--stat", "origin/main", "--",
                 "app/retrieval.py", "app/evidence.py",
                 "config/evidence_calibration.json"])
    dirty = bool(diff.stdout.strip())
    if dirty:
        return False, f"证据闸相关文件被改动:{diff.stdout.strip()[:200]}"
    settings_note = "config/retrieval_route 默认 hybrid_rerank(ADR-2/5)"
    return True, f"app/retrieval.py、app/evidence.py、校准文件与 origin/main 零差异;{settings_note}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ch12 验收")
    parser.add_argument("--ledger", type=Path, default=GLOBAL_LEDGER)
    args = parser.parse_args(argv)

    checks = {
        "1-frozen-unchanged": _check_frozen_unchanged,
        "2-default-route-zero-drift": _check_default_route_zero_drift,
        "3-loop-bounds": _check_loop_bounds,
        "4-steps-table-audit-only": _check_steps_table_audit_only,
        "5-idempotency": _check_idempotency,
        "6-contract-three-modes": _check_contract_three_modes,
        "7-replay-zero-new-calls": _check_replay_zero_new_calls,
        "8-ledger-zero-ch12-spend": lambda: _check_ledger_zero_ch12_spend(args.ledger),
        "9-gate-not-loosened": _check_gate_not_loosened,
    }
    failed = 0
    for name, check in checks.items():
        try:
            ok, detail = check()
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, f"exception: {exc}"
        print(f"[{'ok' if ok else 'fail'}] {name}: {detail}")
        if not ok:
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
