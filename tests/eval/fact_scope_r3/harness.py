"""Frozen three-arm harness for fact identity and evidence-scope evaluation."""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal

import yaml
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.business_facts import policy_version_for_date
from app.chunking import Chunk
from app.config import Settings
from app.kb import KnowledgeBaseStore, vector_text, vectorize_pending
from app.llm import make_extract_model, make_structured_extractor
from app.models import Base
from app.orders import ORDER_CATALOG, extract_order_reference, get_order
from app.retrieval import (
    EvidenceCalibration,
    EvidenceScope,
    RetrievalService,
)
from scripts.build_kb import chunk_corpus
from scripts.ch15_paid_guard import (
    ArmLimit,
    GuardedChatModel,
    PersistentPaidRunBudget,
    paid_call_context,
)
from scripts.eval_cost import PricingTable
from tests.helpers import FakeEmbedding, FakeReranker, FakeVectorStore

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[2]
ENV_FILE = Path(
    os.environ.get("SHOPHELPER_ENV_FILE", "/root/workplace/ShopHelper/.env")
)
MODEL = "deepseek-flash"
ARMS = ("baseline", "identity", "scope")
MAX_EVIDENCE_ITEMS = 6
MAX_CONTEXT_CHARS = 6000
MAX_OUTPUT_TOKENS = 400
PRIOR_DEV_CHARGED_USD = 0.0450606
RUN_CAP_USD = 1.95
EXPERIMENT_ID = "fact-scope-r3-20261027-final"
FROZEN_PATH = PACKAGE_ROOT / "frozen-manifest.json"
RUNS_DIR = PACKAGE_ROOT / "runs"
LEDGER_PATH = RUNS_DIR / "fact-scope-r3-paid-final.sqlite3"
PRICING_PATH = PROJECT_ROOT / "config" / "pricing.yaml"
SYSTEM_PROMPT = """你是电商客服评测中的最终回答器。只使用给定的结构化订单工具结果和知识证据。
规则:
1. 结构化订单结果是当前订单事实的唯一来源;裸数字属于 legacy_demo,完整 ORD-* 属于 corpus。
2. 尾号或后四位同时匹配多个命名空间时不要猜,要求用户提供完整订单号。
3. 知识库只补充政策。没有显式主体/版本时保留泛化政策并说明哪些条件未知。
4. 订单签收时间与用户声称冲突时,以工具结果中的签收时间为准。
5. 只引用 retrieved_evidence 中存在的 chunk_id;不要编造订单、金额、日期或政策版本。"""


class AnswerOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["answer", "clarify", "no_match"]
    order_subject: str
    status: str | None = None
    amount: float | None = None
    policy_version: Literal["v1", "v2", "unknown"] = "unknown"
    window_days: int | None = None
    answer: str
    cited_chunk_ids: list[int] = []


@dataclass(frozen=True)
class CaseContext:
    arm: str
    order_result: dict | None
    resolution: dict | None
    evidence: list[dict]
    retrieval: dict
    prompt: str
    prompt_chars: int
    visible_ids: list[int]
    excluded_ids: list[int]


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}:{number} 不是合法 JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path.name}:{number} 必须是对象")
        rows.append(row)
    return rows


def load_split(split: str) -> tuple[list[dict], dict[str, dict]]:
    if split == "dev":
        cases_path = PACKAGE_ROOT / "cases-dev.jsonl"
        gold_path = PACKAGE_ROOT / "gold-dev.jsonl"
    elif split == "heldout":
        cases_path = PACKAGE_ROOT / "cases-heldout.jsonl"
        gold_path = PACKAGE_ROOT / "gold-heldout.jsonl"
    else:
        raise ValueError(f"未知 split:{split}")
    cases = _read_jsonl(cases_path)
    gold_rows = _read_jsonl(gold_path)
    gold = {row["id"]: row for row in gold_rows}
    if len(gold) != len(gold_rows):
        raise ValueError("gold id 重复")
    case_ids = {row["id"] for row in cases}
    if len(case_ids) != len(cases) or case_ids != set(gold):
        raise ValueError("case/gold id 不一一对应")
    for row in cases:
        if row.get("split") != split:
            raise ValueError(f"{row['id']} split 声明不一致")
        leaked = {"gold", "expected", "expected_behavior", "must_include"} & set(row)
        if leaked:
            raise ValueError(f"{row['id']} 含评分泄漏字段:{sorted(leaked)}")
    return cases, gold


def dataset_manifest() -> dict:
    cases, _ = load_split("dev")
    heldout, _ = load_split("heldout")
    groups = [row["group"] for row in cases + heldout]
    if len(groups) != len(set(groups)):
        raise ValueError("dev/held-out 的模板分组不唯一")
    files = [
        PACKAGE_ROOT / "cases-dev.jsonl",
        PACKAGE_ROOT / "gold-dev.jsonl",
        PACKAGE_ROOT / "cases-heldout.jsonl",
        PACKAGE_ROOT / "gold-heldout.jsonl",
        Path(__file__),
        PRICING_PATH,
    ]
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "model": MODEL,
        "arms": list(ARMS),
        "dev_cases": len(cases),
        "heldout_cases": len(heldout),
        "max_evidence_items": MAX_EVIDENCE_ITEMS,
        "max_context_chars": MAX_CONTEXT_CHARS,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "provider_retries_max": 2,
        "budget_cap_usd": 2.0,
        "prior_dev_charged_usd": PRIOR_DEV_CHARGED_USD,
        "remaining_budget_cap_usd": RUN_CAP_USD,
        "files": {
            str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
        },
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    payload["config_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def freeze_manifest() -> dict:
    payload = dataset_manifest()
    FROZEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    FROZEN_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def load_frozen_manifest() -> dict:
    if not FROZEN_PATH.is_file():
        raise ValueError("缺少 frozen-manifest.json;先运行 --freeze")
    frozen = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
    current = dataset_manifest()
    if frozen != current:
        raise ValueError("题集/评分/提示/配置在冻结后发生变化")
    return frozen


def _knowledge_service() -> tuple[KnowledgeBaseStore, RetrievalService]:
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    kb = KnowledgeBaseStore(factory)
    for doc_name, chunks in chunk_corpus().items():
        kb.replace_doc_chunks(doc_name, chunks)
    vectors = FakeVectorStore()
    vectorize_pending(kb, vectors, FakeEmbedding())
    service = RetrievalService(
        FakeEmbedding(),
        vectors,
        kb,
        reranker=FakeReranker(),
        final_top_k=MAX_EVIDENCE_ITEMS,
        evidence_calibration=EvidenceCalibration(
            top1_floor=0.03,
            effective_score_floor=0.03,
            min_effective_count=1,
            margin_floor=0.0,
            combined_floor=0.0,
        ),
    )
    return kb, service


def _baseline_order_result(order_ref: str | None) -> dict | None:
    if not order_ref:
        return None
    raw = ORDER_CATALOG.get(str(order_ref).strip())
    if raw is None:
        return {"order_id": str(order_ref).strip(), "error": "未找到该订单"}
    return dict(raw)


def _identity_order_result(
    case: dict,
) -> tuple[dict | None, dict | None]:
    reference = extract_order_reference(case["question"])
    if reference.ambiguous:
        return None, {
            "ambiguous": True,
            "reason": reference.reason,
        }
    if reference.order_id is None:
        return None, None
    return get_order(reference.order_id), {
        "ambiguous": False,
        "canonical_id": reference.canonical_id,
    }


def _scope_for(
    case: dict,
    order_result: dict | None,
    resolution: dict | None,
) -> EvidenceScope:
    subject = (order_result or {}).get("subject", {}).get("canonical_id")
    reference_date = None
    policy_version = None
    if order_result and order_result.get("signed_at"):
        reference_date = date.fromisoformat(str(order_result["signed_at"])[:10])
        policy_version = policy_version_for_date(reference_date)
    elif not case.get("order_ref") and case.get("hypothetical_date"):
        reference_date = date.fromisoformat(case["hypothetical_date"])
        policy_version = policy_version_for_date(reference_date)
    return EvidenceScope(
        subject=subject if not (resolution or {}).get("ambiguous") else None,
        policy_version=policy_version,
        reference_date=reference_date,
    )


def build_context(case: dict, arm: str, kb: KnowledgeBaseStore,
                  service: RetrievalService) -> CaseContext:
    resolution = None
    if arm == "baseline":
        order_result = _baseline_order_result(case.get("order_ref"))
    else:
        order_result, resolution = _identity_order_result(case)

    scope = _scope_for(case, order_result, resolution) if arm == "scope" else None
    result = service.retrieve(
        case["policy_query"],
        strategy="hybrid_rerank",
        scope=scope,
    )
    visible_items = [] if result.low_confidence else result.items
    rows = kb.get_chunks([item.chunk_id for item in visible_items])
    evidence = []
    used_chars = 0
    truncated = False
    for item, row in zip(visible_items, rows):
        text = vector_text(row.category, row.questions, row.answer)
        section = row.section_path or ""
        remaining = MAX_CONTEXT_CHARS - used_chars
        if remaining <= 0:
            truncated = True
            break
        body = f"{section}\n{text}"
        if len(body) > remaining:
            body = body[:remaining]
            truncated = True
        evidence.append(
            {
                "chunk_id": item.chunk_id,
                "section_path": section,
                "text": body,
                "score": round(float(item.score), 6),
            }
        )
        used_chars += len(body)

    retrieval = {
        "low_confidence": result.low_confidence,
        "reason": result.reason,
        "filter_fallback": result.filter_fallback,
        "scope_status": (
            result.scope_status if arm == "scope"
            else ("unknown" if scope is not None else "not_requested")
        ),
        "scope_excluded_count": len(result.scope_excluded) if arm == "scope" else 0,
        "truncated": truncated,
    }
    payload = {
        "question": case["question"],
        "order_tool_result": order_result,
        "resolution": resolution,
        "retrieved_evidence": evidence,
        "retrieval": retrieval,
    }
    prompt = (
        f"{SYSTEM_PROMPT}\n"
        "请根据以下 JSON 上下文作出客服决策。"
        "order_subject 使用 canonical_id 或 unknown。\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )
    return CaseContext(
        arm=arm,
        order_result=order_result,
        resolution=resolution,
        evidence=evidence,
        retrieval=retrieval,
        prompt=prompt,
        prompt_chars=len(prompt),
        visible_ids=[item["chunk_id"] for item in evidence],
        excluded_ids=[
            int(item["chunk_id"])
            for item in result.scope_excluded
        ] if arm == "scope" else [],
    )


def relevant_chunk_ids(kb: KnowledgeBaseStore, markers: list[str]) -> list[int]:
    if not markers:
        return []
    ids = kb.doc_chunk_ids_all()
    rows = kb.get_chunks(ids)
    found = set()
    for row in rows:
        haystack = f"{row.section_path or ''}\n{row.questions}\n{row.answer}"
        if any(marker in haystack for marker in markers):
            found.add(row.id)
    return sorted(found)


def _normalized(text: str) -> str:
    return "".join(str(text).casefold().split())


def score_output(
    output: AnswerOutput,
    gold: dict,
    context: CaseContext,
    relevant_ids: list[int],
) -> dict:
    answer = _normalized(output.answer)
    required_missing = [
        term for term in gold.get("required_terms", []) if _normalized(term) not in answer
    ]
    forbidden_present = [
        term for term in gold.get("forbidden_terms", []) if _normalized(term) in answer
    ]
    subject_expected = gold.get("subject") or "unknown"
    subject_correct = output.order_subject == subject_expected
    action_correct = output.action == gold["action"]

    field_mismatches = {}
    for key in ("status", "amount", "policy_version", "window_days"):
        expected = gold.get(key)
        if expected is None:
            continue
        actual = getattr(output, key)
        if actual != expected:
            field_mismatches[key] = {"expected": expected, "actual": actual}

    conditions_ok = not required_missing and not forbidden_present and not field_mismatches
    task_success = subject_correct and action_correct and conditions_ok
    relevant_visible = [chunk_id for chunk_id in relevant_ids if chunk_id in context.visible_ids]
    wrong_filter = [chunk_id for chunk_id in relevant_ids if chunk_id in context.excluded_ids]
    recall = (
        len(relevant_visible) / len(relevant_ids)
        if relevant_ids else None
    )
    citation_ids = set(output.cited_chunk_ids)
    return {
        "subject_correct": subject_correct,
        "action_correct": action_correct,
        "conditions_correct": conditions_ok,
        "task_success": task_success,
        "required_missing": required_missing,
        "forbidden_present": forbidden_present,
        "field_mismatches": field_mismatches,
        "entity_cross_use": (
            gold.get("subject")
            and output.order_subject not in {gold["subject"], "unknown", None}
        ),
        "false_refusal": gold["action"] == "answer" and output.action in {"clarify", "no_match"},
        "wrong_filter_ids": wrong_filter,
        "relevant_ids": relevant_ids,
        "relevant_visible_ids": relevant_visible,
        "recall_at_k": recall,
        "citation_ids_valid": citation_ids.issubset(set(context.visible_ids)),
    }


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return round(ordered[index], 3)


def summarize(results: list[dict]) -> dict:
    summary = {}
    for arm in ARMS:
        rows = [row for row in results if row["arm"] == arm]
        scores = [row["score"] for row in rows]
        r_at_k = [item["recall_at_k"] for item in scores if item["recall_at_k"] is not None]
        latencies = [row["duration_ms"] for row in rows]
        summary[arm] = {
            "cases": len(rows),
            "task_success": sum(item["task_success"] for item in scores),
            "subject_correct": sum(item["subject_correct"] for item in scores),
            "conditions_correct": sum(item["conditions_correct"] for item in scores),
            "action_correct": sum(item["action_correct"] for item in scores),
            "entity_cross_use": sum(bool(item["entity_cross_use"]) for item in scores),
            "false_refusal": sum(item["false_refusal"] for item in scores),
            "wrong_filter_cases": sum(bool(item["wrong_filter_ids"]) for item in scores),
            "r_at_k_numerator": sum(item["relevant_visible_ids"].__len__() for item in scores),
            "r_at_k_denominator": sum(len(item["relevant_ids"]) for item in scores),
            "r_at_k_eligible_cases": len(r_at_k),
            "mean_recall_at_k": (
                round(sum(r_at_k) / len(r_at_k), 6) if r_at_k else None
            ),
            "p95_ms": _p95(latencies),
            "input_tokens": sum(row["usage"].get("input_tokens", 0) for row in rows),
            "output_tokens": sum(row["usage"].get("output_tokens", 0) for row in rows),
            "cost_usd": round(sum(row["cost_usd"] for row in rows), 8),
            "technical_failures": sum(bool(row.get("technical_error")) for row in rows),
        }

    by_case: dict[str, dict[str, bool]] = {}
    for row in results:
        by_case.setdefault(row["case_id"], {})[row["arm"]] = row["score"]["task_success"]
    pairs = {}
    for left, right in (("baseline", "identity"), ("identity", "scope")):
        left_only = sum(
            values.get(left) and not values.get(right)
            for values in by_case.values()
        )
        right_only = sum(
            values.get(right) and not values.get(left)
            for values in by_case.values()
        )
        ties = sum(
            values.get(left) == values.get(right)
            for values in by_case.values()
        )
        pairs[f"{right}_vs_{left}"] = {
            "right_wins": right_only,
            "left_wins": left_only,
            "ties": ties,
        }
    return {"arms": summary, "paired": pairs}


def _settings() -> Settings:
    if not ENV_FILE.is_file():
        raise RuntimeError("根 .env 不存在;不能在内部读取 provider 凭证")
    return Settings(
        _env_file=ENV_FILE,
        openai_model=MODEL,
        llm_max_retries=0,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        extract_temperature=0,
        chat_temperature=0,
        evidence_entailment_enabled=False,
        query_rewrite_enabled=False,
    )


def _extractor_and_ledger(config_hash: str):
    settings = _settings()
    pricing = PricingTable.load(PRICING_PATH)
    limits = {
        arm: ArmLimit(max_generations=30, cap_usd=0.64)
        for arm in ARMS
    }
    ledger = PersistentPaidRunBudget(
        pricing,
        path=LEDGER_PATH,
        experiment_id=EXPERIMENT_ID,
        config_hash=config_hash,
        model=MODEL,
        arm_limits=limits,
        run_cap_usd=RUN_CAP_USD,
        max_input_tokens=20_000,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        allow_paid=True,
    )
    raw: ChatOpenAI = make_extract_model(settings, model=MODEL)
    guarded = GuardedChatModel(raw, ledger, "baseline", "answer")
    extractor = make_structured_extractor(
        settings,
        AnswerOutput,
        model=MODEL,
        base_model=guarded,
    )
    return settings, ledger, extractor


async def _invoke_with_retries(extractor, ledger, arm: str, case_id: str, prompt: str):
    last_error = None
    guarded = getattr(extractor, "model", extractor)
    if hasattr(guarded, "arm"):
        guarded.arm = arm
    for attempt in range(3):
        started = time.perf_counter()
        try:
            with paid_call_context(
                scenario_id=case_id,
                arm=arm,
                turn=1,
                run_id=EXPERIMENT_ID,
            ):
                output = await extractor.ainvoke(prompt)
            return output, int((time.perf_counter() - started) * 1000), attempt
        except Exception as exc:
            last_error = exc
            if attempt >= 2:
                break
            await asyncio.sleep(2 ** attempt)
    return None, 0, last_error


async def run_split(split: str) -> Path:
    frozen = load_frozen_manifest()
    cases, gold = load_split(split)
    if split == "heldout" and not (RUNS_DIR / "live-20261027-dev" / "summary.json").is_file():
        raise RuntimeError("必须先完成 dev 运行并在冻结配置下检查 harness")
    run_dir = RUNS_DIR / f"live-20261027-{split}"
    if run_dir.exists():
        raise RuntimeError(f"{run_dir} 已存在;一次预留运行不得覆盖")
    run_dir.mkdir(parents=True)
    kb, service = _knowledge_service()
    relevance = {
        row["id"]: relevant_chunk_ids(kb, row.get("relevant_markers", []))
        for row in gold.values()
    }
    _, ledger, extractor = _extractor_and_ledger(frozen["config_hash"])
    results: list[dict] = []

    try:
        for case in cases:
            for arm in ARMS:
                context = build_context(case, arm, kb, service)
                output, duration_ms, retries_or_error = await _invoke_with_retries(
                    extractor,
                    ledger,
                    arm,
                    case["id"],
                    context.prompt,
                )
                calls = ledger.calls_for_case(case["id"], arm)
                usage = {
                    "input_tokens": sum(call.get("input_tokens") or 0 for call in calls),
                    "output_tokens": sum(call.get("output_tokens") or 0 for call in calls),
                }
                cost_usd = sum(call.get("charged_usd") or 0.0 for call in calls)
                if output is None:
                    score = {
                        "subject_correct": False,
                        "action_correct": False,
                        "conditions_correct": False,
                        "task_success": False,
                        "required_missing": gold[case["id"]].get("required_terms", []),
                        "forbidden_present": [],
                        "field_mismatches": {},
                        "entity_cross_use": False,
                        "false_refusal": False,
                        "wrong_filter_ids": [],
                        "relevant_ids": relevance[case["id"]],
                        "relevant_visible_ids": [],
                        "recall_at_k": (
                            0.0 if relevance[case["id"]] else None
                        ),
                        "citation_ids_valid": False,
                    }
                    technical_error = type(retries_or_error).__name__
                    retry_count = max(0, len(calls) - 1)
                    model_output = None
                else:
                    score = score_output(
                        output,
                        gold[case["id"]],
                        context,
                        relevance[case["id"]],
                    )
                    technical_error = None
                    retry_count = int(retries_or_error)
                    model_output = output.model_dump()
                results.append(
                    {
                        "case_id": case["id"],
                        "split": split,
                        "group": case["group"],
                        "arm": arm,
                        "prompt_sha256": hashlib.sha256(
                            context.prompt.encode("utf-8")
                        ).hexdigest(),
                        "prompt_chars": context.prompt_chars,
                        "order_tool_result": context.order_result,
                        "resolution": context.resolution,
                        "retrieval": context.retrieval,
                        "visible_chunk_ids": context.visible_ids,
                        "excluded_chunk_ids": context.excluded_ids,
                        "model_output": model_output,
                        "technical_error": technical_error,
                        "retry_count": retry_count,
                        "duration_ms": duration_ms,
                        "usage": usage,
                        "cost_usd": round(cost_usd, 10),
                        "score": score,
                    }
                )
                print(
                    f"{split} {case['id']} {arm}: "
                    f"success={score['task_success']} duration={duration_ms}ms",
                    flush=True,
                )
    finally:
        (run_dir / "results.jsonl").write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in results
            ),
            encoding="utf-8",
        )
        summary = {
            "experiment_id": EXPERIMENT_ID,
            "split": split,
            "config_hash": frozen["config_hash"],
            "frozen_manifest": frozen,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "summary": summarize(results),
            "ledger": ledger.snapshot(),
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return run_dir


def preflight() -> dict:
    cases, gold = load_split("dev")
    heldout, heldout_gold = load_split("heldout")
    for case, labels in [(row, gold) for row in cases] + [
        (row, heldout_gold) for row in heldout
    ]:
        for forbidden in ("gold", "expected", "expected_behavior", "must_include"):
            if forbidden in case:
                raise AssertionError(f"{case['id']} 泄漏字段 {forbidden}")
        if case["id"] not in labels:
            raise AssertionError(f"{case['id']} 缺少 gold")
    return {
        "status": "ok",
        "dev_cases": len(cases),
        "heldout_cases": len(heldout),
        "groups_disjoint": len(
            {row["group"] for row in cases + heldout}
        ) == len(cases + heldout),
        "model": MODEL,
        "arms": list(ARMS),
        "config_hash": dataset_manifest()["config_hash"],
    }
