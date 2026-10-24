"""评测契约 `eval_contract.json`:同 contract 才允许做对照。

`contract_sha256` 排除运行时间/主机等易变字段,保证同一配置得到同一指纹。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

CONTRACT_VERSION = 1
FAILURE_LABELS = (
    "retrieval_miss",
    "entity_confusion",
    "version_conflict_missed",
    "evidence_insufficient",
    "gate_false_accept",
    "gate_false_block",
    "answer_wrong_fact",
    "answer_missing_condition",
    "forbidden_claim",
    "citation_unsupported",
    "over_refusal",
    "under_refusal",
    "clarify_missing_slot",
    "clarify_over_asking",
    "verification_overclaim",
    "tool_error",
    "timeout",
    "cost_over_budget",
    "data_error",
)
VOLATILE_RUN_FIELDS = ("started_at", "host")


class ContractError(ValueError):
    """契约缺字段或版本不兼容。"""


def build_contract(
    *,
    dataset: dict,
    corpus: dict,
    index: dict,
    retrieval: dict,
    rewrite: dict,
    llm: dict,
    judge: dict,
    pricing: dict,
    run: dict,
    harness: dict,
) -> dict:
    return {
        "contract_version": CONTRACT_VERSION,
        "dataset": dataset,
        "corpus": corpus,
        "index": index,
        "retrieval": retrieval,
        "rewrite": rewrite,
        "llm": llm,
        "judge": judge,
        "pricing": pricing,
        "run": run,
        "harness": harness,
    }


def _canonical(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def contract_sha256(contract: dict) -> str:
    stable = json.loads(_canonical(contract))
    run = stable.get("run")
    if isinstance(run, dict):
        for field in VOLATILE_RUN_FIELDS:
            run.pop(field, None)
    return hashlib.sha256(_canonical(stable).encode("utf-8")).hexdigest()


def _flatten(payload: dict, prefix: str = "") -> dict[str, object]:
    flat: dict[str, object] = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(value, name))
        else:
            flat[name] = value
    return flat


def contract_diff(left: dict, right: dict) -> list[str]:
    flat_left, flat_right = _flatten(left), _flatten(right)
    changed = []
    for key in sorted(set(flat_left) | set(flat_right)):
        if key.startswith("run.") and key.split(".", 1)[1] in VOLATILE_RUN_FIELDS:
            continue
        if flat_left.get(key) != flat_right.get(key):
            changed.append(key)
    return changed


def check_contract(contract: dict) -> None:
    if contract.get("contract_version") != CONTRACT_VERSION:
        raise ContractError(
            f"contract_version 不匹配:{contract.get('contract_version')} != {CONTRACT_VERSION}"
        )
    required = {
        "dataset": ("split", "sha256", "n"),
        "corpus": ("docs_sha256", "facts_sha256", "doc_count", "chunk_count"),
        "index": ("backend", "build_id", "embedding_model", "dim"),
        "retrieval": ("strategy", "candidates", "top_k"),
        "llm": ("model", "prompt_sha256"),
        "judge": ("model", "prompt_sha256"),
        "pricing": ("version", "currency", "verified"),
        "run": ("git_commit", "seed"),
        "harness": ("version", "mode"),
    }
    missing = [
        f"{section}.{field}"
        for section, fields in required.items()
        for field in fields
        if field not in (contract.get(section) or {})
    ]
    if missing:
        raise ContractError(f"contract 缺字段:{', '.join(missing)}")


def write_contract(path: Path, contract: dict) -> None:
    check_contract(contract)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
