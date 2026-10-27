"""Frozen dataset and model-input projections for the answer-depth experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.eval_dataset import load_split

ROOT = Path(__file__).resolve().parents[2]
SPLIT_PATHS = {
    "dev": ROOT / "tests" / "eval" / "splits" / "dev.jsonl",
    "frozen-v1": ROOT / "tests" / "eval" / "splits" / "frozen-v1.jsonl",
    "retrieval-slate-holdout-v1": (
        ROOT / "tests" / "eval" / "splits" / "retrieval-slate-holdout-v1.jsonl"
    ),
}
ARM_BASELINE = "baseline_rerank_top10"
ARM_SLATE = "evidence_slate_top10"
ARMS = (ARM_BASELINE, ARM_SLATE)

GENERATION_SYSTEM_PROMPT = """你是 ShopHelper 模拟电商客服。
只使用【对话上下文】和【候选证据】中的业务事实作答，不使用未给出的常识、记忆或猜测。
回答用户当前问题；证据不足时明确说明缺少什么并追问必要槽位，无法可靠回答时明确拒绝猜测。
保留适用于本单的日期、政策版本、商品/订单主体、条件和例外。
每个实质业务事实后用候选编号引用，例如 [1]；不要引用候选编号列表之外的资料。
不要输出分析过程，只输出最终客服答复。
"""

GENERATION_USER_TEMPLATE = """【对话上下文】
{context}

【用户当前问题】
{query}

【候选证据】
{evidence}
"""

JUDGE_SYSTEM_PROMPT = """你是匿名答案质量裁判。
输入只有用户问题、对话上下文、候选证据和一份匿名客服答案，不提供标准答案或方法标签。
你只判断：答案是否依据证据回答了问题，是否遗漏必要条件，是否存在无证据断言，
澄清或拒答是否与证据充分性相符。不要因为答案长度或语气加分。
必须严格输出 JSON 对象，不要代码围栏或额外文字。
"""

JUDGE_SCHEMA = """{
  "disposition": "answer|clarify|refuse",
  "answer_status": "correct|partial|incorrect|not_answerable",
  "grounding": "supported|partly_supported|unsupported|not_applicable",
  "completeness": 0,
  "citation_support": "supported|partial|unsupported|not_applicable",
  "unsupported_claims": [],
  "missing_requirements": [],
  "rationale": "一句话理由"
}"""

JUDGE_USER_TEMPLATE = """【对话上下文】
{context}

【用户当前问题】
{query}

【候选证据】
{evidence}

【候选答案甲】
{answer}

判断规则：
1. 只有候选证据能直接支持的事实才可判为支持；证据没写不等于否定事实。
2. 遗漏必要政策版本、适用日期、订单/商品主体、单位或例外时判 partial 或 incorrect。
3. 证据确实不足时，合理澄清或拒答分别标 clarify/refuse；证据足够却拒答属于错误。
4. unsupported_claims 和 missing_requirements 只列短句，不输出推理过程。
5. completeness 使用 0 到 100 的整数，100 表示当前问题所需事实与条件完整。
6. 必须严格输出以下 JSON：
{schema}
"""


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_depth_cases() -> dict[str, list[dict]]:
    cases = {
        "dev": load_split("dev", root=ROOT),
        "frozen-v1": load_split("frozen-v1", root=ROOT),
        "retrieval-slate-holdout-v1": _read_jsonl(
            SPLIT_PATHS["retrieval-slate-holdout-v1"]
        ),
    }
    seen: set[str] = set()
    for split, rows in cases.items():
        assert rows, f"empty split: {split}"
        for row in rows:
            case_id = str(row["id"])
            assert case_id not in seen, f"duplicate case id: {case_id}"
            seen.add(case_id)
            row.setdefault("split", split)
            row.setdefault("gold_facts", [])
            row.setdefault("required_conditions", [])
            row.setdefault("condition_aliases", {})
            row.setdefault("forbidden_claims", [])
            row.setdefault("clarifying_slots", [])
            row.setdefault("slot_aliases", {})
            row.setdefault("verification_markers", [])
            row.setdefault("conversation_context", [])
    return cases


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dataset_hashes() -> dict[str, str]:
    return {split: file_sha256(path) for split, path in SPLIT_PATHS.items()}


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def case_projection(case: dict) -> dict:
    """Only user-visible conversation may reach an answer model."""
    turns = []
    for turn in case.get("conversation_context") or []:
        role = str(turn.get("role") or "")
        content = str(turn.get("content") or "")
        if role in {"user", "assistant"} and content:
            turns.append({"role": role, "content": content})
    return {"query": str(case["query"]), "conversation_context": turns}


def _format_context(turns: list[dict]) -> str:
    if not turns:
        return "（无）"
    labels = {"user": "用户", "assistant": "客服"}
    return "\n".join(
        f"{labels[str(turn['role'])]}：{str(turn['content']).strip()}" for turn in turns
    )


def format_evidence(evidence: list[dict]) -> str:
    rows = []
    for index, item in enumerate(evidence, start=1):
        section = str(item.get("section_path") or item.get("category") or "").strip()
        text = str(item.get("text") or item.get("answer") or "").strip()
        rows.append(f"[{index}] {section}\n{text}")
    return "\n\n".join(rows) if rows else "（未检索到候选证据）"


def build_generation_messages(
    case: dict, evidence: list[dict], *, arm: str
) -> list[dict[str, str]]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    projection = case_projection(case)
    user = GENERATION_USER_TEMPLATE.format(
        context=_format_context(projection["conversation_context"]),
        query=projection["query"],
        evidence=format_evidence(evidence),
    )
    return [
        {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_judge_prompt(case: dict, evidence: list[dict], answer: str) -> str:
    projection = case_projection(case)
    return JUDGE_USER_TEMPLATE.format(
        context=_format_context(projection["conversation_context"]),
        query=projection["query"],
        evidence=format_evidence(evidence),
        answer=str(answer or "").strip(),
        schema=JUDGE_SCHEMA,
    )
