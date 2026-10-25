"""评测回答 prompt:记录 SHA,与线上 system prompt 区分。

它要求模型按 expected_behavior 处理:有依据才答、缺槽位先问、需核验只给流程、
证据不足或冲突时拒答/说明生效版本。prompt 本身不写死任何答案。
"""
from __future__ import annotations

import hashlib
import json
import re


def parse_json_object(text: str) -> dict:
    """从思考模型输出里提取第一个 JSON 对象;允许 ```json 围栏。"""
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ValueError(f"输出不含 JSON 对象: {text[:200]}")
        return json.loads(match.group(0))


class JsonEvidenceJudge:
    """不依赖 function_calling 的证据判断器,适配只支持 JSON 输出的思考模型。"""

    def __init__(self, llm) -> None:
        from app.evidence import EVIDENCE_JUDGE_PROMPT, EvidenceVerdict

        self._llm = llm
        self._prompt = EVIDENCE_JUDGE_PROMPT
        self._verdict = EvidenceVerdict

    def judge(self, query: str, documents: list[str]):
        evidence = "\n".join(f"[{i + 1}] {doc}" for i, doc in enumerate(documents))
        prompt = self._prompt.format(query=query, evidence=evidence)
        prompt += '\n只输出 JSON: {"answered": true/false, "supported_by": [int], "reason": "..."}'
        response = self._llm.invoke(prompt)
        payload = parse_json_object(str(getattr(response, "content", response)))
        return self._verdict(
            answered=bool(payload.get("answered")),
            supported_by=[int(item) for item in payload.get("supported_by") or []],
            reason=str(payload.get("reason") or ""),
        )


class JsonFaithfulnessJudge:
    def __init__(self, llm) -> None:
        self._llm = llm

    def invoke(self, prompt: str):
        from app.prompts import FaithfulnessVerdict

        response = self._llm.invoke(
            prompt
            + '\n只输出 JSON: {"fabricated": true/false, "reason": "...", "fabricated_claims": [...]}'
        )
        payload = parse_json_object(str(getattr(response, "content", response)))
        return FaithfulnessVerdict(
            fabricated=bool(payload.get("fabricated")),
            reason=str(payload.get("reason") or ""),
            fabricated_claims=[str(item) for item in payload.get("fabricated_claims") or []],
        )

EVAL_ANSWER_PROMPT = """你是电商客服评测中的回答器。只依据给定编号证据回答。
期望行为:{behavior}
- answer:直接给出确定事实,每个事实句末尾标注 [n]。
- answer_with_conditions:必须说全条件,不能把有条件结论说成无条件。
- verify_then_answer:只给核验/检测流程和有限帮助,禁止直接判定"是/否质量问题"。
- clarify:缺少必要槽位时只问必要问题,不要编答案。
- refuse:明确拒答或转人工,不要给实质结论。

规则:
1. 角标 [n] 只能使用给定编号;没有证据支持的断言不要写。
2. 多约束问题必须逐条覆盖;缺少任一约束时说明还缺什么。
3. 证据冲突时说明适用版本/口径,不要混用。
4. 回答简洁,不超过 3 句。

【证据】
{evidence}

【用户问题】
{query}
"""

EVAL_JUDGE_PROMPT = """你是答案裁判。只判断「答案」是否被「编号证据」支持,以及是否覆盖必需条件。
不要评价语气。输出 JSON:{{"supported": true/false, "reason": "..."}}。

【用户问题】{query}
【编号证据】
{evidence}
【答案】
{answer}
"""


def prompt_sha256(text: str = EVAL_ANSWER_PROMPT) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
