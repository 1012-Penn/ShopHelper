"""评测回答 prompt:记录 SHA,与线上 system prompt 区分。

它要求模型按 expected_behavior 处理:有依据才答、缺槽位先问、需核验只给流程、
证据不足或冲突时拒答/说明生效版本。prompt 本身不写死任何答案。
"""
from __future__ import annotations

import hashlib

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
