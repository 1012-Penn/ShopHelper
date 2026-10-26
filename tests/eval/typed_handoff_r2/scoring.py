"""Deterministic scorer. Gold is passed here and never to model-facing projections."""

from __future__ import annotations

import re
from typing import Any


SCORING_VERSION = "typed-handoff-r2-v3"
KNOWN_ORDER_IDS = tuple(str(value) for value in range(1001, 1011))

ALTERNATIVES = {
    "7天": ["7天", "七天"],
    "15天": ["15天", "十五天"],
    "买家承担": ["买家承担", "由买家承担"],
    "商家承担": ["商家承担", "由商家承担"],
    "物流快照已过期": [
        "物流快照已过期", "物流状态已过期", "无法确认最新物流",
        "已过期", "过期快照",
    ],
    "不会提交": ["不会提交", "未提交", "没有提交"],
}

BACKGROUND_CONTEXTS = {
    "15天": ("v1", "旧版", "旧版本", "已过期", "已于", "不再适用"),
    "商家承担": ("质量问题",),
    "买家承担": ("非质量问题",),
}


def _normalized(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _negated_order_mention(text: str, start: int) -> bool:
    prefix = text[max(0, start - 24):start]
    prefix = re.split(r"[。！？；\n]", prefix)[-1]
    return bool(re.search(
        r"(?:未使用|未带入|未采用|不包含|不要使用|不涉及|与.{0,8}无关)",
        prefix,
    ))


def order_ids(text: str) -> list[str]:
    found = []
    for order_id in KNOWN_ORDER_IDS:
        pattern = re.compile(rf"(?<!\d){re.escape(order_id)}(?!\d)")
        for match in pattern.finditer(text or ""):
            if not _negated_order_mention(text, match.start()):
                found.append(order_id)
    return list(dict.fromkeys(found))


def actual_outcome(reply: str, disposition: str = "") -> str:
    if disposition in {"clarify", "refuse", "handoff"}:
        return disposition
    if any(term in (reply or "") for term in ("无法回答", "不能提供医疗")):
        return "refuse"
    if "转人工" in (reply or ""):
        return "handoff"
    if re.search(
        r"(已签收|已发货|待发货|派送中|7天|15天|买家承担|商家承担|"
        r"原包装|3个工作日|可以申请取消|不能直接取消|商品|预览)",
        reply or "",
    ):
        return "answer"
    if re.search(r"(请|需要|麻烦).{0,12}(提供|补充|告知).{0,8}(订单号|单号)", reply or ""):
        return "clarify"
    return "answer"


def _tool_parameter_errors(tool_calls: list[dict[str, Any]]) -> int:
    errors = 0
    for call in tool_calls:
        name = str(call.get("name") or "")
        args = call.get("arguments") or {}
        if not isinstance(args, dict):
            errors += 1
        elif name == "query_order" and not str(args.get("order_id") or "").strip():
            errors += 1
        elif name == "query_faq" and not str(args.get("keyword") or "").strip():
            errors += 1
        elif name == "create_ticket":
            errors += int(args.get("conversation_id") in {None, ""})
            errors += int(not str(args.get("description") or "").strip())
            errors += int(args.get("ticket_type") not in {"售后", "投诉", "咨询"})
    return errors


def _contains_order(text: str, order_id: str) -> bool:
    return bool(re.search(rf"(?<!\d){re.escape(order_id)}(?!\d)", text))


def _relation_present(reply: str, relation: dict[str, Any]) -> bool:
    order_id = str(relation["order_id"])
    text = reply or ""
    pattern = re.compile(rf"(?<!\d){re.escape(order_id)}(?!\d)")
    for match in pattern.finditer(text):
        line_start = text.rfind("\n", 0, match.start()) + 1
        line_end = text.find("\n", match.end())
        line = text[line_start:line_end if line_end >= 0 else len(text)]
        next_starts = [
            next_match.start()
            for other in KNOWN_ORDER_IDS if other != order_id
            for next_match in [
                re.compile(rf"(?<!\d){re.escape(other)}(?!\d)").search(
                    text, match.end())
            ] if next_match is not None
        ]
        end = min(next_starts) if next_starts else len(text)
        block = text[match.start():min(end, match.start() + 1200)]
        line_has_other_order = any(
            _contains_order(line, other)
            for other in KNOWN_ORDER_IDS if other != order_id
        )
        for segment in ((block,) if line_has_other_order else (line, block)):
            if any(
                not any(_normalized(option) in _normalized(segment) for option in
                        ALTERNATIVES.get(product, [product]))
                for product in relation.get("products", [])
            ):
                continue
            if any(
                not any(_normalized(option) in _normalized(segment) for option in
                        ALTERNATIVES.get(status, [status]))
                for status in relation.get("statuses", [])
            ):
                continue
            return True
    return False


def _forbidden_present(phrase: str, reply: str) -> bool:
    if phrase in KNOWN_ORDER_IDS:
        pattern = re.compile(rf"(?<!\d){re.escape(phrase)}(?!\d)")
        mentions = list(pattern.finditer(reply or ""))
        return any(not _negated_order_mention(reply, match.start())
                   for match in mentions)
    normalized_phrase = _normalized(phrase)
    segments = re.split(r"[。！？；\n]", reply or "")
    matched = [segment for segment in segments
               if normalized_phrase in _normalized(segment)]
    if not matched:
        return False
    contexts = BACKGROUND_CONTEXTS.get(phrase, ())
    for segment in matched:
        if contexts and any(_normalized(item) in _normalized(segment)
                            for item in contexts):
            continue
        return True
    return False


def score_case(scenario: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    gold = scenario["gold"]
    reply = str(result.get("reply") or "")
    normalized_reply = _normalized(reply)
    tool_calls = list(result.get("tool_calls") or [])
    tool_names = [str(call.get("name") or "") for call in tool_calls]
    outcome = actual_outcome(reply, str(result.get("disposition") or ""))

    missing_conditions = []
    for phrase in gold.get("must_include", []):
        choices = ALTERNATIVES.get(phrase, [phrase])
        if not any(_normalized(item) in normalized_reply for item in choices):
            missing_conditions.append(phrase)
    forbidden_conditions = [
        phrase for phrase in gold.get("must_not_include", [])
        if _forbidden_present(phrase, reply)
    ]
    final_orders = order_ids(reply)
    queried_orders = {
        str((call.get("arguments") or {}).get("order_id"))
        for call in tool_calls if call.get("name") == "query_order"
    }
    required_orders = [str(item) for item in gold.get("required_order_ids", [])]
    forbidden_orders = {str(item) for item in gold.get("forbidden_final_order_ids", [])}
    entity_errors = [
        *[item for item in required_orders if item not in final_orders],
        *[item for item in final_orders if item in forbidden_orders],
        *[item for item in required_orders if item not in queried_orders],
        *[item for item in final_orders
          if required_orders and item not in required_orders
          and item not in forbidden_orders],
    ]
    for relation in gold.get("entity_relations", []):
        order_id = str(relation["order_id"])
        if not _relation_present(reply, relation):
            entity_errors.append(f"{order_id}:relation_mismatch")
    required_tools = list(gold.get("required_tools", []))
    source_missing = [name for name in required_tools if name not in tool_names]
    parameter_errors = _tool_parameter_errors(tool_calls)
    previews = list(result.get("write_previews") or [])
    confirmed = int(result.get("confirmed_write_count") or 0)
    unconfirmed = int(result.get("unconfirmed_write_count") or 0)
    write_mode = gold.get("write_mode", "none")
    write_correct = (
        (write_mode == "none" and not previews and confirmed == 0 and unconfirmed == 0)
        or (write_mode == "confirm_once" and bool(previews)
            and confirmed == 1 and unconfirmed == 0)
        or (write_mode == "cancel" and bool(previews)
            and confirmed == 0 and unconfirmed == 0)
    )
    handoff_omitted = 0
    for predicate in gold.get("required_handoff_predicates", []):
        visible = {
            str(claim.get("predicate"))
            for claim in result.get("action_visible_claims", [])
        }
        handoff_omitted += int(predicate not in visible)
    task_success = (
        outcome == gold["outcome"]
        and not missing_conditions
        and not forbidden_conditions
        and not entity_errors
        and not source_missing
        and parameter_errors == 0
        and write_correct
    )
    return {
        "scoring_version": SCORING_VERSION,
        "task_success": bool(task_success),
        "actual_outcome": outcome,
        "outcome_correct": outcome == gold["outcome"],
        "condition_error_count": len(missing_conditions) + len(forbidden_conditions),
        "entity_error_count": len(entity_errors),
        "source_missing_count": len(source_missing),
        "parameter_error_count": parameter_errors,
        "handoff_omission_count": handoff_omitted,
        "missing_conditions": missing_conditions,
        "forbidden_conditions": forbidden_conditions,
        "entity_errors": entity_errors,
        "source_missing_tools": source_missing,
        "write_correct": bool(write_correct),
        "preview_count": len(previews),
        "confirmed_write_count": confirmed,
        "unconfirmed_write_count": unconfirmed,
    }
