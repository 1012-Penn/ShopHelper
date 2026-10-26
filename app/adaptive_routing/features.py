"""Explainable routing signals derived only from visible Agent input."""

from __future__ import annotations

import re
from collections.abc import Sequence

from app.adaptive_routing.models import HistoryTurn, RoutingFeatures

_ORDER_CONTEXTUAL = re.compile(r"(?:订单|order)\s*号?\s*(\d{3,6})", re.IGNORECASE)
_ORDER_BARE = re.compile(r"(?<!\d)(\d{4,6})(?!\d)")
_COMPARISON = re.compile(
    r"比较|对比|哪个|哪一个|分别|两者|二者|各自|谁更|还是|排行|排序"
)
_POLICY_ORDER_ACTION = re.compile(
    r"取消|退货|退款|退换|运费|包邮|售后|工单|换货|拒收|签收|发货|"
    r"政策|规则|怎么处理|怎么办|能否|能不能|可以吗|允许吗"
)
_CORRECTION = re.compile(
    r"更正|改问|改成|改为|刚才问错|重新问|换一个问题|不是.*(?:而是|改问)|算了"
)
_DEMONSTRATIVE_ORDER = re.compile(
    r"这个|这单|那单|这笔|那笔|这个订单|那个订单|该订单|这件商品|那个商品"
)
_ORDER_ACTION = re.compile(
    r"取消|退货|退款|退换|售后|换货|拒收|签收|发货|工单|能退|能换|怎么退"
)
_WRITE_RISK = re.compile(
    r"创建.*工单|开具.*工单|提交|确认|申请退款|退款申请|取消订单|写入|开单"
)


def _history_text(history: Sequence[HistoryTurn]) -> str:
    return "\n".join(item.content for item in history)


def _order_ids(text: str) -> tuple[str, ...]:
    found: list[str] = []
    for match in _ORDER_CONTEXTUAL.finditer(text):
        found.append(match.group(1))
    for match in _ORDER_BARE.finditer(text):
        value = match.group(1)
        # Bare years are common in policy questions and are not order entities.
        if re.fullmatch(r"(?:19|20)\d{2}", value):
            continue
        found.append(value)
    return tuple(dict.fromkeys(found))


def extract_routing_features(
    question: str, history: Sequence[HistoryTurn] | None = None
) -> RoutingFeatures:
    """Extract pre-execution structural features without consulting gold metadata."""

    history = list(history or [])
    history_text = _history_text(history)
    visible_text = "\n".join(item for item in (history_text, question) if item)
    current = question or ""
    order_ids = _order_ids(visible_text)
    entity_count = len(order_ids)
    has_order_context = bool(order_ids) or bool(re.search(r"订单|order", visible_text, re.IGNORECASE))
    has_policy_action = bool(_POLICY_ORDER_ACTION.search(current))
    policy_order_dependency = has_order_context and has_policy_action
    cross_order_comparison = entity_count >= 2 and bool(_COMPARISON.search(current))
    correction = bool(_CORRECTION.search(current))
    missing_slot = (
        entity_count == 0
        and bool(_ORDER_ACTION.search(current))
        and bool(_DEMONSTRATIVE_ORDER.search(current))
    )
    write_confirmation_risk = bool(_WRITE_RISK.search(current))
    return RoutingFeatures(
        visible_order_ids=order_ids,
        entity_count=entity_count,
        cross_order_comparison=cross_order_comparison,
        policy_order_dependency=policy_order_dependency,
        correction=correction,
        missing_slot=missing_slot,
        write_confirmation_risk=write_confirmation_risk,
    )
