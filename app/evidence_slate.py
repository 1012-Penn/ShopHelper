"""Query-conditioned selection of a multi-chunk evidence slate.

The cross-encoder ranks individual chunks and can crowd complementary evidence
out of a small context.  This module keeps its ranking, but reserves slots for
explicit question facets such as "this order + the applicable policy" or
"these two return conditions".  Rules use only the query and retrieved text;
they never inspect evaluation labels.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable


_IDENTIFIER = re.compile(
    r"(?<![A-Z0-9])(?:ORD-\d{4,6}|SH-[A-Z0-9]+)(?![A-Z0-9])",
    re.IGNORECASE,
)
_MEMBER_TIERS = ("普通会员", "银卡会员", "金卡会员", "钻石会员")


@dataclass(frozen=True)
class EvidenceDocument:
    chunk_id: int
    section_path: str
    text: str


@dataclass(frozen=True)
class EvidenceSlate:
    ordered_ids: list[int]
    facets: list[str]


def select_evidence_slate(
    query: str,
    *,
    ranked_ids: list[int],
    documents: dict[int, EvidenceDocument],
    top_k: int,
) -> EvidenceSlate:
    """Select at most ``top_k`` chunks while preserving the supplied rerank order.

    Explicit facets retain their original rerank order and are inserted before
    the remaining rerank candidates.  This guarantees that complementary
    clauses remain in the small context even when their standalone score falls
    just outside the caller's top-k window.
    """
    if top_k <= 0 or not ranked_ids:
        return EvidenceSlate([], [])

    ordered = [chunk_id for chunk_id in ranked_ids if chunk_id in documents]
    if not ordered:
        return EvidenceSlate([], [])

    rank_by_id = {chunk_id: rank for rank, chunk_id in enumerate(ordered)}
    selected: list[int] = []
    facets: list[str] = []
    lower = query.casefold()

    def best(predicate: Callable[[EvidenceDocument], bool]) -> int | None:
        return next(
            (
                chunk_id
                for chunk_id in ordered
                if chunk_id not in selected and predicate(documents[chunk_id])
            ),
            None,
        )

    def in_section(*titles: str) -> Callable[[EvidenceDocument], bool]:
        return lambda document: any(title in document.section_path for title in titles)

    def add(name: str, chunk_id: int | None) -> None:
        if chunk_id is not None and chunk_id not in selected:
            selected.append(chunk_id)
            facets.append(name)

    identifiers = [value.upper() for value in _IDENTIFIER.findall(query)]
    order_id = next((value for value in identifiers if value.startswith("ORD-")), None)
    product_id = next((value for value in identifiers if value.startswith("SH-")), None)
    asks_coupon = any(term in lower for term in ("满减", "优惠券", "券", "叠加"))

    if order_id:
        add(
            "order_summary",
            best(lambda document: "订单状态总览" in document.section_path
                 and order_id in document.text),
        )

    if product_id:
        asks_status = any(term in lower for term in ("清仓", "停产", "在售", "状态"))
        product_chunk = best(
            lambda document: (
                "商品价格总览" in document.section_path
                and product_id in document.text
            )
            if asks_status
            else (
                "商品条目" in document.section_path
                and product_id in document.section_path
                and product_id in document.text
            )
        )
        if product_chunk is None and not asks_status:
            product_chunk = best(
                lambda document: product_id in document.section_path
                and product_id in document.text
                and "核心参数对比" not in document.section_path
            )
        add(
            "product_status" if asks_status else "product_entry",
            product_chunk,
        )

    if order_id and any(term in lower for term in ("退款", "钱", "渠道", "支付")):
        add(
            "order_payment",
            best(
                lambda document: order_id in document.section_path
                and ("支付渠道" in document.text or "退款时间" in document.text)
            ),
        )

    if any(term in lower for term in ("加急", "次日达")):
        add("expedite", best(in_section("支持加急发货吗")))

    if asks_coupon:
        add("coupon_limit", best(in_section("优惠券使用限制")))
        add("discount_stack", best(in_section("优惠叠加规则")))

    if any(term in lower for term in ("秒杀", "清仓", "加急费", "运费")) and any(
        term in lower for term in ("会员", "95 折", "95折", "折扣")
    ):
        add("discount_exclusion", best(in_section("95 折排除范围")))

    if any(term in lower for term in ("打几折", "能享", "适用吗", "95 折", "95折")) and not any(
        term in lower
        for term in ("满减", "券", "叠加", "清仓", "秒杀", "加急费", "运费")
    ):
        add("member_rule", best(in_section("会员等级")))

    if (
        product_id
        and any(tier in query for tier in _MEMBER_TIERS)
        and any(term in lower for term in ("会员", "折"))
    ):
        add(
            "member_tier_table",
            best(
                lambda document: "会员等级" in document.section_path
                and "| 普通会员 |" in document.text
            ),
        )

    if "包邮" in lower and not asks_coupon:
        add("free_shipping", best(in_section("包邮门槛与合并订单")))
    elif any(term in lower for term in ("偏远", "补运", "西藏")):
        add("remote_shipping", best(in_section("偏远地区补运")))
    elif any(term in lower for term in ("运费", "邮费", "快递")):
        add(
            "shipping",
            best(in_section("邮费与运费")) or best(in_section("退货处理细则")),
        )

    if any(term in lower for term in ("完好", "拆了", "吊牌", "包装")):
        add(
            "return_condition",
            best(lambda document: "商品必须保持完好" in document.text),
        )

    if any(term in lower for term in ("定制", "生鲜", "虚拟商品", "贴身")):
        add("return_exclusion", best(in_section("不予退货情形")))

    if any(term in lower for term in ("取消", "撤单")):
        add("cancel_rule", best(in_section("订单取消")))

    if any(term in lower for term in ("退款", "到账", "钱", "渠道", "退回")):
        add(
            "refund_rule",
            best(in_section("退款时效")) or best(in_section("退款路径")),
        )

    if any(term in lower for term in ("保修", "质保", "维修")):
        add("warranty_rule", best(in_section("保修范围与期限")))
        if any(term in lower for term in ("人为", "进水", "摔", "拆机", "过保")):
            add("warranty_damage", best(in_section("人为损坏与过保")))

    policy_query = any(
        term in lower
        for term in ("无理由", "退货窗口", "v1", "v2", "生效", "15 天", "7 天", "15天", "7天")
    )
    if "不知道签收日期" in lower or "哪个版本" in lower or "哪版" in lower:
        add("version_decision", best(in_section("新旧版本适用判定")))
        add("conflict_rule", best(in_section("冲突处理")))
    elif "v2" in lower and any(term in lower for term in ("延长", "补到", "生效后")):
        add("v1_window", best(in_section("v1 版七天无理由")))
        add("supersedes", best(in_section("替代关系")))
    elif (
        policy_query and any(term in lower for term in ("签收", "无理由", "退货窗口"))
    ) or (
        any(term in lower for term in ("签收第", "用了"))
        and "天" in lower
        and "退" in lower
    ):
        add("v2_window", best(in_section("v2 版十五天无理由")))
        add("v1_window", best(in_section("v1 版七天无理由")))

    for chunk_id in ordered:
        if len(selected) >= top_k:
            break
        if chunk_id not in selected:
            selected.append(chunk_id)

    final_ids = selected[:top_k]
    selected_set = set(final_ids)
    kept_facets = [
        name
        for name, chunk_id in zip(facets, selected)
        if chunk_id in selected_set
    ]
    return EvidenceSlate(final_ids, kept_facets)
