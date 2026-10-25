"""ch11 评测题生成器:从 FactBook/evidence 单元拼装 v2 JSONL。

题目与期望行为在这里人工定义;证据 quote 与 gold fact 从 `corpus/` 解析,
避免手抄 quote 造成漂移。生成后由 `eval_dataset.validate_cases` 校验。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.corpus_facts import FactBook, load_factbook
from scripts.eval_dataset import validate_cases
from scripts.frozen_case_specs import FROZEN_CASES

ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = ROOT / "tests" / "eval" / "splits"

SLOT_ALIASES = {
    "sku": ["型号", "哪款", "哪个商品", "商品"],
    "order_id": ["订单号", "订单编号", "哪笔订单"],
    "member_status": ["会员", "会员等级"],
    "region": ["地区", "收货地"],
}


def _turn(role: str, content: str) -> dict[str, str]:
    return {"role": role, "content": content}


# 每个 spec 的字段:
# id / bucket / query / evidence(evidence_id 列表) / gold(fact_id 列表)
# behavior / conditions / forbidden / slots / clarifying / context / as_of
# evidence_state(可选) / note(可选)
DEV_CASES: list[dict[str, Any]] = [
    # ---------- single_hop 12 ----------
    {"id": "DEV-SINGLE-01", "bucket": "single_hop", "query": "SH-E300 多少钱？",
     "evidence": ["EV-CATALOG-SH-E300-ENTRY"], "gold": ["CATALOG-SH-E300-PRICE"]},
    {"id": "DEV-SINGLE-02", "bucket": "single_hop", "query": "SH-E120 续航多久？",
     "evidence": ["EV-CATALOG-SH-E120-ENTRY"], "gold": ["CATALOG-SH-E120-BATTERY-HOURS"]},
    {"id": "DEV-SINGLE-03", "bucket": "single_hop", "query": "SH-K104 有多少键？",
     "evidence": ["EV-CATALOG-SH-K104-ENTRY"], "gold": ["CATALOG-SH-K104-KEY-COUNT"]},
    {"id": "DEV-SINGLE-04", "bucket": "single_hop", "query": "SH-C60 数据线质保多久？",
     "evidence": ["EV-CATALOG-SH-C60-ENTRY"], "gold": ["CATALOG-SH-C60-WARRANTY-MONTHS"],
     "behavior": "answer_with_conditions", "conditions": ["配件类"]},
    {"id": "DEV-SINGLE-05", "bucket": "single_hop", "query": "SH-W20 支持 GPS 吗？",
     "evidence": ["EV-CATALOG-SH-W20-ENTRY"], "gold": ["CATALOG-SH-W20-GPS"]},
    {"id": "DEV-SINGLE-06", "bucket": "single_hop", "query": "普通订单邮费多少？",
     "evidence": ["EV-POLICY-SHIPPING"], "gold": ["POLICY-SHIPPING-ORDINARY-FEE"]},
    {"id": "DEV-SINGLE-07", "bucket": "single_hop", "query": "微信支付退款多久到账？",
     "evidence": ["EV-POLICY-REFUND"], "gold": ["POLICY-REFUND-WORKDAYS"],
     "behavior": "answer_with_conditions", "conditions": ["微信/支付宝"]},
    {"id": "DEV-SINGLE-08", "bucket": "single_hop", "query": "退货申请多久审核？",
     "evidence": ["EV-AFTERSALES-RETURN-REVIEW"], "gold": ["AFTERSALES-RETURN-REVIEW-HOURS"]},
    {"id": "DEV-SINGLE-09", "bucket": "single_hop", "query": "商品是否有质量问题，以什么为准？",
     "evidence": ["EV-AFTERSALES-QUALITY-DETERMINATION"],
     "gold": ["AFTERSALES-QUALITY-DETERMINATION"]},
    {"id": "DEV-SINGLE-10", "bucket": "single_hop", "query": "西藏首重运费多少？",
     "evidence": ["EV-REGION-TIBET"], "gold": ["REGION-TIBET-FIRST-WEIGHT-FEE"]},
    {"id": "DEV-SINGLE-11", "bucket": "single_hop", "query": "ORD-1002 是什么状态？",
     "evidence": ["EV-ORDER-ORD-1002-ROW"], "gold": ["ORDER-ORD-1002-STATUS"]},
    {"id": "DEV-SINGLE-12", "bucket": "single_hop", "query": "ORD-1002 用什么渠道支付？",
     "evidence": ["EV-ORDER-ORD-1002-PAYMENT"], "gold": ["ORDER-ORD-1002-PAYMENT-CHANNEL"]},

    # ---------- multi_hop 11 ----------
    {"id": "DEV-MULTI-01", "bucket": "multi_hop",
     "query": "ORD-1002 是银卡会员，商品 328 元，会员能打几折？",
     "evidence": ["EV-ORDER-ORD-1002-ROW", "EV-PROMO-MEMBER-DISCOUNT"],
     "gold": ["ORDER-ORD-1002-PRODUCT-AMOUNT", "PROMO-MEMBER-DISCOUNT"],
     "behavior": "answer_with_conditions", "conditions": ["所有会员等级", "商品原价"]},
    {"id": "DEV-MULTI-02", "bucket": "multi_hop",
     "query": "SH-E300 售价 299 元，这一单包邮吗？",
     "evidence": ["EV-CATALOG-SH-E300-ENTRY", "EV-PROMO-SHIPPING"],
     "gold": ["CATALOG-SH-E300-PRICE", "PROMO-FREE-SHIPPING-THRESHOLD"]},
    {"id": "DEV-MULTI-03", "bucket": "multi_hop",
     "query": "SH-K104 出现质量问题要退货，退货运费谁出？",
     "evidence": ["EV-CATALOG-SH-K104-ENTRY", "EV-POLICY-SHIPPING"],
     "gold": ["CATALOG-SH-K104-PRODUCT-TYPE", "POLICY-SHIPPING-QUALITY-PAYER"],
     "behavior": "answer_with_conditions", "conditions": ["质量问题"]},
    {"id": "DEV-MULTI-04", "bucket": "multi_hop",
     "query": "ORD-1004 签收于 2026-06-05，适用几天无理由退货？",
     "evidence": ["EV-ORDER-ORD-1004-ROW", "EV-POLICY-V2-WINDOW"],
     "gold": ["ORDER-ORD-1004-STATUS", "POLICY-RETURN-WINDOW-V2"],
     "behavior": "answer_with_conditions", "conditions": ["签收日期为 2026-06-01 或之后", "15 天内"],
     "as_of": "2026-06-05", "evidence_state": "conflicting"},
    {"id": "DEV-MULTI-05", "bucket": "multi_hop",
     "query": "SH-M10 是清仓商品，会员还能享 95 折吗？",
     "evidence": ["EV-CATALOG-SH-M10-STATUS", "EV-PROMO-DISCOUNT-EXCLUSIONS"],
     "gold": ["CATALOG-SH-M10-STATUS"], "behavior": "answer_with_conditions",
     "conditions": ["清仓与停产库存商品不参与"], "forbidden": ["清仓商品也打 95 折"]},
    {"id": "DEV-MULTI-06", "bucket": "multi_hop",
     "query": "ORD-1002 已支付但未发货，可以取消吗？",
     "evidence": ["EV-ORDER-ORD-1002-ROW", "EV-POLICY-CANCEL"],
     "gold": ["ORDER-ORD-1002-STATUS", "POLICY-CANCEL-UNSHIPPED-ALLOWED"],
     "behavior": "answer_with_conditions", "conditions": ["未发货"]},
    {"id": "DEV-MULTI-07", "bucket": "multi_hop",
     "query": "ORD-1003 已经发货了，能直接取消吗？",
     "evidence": ["EV-ORDER-ORD-1003-ROW", "EV-POLICY-CANCEL"],
     "gold": ["ORDER-ORD-1003-STATUS", "POLICY-CANCEL-SHIPPED-ALLOWED"],
     "behavior": "answer_with_conditions", "conditions": ["已发货只能拒收或退货"],
     "forbidden": ["已发货可以直接取消"]},
    {"id": "DEV-MULTI-08", "bucket": "multi_hop",
     "query": "退货审核通过后，几天内要把商品寄回？",
     "evidence": ["EV-AFTERSALES-RETURN-REVIEW"],
     "gold": ["AFTERSALES-RETURN-SHIP-DAYS", "AFTERSALES-RETURN-REVIEW-HOURS"],
     "behavior": "answer_with_conditions", "conditions": ["审核通过后"]},
    {"id": "DEV-MULTI-09", "bucket": "multi_hop",
     "query": "过保维修要先报价再修吗？",
     "evidence": ["EV-AFTERSALES-WARRANTY", "EV-AFTERSALES-REPAIR"],
     "gold": ["AFTERSALES-OVER-WARRANTY-QUOTE-FIRST"],
     "behavior": "answer_with_conditions", "conditions": ["过保"]},
    {"id": "DEV-MULTI-10", "bucket": "multi_hop",
     "query": "西藏订单满 99 元包邮后，还要补运费吗？",
     "evidence": ["EV-PROMO-SHIPPING", "EV-PROMO-REMOTE"],
     "gold": ["PROMO-FREE-SHIPPING-THRESHOLD", "PROMO-REMOTE-SURCHARGE"],
     "behavior": "answer_with_conditions", "conditions": ["包邮基础上", "偏远地区"]},
    {"id": "DEV-MULTI-11", "bucket": "multi_hop",
     "query": "金卡会员买 SH-B5，会员折扣适用吗？",
     "evidence": ["EV-PROMO-TIER-TABLE", "EV-CATALOG-SH-B5-ENTRY", "EV-PROMO-MEMBER-DISCOUNT"],
     "gold": ["PROMO-TIER-GOLD-GROWTH", "CATALOG-SH-B5-PRICE", "PROMO-MEMBER-DISCOUNT"],
     "behavior": "answer_with_conditions", "conditions": ["所有会员等级", "在售的正价商品"]},

    # ---------- multi_constraint 9 ----------
    {"id": "DEV-CONSTRAINT-01", "bucket": "multi_constraint",
     "query": "SH-E300 签收第 6 天，包装拆了但商品完好，能退吗？",
     "evidence": ["EV-POLICY-V2-WINDOW", "EV-POLICY-RETURN-CONDITION"],
     "gold": ["POLICY-RETURN-WINDOW-V2", "POLICY-RETURN-CONDITION"],
     "behavior": "answer_with_conditions",
     "conditions": ["15 天内", "商品完好", "吊牌、包装齐全"],
     "forbidden": ["拆了包装就一定不能退"]},
    {"id": "DEV-CONSTRAINT-02", "bucket": "multi_constraint",
     "query": "定制键盘用了 10 天，能退吗？",
     "evidence": ["EV-POLICY-NON-RETURNABLE", "EV-POLICY-V2-WINDOW"],
     "gold": ["POLICY-CUSTOM-QUALITY-EXCEPTION"],
     "behavior": "answer_with_conditions",
     "conditions": ["定制类商品仅在有质量问题时可退"], "forbidden": ["定制商品 15 天内都能无理由退"]},
    {"id": "DEV-CONSTRAINT-03", "bucket": "multi_constraint",
     "query": "生鲜第二天发现质量问题，能退吗？",
     "evidence": ["EV-POLICY-NON-RETURNABLE"],
     "gold": ["POLICY-FRESH-SAME-DAY-QUALITY-ONLY"],
     "behavior": "answer_with_conditions",
     "conditions": ["仅限签收当日", "质量问题"], "forbidden": ["生鲜第二天也能无理由退"]},
    {"id": "DEV-CONSTRAINT-04", "bucket": "multi_constraint",
     "query": "已经激活的虚拟商品可以退款吗？",
     "evidence": ["EV-POLICY-VIRTUAL"],
     "gold": ["POLICY-VIRTUAL-ACTIVATED-NONREFUNDABLE"]},
    {"id": "DEV-CONSTRAINT-05", "bucket": "multi_constraint",
     "query": "会员 95 折后 98 元，还能用满 99 减 10 的券吗？",
     "evidence": ["EV-PROMO-STACKING", "EV-PROMO-SHIPPING"],
     "gold": ["PROMO-STACKING-THRESHOLD-BASE", "PROMO-FREE-SHIPPING-THRESHOLD"],
     "behavior": "answer_with_conditions",
     "conditions": ["按 95 折后的商品金额计算"], "forbidden": ["按原价算满减门槛"]},
    {"id": "DEV-CONSTRAINT-06", "bucket": "multi_constraint",
     "query": "加急费 10 元能参与会员 95 折吗？",
     "evidence": ["EV-PROMO-EXPEDITE", "EV-PROMO-DISCOUNT-EXCLUSIONS"],
     "gold": ["PROMO-EXPEDITE-FEE"], "behavior": "answer_with_conditions",
     "conditions": ["加急费不参与折扣"], "forbidden": ["加急费也打 95 折"]},
    {"id": "DEV-CONSTRAINT-07", "bucket": "multi_constraint",
     "query": "西藏补运的 5 元算进满减门槛吗？",
     "evidence": ["EV-PROMO-REMOTE", "EV-PROMO-STACKING"],
     "gold": ["PROMO-REMOTE-SURCHARGE", "PROMO-REMOTE-SURCHARGE-NOT-DISCOUNTED"],
     "behavior": "answer_with_conditions", "conditions": ["偏远补运不参与满减"],
     "forbidden": ["补运计入满减门槛"]},
    {"id": "DEV-CONSTRAINT-08", "bucket": "multi_constraint",
     "query": "非质量问题退货，商品完好，退回运费谁出？",
     "evidence": ["EV-POLICY-RETURN-CONDITION", "EV-POLICY-SHIPPING"],
     "gold": ["POLICY-RETURN-CONDITION", "POLICY-SHIPPING-NONQUALITY-PAYER"],
     "behavior": "answer_with_conditions", "conditions": ["非质量问题", "商品完好"]},
    {"id": "DEV-CONSTRAINT-09", "bucket": "multi_constraint",
     "query": "无理由拒收，退货运费谁承担？",
     "evidence": ["EV-POLICY-REJECT"],
     "gold": ["POLICY-REJECT-NO-REASON-PAYER"],
     "behavior": "answer_with_conditions", "conditions": ["无理由拒收"]},

    # ---------- entity_mismatch 7 ----------
    {"id": "DEV-ENTITY-01", "bucket": "entity_mismatch",
     "query": "SH-E300 是 87 键机械键盘吗？",
     "evidence": ["EV-CATALOG-SH-E300-ENTRY"],
     "gold": ["CATALOG-SH-E300-PRODUCT-TYPE"], "behavior": "answer",
     "forbidden": ["SH-E300 是 87 键键盘"]},
    {"id": "DEV-ENTITY-02", "bucket": "entity_mismatch",
     "query": "SH-K870 是 104 键吗？",
     "evidence": ["EV-CATALOG-SH-K870-ENTRY"],
     "gold": ["CATALOG-SH-K870-KEY-COUNT"], "behavior": "answer",
     "forbidden": ["SH-K870 是 104 键"]},
    {"id": "DEV-ENTITY-03", "bucket": "entity_mismatch",
     "query": "ORD-9999 到哪了？", "evidence": [], "gold": [],
     "behavior": "clarify", "clarifying": ["order_id"], "evidence_state": "insufficient"},
    {"id": "DEV-ENTITY-04", "bucket": "entity_mismatch",
     "query": "SH-T50 是整机一年质保吗？",
     "evidence": ["EV-CATALOG-SH-T50-ENTRY"],
     "gold": ["CATALOG-SH-T50-WARRANTY-MONTHS"],
     "behavior": "answer_with_conditions", "conditions": ["配件类", "例外以商品目录为准"],
     "forbidden": ["SH-T50 整机一年质保"]},
    {"id": "DEV-ENTITY-05", "bucket": "entity_mismatch",
     "query": "SH-B5 支持血氧监测吗？",
     "evidence": ["EV-CATALOG-SH-B5-ENTRY", "EV-CATALOG-SH-B7-ENTRY"],
     "gold": ["CATALOG-SH-B5-HEART-RATE-MONITORING", "CATALOG-SH-B7-BLOOD-OXYGEN-MONITORING"],
     "behavior": "answer", "forbidden": ["SH-B5 支持血氧监测"]},
    {"id": "DEV-ENTITY-06", "bucket": "entity_mismatch",
     "query": "SH-W20 能独立通话吗？",
     "evidence": ["EV-CATALOG-SH-W20-ENTRY", "EV-CATALOG-SH-W30-ENTRY"],
     "gold": ["CATALOG-SH-W20-GPS", "CATALOG-SH-W30-STANDALONE-CALL"],
     "behavior": "answer", "forbidden": ["SH-W20 支持独立通话"]},
    {"id": "DEV-ENTITY-07", "bucket": "entity_mismatch",
     "query": "SH-E120 续航是 30 小时吗？",
     "evidence": ["EV-CATALOG-SH-E120-ENTRY"],
     "gold": ["CATALOG-SH-E120-BATTERY-HOURS"], "behavior": "answer",
     "forbidden": ["SH-E120 续航 30 小时"]},

    # ---------- policy_conflict 5 ----------
    {"id": "DEV-CONFLICT-01", "bucket": "policy_conflict",
     "query": "2026-05-20 签收的订单，适用 7 天还是 15 天无理由？",
     "evidence": ["EV-POLICY-V1-WINDOW", "EV-POLICY-V2-WINDOW"],
     "gold": ["POLICY-RETURN-WINDOW-V1"], "behavior": "answer_with_conditions",
     "conditions": ["签收日期早于 2026-06-01", "7 天内"],
     "as_of": "2026-05-20", "evidence_state": "conflicting"},
    {"id": "DEV-CONFLICT-02", "bucket": "policy_conflict",
     "query": "2026-06-05 签收的订单，能享受 15 天无理由吗？",
     "evidence": ["EV-POLICY-V1-WINDOW", "EV-POLICY-V2-WINDOW"],
     "gold": ["POLICY-RETURN-WINDOW-V2"], "behavior": "answer_with_conditions",
     "conditions": ["签收日期为 2026-06-01 或之后", "15 天内"],
     "as_of": "2026-06-05", "evidence_state": "conflicting"},
    {"id": "DEV-CONFLICT-03", "bucket": "policy_conflict",
     "query": "2026-05-20 签收的订单，v2 生效后会自动延长到 15 天吗？",
     "evidence": ["EV-POLICY-V1-WINDOW", "EV-POLICY-SUPERSEDES"],
     "gold": ["POLICY-NO-RETROACTIVE-EXTENSION"], "behavior": "answer",
     "forbidden": ["v2 生效后旧订单自动延长"], "as_of": "2026-05-20",
     "evidence_state": "conflicting"},
    {"id": "DEV-CONFLICT-04", "bucket": "policy_conflict",
     "query": "不知道签收日期，应该按 7 天还是 15 天处理？",
     "evidence": ["EV-POLICY-DECISION", "EV-POLICY-CONFLICT"],
     "gold": ["POLICY-MISSING-SIGN-TIME-ACTION"], "behavior": "clarify",
     "clarifying": ["order_id"], "as_of": "2026-10-25", "evidence_state": "conflicting"},
    {"id": "DEV-CONFLICT-05", "bucket": "policy_conflict",
     "query": "ORD-1004 签收于 2026-06-05，退货窗口是多久？",
     "evidence": ["EV-ORDER-ORD-1004-ROW", "EV-POLICY-V2-WINDOW"],
     "gold": ["ORDER-ORD-1004-STATUS", "POLICY-RETURN-WINDOW-V2"],
     "behavior": "answer_with_conditions", "conditions": ["15 天内"],
     "as_of": "2026-06-05", "evidence_state": "conflicting"},

    # ---------- hard_negative 7 ----------
    {"id": "DEV-HARD-01", "bucket": "hard_negative",
     "query": "SH-E300 支持无线充电吗？", "evidence": [], "gold": [],
     "behavior": "refuse", "refusal_reason": "知识库只写蓝牙和续航，没有充电方式"},
    {"id": "DEV-HARD-02", "bucket": "hard_negative",
     "query": "会员生日礼包里面有什么？", "evidence": [], "gold": [],
     "behavior": "refuse", "refusal_reason": "知识库只写礼包存在，没有内容清单"},
    {"id": "DEV-HARD-03", "bucket": "hard_negative",
     "query": "发票抬头能改成公司名吗？", "evidence": [], "gold": [],
     "behavior": "refuse", "refusal_reason": "知识库没有发票抬头修改规则"},
    {"id": "DEV-HARD-04", "bucket": "hard_negative",
     "query": "手表怎么安装微信？", "evidence": [], "gold": [],
     "behavior": "refuse", "refusal_reason": "知识库没有手表安装微信的说明"},
    {"id": "DEV-HARD-05", "bucket": "hard_negative",
     "query": "维修屏幕要多少钱？", "evidence": [], "gold": [],
     "behavior": "refuse", "refusal_reason": "知识库没有维修价格，需检测后报价"},
    {"id": "DEV-HARD-06", "bucket": "hard_negative",
     "query": "会飞的手机保修多久？", "evidence": [], "gold": [],
     "behavior": "refuse", "refusal_reason": "知识库没有该商品"},
    {"id": "DEV-HARD-07", "bucket": "hard_negative",
     "query": "SH-E300 的防水等级是多少？", "evidence": [], "gold": [],
     "behavior": "refuse", "refusal_reason": "知识库没有 SH-E300 的防水等级"},

    # ---------- clarify 5 ----------
    {"id": "DEV-CLARIFY-01", "bucket": "clarify", "query": "这个多少钱？",
     "evidence": [], "gold": [], "behavior": "clarify",
     "clarifying": ["sku"], "evidence_state": "insufficient"},
    {"id": "DEV-CLARIFY-02", "bucket": "clarify", "query": "我的订单到哪了？",
     "evidence": [], "gold": [], "behavior": "clarify",
     "clarifying": ["order_id"], "evidence_state": "insufficient"},
    {"id": "DEV-CLARIFY-03", "bucket": "clarify", "query": "保修多久？",
     "evidence": [], "gold": [], "behavior": "clarify",
     "clarifying": ["sku"], "evidence_state": "insufficient"},
    {"id": "DEV-CLARIFY-04", "bucket": "clarify", "query": "什么时候能到？",
     "evidence": [], "gold": [], "behavior": "clarify",
     "clarifying": ["order_id", "region"], "evidence_state": "insufficient"},
    {"id": "DEV-CLARIFY-05", "bucket": "clarify",
     "query": "怎么退？", "evidence": [], "gold": [], "behavior": "clarify",
     "clarifying": ["order_id"], "evidence_state": "insufficient",
     "context": [_turn("user", "我要退货"), _turn("assistant", "好的，请提供订单号。")]},

    # ---------- ood 4 ----------
    {"id": "DEV-OOD-01", "bucket": "ood", "query": "今天天气怎么样？",
     "evidence": [], "gold": [], "behavior": "refuse", "refusal_reason": "超出店铺服务范围"},
    {"id": "DEV-OOD-02", "bucket": "ood", "query": "怎么用微波炉煮火锅？",
     "evidence": [], "gold": [], "behavior": "refuse", "refusal_reason": "超出店铺服务范围"},
    {"id": "DEV-OOD-03", "bucket": "ood", "query": "推荐一只股票？",
     "evidence": [], "gold": [], "behavior": "refuse", "refusal_reason": "超出店铺服务范围"},
    {"id": "DEV-OOD-04", "bucket": "ood", "query": "帮我写一首诗。",
     "evidence": [], "gold": [], "behavior": "refuse", "refusal_reason": "超出店铺服务范围"},
]

# 多轮:给已有单轮的 multi_hop/multi_constraint/clarify 题补前文。
DEV_CONTEXT: dict[str, list[dict[str, str]]] = {
    "DEV-MULTI-06": [_turn("user", "ORD-1002 是什么状态？"),
                      _turn("assistant", "ORD-1002 已支付，尚未发货。")],
    "DEV-MULTI-02": [_turn("user", "SH-E300 多少钱？"),
                      _turn("assistant", "SH-E300 售价 299 元。")],
    "DEV-MULTI-11": [_turn("user", "我是金卡会员。"),
                      _turn("assistant", "好的，金卡会员享商品 95 折。")],
    "DEV-CONSTRAINT-01": [_turn("user", "SH-E300 已经签收 10 天了。"),
                           _turn("assistant", "收到，请确认包装和商品状态。")],
    "DEV-CONSTRAINT-07": [_turn("user", "我收货地址在西藏。"),
                           _turn("assistant", "西藏属于偏远地区。")],
    "DEV-CLARIFY-05": [_turn("user", "我要退货"),
                        _turn("assistant", "好的，请提供订单号。")],
}


def _build_case(spec: dict[str, Any], book: FactBook, split: str) -> dict[str, Any]:
    behavior = spec.get("behavior", "answer")
    evidence_ids = spec.get("evidence") or []
    refs = []
    for evidence_id in evidence_ids:
        unit = book.evidence[evidence_id]
        refs.append({
            "doc": unit.doc,
            "section": unit.section,
            "quotes": list(unit.quotes),
            "fact_ids": list(unit.fact_ids),
        })
    gold_facts = []
    for fact_id in spec.get("gold") or []:
        fact = book.fact(fact_id)
        gold_facts.append({
            "fact_id": fact.fact_id,
            "key": fact.key,
            "value": fact.value,
            "unit": fact.unit,
            "conditions": list(fact.conditions),
        })
    answerable = behavior != "refuse"
    evidence_state = spec.get("evidence_state")
    if evidence_state is None:
        if behavior == "refuse":
            evidence_state = "insufficient"
        elif behavior in {"verify_then_answer", "clarify"}:
            evidence_state = "partial" if refs else "insufficient"
        else:
            evidence_state = "sufficient"
    if behavior == "policy_conflict":
        evidence_state = "conflicting"
    case = {
        "schema_version": 2,
        "id": spec["id"],
        "split": split,
        "bucket": spec["bucket"],
        "label_version": "v2",
        "query": spec["query"],
        "conversation_context": list(
            spec.get("context") or DEV_CONTEXT.get(spec["id"]) or []
        ),
        "expected_behavior": behavior,
        "answerable": answerable,
        "evidence_state": evidence_state,
        "required_slots": list(spec.get("slots") or []),
        "clarifying_slots": list(spec.get("clarifying") or []),
        "gold_facts": gold_facts,
        "required_evidence": refs,
        "required_conditions": list(spec.get("conditions") or []),
        "condition_aliases": dict(spec.get("condition_aliases") or {}),
        "slot_aliases": {slot: SLOT_ALIASES[slot] for slot in spec.get("clarifying") or [] if slot in SLOT_ALIASES},
        "verification_markers": list(spec.get("verification") or []),
        "forbidden_claims": list(spec.get("forbidden") or []),
        "expected_refusal_reason": str(spec.get("refusal_reason") or ""),
        "as_of_date": spec.get("as_of"),
        "requires_tools": [tool["tool"] for tool in spec.get("tools") or []],
        "tool_fixtures": list(spec.get("tools") or []),
        "source": "synthetic",
        "source_ref": ",".join(evidence_ids) or "corpus/facts",
        "annotator": "claude-code",
        "reviewer": "codex",
        "review_status": "frozen" if split == "frozen-v1" else "reviewed",
        "notes": str(spec.get("note") or ""),
    }
    return case


SPECS_BY_SPLIT = {"dev": DEV_CASES, "frozen-v1": FROZEN_CASES}


def build_split(split: str, book: FactBook | None = None) -> list[dict[str, Any]]:
    if split not in SPECS_BY_SPLIT:
        raise ValueError(f"未知 split: {split}")
    book = book or load_factbook()
    return [_build_case(spec, book, split) for spec in SPECS_BY_SPLIT[split]]


def write_split(split: str, out: Path) -> int:
    cases = build_split(split)
    validate_cases(cases, split=split)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for case in cases:
            fh.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
    return len(cases)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = args.out or (SPLITS_DIR / f"{args.split}.jsonl")
    count = write_split(args.split, out)
    print(f"wrote {count} cases to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
