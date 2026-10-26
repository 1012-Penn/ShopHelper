"""Deterministic checks that semantic backends are not allowed to override."""

from __future__ import annotations

import re
from datetime import date

from app.claim_grounding.models import ClaimInput, GroundingStatus

_MODEL_RE = re.compile(r"\bSH-[A-Za-z0-9]+\b", re.IGNORECASE)
_ORDER_RE = re.compile(r"\bORD-[A-Za-z0-9]+\b", re.IGNORECASE)
_DATE_RE = re.compile(r"(?<!\d)(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})日?")
_VERSION_RE = re.compile(r"\bv([0-9]+)\b", re.IGNORECASE)
_CJK_NUMBER = "零〇一二三四五六七八九十百两"
_NUM = rf"(?:\d+(?:\.\d+)?|[{_CJK_NUMBER}行半]+)"

_ATTRIBUTE_PATTERNS = {
    "price": re.compile(
        rf"(?:售价|标价|报价|价格为|售价为|卖)[^。；;，,\n]{{0,12}}?({_NUM})\s*元"
    ),
    "battery": re.compile(rf"(?:续航|电池)[^。；;，,\n]{{0,12}}?({_NUM})\s*(小时|天)"),
    "warranty": re.compile(
        rf"(?:质保|保修)[^。；;，,\n]{{0,12}}?({_NUM})\s*(年|个月|月|天)"
    ),
    "shipping": re.compile(rf"(?:运费|邮费)[^。；;，,\n]{{0,12}}?({_NUM})\s*元"),
}
_DEFAULT_ATTRIBUTE_UNIT = {"price": "元", "shipping": "元"}

_CAPABILITY_TERMS = (
    "主动降噪",
    "GPS定位",
    "GPS",
    "血氧监测",
    "心率监测",
    "NFC刷卡",
    "NFC",
    "独立通话",
    "无线充电",
)


def _chinese_number(value: str) -> float | None:
    value = value.strip()
    if value in {"半", "半行"}:
        return 0.5
    if value == "行":
        return 0.0
    if value.isdigit():
        return float(value)
    digits = {
        "零": 0,
        "〇": 0,
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    if value in digits:
        return float(digits[value])
    if "百" in value:
        left, _, right = value.partition("百")
        hundreds = digits.get(left[-1:], 1) if left else 1
        remainder = _chinese_number(right) if right else 0
        return float(hundreds * 100 + int(remainder or 0))
    if "十" in value:
        left, _, right = value.partition("十")
        tens = digits.get(left[-1:], 1) if left else 1
        ones = digits.get(right[:1], 0) if right else 0
        return float(tens * 10 + ones)
    return None


def _canonical(attribute: str, number: str, unit: str) -> tuple[float, str] | None:
    value = _chinese_number(number)
    if value is None:
        return None
    if attribute == "warranty" and unit == "年":
        return value * 12, "month"
    if attribute == "warranty" and unit in {"个月", "月"}:
        return value, "month"
    if attribute == "warranty" and unit == "天":
        return value, "day"
    if unit == "小时":
        return value, "hour"
    if unit == "天":
        return value, "day"
    if unit == "元":
        return value, "yuan"
    return value, unit


def _attributes(text: str) -> dict[str, tuple[float, str]]:
    values: dict[str, tuple[float, str]] = {}
    for attribute, pattern in _ATTRIBUTE_PATTERNS.items():
        match = pattern.search(text)
        if not match:
            continue
        number = match.group(1)
        unit = (
            match.group(2)
            if (match.lastindex or 0) >= 2
            else _DEFAULT_ATTRIBUTE_UNIT.get(attribute, "")
        )
        canonical = _canonical(attribute, number, unit)
        if canonical is not None:
            values[attribute] = canonical
    return values


def _dates(text: str) -> list[date]:
    values = []
    for year, month, day in _DATE_RE.findall(text):
        try:
            values.append(date(int(year), int(month), int(day)))
        except ValueError:
            continue
    return values


def _expected_policy_versions(text: str, as_of: date) -> set[str]:
    expected: set[str] = set()
    segments = re.split(r"[\n。；;]", text)
    for segment in segments:
        versions = {f"v{item.lower()}" for item in _VERSION_RE.findall(segment)}
        if len(versions) != 1:
            continue
        version = next(iter(versions))
        dates = _dates(segment)
        if not dates:
            continue
        if len(dates) >= 2:
            start, end = min(dates), max(dates)
            if start <= as_of <= end:
                expected.add(version)
            continue
        boundary = dates[0]
        if any(token in segment for token in ("起", "之后", "自")):
            if as_of >= boundary:
                expected.add(version)
        elif (
            any(token in segment for token in ("早于", "之前", "以前"))
            and as_of < boundary
        ):
            expected.add(version)
    return expected


def _capability_polarity(text: str, term: str) -> bool | None:
    compact = re.sub(r"\s+", "", text)
    if term not in compact:
        return None
    negative = re.search(
        rf"(?:不支持|不能|不具备|不提供|没有|不享).{{0,6}}{re.escape(term)}",
        compact,
    )
    if negative:
        return False
    positive = re.search(
        rf"(?:支持|具备|提供|享).{{0,6}}{re.escape(term)}",
        compact,
    )
    if positive:
        return True
    return None


def deterministic_status(
    claim: ClaimInput,
) -> tuple[GroundingStatus | None, str]:
    if not claim.evidence:
        return None, ""

    claim_text = claim.claim
    evidence_text = "\n".join(item.text for item in claim.evidence)
    claim_models = {item.upper() for item in _MODEL_RE.findall(claim_text)}
    if claim.model_number:
        claim_models.add(claim.model_number.upper())
    evidence_models = {item.upper() for item in _MODEL_RE.findall(evidence_text)}
    if claim_models and evidence_models and claim_models.isdisjoint(evidence_models):
        return (
            "unsupported",
            f"候选证据仅绑定其他型号:{','.join(sorted(evidence_models))}",
        )

    claim_orders = {item.upper() for item in _ORDER_RE.findall(claim_text)}
    if claim.order_id:
        claim_orders.add(claim.order_id.upper())
    evidence_orders = {item.upper() for item in _ORDER_RE.findall(evidence_text)}
    if claim_orders and evidence_orders and claim_orders.isdisjoint(evidence_orders):
        return (
            "unsupported",
            f"跨订单证据:{','.join(sorted(evidence_orders))}",
        )

    # ``as_of_date`` is experiment context, not necessarily an asserted order
    # fact. Only dates written inside the claim can contradict evidence dates.
    claim_dates = _dates(claim_text)
    evidence_dates = _dates(evidence_text)
    if claim_orders and claim_dates and evidence_dates:
        for order_id in claim_orders:
            if (
                order_id in evidence_text.upper()
                and claim_dates[0] not in evidence_dates
            ):
                return (
                    "unsupported",
                    (
                        f"{order_id} 的日期与证据不一致:"
                        f"claim={claim_dates[0]},"
                        f"evidence={sorted(set(evidence_dates))}"
                    ),
                )

    claim_version = claim.policy_version
    if not claim_version:
        versions = _VERSION_RE.findall(claim_text)
        if versions:
            claim_version = f"v{versions[0]}"
    if claim.as_of_date and claim_version:
        values = _dates(claim.as_of_date)
        if values:
            expected = _expected_policy_versions(evidence_text, values[0])
            if expected and claim_version.lower() not in expected:
                return (
                    "unsupported",
                    (
                        f"{values[0]} 应适用 {','.join(sorted(expected))},"
                        f"claim 使用 {claim_version}"
                    ),
                )

    claim_attributes = _attributes(claim_text)
    evidence_attributes = _attributes(evidence_text)
    for attribute, claim_value in claim_attributes.items():
        evidence_value = evidence_attributes.get(attribute)
        if evidence_value is None:
            continue
        if claim_value != evidence_value:
            return (
                "unsupported",
                (
                    f"{attribute} 数值或单位冲突:"
                    f"claim={claim_value},evidence={evidence_value}"
                ),
            )

    for term in _CAPABILITY_TERMS:
        claim_polarity = _capability_polarity(claim_text, term)
        evidence_polarity = _capability_polarity(evidence_text, term)
        if (
            claim_polarity is not None
            and evidence_polarity is not None
            and claim_polarity != evidence_polarity
        ):
            return "unsupported", f"{term} 的支持/否定与证据相反"

    compact_claim = re.sub(r"\s+", "", claim_text)
    compact_evidence = re.sub(r"\s+", "", evidence_text)
    if any(
        term in compact_claim for term in ("已经发货", "当前已发货", "已发出")
    ) and any(
        term in compact_evidence for term in ("当前未发货", "尚未发货", "未发货")
    ):
        return "unsupported", "订单发货状态与证据相反"
    if any(term in compact_claim for term in ("当前未发货", "尚未发货")) and any(
        term in compact_evidence for term in ("已经发货", "当前已发货", "已发出")
    ):
        return "unsupported", "订单发货状态与证据相反"

    payment_terms = {"微信支付": "wechat", "支付宝": "alipay", "银联卡": "unionpay"}
    claim_payments = {
        label for term, label in payment_terms.items() if term in claim_text
    }
    evidence_payments = {
        label for term, label in payment_terms.items() if term in evidence_text
    }
    if (
        claim_payments
        and evidence_payments
        and claim_payments.isdisjoint(evidence_payments)
    ):
        return "unsupported", "支付渠道与证据冲突"

    if claim_attributes and claim_attributes == evidence_attributes:
        return "supported", "型号与数值/单位由候选证据直接确认"
    return None, ""
