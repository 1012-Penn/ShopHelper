"""回答层/行为层确定性检查:数值、条件、禁用断言、引用、澄清槽位。

原则:确定性检查优先;LLM 裁判只处理同义转述,不覆盖确定性结论。
不合成任何单一总分。
"""
from __future__ import annotations

import re
import unicodedata
from datetime import date
from dataclasses import asdict, dataclass

from app.guard import REFUSAL_MARKER

DEFAULT_SLOT_ALIASES = {
    "sku": ("型号", "哪款", "哪个商品", "商品", "款式"),
    "order_id": ("订单号", "订单编号", "哪笔订单", "下单手机号"),
    "member_status": ("会员", "会员等级", "是不是会员"),
    "region": ("地区", "收货地", "寄到哪", "省份"),
    "purchase_date": ("购买日期", "下单时间", "签收日期"),
    "damage_photo": ("照片", "图片", "凭证"),
    "damage_video": ("视频", "录像"),
}
DEFAULT_CONDITION_ALIASES = {
    "会员身份": ("会员",),
    "商品参与活动": ("参与活动", "活动商品"),
    "最终价格以页面为准": ("页面为准", "以页面"),
    "微信/支付宝": ("微信", "支付宝"),
    "银联卡/银行卡": ("银行卡", "银联卡"),
    "7 天内": ("7天", "七天"),
    "15 天内": ("15天", "十五天"),
    "商品完好": ("完好",),
    "吊牌、包装齐全": ("吊牌", "包装齐全"),
    "签收日期早于 2026-06-01": ("2026-06-01 之前", "早于 2026-06-01"),
    "签收日期为 2026-06-01 或之后": ("2026-06-01 之后", "6 月 1 日之后"),
    "所有会员等级": ("所有等级",),
    "商品原价": ("原价",),
    "在售的正价商品": ("在售", "正价"),
    "未发货": ("未发货",),
    "已发货只能拒收或退货": ("拒收", "退货"),
    "审核通过后": ("审核通过",),
    "包邮基础上": ("包邮",),
    "偏远地区": ("偏远",),
    "按 95 折后的商品金额计算": ("95 折后", "折后"),
    "合并后的商品总额": ("合并后", "合并订单"),
    "拆单后各子订单分别计算": ("拆单", "分别计算"),
    "偏远补运": ("补运",),
    "不参与折扣": ("不参与", "不享受"),
    "责任方承担": ("责任方",),
    "人为损坏按检测结果收费": ("检测", "收费"),
    "提供订单号": ("订单号",),
    "仅限签收当日": ("当日", "当天"),
    "定制类仅在有质量问题时可退": ("定制", "质量问题"),
    "超过 15 天无理由窗口": ("超过 15 天",),
    "以检测结果为准": ("检测结果",),
    "整机": ("整机",),
    "签收之日起": ("签收",),
    "秒杀商品不叠加": ("秒杀", "不叠加"),
    "会员 95 折": ("95 折",),
    "店铺满减券": ("满减",),
    "运费券": ("运费券",),
    "配件类": ("配件",),
    "例外以商品目录为准": ("商品目录",),
    "首重 1kg": ("首重",),
    "原路返回": ("原路",),
    "支付渠道": ("渠道",),
}
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_CITE_RE = re.compile(r"\[(\d+)\]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])")
_DATE_TEXT = r"\d{4}\s*[-/年]\s*\d{1,2}\s*[-/月]\s*\d{1,2}\s*日?"
_DATE_RE = re.compile(rf"(?<!\d)({_DATE_TEXT})")
_CN_DIGITS = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


@dataclass(frozen=True)
class CheckOutcome:
    name: str
    passed: bool
    score: float = 0.0
    detail: str = ""
    status: str = ""


def normalize_answer(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


def _segments(text: str) -> list[str]:
    return [
        part
        for part in re.split(r"[、,，;；/\s]+", normalize_answer(text))
        if len(part) >= 2
    ]


def _contains(haystack: str, needle: str) -> bool:
    target = normalize_answer(needle)
    if not target:
        return False
    if target in haystack:
        return True
    segments = _segments(target)
    return bool(segments) and all(segment in haystack for segment in segments)


def _loose_contains(haystack: str, needle: str) -> bool:
    """列表事实用:允许「清仓与停产库存商品」被答案里的「清仓」覆盖。"""
    target = normalize_answer(needle)
    if not target:
        return False
    if target in haystack:
        return True
    parts = [part for part in re.split(r"[与和、,，/\s]+", target) if len(part) >= 2]
    return any(part in haystack for part in parts)


def _cn_integer(text: str) -> int | None:
    if text == "十":
        return 10
    if "十" in text:
        left, _, right = text.partition("十")
        tens = _CN_DIGITS.get(left, 1) if left else 1
        ones = _CN_DIGITS.get(right, 0) if right else 0
        return tens * 10 + ones
    if len(text) == 1 and text in _CN_DIGITS:
        return _CN_DIGITS[text]
    return None


def _quantity_values(text: str, unit: str) -> set[float]:
    normalized = normalize_answer(text)
    values: set[float] = set()
    canonical = normalize_answer(unit)
    if canonical in {"月", "个月"}:
        if "半年" in normalized:
            values.update({0.5, 6.0})
        if "一年" in normalized:
            values.update({1.0, 12.0})
        patterns = (
            (r"(\d+(?:\.\d+)?)\s*个?月", 1),
            (r"(\d+(?:\.\d+)?)\s*年", 12),
            (r"([零一二两三四五六七八九十]+)\s*个?月", 1),
            (r"([零一二两三四五六七八九十]+)\s*年", 12),
        )
        for pattern, multiplier in patterns:
            for match in re.finditer(pattern, normalized):
                raw = match.group(1)
                number = float(raw) if re.fullmatch(r"\d+(?:\.\d+)?", raw) else _cn_integer(raw)
                if number is not None:
                    values.add(number * multiplier)
        return values
    aliases = {
        "元": r"(?:元|块)",
        "天": r"(?:天|日)",
        "小时": r"(?:小时|h)",
        "工作日": r"(?:工作日|个工作日)",
        "键": r"键",
        "米": r"米",
    }
    unit_pattern = aliases.get(canonical, re.escape(canonical))
    for match in re.finditer(rf"(\d+(?:\.\d+)?)\s*{unit_pattern}", normalized):
        values.add(float(match.group(1)))
    for match in re.finditer(rf"([零一二两三四五六七八九十]+)\s*{unit_pattern}", normalized):
        number = _cn_integer(match.group(1))
        if number is not None:
            values.add(float(number))
    return values


def _extract_dates(text: str) -> list[date]:
    dates: list[date] = []
    for match in _DATE_RE.finditer(normalize_answer(text)):
        value = match.group(1)
        parts = re.findall(r"\d+", value)
        if len(parts) < 3:
            continue
        try:
            dates.append(date(int(parts[0]), int(parts[1]), int(parts[2])))
        except ValueError:
            continue
    return dates


def _signed_dates(text: str) -> list[date]:
    normalized = normalize_answer(text)
    values: list[date] = []
    patterns = (
        rf"签收(?:日期|时间)?\s*(?:为|于)?\s*({_DATE_TEXT})",
        rf"于\s*({_DATE_TEXT})\s*签收",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, normalized):
            parsed = _extract_dates(match.group(1))
            values.extend(parsed)
    return list(dict.fromkeys(values))


def number_covered(answer: str, value, unit: str = "", tolerance: float = 0.0) -> bool:
    normalized = normalize_answer(answer)
    if isinstance(value, bool):
        yes = ("支持", "是", "有", "可以", "能", "适用", "参与", "享受", "计入")
        no = ("不支持", "不是", "没有", "不可以", "不能", "不参与", "不适用", "不享受", "不计入", "不叠加")

        def _hit(term: str) -> bool:
            for match in re.finditer(term, normalized):
                prefix = normalized[max(0, match.start() - 2):match.start()]
                if value:
                    # “不支持/不参与”里的“支持/参与”不算肯定。
                    if not prefix.endswith(("不", "没", "非")):
                        return True
                else:
                    # “是否支持”是提问,不是事实。
                    if not prefix.endswith("是"):
                        return True
            return False

        return any(_hit(term) for term in (yes if value else no))
    if isinstance(value, list):
        # 列表事实只要求答案覆盖其中一项;完整逐项覆盖交给 required_conditions。
        return any(
            _loose_contains(normalized, str(item)) if isinstance(item, str)
            else number_covered(answer, item, unit, tolerance)
            for item in value
        )
    if isinstance(value, (int, float)):
        unit_norm = normalize_answer(unit)
        numbers = (_quantity_values(normalized, unit_norm) if unit_norm else
                   {float(match) for match in _NUM_RE.findall(normalized)})
        return any(abs(number - float(value)) <= float(tolerance) for number in numbers)
    return _contains(normalized, str(value))


def _condition_matches(answer: str, condition: str) -> bool:
    normalized_answer = normalize_answer(answer)
    normalized_condition = normalize_answer(condition)
    condition_dates = _extract_dates(normalized_condition)
    answer_dates = _signed_dates(normalized_answer) or _extract_dates(normalized_answer)
    if condition_dates and answer_dates:
        boundary = condition_dates[0]
        if "早于" in normalized_condition or "之前" in normalized_condition:
            return any(item < boundary for item in answer_dates)
        if "之后" in normalized_condition:
            return any(item >= boundary for item in answer_dates)
        return any(item == boundary for item in answer_dates)
    window = re.search(r"(\d+)\s*天\s*(?:内|之内)", normalized_condition)
    if window:
        limit = int(window.group(1))
        if re.search(rf"(?:超过|超出|晚于)\s*{limit}\s*天", normalized_answer):
            return False
        days = [int(value) for value in re.findall(r"第\s*(\d+)\s*天", normalized_answer)]
        if days and any(value <= limit for value in days):
            return any(term in normalized_answer
                       for term in ("窗口内", "天内", "还能退", "可以退"))
        if f"{limit}天内" in normalized_answer or f"{limit}天之内" in normalized_answer:
            return True
        return False
    return False


def condition_coverage(
    answer: str,
    conditions: list[str],
    aliases: dict[str, list[str]] | None = None,
) -> CheckOutcome:
    if not conditions:
        return CheckOutcome("condition_coverage", True, 1.0, "no_conditions")
    normalized = normalize_answer(answer)
    hits = []
    for condition in conditions:
        if _condition_matches(answer, condition):
            hits.append(True)
            continue
        required_numbers = _NUM_RE.findall(normalize_answer(condition))
        is_date_condition = bool(re.search(r"\d{4}\s*[-/年]\s*\d{1,2}", condition))
        if (
            required_numbers
            and not is_date_condition
            and not all(number in normalized for number in required_numbers)
        ):
            hits.append(False)
            continue
        candidates = [
            condition,
            *DEFAULT_CONDITION_ALIASES.get(condition, ()),
            *(aliases or {}).get(condition, []),
        ]
        hits.append(any(_contains(normalized, candidate) for candidate in candidates))
    score = sum(hits) / len(hits)
    return CheckOutcome("condition_coverage", all(hits), score, f"{sum(hits)}/{len(hits)}")


def forbidden_claims_found(answer: str, forbidden: list[str]) -> list[str]:
    # 完整短语匹配 + 否定前缀豁免:避免把“不是所有商品 95 折”当成肯定断言。
    normalized = normalize_answer(answer)
    negations = ("不是", "并非", "不可以", "不支持", "不能", "不会", "不属于")
    found: list[str] = []
    for claim in forbidden:
        target = normalize_answer(claim)
        if not target:
            continue
        start = 0
        while True:
            index = normalized.find(target, start)
            if index < 0:
                break
            prefix = normalized[max(0, index - 8):index]
            if not any(neg in prefix for neg in negations):
                found.append(claim)
                break
            start = index + 1
    return found


def _slot_terms(slot: str, aliases: dict[str, list[str]] | None) -> tuple[str, ...]:
    return tuple((aliases or {}).get(slot) or DEFAULT_SLOT_ALIASES.get(slot, (slot,)))


def asked_slots(answer: str, slots: list[str], aliases=None) -> list[str]:
    normalized = normalize_answer(answer)
    return [slot for slot in slots if any(term in normalized for term in _slot_terms(slot, aliases))]


def clarifying_slot_recall(
    answer: str, required_slots: list[str], aliases=None
) -> CheckOutcome:
    if not required_slots:
        return CheckOutcome("clarify_slot_recall", True, 1.0, "no_slots")
    hit = asked_slots(answer, required_slots, aliases)
    score = len(hit) / len(required_slots)
    return CheckOutcome("clarify_slot_recall", len(hit) == len(required_slots), score, f"{len(hit)}/{len(required_slots)}")


def clarifying_precision(
    answer: str, required_slots: list[str], aliases=None
) -> CheckOutcome:
    if not required_slots:
        return CheckOutcome("clarify_precision", True, 1.0, "no_slots")
    all_slots = list(dict.fromkeys([*required_slots, *DEFAULT_SLOT_ALIASES]))
    asked = asked_slots(answer, all_slots, aliases)
    matched = [slot for slot in asked if slot in required_slots]
    if not asked:
        return CheckOutcome("clarify_precision", False, 0.0, "no_slot_asked")
    return CheckOutcome("clarify_precision", len(asked) == len(matched),
                        len(matched) / len(asked), f"{len(matched)}/{len(asked)}")


def gold_fact_accuracy(
    answer: str, gold_facts: list[dict], fact_aliases: dict[str, list[str]] | None = None
) -> CheckOutcome:
    if not gold_facts:
        return CheckOutcome("gold_fact_accuracy", True, 1.0, "0 gold facts; not scored")
    hits = 0
    for fact in gold_facts:
        fact_id = str(fact.get("fact_id") or "")
        aliases = (fact_aliases or {}).get(fact_id, [])
        if (
            number_covered(
            answer,
            fact.get("value"),
            str(fact.get("unit") or ""),
            float(fact.get("tolerance") or 0.0),
            )
            or any(_contains(normalize_answer(answer), alias) for alias in aliases)
        ):
            hits += 1
    return CheckOutcome(
        "gold_fact_accuracy", hits == len(gold_facts), hits / len(gold_facts),
        f"{hits}/{len(gold_facts)}",
    )


def parse_citations(answer: str) -> list[tuple[str, int]]:
    citations: list[tuple[str, int]] = []
    owned = re.sub(r"([。！？!?；;])(\s*(?:\[\d+\])+)", r"\2\1", answer)
    for sentence in _SENTENCE_SPLIT_RE.split(owned):
        for match in _CITE_RE.finditer(sentence):
            citations.append((sentence.strip(), int(match.group(1))))
    return citations


def _semantic_surface(text: str) -> str:
    text = normalize_answer(_CITE_RE.sub("", text))
    replacements = (
        ("价格为", "售价"), ("售价为", "售价"), ("为", "是"),
        ("可以", "可"), ("能够", "能"), ("之内", "内"),
        ("个工作日", "工作日"), ("个月", "月"),
    )
    for old, new in replacements:
        text = text.replace(old, new)
    return "".join(char for char in text if char not in "，,。.!！?？；;：: ")


def _sentence_support_status(sentence: str, evidence: str) -> str:
    """Tri-state support. Character overlap is not evidence of semantics."""
    clean = _CITE_RE.sub("", sentence)
    numeric_claim = re.sub(
        r"(?:SH|ORD)-[A-Za-z0-9]+", "", normalize_answer(clean), flags=re.I
    )
    sentence_numbers = set(_NUM_RE.findall(numeric_claim))
    evidence_numbers = set(_NUM_RE.findall(normalize_answer(evidence)))
    if not sentence_numbers.issubset(evidence_numbers):
        return "unsupported"
    answer_entities = set(re.findall(r"(?:SH|ORD)-[A-Za-z0-9]+", clean, flags=re.I))
    evidence_entities = set(re.findall(r"(?:SH|ORD)-[A-Za-z0-9]+", evidence, flags=re.I))
    if answer_entities and evidence_entities and not answer_entities & evidence_entities:
        return "unsupported"

    evidence_surface = _semantic_surface(evidence)
    units = [part for part in re.split(r"[，,；;。！？!?]", clean) if len(part.strip()) >= 2]
    if units and all(_semantic_surface(part) in evidence_surface for part in units):
        return "supported"
    surface = _semantic_surface(clean)
    if surface and surface in evidence_surface:
        return "supported"
    return "uncertain"


def citation_support_rate(
    answer: str, evidence_by_n: dict[int, str], threshold: float = 0.30
) -> CheckOutcome:
    citations = parse_citations(answer)
    if not citations:
        if REFUSAL_MARKER in answer or classify_behavior(answer) == "refuse":
            return CheckOutcome("citation_support", True, 1.0, "refusal_without_citation")
        return CheckOutcome("citation_support", False, 0.0, "no_citation")
    statuses = [
        _sentence_support_status(sentence, evidence_by_n[n])
        if n in evidence_by_n else "unsupported"
        for sentence, n in citations
    ]
    supported = statuses.count("supported")
    score = supported / len(citations)
    if "unsupported" in statuses:
        status = "unsupported"
    elif "uncertain" in statuses:
        status = "uncertain"
    else:
        status = "supported"
    detail = f"{supported}/{len(citations)}"
    if status == "uncertain":
        detail += "; uncertain"
    return CheckOutcome("citation_support", status == "supported", score, detail, status)


def classify_behavior(answer: str) -> str:
    normalized = normalize_answer(answer)
    marker = normalize_answer(REFUSAL_MARKER)
    body = (
        normalized[len(marker):].lstrip(" :：")
        if normalized.startswith(marker)
        else normalized
    )
    asks = "?" in answer or "？" in answer
    slot_terms = ("请提供", "请问", "哪款", "订单号", "订单编号", "型号", "会员等级")
    requests_slot = body.startswith(("请提供", "请告知", "请告诉我", "请问", "麻烦提供"))
    if (asks and any(term in normalized for term in slot_terms)) or requests_slot:
        return "clarify"
    if answer.lstrip().startswith(REFUSAL_MARKER) or normalized.startswith(("无法回答", "抱歉")):
        return "refuse"
    if "查不到" in normalized and not asks:
        return "refuse"
    if normalized.startswith(("请提供", "请告知", "请告诉我", "麻烦提供", "请问")):
        return "clarify"
    refusal_phrases = (
        "我无法", "无法根据", "无法依据", "无法提供", "无法给出", "无法告知",
        "无法协助", "该请求不在", "该问题不在",
    )
    explicit_refusal_prefix = normalized.startswith(
        ("无法确认", "无法判断", "无法核实")
    )
    evidence_limited = normalized.startswith(
        ("现有证据", "现有资料", "现有商品", "证据仅")
    ) and any(
        term in normalized
        for term in (
            "未提供", "未说明", "未包含", "未涉及", "未覆盖", "未列出", "未提及", "无法"
        )
    )
    out_of_scope = normalized.startswith("该问题属于") and "不在" in normalized
    if (
        any(term in normalized for term in refusal_phrases)
        or explicit_refusal_prefix
        or evidence_limited
        or out_of_scope
    ):
        return "refuse"
    return "answer"


def _claim_counts(answer: str, evidence_by_n: dict[int, str]) -> tuple[int, int, int]:
    if REFUSAL_MARKER in answer or classify_behavior(answer) == "refuse":
        return 0, 0, 0
    citations = parse_citations(answer)
    support_status: dict[str, str] = {}
    for sentence, n in citations:
        status = (_sentence_support_status(sentence, evidence_by_n[n])
                  if n in evidence_by_n else "unsupported")
        prior = support_status.get(sentence)
        if prior == "supported" or status == "supported":
            support_status[sentence] = "supported"
        elif prior == "unsupported" or status == "unsupported":
            support_status[sentence] = "unsupported"
        else:
            support_status[sentence] = "uncertain"
    owned = re.sub(r"([。！？!?；;])(\s*(?:\[\d+\])+)", r"\2\1", answer)
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(owned) if len(s.strip()) >= 6]
    unsupported = sum(support_status.get(sentence) != "supported" for sentence in sentences)
    uncertain = sum(support_status.get(sentence) == "uncertain" for sentence in sentences)
    return unsupported, uncertain, len(sentences)


def unsupported_claim_rate(answer: str, evidence_by_n: dict[int, str]) -> float:
    unsupported, _, total = _claim_counts(answer, evidence_by_n)
    return 0.0 if total == 0 else unsupported / total


def uncertain_claim_rate(answer: str, evidence_by_n: dict[int, str]) -> float:
    _, uncertain, total = _claim_counts(answer, evidence_by_n)
    return 0.0 if total == 0 else uncertain / total


def check_case_deterministic(
    case: dict, answer: str, evidence_by_n: dict[int, str], *,
    scoring_overlay: dict | None = None,
) -> dict:
    overlay = scoring_overlay or {}
    case_overlay = (overlay.get("cases") or {}).get(case.get("id"), {})
    background_fact_ids = set(case_overlay.get("background_gold_fact_ids") or [])
    background_conditions = set(case_overlay.get("background_conditions") or [])
    expected = case.get("expected_behavior", "answer")
    all_gold = case.get("gold_facts") or []
    gold = [
        fact for fact in all_gold
        if str(fact.get("fact_id") or "") not in background_fact_ids
    ]
    all_conditions = case.get("required_conditions") or []
    conditions = [item for item in all_conditions if item not in background_conditions]
    condition_aliases = {
        **((overlay.get("condition_aliases") or {})),
        **(case.get("condition_aliases") or {}),
    }
    gold_outcome = gold_fact_accuracy(
        answer, gold, case_overlay.get("fact_aliases") or {}
    )
    condition_outcome = condition_coverage(
        answer, conditions, condition_aliases
    )
    forbidden = forbidden_claims_found(answer, case.get("forbidden_claims") or [])
    slot_recall = clarifying_slot_recall(
        answer, case.get("clarifying_slots") or [], case.get("slot_aliases")
    )
    slot_precision = clarifying_precision(
        answer, case.get("clarifying_slots") or [], case.get("slot_aliases")
    )
    citation = citation_support_rate(answer, evidence_by_n)
    unsupported_claims, uncertain_claims, substantive_claims = _claim_counts(
        answer, evidence_by_n
    )
    unsupported = unsupported_claim_rate(answer, evidence_by_n)
    uncertain = uncertain_claim_rate(answer, evidence_by_n)
    behavior = classify_behavior(answer)
    verification_markers = case.get("verification_markers") or []
    verification_ok = (
        not verification_markers
        or any(marker in answer for marker in verification_markers)
    )

    failure_labels: list[str] = []
    if forbidden:
        failure_labels.append("forbidden_claim")
    if gold and gold_outcome.score < 1.0:
        failure_labels.append("answer_wrong_fact")
    if conditions and condition_outcome.score < 1.0:
        failure_labels.append("answer_missing_condition")
    if expected == "clarify" and slot_recall.score < 1.0:
        failure_labels.append("clarify_missing_slot")
    if expected == "clarify" and slot_precision.score < 1.0:
        failure_labels.append("clarify_over_asking")
    if citation.status == "unsupported" and expected in {"answer", "answer_with_conditions"}:
        failure_labels.append("citation_unsupported")
    if citation.status == "uncertain" and expected in {"answer", "answer_with_conditions"}:
        failure_labels.append("citation_uncertain")
    if expected == "verify_then_answer" and not verification_ok:
        failure_labels.append("verification_overclaim")
    if expected != "refuse" and behavior == "refuse":
        failure_labels.append("over_refusal")
    if expected == "refuse" and behavior != "refuse":
        failure_labels.append("under_refusal")

    return {
        "gold_facts_checked": len(gold),
        "gold_facts_excluded_as_background": sorted(background_fact_ids),
        "conditions_excluded_as_background": sorted(background_conditions),
        "scoring_overlay": {
            "version": str(overlay.get("version") or "rag-quality-v3"),
            **case_overlay,
        },
        "gold_fact_accuracy": asdict(gold_outcome),
        "condition_coverage": asdict(condition_outcome),
        "forbidden_claims": forbidden,
        "clarify_slot_recall": asdict(slot_recall),
        "clarify_precision": asdict(slot_precision),
        "citation_support": asdict(citation),
        "unsupported_claim_rate": unsupported,
        "uncertain_claim_rate": uncertain,
        "unsupported_claims": unsupported_claims,
        "uncertain_claims": uncertain_claims,
        "substantive_claims": substantive_claims,
        "verification_compliant": verification_ok,
        "behavior_pred": behavior,
        "refused": behavior == "refuse",
        "failure_labels": failure_labels,
    }
