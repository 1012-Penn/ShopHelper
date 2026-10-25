"""回答层/行为层确定性检查:数值、条件、禁用断言、引用、澄清槽位。

原则:确定性检查优先;LLM 裁判只处理同义转述,不覆盖确定性结论。
不合成任何单一总分。
"""
from __future__ import annotations

import re
import unicodedata
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


@dataclass(frozen=True)
class CheckOutcome:
    name: str
    passed: bool
    score: float = 0.0
    detail: str = ""


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


def number_covered(answer: str, value, unit: str = "", tolerance: float = 0.0) -> bool:
    normalized = normalize_answer(answer)
    if isinstance(value, bool):
        yes = ("支持", "是", "有", "可以", "能", "适用", "参与", "享受", "计入")
        no = ("不支持", "不是", "没有", "不可以", "不能", "不参与", "不适用", "不享受", "不计入", "不叠加")
        # 先看否定,避免“不支持”因包含“支持”被当成 true。
        if any(term in normalized for term in no):
            return False if value else True
        return any(term in normalized for term in yes) if value else False
    if isinstance(value, list):
        # 列表事实只要求答案覆盖其中一项;完整逐项覆盖交给 required_conditions。
        return any(
            _loose_contains(normalized, str(item)) if isinstance(item, str)
            else number_covered(answer, item, unit, tolerance)
            for item in value
        )
    if isinstance(value, (int, float)):
        numbers = [float(match) for match in _NUM_RE.findall(normalized)]
        if not any(abs(number - float(value)) <= float(tolerance) for number in numbers):
            return False
        unit_norm = normalize_answer(unit)
        if unit_norm and unit_norm not in normalized:
            unit_aliases = {"元": ("元", "块"), "天": ("天",), "小时": ("小时", "h")}
            aliases = unit_aliases.get(unit_norm, (unit_norm,))
            if not any(alias in normalized for alias in aliases):
                return False
        return True
    return _contains(normalized, str(value))


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
        required_numbers = _NUM_RE.findall(normalize_answer(condition))
        if required_numbers and not all(number in normalized for number in required_numbers):
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


def gold_fact_accuracy(answer: str, gold_facts: list[dict]) -> CheckOutcome:
    if not gold_facts:
        return CheckOutcome("gold_fact_accuracy", True, 1.0, "0 gold facts; not scored")
    hits = 0
    for fact in gold_facts:
        if number_covered(
            answer,
            fact.get("value"),
            str(fact.get("unit") or ""),
            float(fact.get("tolerance") or 0.0),
        ):
            hits += 1
    return CheckOutcome(
        "gold_fact_accuracy", hits == len(gold_facts), hits / len(gold_facts),
        f"{hits}/{len(gold_facts)}",
    )


def parse_citations(answer: str) -> list[tuple[str, int]]:
    citations: list[tuple[str, int]] = []
    for sentence in _SENTENCE_SPLIT_RE.split(answer):
        for match in _CITE_RE.finditer(sentence):
            citations.append((sentence.strip(), int(match.group(1))))
    return citations


def _char_overlap(left: str, right: str, n: int = 4) -> float:
    left_norm, right_norm = normalize_answer(left), normalize_answer(right)
    left_grams = {left_norm[i : i + n] for i in range(max(0, len(left_norm) - n + 1))}
    if not left_grams:
        return 0.0
    right_grams = {right_norm[i : i + n] for i in range(max(0, len(right_norm) - n + 1))}
    return len(left_grams & right_grams) / len(left_grams)


def _sentence_supported(sentence: str, evidence: str, threshold: float = 0.30) -> bool:
    """证据要同时满足:数字不冲突,且字符 4-gram 重叠达到阈值。"""
    clean = _CITE_RE.sub("", sentence)
    sentence_numbers = set(_NUM_RE.findall(normalize_answer(clean)))
    evidence_numbers = set(_NUM_RE.findall(normalize_answer(evidence)))
    if not sentence_numbers.issubset(evidence_numbers):
        return False
    return _char_overlap(clean, evidence) >= threshold


def citation_support_rate(
    answer: str, evidence_by_n: dict[int, str], threshold: float = 0.30
) -> CheckOutcome:
    citations = parse_citations(answer)
    if not citations:
        if REFUSAL_MARKER in answer or classify_behavior(answer) == "refuse":
            return CheckOutcome("citation_support", True, 1.0, "refusal_without_citation")
        return CheckOutcome("citation_support", False, 0.0, "no_citation")
    supported = sum(
        1
        for sentence, n in citations
        if n in evidence_by_n and _sentence_supported(sentence, evidence_by_n[n], threshold)
    )
    score = supported / len(citations)
    return CheckOutcome("citation_support", supported == len(citations), score,
                        f"{supported}/{len(citations)}")


def classify_behavior(answer: str) -> str:
    normalized = normalize_answer(answer)
    asks = "?" in answer or "？" in answer
    slot_terms = ("请提供", "请问", "哪款", "订单号", "订单编号", "型号", "会员等级")
    if asks and any(term in normalized for term in slot_terms):
        return "clarify"
    if answer.lstrip().startswith(REFUSAL_MARKER) or normalized.startswith(("无法回答", "抱歉")):
        return "refuse"
    if any(term in normalized for term in ("转人工", "查不到")) and not asks:
        return "refuse"
    if normalized.startswith(("请提供", "请告知", "麻烦提供", "请问")):
        return "clarify"
    return "answer"


def _claim_counts(answer: str, evidence_by_n: dict[int, str]) -> tuple[int, int]:
    if REFUSAL_MARKER in answer or classify_behavior(answer) == "refuse":
        return 0, 0
    citations = parse_citations(answer)
    supported_sentences = {
        sentence for sentence, n in citations
        if n in evidence_by_n and _sentence_supported(sentence, evidence_by_n[n])
    }
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(answer) if len(s.strip()) >= 6]
    unsupported = [s for s in sentences if s not in supported_sentences]
    return len(unsupported), len(sentences)


def unsupported_claim_rate(answer: str, evidence_by_n: dict[int, str]) -> float:
    unsupported, total = _claim_counts(answer, evidence_by_n)
    return 0.0 if total == 0 else unsupported / total


def check_case_deterministic(
    case: dict, answer: str, evidence_by_n: dict[int, str]
) -> dict:
    expected = case.get("expected_behavior", "answer")
    gold = case.get("gold_facts") or []
    gold_outcome = gold_fact_accuracy(answer, gold)
    condition_outcome = condition_coverage(
        answer, case.get("required_conditions") or [], case.get("condition_aliases")
    )
    forbidden = forbidden_claims_found(answer, case.get("forbidden_claims") or [])
    slot_recall = clarifying_slot_recall(
        answer, case.get("clarifying_slots") or [], case.get("slot_aliases")
    )
    slot_precision = clarifying_precision(
        answer, case.get("clarifying_slots") or [], case.get("slot_aliases")
    )
    citation = citation_support_rate(answer, evidence_by_n)
    unsupported_claims, substantive_claims = _claim_counts(answer, evidence_by_n)
    unsupported = unsupported_claim_rate(answer, evidence_by_n)
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
    if (case.get("required_conditions") or []) and condition_outcome.score < 1.0:
        failure_labels.append("answer_missing_condition")
    if expected == "clarify" and slot_recall.score < 1.0:
        failure_labels.append("clarify_missing_slot")
    if expected == "clarify" and slot_precision.score < 1.0:
        failure_labels.append("clarify_over_asking")
    if citation.score < 1.0 and expected in {"answer", "answer_with_conditions"}:
        failure_labels.append("citation_unsupported")
    if expected == "verify_then_answer" and not verification_ok:
        failure_labels.append("verification_overclaim")
    if expected != "refuse" and behavior == "refuse":
        failure_labels.append("over_refusal")
    if expected == "refuse" and behavior != "refuse":
        failure_labels.append("under_refusal")

    return {
        "gold_facts_checked": len(gold),
        "gold_fact_accuracy": asdict(gold_outcome),
        "condition_coverage": asdict(condition_outcome),
        "forbidden_claims": forbidden,
        "clarify_slot_recall": asdict(slot_recall),
        "clarify_precision": asdict(slot_precision),
        "citation_support": asdict(citation),
        "unsupported_claim_rate": unsupported,
        "unsupported_claims": unsupported_claims,
        "substantive_claims": substantive_claims,
        "verification_compliant": verification_ok,
        "behavior_pred": behavior,
        "refused": behavior == "refuse",
        "failure_labels": failure_labels,
    }
