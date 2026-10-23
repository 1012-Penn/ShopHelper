"""严格检索判定纯函数测试。

重点回归扩库后出现的泛词假阳性：只含 `24 小时`/`7 天`/`定制` 但没有
完整证据 quote 的 chunk，不得再被算作相关。
"""
from dataclasses import dataclass

import pytest

from scripts.eval_metrics import (
    EvalDataError,
    chunk_covers,
    coverage_rank,
    evidence_group_recall,
    needs_from_case,
    normalize_span,
    resolve_gold_chunks,
    single_chunk_rank,
)


@dataclass(frozen=True)
class Row:
    id: int
    section_path: str
    questions: str
    answer: str


ROWS = {
    1: Row(1, "商品FAQ.md > 商品FAQ > 售后 > 怎么申请退货?", "怎么申请退货?",
           "在订单详情页点击「申请售后」,选择退货原因后提交,客服会在 24 小时内审核。"),
    2: Row(2, "商品FAQ.md > 商品FAQ > 商品型号 > SH-W20 智能手表有什么功能?",
           "SH-W20 智能手表有什么功能?",
           "SH-W20 智能手表,售价 399 元,支持血氧监测、GPS 定位、NFC 刷卡,续航 7 天,一年质保。"),
    3: Row(3, "商品FAQ.md > 商品FAQ > 售后 > 换货和维修怎么选?", "换货和维修怎么选?",
           "签收后 7 天内可换货;超过 7 天走保修维修。"),
    4: Row(4, "退货政策.md > 退货政策 > 不予退货情形", "不予退货情形",
           "定制类商品、贴身衣物、已激活的虚拟商品不予退货。生鲜食品仅限质量问题退货。"),
    5: Row(5, "退货政策.md > 退货政策 > 邮费与运费", "邮费与运费",
           "普通订单邮费 8 元,单笔满 99 元包邮。质量问题退货,运费由商家承担;"
           "非质量问题退货,运费由买家自行承担。偏远地区额外补收 5 元运费。"),
    6: Row(6, "退货政策.md > 退货政策 > 退款时效", "退款时效",
           "退货商品验收通过后,退款原路返回,1-3 个工作日到账。"),
}


def _case(evidence, **extra):
    return {"id": "T", "answerable": True, "required_evidence": evidence, **extra}


def test_normalize_span_nfkc_and_whitespace():
    assert normalize_span("Ａ  B\n C") == "A B C"
    assert normalize_span("普通订单邮费 8 元,") == "普通订单邮费 8 元,"


def test_chunk_covers_requires_doc_section_and_all_quotes():
    a07 = {"doc": "商品FAQ.md", "section": "怎么申请退货?",
           "quotes": ["客服会在 24 小时内审核。"]}
    assert chunk_covers(ROWS[1], needs_from_case(_case([a07]))[0])
    # 只含 24 小时、但不是审核句：不算命中
    assert not chunk_covers(ROWS[3], needs_from_case(_case([a07]))[0])
    # section 不匹配
    assert not chunk_covers(ROWS[2], needs_from_case(_case([a07]))[0])
    # doc 不匹配
    assert not chunk_covers(ROWS[4], needs_from_case(_case([a07]))[0])


def test_b07_quote_pins_w20_not_generic_7_days():
    b07 = {"doc": "商品FAQ.md", "section": "SH-W20",
           "quotes": ["SH-W20 智能手表,售价 399 元,支持血氧监测、GPS 定位、NFC 刷卡,续航 7 天"]}
    need = needs_from_case(_case([b07]))[0]
    assert chunk_covers(ROWS[2], need)
    assert not chunk_covers(ROWS[3], need)  # 换货条目只含 7 天


def test_a04_quote_pins_exclusion_not_generic_custom():
    a04 = {"doc": "退货政策.md", "section": "不予退货情形",
           "quotes": ["定制类商品、贴身衣物、已激活的虚拟商品不予退货。"]}
    need = needs_from_case(_case([a04]))[0]
    assert chunk_covers(ROWS[4], need)
    assert not chunk_covers(ROWS[5], need)


def test_multi_quote_same_chunk_is_and():
    need = needs_from_case(_case([{
        "doc": "退货政策.md", "section": "邮费与运费",
        "quotes": ["单笔满 99 元包邮", "偏远地区额外补收 5 元运费"],
    }]))[0]
    assert chunk_covers(ROWS[5], need)
    assert not chunk_covers(ROWS[6], need)


def test_resolve_gold_chunks_and_missing_evidence():
    case = _case([
        {"doc": "商品FAQ.md", "section": "怎么申请退货?", "quotes": ["客服会在 24 小时内审核。"]},
    ])
    assert resolve_gold_chunks(case, ROWS) == [frozenset({1})]
    bad = _case([{"doc": "商品FAQ.md", "section": "怎么申请退货?", "quotes": ["不存在的句子"]}])
    with pytest.raises(EvalDataError):
        resolve_gold_chunks(bad, ROWS)


def test_coverage_rank_takes_max_group_rank_and_respects_cap():
    case = _case([
        {"doc": "商品FAQ.md", "section": "怎么申请退货?", "quotes": ["客服会在 24 小时内审核。"]},
        {"doc": "退货政策.md", "section": "退款时效", "quotes": ["1-3 个工作日到账"]},
    ])
    gold = resolve_gold_chunks(case, ROWS)
    assert coverage_rank([5, 1, 6], gold) == 3
    assert coverage_rank([5, 1, 6], gold, cap=2) is None
    assert coverage_rank([5, 1], gold) is None


def test_single_chunk_rank_requires_one_chunk_for_all_evidence():
    split = _case([
        {"doc": "退货政策.md", "section": "邮费与运费", "quotes": ["质量问题退货,运费由商家承担"]},
        {"doc": "退货政策.md", "section": "退款时效", "quotes": ["1-3 个工作日到账"]},
    ])
    assert single_chunk_rank([5, 6], split, ROWS) is None

    same = _case([{
        "doc": "退货政策.md", "section": "邮费与运费",
        "quotes": ["单笔满 99 元包邮", "偏远地区额外补收 5 元运费"],
    }])
    assert single_chunk_rank([5, 6], same, ROWS) == 1


def test_evidence_group_recall():
    case = _case([
        {"doc": "商品FAQ.md", "section": "怎么申请退货?", "quotes": ["客服会在 24 小时内审核。"]},
        {"doc": "退货政策.md", "section": "退款时效", "quotes": ["1-3 个工作日到账"]},
    ])
    gold = resolve_gold_chunks(case, ROWS)
    assert evidence_group_recall([1, 5], gold) == 0.5
    assert evidence_group_recall([1, 6], gold) == 1.0
