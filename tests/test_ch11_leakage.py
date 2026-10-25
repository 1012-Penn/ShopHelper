"""泄漏扫描:题目/ID/禁用断言/评测标记不得进入知识文档。"""
from __future__ import annotations

from scripts.leakage_scan import LeakFinding, scan_all, scan_cases, scan_split


def _case(**overrides) -> dict:
    case = {
        "id": "LEAK-01",
        "query": "完全无关的测试问法",
        "gold_facts": [],
        "required_evidence": [],
        "forbidden_claims": [],
    }
    case.update(overrides)
    return case


def test_exact_query_in_doc_is_blocking():
    findings = scan_cases(
        [_case(query="SH-E300 多少钱")],
        {"商品目录.md": "SH-E300 多少钱？我们售价 299 元。"},
        split="dev",
    )
    assert any(item.kind == "exact_query" and item.severity == "blocking" for item in findings)


def test_case_id_and_fact_id_in_doc_are_blocking():
    findings = scan_cases(
        [_case(gold_facts=[{"fact_id": "CATALOG-SH-E300-PRICE", "value": 299}])],
        {"商品目录.md": "LEAK-01 与 CATALOG-SH-E300-PRICE 都是内部编号"},
        split="dev",
    )
    assert any(item.kind == "case_id" and item.severity == "blocking" for item in findings)
    assert any(item.kind == "fact_id" and item.severity == "blocking" for item in findings)


def test_forbidden_claim_in_doc_is_blocking():
    findings = scan_cases(
        [_case(forbidden_claims=["所有商品 95 折"])],
        {"会员与优惠.md": "所有商品 95 折是错误说法。"},
        split="dev",
    )
    assert any(item.kind == "forbidden_claim" for item in findings)


def test_paraphrase_is_not_blocking():
    findings = scan_cases(
        [_case(query="请问这个耳机卖多少钱呢")],
        {"商品目录.md": "SH-E300 头戴式无线降噪耳机,售价 299 元,支持主动降噪。"},
        split="dev",
    )
    assert not any(item.severity == "blocking" for item in findings)


def test_fact_evidence_quotes_are_not_findings():
    # evidence quote 本来就应该在文档中,扫描器只看题目/断言/ID,不看证据 quote。
    findings = scan_cases(
        [_case(
            query="这个耳机多少钱",
            required_evidence=[{
                "doc": "商品目录.md",
                "section": "SH-E300",
                "quotes": ["SH-E300 头戴式无线降噪耳机,售价 299 元,支持主动降噪。"],
                "fact_ids": [],
            }],
        )],
        {"商品目录.md": "SH-E300 头戴式无线降噪耳机,售价 299 元,支持主动降噪。"},
        split="dev",
    )
    assert not any(item.kind in {"exact_query", "fact_id"} and item.severity == "blocking" for item in findings)


def test_real_dev_split_has_no_blocking_findings():
    findings = scan_split("dev")
    assert not any(item.severity == "blocking" for item in findings)


def test_real_splits_have_no_blocking_findings():
    findings = scan_all()
    assert not [item for item in findings if item.severity == "blocking"]
