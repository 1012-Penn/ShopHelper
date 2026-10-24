"""ch11 结构化事实 schema:加载、稳定哈希、引用完整性与冲突校验。"""
from pathlib import Path

import pytest

from scripts.corpus_facts import (
    CorpusDataError,
    EvidenceUnit,
    Fact,
    FactBook,
    Source,
    load_factbook,
    validate_factbook,
)

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "knowledge"

RETURN_QUOTE = "商品自签收之日起 7 天内,支持七天无理由退货。"
SHIPPING_QUOTE = "普通订单邮费 8 元,单笔满 99 元包邮。"


def _source(source_id: str = "SRC-TEST") -> Source:
    return Source(
        source_id=source_id,
        type="internal_rule",
        title="测试来源",
        license="内部演示资料",
    )


def _fact(
    fact_id: str,
    *,
    entity_type: str = "policy",
    entity_id: str = "returns",
    key: str = "return_window_days",
    value=7,
    source_ref: str = "SRC-TEST",
    evidence_ids: list[str] | None = None,
    **overrides,
) -> Fact:
    return Fact(
        fact_id=fact_id,
        entity_type=entity_type,
        entity_id=entity_id,
        key=key,
        value=value,
        source_ref=source_ref,
        evidence_ids=evidence_ids or [f"EV-{fact_id}"],
        **overrides,
    )


def _evidence(
    evidence_id: str,
    fact_ids: list[str],
    *,
    doc: str = "退货政策.md",
    section: str = "七天无理由退货",
    quotes: tuple[str, ...] = (RETURN_QUOTE,),
) -> EvidenceUnit:
    return EvidenceUnit(
        evidence_id=evidence_id,
        fact_ids=fact_ids,
        doc=doc,
        section=section,
        quotes=list(quotes),
    )


def _book(
    facts: tuple[Fact, ...] = (),
    evidence: tuple[EvidenceUnit, ...] = (),
    sources: tuple[Source, ...] | None = None,
) -> FactBook:
    sources = sources if sources is not None else (_source(),)
    return FactBook(
        facts={fact.fact_id: fact for fact in facts},
        evidence={unit.evidence_id: unit for unit in evidence},
        sources={source.source_id: source for source in sources},
    )


def _write_corpus(tmp_path: Path, *, sources: str, facts: dict[str, str]) -> Path:
    (tmp_path / "facts").mkdir()
    (tmp_path / "sources.yaml").write_text(sources, encoding="utf-8")
    for name, text in facts.items():
        (tmp_path / "facts" / name).write_text(text, encoding="utf-8")
    return tmp_path


SOURCES_YAML = """
sources:
  - source_id: SRC-TEST
    type: internal_rule
    title: 测试来源
    license: 内部演示资料
evidence_units: []
"""

FACT_YAML = """
facts:
  - fact_id: POLICY-RETURN-WINDOW
    entity_type: policy
    entity_id: returns
    key: return_window_days
    value: 7
    source_ref: SRC-TEST
    evidence_ids: [EV-POLICY-RETURN-WINDOW]
"""


def test_loads_factbook_and_sha_is_stable():
    book = load_factbook()

    assert book.count() >= 10
    assert book.sha256() == book.sha256()
    assert book.fact("POLICY-RETURN-WINDOW").value == 7
    assert book.evidence_for_fact("POLICY-RETURN-WINDOW")
    assert [fact.fact_id for fact in book.facts_for_entity("sku", "SH-E300")] == [
        "CATALOG-SH-E300-PRICE"
    ]
    assert {fact.entity_type for fact in book.facts.values()} >= {
        "sku",
        "order",
        "policy",
        "promotion",
        "after_sales",
    }

    # 哈希只取决于内容,与加载/插入顺序无关
    reordered = FactBook(
        facts=dict(reversed(list(book.facts.items()))),
        evidence=dict(reversed(list(book.evidence.items()))),
        sources=dict(reversed(list(book.sources.items()))),
    )
    assert reordered.sha256() == book.sha256()

    assert validate_factbook(book) == []


def test_load_rejects_malformed_yaml(tmp_path):
    root = _write_corpus(
        tmp_path,
        sources=SOURCES_YAML,
        facts={"broken.yaml": "facts: [\n  - fact_id: ["},
    )

    with pytest.raises(CorpusDataError, match="YAML"):
        load_factbook(root)


def test_non_utf8_file_rejected(tmp_path):
    root = _write_corpus(tmp_path, sources=SOURCES_YAML, facts={"bad.yaml": FACT_YAML})
    (root / "facts" / "bad.yaml").write_bytes(b"facts:\n  - fact_id: \xff\xfe\n")

    with pytest.raises(CorpusDataError, match="读取"):
        load_factbook(root)


def test_duplicate_fact_id_rejected(tmp_path):
    root = _write_corpus(
        tmp_path,
        sources=SOURCES_YAML,
        facts={"a.yaml": FACT_YAML, "b.yaml": FACT_YAML},
    )

    with pytest.raises(CorpusDataError, match="重复"):
        load_factbook(root)


def test_duplicate_evidence_id_rejected(tmp_path):
    sources = """
sources:
  - source_id: SRC-TEST
    type: internal_rule
    title: 测试来源
    license: 内部演示资料
evidence_units:
  - evidence_id: EV-DUP
    fact_ids: [POLICY-RETURN-WINDOW]
    doc: 退货政策.md
    section: 七天无理由退货
    quotes: [商品自签收之日起 7 天内,支持七天无理由退货。]
  - evidence_id: EV-DUP
    fact_ids: [POLICY-REFUND-WINDOW]
    doc: 退货政策.md
    section: 退款时效
    quotes: [退货商品验收通过后,退款原路返回,1-3 个工作日到账。]
"""
    facts = {
        "a.yaml": FACT_YAML,
        "c.yaml": FACT_YAML.replace("POLICY-RETURN-WINDOW", "POLICY-REFUND-WINDOW"),
    }
    root = _write_corpus(tmp_path, sources=sources, facts=facts)

    with pytest.raises(CorpusDataError, match="重复"):
        load_factbook(root)


def test_unregistered_evidence_doc_rejected():
    fact = _fact("POLICY-RETURN-WINDOW")
    unit = _evidence(
        "EV-POLICY-RETURN-WINDOW",
        [fact.fact_id],
        doc="未登记文档.md",
        section="七天无理由退货",
        quotes=(RETURN_QUOTE,),
    )

    issues = validate_factbook(_book((fact,), (unit,)))

    assert any("未登记" in issue for issue in issues)


def test_missing_source_ref_rejected():
    fact = _fact("POLICY-RETURN-WINDOW", source_ref="SRC-MISSING")
    unit = _evidence("EV-POLICY-RETURN-WINDOW", [fact.fact_id])

    issues = validate_factbook(_book((fact,), (unit,)))

    assert any("source_ref" in issue for issue in issues)


def test_evidence_quote_must_exist_verbatim():
    fact = _fact("POLICY-RETURN-WINDOW")
    unit = _evidence(
        "EV-POLICY-RETURN-WINDOW",
        [fact.fact_id],
        quotes=("商品自签收之日起 30 天内,支持七天无理由退货。",),
    )

    issues = validate_factbook(_book((fact,), (unit,)))

    assert any("逐字" in issue for issue in issues)


def test_evidence_quote_must_land_in_one_chunk():
    fact = _fact("POLICY-RETURN-WINDOW")
    unit = _evidence(
        "EV-POLICY-RETURN-WINDOW",
        [fact.fact_id],
        section="七天无理由退货",
        # 两句分别落在「七天无理由退货」与「邮费与运费」两个 chunk
        quotes=(RETURN_QUOTE, SHIPPING_QUOTE),
    )

    issues = validate_factbook(_book((fact,), (unit,)))

    assert any("chunk" in issue for issue in issues)


def test_evidence_section_must_be_in_covering_chunk():
    fact = _fact("POLICY-RETURN-WINDOW")
    unit = _evidence(
        "EV-POLICY-RETURN-WINDOW",
        [fact.fact_id],
        section="邮费与运费",
    )

    issues = validate_factbook(_book((fact,), (unit,)))

    assert any("section_path" in issue for issue in issues)


def test_missing_evidence_id_rejected():
    fact = _fact("POLICY-RETURN-WINDOW", evidence_ids=["EV-NOT-EXIST"])
    unit = _evidence("EV-POLICY-RETURN-WINDOW", [fact.fact_id])

    issues = validate_factbook(_book((fact,), (unit,)))

    assert any("EV-NOT-EXIST" in issue for issue in issues)


def test_effective_range_must_be_ordered():
    fact = _fact(
        "POLICY-RETURN-WINDOW",
        effective_from="2026-01-01",
        effective_to="2025-01-01",
    )
    unit = _evidence("EV-POLICY-RETURN-WINDOW", [fact.fact_id])

    issues = validate_factbook(_book((fact,), (unit,)))

    assert any("effective_to" in issue for issue in issues)


def test_supersedes_cycle_rejected():
    fact_a = _fact("POLICY-RETURN-WINDOW", key="a", supersedes=["POLICY-REFUND-WINDOW"])
    fact_b = _fact("POLICY-REFUND-WINDOW", key="b", supersedes=["POLICY-RETURN-WINDOW"])
    evidence = (
        _evidence("EV-POLICY-RETURN-WINDOW", [fact_a.fact_id]),
        _evidence("EV-POLICY-REFUND-WINDOW", [fact_b.fact_id]),
    )

    issues = validate_factbook(_book((fact_a, fact_b), evidence))

    assert any("环" in issue for issue in issues)


def test_supersedes_missing_target_rejected():
    fact = _fact("POLICY-RETURN-WINDOW", supersedes=["POLICY-NOT-EXIST"])
    unit = _evidence("EV-POLICY-RETURN-WINDOW", [fact.fact_id])

    issues = validate_factbook(_book((fact,), (unit,)))

    assert any("POLICY-NOT-EXIST" in issue for issue in issues)


def test_overlapping_conflicting_values_rejected():
    fact_old = _fact("POLICY-RETURN-WINDOW", value=7)
    fact_new = _fact("POLICY-RETURN-WINDOW-NEW", value=30)
    evidence = (
        _evidence("EV-POLICY-RETURN-WINDOW", [fact_old.fact_id]),
        _evidence("EV-POLICY-RETURN-WINDOW-NEW", [fact_new.fact_id]),
    )

    issues = validate_factbook(_book((fact_old, fact_new), evidence))

    assert any("重叠" in issue for issue in issues)


def test_disjoint_one_sided_ranges_do_not_conflict():
    fact_old = _fact("POLICY-RETURN-WINDOW", value=7, effective_to="2024-12-31")
    fact_new = _fact(
        "POLICY-RETURN-WINDOW-NEW", value=30, effective_from="2025-01-01"
    )
    evidence = (
        _evidence("EV-POLICY-RETURN-WINDOW", [fact_old.fact_id]),
        _evidence("EV-POLICY-RETURN-WINDOW-NEW", [fact_new.fact_id]),
    )

    assert validate_factbook(_book((fact_old, fact_new), evidence)) == []


def test_same_conflict_group_allows_overlap():
    fact_old = _fact(
        "POLICY-RETURN-WINDOW", value=7, conflict_group="CG-RETURN-WINDOW"
    )
    fact_new = _fact(
        "POLICY-RETURN-WINDOW-NEW", value=30, conflict_group="CG-RETURN-WINDOW"
    )
    evidence = (
        _evidence("EV-POLICY-RETURN-WINDOW", [fact_old.fact_id]),
        _evidence("EV-POLICY-RETURN-WINDOW-NEW", [fact_new.fact_id]),
    )

    assert validate_factbook(_book((fact_old, fact_new), evidence)) == []


@pytest.mark.parametrize(
    ("left_group", "right_group"),
    [("CG-A", "CG-B"), ("", "CG-A"), ("", "")],
)
def test_different_or_empty_conflict_group_rejected(left_group, right_group):
    fact_old = _fact("POLICY-RETURN-WINDOW", value=7, conflict_group=left_group)
    fact_new = _fact(
        "POLICY-RETURN-WINDOW-NEW", value=30, conflict_group=right_group
    )
    evidence = (
        _evidence("EV-POLICY-RETURN-WINDOW", [fact_old.fact_id]),
        _evidence("EV-POLICY-RETURN-WINDOW-NEW", [fact_new.fact_id]),
    )

    issues = validate_factbook(_book((fact_old, fact_new), evidence))

    assert any("重叠" in issue for issue in issues)


def test_fact_declared_evidence_must_reciprocate():
    fact = _fact("POLICY-RETURN-WINDOW")
    other = _fact("POLICY-OTHER", key="other").model_copy(
        update={"evidence_ids": ["EV-POLICY-RETURN-WINDOW"]}
    )
    # EV-POLICY-RETURN-WINDOW 只声明了 other,而 fact 却引用了它
    unit = _evidence("EV-POLICY-RETURN-WINDOW", [other.fact_id])

    issues = validate_factbook(_book((fact, other), (unit,)))

    assert any("互惠" in issue and "POLICY-RETURN-WINDOW" in issue for issue in issues)


def test_evidence_declared_fact_must_reciprocate():
    fact = _fact("POLICY-OTHER", key="other")
    own_evidence = _evidence("EV-POLICY-OTHER", [fact.fact_id])
    extra_evidence = _evidence("EV-EXTRA", [fact.fact_id])

    issues = validate_factbook(_book((fact,), (own_evidence, extra_evidence)))

    assert any("互惠" in issue and "EV-EXTRA" in issue for issue in issues)


def test_identical_values_may_overlap():
    fact_a = _fact("POLICY-RETURN-WINDOW", value=7)
    fact_b = _fact("POLICY-RETURN-WINDOW-NEW", value=7)
    evidence = (
        _evidence("EV-POLICY-RETURN-WINDOW", [fact_a.fact_id]),
        _evidence("EV-POLICY-RETURN-WINDOW-NEW", [fact_b.fact_id]),
    )

    assert validate_factbook(_book((fact_a, fact_b), evidence)) == []
