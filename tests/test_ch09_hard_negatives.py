"""ch09 负样本集：8 条 OOD + 5 条同域硬负例。

硬负例必须满足：
- answerable=false，且没有 required_evidence；
- topical_docs 存在，tempting_sections 出现在这些文档里；
- absent_terms 不出现在任何知识文档中，防止未来补文档后悄悄变成可答。
"""
import json
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROWS = [json.loads(ln) for ln in
        (ROOT / "tests" / "eval" / "ch09_junk_set.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()]
DOCS = {p.name: p.read_text(encoding="utf-8")
        for p in (ROOT / "knowledge").glob("*.md")}


def _norm(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def test_13_negative_cases_and_labels():
    assert len(ROWS) == 13
    assert Counter(r["bucket"] for r in ROWS) == {"junk": 7, "H_hard_negative": 6}
    assert len({r["id"] for r in ROWS}) == 13
    assert all(r["answerable"] is False for r in ROWS)
    assert all("required_evidence" not in r for r in ROWS)


def test_hard_negatives_are_in_domain_and_absent():
    hard = [r for r in ROWS if r["bucket"] == "H_hard_negative"]
    assert len(hard) == 6
    for r in hard:
        assert r["absent_terms"], r["id"]
        assert r["unanswerable_because"].strip(), r["id"]
        for doc in r.get("topical_docs", []):
            assert doc in DOCS, f"{r['id']}: {doc} 不存在"
        for section in r.get("tempting_sections", []):
            assert any(section in DOCS[doc] for doc in r.get("topical_docs", [])), \
                f"{r['id']}: tempting section「{section}」不在 topical_docs"
        for term in r["absent_terms"]:
            for doc, text in DOCS.items():
                assert _norm(term) not in _norm(text), \
                    f"{r['id']}: absent term「{term}」出现在 {doc}"
