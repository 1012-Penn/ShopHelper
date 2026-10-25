"""ch04 评估集数据验证：47 题、五桶、逐字证据、可映射 chunk。

判定真值只看 required_evidence（AND 语义）；不再有关键词 ANY 分支。
"""
import json
import unicodedata
from collections import Counter
from pathlib import Path

from app.chunking import chunk_markdown
from scripts.corpus_manifest import doc_types

ROOT = Path(__file__).resolve().parent.parent
CASES = [json.loads(ln) for ln in
         (ROOT / "tests" / "eval" / "ch04_eval_set.jsonl").read_text(encoding="utf-8").splitlines()
         if ln.strip()]
BUCKETS = {"A_policy": 8, "B_model": 8, "C_colloquial": 8, "E_multi": 8, "F_service": 15}
# 文档名 → content_type 的唯一真源是 corpus/manifest.yaml
DOC_TYPES = doc_types()


def _norm(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def _doc_text(doc: str) -> str:
    return _norm((ROOT / "knowledge" / doc).read_text(encoding="utf-8"))


def test_47_cases_five_buckets():
    assert len(CASES) == 47
    assert Counter(c["bucket"] for c in CASES) == BUCKETS
    assert len({c["id"] for c in CASES}) == 47
    assert len({c["query"] for c in CASES}) == 47


def test_all_answerable_and_have_evidence():
    for c in CASES:
        assert c["answerable"] is True, c["id"]
        assert c["required_evidence"], c["id"]
        assert "expect_keywords" not in c, c["id"]
        docs = sorted({e["doc"] for e in c["required_evidence"]})
        assert docs == sorted(c["expect_doc"]), c["id"]


def test_evidence_docs_and_quotes_exist_verbatim():
    for c in CASES:
        for e in c["required_evidence"]:
            doc = e["doc"]
            assert doc in DOC_TYPES, f"{c['id']}: {doc} 未登记"
            text = _doc_text(doc)
            assert e["section"] in text, f"{c['id']}: section「{e['section']}」不在 {doc}"
            for quote in e["quotes"]:
                assert _norm(quote) in text, f"{c['id']}: quote「{quote}」不在 {doc}"


def test_ground_truth_reachable_by_chunker():
    """每组 evidence 必须能被某个真实 chunk 同时覆盖。"""
    for c in CASES:
        for e in c["required_evidence"]:
            doc = e["doc"]
            chunks = chunk_markdown(
                (ROOT / "knowledge" / doc).read_text(encoding="utf-8"),
                doc, DOC_TYPES[doc],
            )
            matched = [
                ch for ch in chunks
                if e["section"] in ch.section_path
                and all(_norm(q) in _norm(ch.questions + "\n" + ch.answer) for q in e["quotes"])
            ]
            assert matched, f"{c['id']}: 没有 chunk 能覆盖 {doc} / {e['section']} / {e['quotes']}"
