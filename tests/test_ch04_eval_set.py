"""ch04 评估集数据验证(工作要求 1:数据类任务以标注样例验证替代 TDD)。

四桶各 8 题;expect_doc 真实存在;每个 expect_keyword 都能在期望文档原文里找到
(ground truth 必须真实可命中,防「评估集飘在知识库外」);切分后必有 chunk 能命中关键词。
"""
import json
from pathlib import Path

from app.chunking import chunk_markdown

ROOT = Path(__file__).resolve().parent.parent
CASES = [json.loads(ln) for ln in
         (ROOT / "tests" / "eval" / "ch04_eval_set.jsonl").read_text(encoding="utf-8").splitlines()
         if ln.strip()]
BUCKETS = {"A_policy", "B_model", "C_colloquial", "E_multi"}


def test_32_cases_four_buckets():
    assert len(CASES) == 32
    assert {c["bucket"] for c in CASES} == BUCKETS
    for b in BUCKETS:
        assert sum(1 for c in CASES if c["bucket"] == b) == 8
    assert len({c["id"] for c in CASES}) == 32


def test_expect_docs_exist():
    for c in CASES:
        docs = c["expect_doc"] if isinstance(c["expect_doc"], list) else [c["expect_doc"]]
        for d in docs:
            assert (ROOT / "knowledge" / d).exists(), f"{c['id']}: {d} 不存在"


def test_keywords_exist_in_expected_docs():
    for c in CASES:
        docs = c["expect_doc"] if isinstance(c["expect_doc"], list) else [c["expect_doc"]]
        corpus = "".join((ROOT / "knowledge" / d).read_text(encoding="utf-8") for d in docs)
        for kw in c["expect_keywords"]:
            assert kw in corpus, f"{c['id']}: 关键词「{kw}」不在期望文档原文里"


def test_ground_truth_reachable_by_chunker():
    """切分后存在这样的 chunk:路径含期望文档且命中任一关键词——保证评估集可被检索命中。"""
    for c in CASES:
        docs = c["expect_doc"] if isinstance(c["expect_doc"], list) else [c["expect_doc"]]
        found = False
        for d in docs:
            text = (ROOT / "knowledge" / d).read_text(encoding="utf-8")
            for chunk in chunk_markdown(text, d, "faq"):
                blob = f"{chunk.questions}\n{chunk.answer}"
                if any(kw in blob for kw in c["expect_keywords"]):
                    found = True
        assert found, f"{c['id']}: 没有任何 chunk 能命中关键词"
