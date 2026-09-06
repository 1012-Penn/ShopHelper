"""Query 改写器:结构化归一 + 同义词;LLM 异常/空归一兜底原句。"""
from app.rewrite import QueryRewriter, bm25_query_text
from tests.helpers import FakeRewriter


class _OkLLM:
    def invoke(self, prompt):
        assert "退了" in prompt  # prompt 里带原 query
        return {"normalized": "怎么申请退货", "synonyms": ["退货流程", "怎么退", "退换货"]}


class _BoomLLM:
    def invoke(self, prompt):
        raise RuntimeError("上游挂了")


class _EmptyLLM:
    def invoke(self, prompt):
        return {"normalized": "  ", "synonyms": []}


def test_rewrite_normalizes_and_expands():
    qr = QueryRewriter(_OkLLM()).rewrite("东西退了")
    assert qr.normalized == "怎么申请退货"
    assert bm25_query_text(qr) == "怎么申请退货 退货流程 怎么退 退换货"


def test_rewrite_boom_falls_back():
    qr = QueryRewriter(_BoomLLM()).rewrite("东西退了")
    assert qr.normalized == "东西退了" and qr.synonyms == []


def test_rewrite_blank_falls_back():
    qr = QueryRewriter(_EmptyLLM()).rewrite("邮费多少")
    assert qr.normalized == "邮费多少"


def test_synonyms_capped_at_three():
    class _Many(_OkLLM):
        def invoke(self, prompt):
            return {"normalized": "n", "synonyms": ["a", "b", "c", "d", "e"]}

    qr = QueryRewriter(_Many()).rewrite("q")
    assert qr.synonyms == ["a", "b", "c"]


def test_fake_rewriter_identity_default():
    assert FakeRewriter().rewrite("随便").normalized == "随便"
