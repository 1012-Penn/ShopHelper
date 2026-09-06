"""忠实度评估:证据快照组装 + 裁判落台账(替身全链,SQLite),复发累加。"""
import pytest

from app.chunking import Chunk
from app.kb import FaithCaseLedger, KnowledgeBaseStore
from app.models import FaithCase
from app.prompts import FaithfulnessVerdict
from app.retrieval import RetrievalResult, RetrievalService, Retrieved
from scripts.eval_faithfulness import make_evidence, run_case
from tests.helpers import FakeEmbedding, FakeReranker, FakeRewriter, FakeVectorStore


class SimpleRow:
    questions = "q"
    answer = "a"
    section_path = "d > s"
    category = "c"


class _JudgeFab:
    def invoke(self, prompt):
        return FaithfulnessVerdict(fabricated=True, reason="到账时间无证据", fabricated_claims=["3 天到账"])


class _JudgeOk:
    def invoke(self, prompt):
        return FaithfulnessVerdict(fabricated=False, reason="", fabricated_claims=[])


class _Chat:
    def invoke(self, prompt):
        return type("Msg", (), {"content": "退款一般 3 天到账 [1]"})()


@pytest.fixture
def env(db_session_factory):
    kb = KnowledgeBaseStore(db_session_factory)
    kb.replace_doc_chunks("d.md", [Chunk(category="售后政策", questions="退款说明",
                                         answer="退款原路返回,3-5 个工作日。",
                                         section_path="d.md > 退款说明",
                                         content_type="policy", is_key_clause=False)])
    vectors = FakeVectorStore()
    from app.kb import vectorize_pending

    vectorize_pending(kb, vectors, FakeEmbedding())
    service = RetrievalService(FakeEmbedding(), vectors, kb,
                               rewriter=FakeRewriter(), reranker=FakeReranker())
    return kb, service, db_session_factory


def test_make_evidence_arrangement():
    result = RetrievalResult(items=[Retrieved(11, 0.5), Retrieved(12, 0.4), Retrieved(13, 0.3)])
    rows = [SimpleRow(), SimpleRow(), SimpleRow()]  # 与 items 平行(get_chunks 返回序)
    evidence = make_evidence(result, rows)
    assert [e["n"] for e in evidence] == [1, 3, 2]  # 反漏斗摆放,n 不变
    assert evidence[0]["chunk_id"] == 11            # n=1 恒为精排第一名


def test_run_case_fabricated_writes_ledger(env):
    kb, service, factory = env
    case = {"id": "A01", "bucket": "A_policy", "query": "退款多久到账"}
    fab, reason = run_case(case, service, kb, _Chat(), _JudgeFab(), FaithCaseLedger(factory),
                           judge_model="stub")
    assert fab and reason
    with factory() as s:
        row = s.query(FaithCase).one()
    assert row.eval_id == "A01" and row.seen_count == 1 and row.judge_model == "stub"
    assert row.citations and row.citations[0]["n"] == 1  # citations 存证据全集快照
    # 复跑一轮:同题累加,不新增行
    run_case(case, service, kb, _Chat(), _JudgeFab(), FaithCaseLedger(factory), judge_model="stub")
    with factory() as s:
        assert s.query(FaithCase).one().seen_count == 2


def test_run_case_ok_no_ledger(env):
    kb, service, factory = env
    case = {"id": "A02", "bucket": "A_policy", "query": "退款多久到账"}
    fab, _ = run_case(case, service, kb, _Chat(), _JudgeOk(), FaithCaseLedger(factory),
                      judge_model="stub")
    assert not fab
    with factory() as s:
        assert s.query(FaithCase).count() == 0
