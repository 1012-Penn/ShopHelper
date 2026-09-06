"""检索评估指标:命中判定与 Recall@K / MRR 纯函数(不发网络、不碰库)。"""
from types import SimpleNamespace

from scripts.eval_retrieval import chunk_is_relevant, recall_mrr

ROW = SimpleNamespace(section_path="退货政策.md > 邮费与运费",
                      questions="邮费与运费", answer="普通订单邮费 8 元,满 99 元包邮。")
ROW_OTHER = SimpleNamespace(section_path="商品FAQ.md > 物流", questions="多久能发货", answer="48 小时内。")
CASE = {"expect_doc": "退货政策.md", "expect_keywords": ["邮费"]}


def test_chunk_is_relevant():
    assert chunk_is_relevant(ROW, CASE)
    assert not chunk_is_relevant(ROW_OTHER, CASE)


def test_chunk_is_relevant_section_pin():
    case = {**CASE, "expect_section": "邮费与运费"}
    assert chunk_is_relevant(ROW, case)
    row_other_section = SimpleNamespace(section_path="退货政策.md > 退款说明",
                                        questions="退款", answer="邮费说明")
    assert not chunk_is_relevant(row_other_section, case)


def test_recall_and_mrr():
    # rank 4(pos=3)才命中:Recall@3 未中、@5/@10 中,MRR = 1/4
    rows_by_id = {1: ROW_OTHER, 2: ROW_OTHER, 3: ROW_OTHER, 4: ROW}
    hits, rr = recall_mrr([(1, 0.9), (2, 0.8), (3, 0.7), (4, 0.6)], rows_by_id, CASE)
    assert hits == {3: False, 5: True, 10: True}
    assert rr == 1.0 / 4


def test_recall_hit_at_rank3_counts_for_k3():
    # rank 3(pos=2)∈ top3:Recall@3 应命中
    rows_by_id = {1: ROW_OTHER, 2: ROW_OTHER, 3: ROW}
    hits, rr = recall_mrr([(1, 0.9), (2, 0.8), (3, 0.7)], rows_by_id, CASE)
    assert hits == {3: True, 5: True, 10: True}
    assert rr == 1.0 / 3


def test_recall_miss_all():
    rows_by_id = {1: ROW_OTHER}
    hits, rr = recall_mrr([(1, 0.9)], rows_by_id, CASE)
    assert hits == {3: False, 5: False, 10: False} and rr == 0.0
