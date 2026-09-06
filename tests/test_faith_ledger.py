"""faith_cases 台账:一题一行 / seen_count 跨轮累加 / 复发退回未解决。"""
from app.kb import FaithCaseLedger
from app.models import FaithCase


def test_first_insert_then_accumulate(db_session_factory):
    led = FaithCaseLedger(db_session_factory)
    led.upsert_case(eval_id="A01", bucket="A_policy", query="q1", strategy="hybrid_rerank",
                    answer="a1", reason="编了一句", citations=[{"n": 1}], judge_model="m1")
    led.upsert_case(eval_id="A01", bucket="A_policy", query="q1", strategy="hybrid_rerank",
                    answer="a2", reason="又编了", citations=[{"n": 2}], judge_model="m2")
    with db_session_factory() as s:
        rows = s.query(FaithCase).all()
    assert len(rows) == 1
    row = rows[0]
    assert row.seen_count == 2 and row.answer == "a2" and row.reason == "又编了"
    assert row.citations == [{"n": 2}] and row.judge_model == "m2"
    assert row.status == "未解决"


def test_resolved_case_relapses(db_session_factory):
    led = FaithCaseLedger(db_session_factory)
    led.upsert_case(eval_id="B03", bucket="B_model", query="q", strategy="hybrid_rerank",
                    answer="a", reason="r", citations=None, judge_model="m")
    with db_session_factory() as s:
        row = s.query(FaithCase).one()
        row.status = "已解决"
        row.resolution = "补了知识"
        row.resolved_at = row.last_seen_at
        s.commit()
    led.upsert_case(eval_id="B03", bucket="B_model", query="q", strategy="hybrid_rerank",
                    answer="a2", reason="复发", citations=None, judge_model="m")
    with db_session_factory() as s:
        row = s.query(FaithCase).one()
    assert row.status == "未解决"
    assert row.resolution is None       # 复发清空处置交代
    assert row.resolved_at is not None  # 保留作复发标记
    assert row.seen_count == 2 and row.reason == "复发"
