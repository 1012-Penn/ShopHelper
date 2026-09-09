from app.evaluation import EvalRunStore
from app.models import EvalRun


def test_eval_run_store_records_and_returns_time_series(db_session_factory):
    store = EvalRunStore(db_session_factory)
    first = store.record("手动", 32, {"recall_at_k": 0.82, "mrr": 0.71, "faithfulness": 0.90})
    second = store.record("定时", 32, {"recall_at_k": 0.78, "mrr": 0.65, "faithfulness": 0.88})
    assert second > first
    rows = store.list_runs()
    assert [row["id"] for row in rows] == [first, second]
    assert rows[-1]["metrics"]["mrr"] < rows[0]["metrics"]["mrr"]

    with db_session_factory() as session:
        assert session.get(EvalRun, second).triggered_by == "定时"
