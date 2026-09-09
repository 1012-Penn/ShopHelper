"""ch09 回捞:当轮召回片段快照的会话级暂存,供 👎 反馈事后取用。"""
from app.round_capture import RoundCapture


def test_lookup_returns_snapshot_of_matching_round():
    capture = RoundCapture()
    snapshot = [{"rank": 1, "chunk_id": 7, "score": 0.88, "question": "邮费", "answer": "8 元"}]
    capture.record(9, "  国际件运费怎么算 ", snapshot)

    assert capture.lookup(9, "国际件运费怎么算") == snapshot


def test_lookup_empty_snapshot_marks_round_without_retrieval():
    capture = RoundCapture()
    capture.record(9, "你好呀", [])

    assert capture.lookup(9, "你好呀") == []


def test_lookup_miss_returns_none():
    capture = RoundCapture()
    capture.record(9, "国际件运费怎么算", [{"rank": 1, "chunk_id": 7, "score": 0.5}])

    assert capture.lookup(9, "没问过的问题") is None
    assert capture.lookup(404, "国际件运费怎么算") is None


def test_session_keeps_most_recent_rounds_and_evicts_oldest():
    capture = RoundCapture(rounds_per_session=2)
    capture.record(9, "第一问", [{"rank": 1}])
    capture.record(9, "第二问", [])
    capture.record(9, "第三问", [{"rank": 2}])

    assert capture.lookup(9, "第一问") is None
    assert capture.lookup(9, "第二问") == []
    assert capture.lookup(9, "第三问") == [{"rank": 2}]


def test_capacity_evicts_least_recently_touched_session():
    capture = RoundCapture(max_sessions=2)
    capture.record(1, "q1", [{"rank": 1}])
    capture.record(2, "q2", [{"rank": 2}])
    capture.lookup(1, "q1")  # 触碰会话 1,应比会话 2 更晚被逐出
    capture.record(3, "q3", [{"rank": 3}])

    assert capture.lookup(2, "q2") is None
    assert capture.lookup(1, "q1") == [{"rank": 1}]
    assert capture.lookup(3, "q3") == [{"rank": 3}]
