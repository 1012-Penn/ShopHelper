# tests/test_history.py
from app.history import estimate_tokens, trim_history


def test_estimate_empty():
    assert estimate_tokens("") == 0


def test_estimate_ascii_ceil_per_4_chars():
    assert estimate_tokens("hello") == 2      # ceil(5/4)
    assert estimate_tokens("abcd") == 1


def test_estimate_cjk_one_each():
    assert estimate_tokens("你好") == 2


def test_estimate_mixed():
    # 2 个 CJK + 5 个 ASCII → 2 + ceil(5/4) = 4
    assert estimate_tokens("你好world") == 4


def test_estimate_fullwidth_counts_as_cjk():
    assert estimate_tokens("！？") == 2


def test_trim_keeps_system_always():
    msgs = [
        {"role": "system", "content": "系统提示"},
        {"role": "user", "content": "很长的老消息" * 10},
    ]
    out = trim_history(msgs, budget=5)
    assert out == [{"role": "system", "content": "系统提示"}]


def test_trim_drops_oldest_keeps_newest():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "aaaa"},   # 1 token
        {"role": "assistant", "content": "bbbb"},  # 1 token
        {"role": "user", "content": "你好世界"},    # 4 token
    ]
    out = trim_history(msgs, budget=5)
    # 预算 5:装得下最后两条(1+4),装不下更老的
    assert out == [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": "bbbb"},
        {"role": "user", "content": "你好世界"},
    ]


def test_trim_everything_fits():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    assert trim_history(msgs, budget=100) == msgs


def test_trim_single_oversized_history_dropped():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "太长" * 100},
    ]
    assert trim_history(msgs, budget=10) == [{"role": "system", "content": "sys"}]


def test_trim_empty_history():
    assert trim_history([{"role": "system", "content": "sys"}], budget=10) == [
        {"role": "system", "content": "sys"}
    ]
