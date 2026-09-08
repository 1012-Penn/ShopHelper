# tests/test_history.py
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.history import estimate_tokens, rows_to_messages


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


def _rows():
    return [
        {"id": 7, "role": "user", "content": "旧问", "tool_calls": None, "tool_call_id": None},
        {"id": 8, "role": "assistant", "content": None,
         "tool_calls": [{"name": "query_order", "args": {"order_id": "1"}, "id": "c1"}],
         "tool_call_id": None},
        {"id": 9, "role": "tool", "content": '{"order_id": "1"}',
         "tool_calls": None, "tool_call_id": "c1"},
        {"id": 10, "role": "assistant", "content": "旧答", "tool_calls": None, "tool_call_id": None},
    ]


def test_rows_to_messages_restores_tool_trace_with_ids():
    msgs = rows_to_messages(_rows())
    assert [m.id for m in msgs] == ["7", "8", "9", "10"]
    assert isinstance(msgs[0], HumanMessage)
    assert isinstance(msgs[1], AIMessage) and msgs[1].tool_calls[0]["id"] == "c1"
    assert isinstance(msgs[2], ToolMessage) and msgs[2].tool_call_id == "c1"
    assert isinstance(msgs[3], AIMessage)


def test_rows_to_messages_skips_empty_text_assistant_without_tool_calls():
    rows = [{"id": 1, "role": "user", "content": "u"},
            {"id": 2, "role": "assistant", "content": None, "tool_calls": None}]
    msgs = rows_to_messages(rows)
    assert [m.id for m in msgs] == ["1"]
