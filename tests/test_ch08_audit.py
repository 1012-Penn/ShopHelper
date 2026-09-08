"""ch08 工具调用审计:落行全字段、截断、被拒/被拦同样落。"""
from app.models import ToolAuditLog
from app.tools.audit import ToolAuditStore


def _rows(factory):
    with factory() as session:
        return session.query(ToolAuditLog).all()


def _log(store, **kw):
    kw.setdefault("tool_name", "query_order")
    kw.setdefault("tool_source", "builtin")
    kw.setdefault("status", "成功")
    store.log(**kw)


def test_log_persists_all_fields(db_session_factory):
    store = ToolAuditStore(db_session_factory)
    _log(store, conversation_id=3, tool_call_id="call_1", arguments={"order_id": "1001"},
         result_summary='{"order_id": "1001"}', retry_count=1, duration_ms=123)
    (row,) = _rows(db_session_factory)
    assert row.tool_name == "query_order" and row.tool_source == "builtin"
    assert row.conversation_id == 3 and row.tool_call_id == "call_1"
    assert row.arguments == {"order_id": "1001"}
    assert row.status == "成功" and row.retry_count == 1 and row.duration_ms == 123


def test_log_truncates_long_summary(db_session_factory):
    store = ToolAuditStore(db_session_factory)
    _log(store, result_summary="长" * 2000)
    (row,) = _rows(db_session_factory)
    assert len(row.result_summary) == 500


def test_log_denied_and_blocked_statuses(db_session_factory):
    store = ToolAuditStore(db_session_factory)
    _log(store, tool_name="create_ticket", status="权限拒绝", error_message="写入操作未获用户确认")
    _log(store, tool_name="query_faq", status="校验拦下", error_message="keyword: 必填缺失")
    assert [r.status for r in _rows(db_session_factory)] == ["权限拒绝", "校验拦下"]
