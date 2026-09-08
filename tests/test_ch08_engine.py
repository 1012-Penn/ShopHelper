"""ch08 执行引擎管线:校验拦下/权限把门/超时重试分类/业务空结果不重试/格式化/审计兜底。"""
import json
import time

import pytest
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.models import ToolAuditLog
from app.tools.audit import ToolAuditStore
from app.tools.base import ToolRecord, ToolRegistryV2
from app.tools.engine import ToolEngine, format_result

pytestmark = pytest.mark.asyncio


class _OrderIn(BaseModel):
    order_id: str = Field(description="订单号")


def _mk_tool(name, func):
    return StructuredTool.from_function(
        func=func, name=name, description="测试工具", args_schema=_OrderIn)


def _engine(factory, tools, *, timeout=5.0, retries=1, audit_raises=False):
    reg = ToolRegistryV2()
    for rec in tools:
        reg.register(rec)
    if audit_raises:
        class BoomStore:
            def log(self, **kw):
                raise RuntimeError("审计库挂了")
        store = BoomStore()
    else:
        store = ToolAuditStore(factory)
    return ToolEngine(reg, store, timeout_seconds=timeout, retry_attempts=retries)


def _rows(factory):
    with factory() as session:
        return list(session.query(ToolAuditLog).order_by(ToolAuditLog.id))


def _rec(name, func, access="read"):
    return ToolRecord(_mk_tool(name, func), source="builtin", access=access)


# ---- 回灌口径 ----

async def test_unregistered_tool_returns_error_without_audit(db_session_factory):
    engine = _engine(db_session_factory, [])
    out = await engine.execute("nope", "{}")
    assert "未注册" in out and _rows(db_session_factory) == []


async def test_invalid_json_blocked(db_session_factory):
    rec = _rec("t", lambda order_id: "ok")
    engine = _engine(db_session_factory, [rec])
    out = await engine.execute("t", "{bad json")
    (row,) = _rows(db_session_factory)
    assert out.startswith("错误:") and "合法 JSON" in out
    assert row.status == "校验拦下"


async def test_missing_required_blocked(db_session_factory):
    rec = _rec("t", lambda order_id: "ok")
    engine = _engine(db_session_factory, [rec])
    out = await engine.execute("t", "{}")
    (row,) = _rows(db_session_factory)
    assert "参数校验失败" in out and "order_id" in out
    assert row.status == "校验拦下" and "order_id" in (row.error_message or "")


async def test_wrong_type_blocked(db_session_factory):
    rec = _rec("t", lambda order_id: "ok")
    engine = _engine(db_session_factory, [rec])
    out = await engine.execute("t", json.dumps({"order_id": 123}))
    assert "参数校验失败" in out


# ---- 权限把门 ----

async def test_write_without_confirm_rejected_and_preview_emitted(db_session_factory):
    calls = []

    def _create(order_id: str) -> str:
        calls.append(order_id)
        return "已创建"

    rec = _rec("create_ticket_x", _create, access="write")
    engine = _engine(db_session_factory, [rec])
    previews = []
    out = await engine.execute("create_ticket_x", json.dumps({"order_id": "1001"}),
                               tool_call_id="call_9",
                               on_write_intercept=lambda r, a: previews.append((r, a)))
    assert calls == []                                   # 没有真实执行
    assert "前端确认" in out
    assert previews and previews[0][0].name == "create_ticket_x"
    assert previews[0][1] == {"order_id": "1001"}
    (row,) = _rows(db_session_factory)
    assert row.status == "权限拒绝" and row.tool_call_id == "call_9"
    assert row.arguments == {"order_id": "1001"}


async def test_write_confirmed_executes(db_session_factory):
    rec = _rec("w", lambda order_id: json.dumps({"ticket_no": "T1"}), access="write")
    engine = _engine(db_session_factory, [rec])
    out = await engine.execute_confirmed("w", json.dumps({"order_id": "1001"}))
    assert "T1" in out
    (row,) = _rows(db_session_factory)
    assert row.status == "成功"


# ---- 超时/重试分类 ----

def _sleeping_rec(name, seconds, counter, access="read"):
    def _f(order_id: str) -> str:
        counter.append(1)
        time.sleep(seconds)
        return "done"
    return _rec(name, _f, access=access)


async def test_read_timeout_retries_once_then_timeout(db_session_factory):
    counter = []
    rec = _sleeping_rec("slow_read", 1.0, counter)
    engine = _engine(db_session_factory, [rec], timeout=0.2, retries=1)
    out = await engine.execute("slow_read", json.dumps({"order_id": "1"}))
    rows = _rows(db_session_factory)
    assert len(counter) == 2            # 首跑 + 重试 1 次
    assert rows[-1].status == "超时" and rows[-1].retry_count == 1
    assert rows[-1].duration_ms is not None and rows[-1].duration_ms >= 200
    assert "超时" in out


async def test_write_timeout_never_retries(db_session_factory):
    counter = []
    rec = _sleeping_rec("slow_write", 1.0, counter, access="write")
    engine = _engine(db_session_factory, [rec], timeout=0.2, retries=1)
    out = await engine.execute_confirmed("slow_write", json.dumps({"order_id": "1"}))
    rows = _rows(db_session_factory)
    assert len(counter) == 1            # 写操作不自动重试(超时未必没执行)
    assert rows[-1].status == "超时" and rows[-1].retry_count == 0
    assert "超时" in out


async def test_business_empty_result_no_retry(db_session_factory):
    counter = []

    def _empty(order_id: str) -> str:
        counter.append(1)
        return json.dumps({"items": []})

    rec = _rec("q", _empty)
    engine = _engine(db_session_factory, [rec], retries=3)
    out = await engine.execute("q", json.dumps({"order_id": "1"}))
    assert len(counter) == 1            # 业务空结果不重试
    (row,) = _rows(db_session_factory)
    assert row.status == "成功" and row.retry_count == 0
    assert "items" in out


async def test_real_failure_no_retry_but_audited(db_session_factory):
    counter = []

    def _boom(order_id: str) -> str:
        counter.append(1)
        raise ValueError("内部参数错乱")

    rec = _rec("q", _boom)
    engine = _engine(db_session_factory, [rec], retries=1)
    out = await engine.execute("q", json.dumps({"order_id": "1"}))
    assert len(counter) == 1
    (row,) = _rows(db_session_factory)
    assert row.status == "失败" and "内部参数错乱" in row.error_message


# ---- 格式化 ----

def test_format_result_mcp_blocks():
    blocks = [{"type": "text", "text": "节点A"}, {"type": "text", "text": "节点B"}]
    assert format_result(blocks) == "节点A节点B"
    assert format_result("原文") == "原文"
    assert "运费" in format_result({"fee": "运费"})


def test_format_result_chinese_not_escaped():
    out = format_result({"city": "杭州"})
    assert "杭州" in out and "\\u" not in out


# ---- 审计写失败不拦执行 ----

async def test_audit_failure_does_not_block_execution(db_session_factory):
    rec = _rec("t", lambda order_id: "结果")
    engine = _engine(db_session_factory, [rec], audit_raises=True)
    out = await engine.execute("t", json.dumps({"order_id": "1"}))
    assert out == "结果"


async def test_mcp_dict_schema_tool_executes(db_session_factory):
    """回归:args_schema 为原始 JSON Schema dict 的 MCP 型工具走完整管线。"""
    from app.tools.base import ToolRecord

    class McpLikeTool:
        name = "mcp_like"
        description = "dict schema 工具"
        args_schema = {"type": "object",
                       "properties": {"order_id": {"type": "string"}},
                       "required": ["order_id"]}

        async def ainvoke(self, args):
            return [{"type": "text", "text": "轨迹节点一"}]

    rec = ToolRecord(McpLikeTool(), source="mcp", access="read", mcp_server="logistics")
    engine = _engine(db_session_factory, [rec])
    out = await engine.execute("mcp_like", json.dumps({"order_id": "1001"}))
    assert out == "轨迹节点一"
    (row,) = _rows(db_session_factory)
    assert row.status == "成功" and row.tool_source == "mcp" and row.mcp_server == "logistics"

    # 缺必填 → 校验拦下(dict schema 同样生效)
    out = await engine.execute("mcp_like", "{}")
    (row,) = [_r for _r in _rows(db_session_factory)][-1:]
    assert row.status == "校验拦下"
