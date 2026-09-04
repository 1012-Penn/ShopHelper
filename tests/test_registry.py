import asyncio
import json

from langchain.tools import tool

from app.tools.registry import ToolRegistry


@tool
def ok_tool(x: int) -> str:
    """doubles x"""
    return json.dumps({"result": x * 2})


@tool
def bad_tool(x: int) -> str:
    """always raises"""
    raise RuntimeError("下游炸了")


@tool
def slow_tool(x: int) -> str:
    """sleeps past timeout"""
    import time

    time.sleep(5)
    return json.dumps({"ok": True})


def test_execute_ok():
    reg = ToolRegistry([ok_tool])
    assert json.loads(asyncio.run(reg.execute("ok_tool", '{"x": 21}')))["result"] == 42


def test_unknown_tool_returns_error_string():
    reg = ToolRegistry([ok_tool])
    out = asyncio.run(reg.execute("nope", "{}"))
    assert "nope" in out and "未注册" in out


def test_bad_json_returns_error_string():
    reg = ToolRegistry([ok_tool])
    out = asyncio.run(reg.execute("ok_tool", "{not json"))
    assert "参数" in out


def test_invalid_args_returns_error_string():
    reg = ToolRegistry([ok_tool])
    out = asyncio.run(reg.execute("ok_tool", '{"x": "not-a-number"}'))
    assert "参数" in out


def test_runtime_error_returns_error_string_after_retry():
    reg = ToolRegistry([bad_tool])
    out = asyncio.run(reg.execute("bad_tool", '{"x": 1}'))
    assert "下游炸了" in out


def test_timeout_returns_error_string():
    reg = ToolRegistry([slow_tool])
    reg.TOOL_TIMEOUT_SECONDS = 0.1  # 测试里收紧超时
    out = asyncio.run(reg.execute("slow_tool", '{"x": 1}'))
    assert "超时" in out


def test_labels():
    reg = ToolRegistry([ok_tool])
    # TOOL_LABELS 没有的工具回退为工具名本身
    assert reg.labels() == {"ok_tool": "ok_tool"}


def test_tools_property_returns_raw_tools():
    reg = ToolRegistry([ok_tool])
    assert reg.tools == [ok_tool]
