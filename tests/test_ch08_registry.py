"""ch08 注册中心:ToolRecord 三件套、重名策略(MCP 跳过/内置 ValueError)、labels、插件 autoload。"""
import textwrap

import pytest
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.tools.base import ToolContext, ToolRecord, ToolRegistryV2
from app.tools.plugins import load_plugins


def _tool(name, desc="工具描述"):
    class _In(BaseModel):
        order_id: str = Field(description="订单号")

    return StructuredTool.from_function(
        func=lambda order_id: "ok", name=name, description=desc, args_schema=_In)


def _rec(name, source="builtin", access="read", label="", mcp_server=None):
    return ToolRecord(_tool(name), source=source, access=access, label=label,
                      mcp_server=mcp_server)


def test_record_exposes_name_and_json_schema():
    rec = _rec("query_order")
    assert rec.name == "query_order"
    schema = rec.json_schema
    assert schema["properties"]["order_id"]["description"] == "订单号"
    assert "order_id" in schema.get("required", []) or schema["properties"]


def test_register_and_bind():
    reg = ToolRegistryV2()
    assert reg.register(_rec("a"))
    assert reg.register(_rec("b", source="mcp", mcp_server="s1"))
    assert reg.names() == ["a", "b"]
    assert [t.name for t in reg.bind_tools()] == ["a", "b"]
    assert reg.get("a").access == "read"


def test_register_mcp_duplicate_skips_builtin_wins():
    reg = ToolRegistryV2()
    reg.register(_rec("query_x", label="内置版"))
    assert reg.register(_rec("query_x", source="mcp", mcp_server="s1")) is False
    assert reg.get("query_x").source == "builtin" and reg.get("query_x").label == "内置版"


def test_register_builtin_duplicate_raises():
    reg = ToolRegistryV2()
    reg.register(_rec("a"))
    with pytest.raises(ValueError):
        reg.register(_rec("a"))


def test_unregister():
    reg = ToolRegistryV2()
    reg.register(_rec("a"))
    reg.unregister("a")
    assert reg.get("a") is None


def test_labels_fallback_to_name():
    reg = ToolRegistryV2()
    reg.register(_rec("a", label="甲工具"))
    reg.register(_rec("b"))
    assert reg.labels() == {"a": "甲工具", "b": "b"}


def test_load_plugins_registers_demo_echo(tmp_path):
    # 临时插件目录:落一个新文件即注册,不动核心代码
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    (plugin_dir / "demo_echo.py").write_text(textwrap.dedent('''
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel, Field

        from app.tools.base import ToolRecord

        class _In(BaseModel):
            text: str = Field(description="原文")

        def _echo(text: str) -> str:
            return text

        def register(registry, ctx):
            registry.register(ToolRecord(
                StructuredTool.from_function(func=_echo, name="demo_echo",
                                             description="原样返回", args_schema=_In),
                source="builtin", access="read", label="回声"))
    '''))
    reg = ToolRegistryV2()
    loaded = load_plugins(reg, ToolContext(), directory=plugin_dir)
    assert loaded == ["demo_echo"]
    assert reg.get("demo_echo") is not None and reg.labels()["demo_echo"] == "回声"


def test_real_plugins_dir_loads_demo_time():
    reg = ToolRegistryV2()
    loaded = load_plugins(reg, ToolContext())
    assert "demo_time" in loaded
    assert reg.get("query_server_time") is not None


class FakeMcpLikeTool:
    """langchain-mcp-adapters 形态:args_schema 为原始 JSON Schema dict。"""

    name = "mcp_like"
    description = "dict schema 工具"
    args_schema = {"type": "object",
                   "properties": {"order_id": {"type": "string"}},
                   "required": ["order_id"]}

    async def ainvoke(self, args):
        return "done"


def test_record_json_schema_accepts_raw_dict():
    """I 回归:MCP 工具 args_schema 为 dict 时不得调用 model_json_schema。"""
    rec = ToolRecord(FakeMcpLikeTool(), source="mcp", mcp_server="aftersales")
    schema = rec.json_schema
    assert schema["properties"]["order_id"]["type"] == "string"
