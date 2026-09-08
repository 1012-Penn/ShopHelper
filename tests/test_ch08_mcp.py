"""ch08 McpService 同步:发现登记/消失下线/重名内置优先/TTL 防抖/连接失败降级。"""
import pytest
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.tools.base import ToolRecord, ToolRegistryV2
from app.tools.mcp_client import McpService

pytestmark = pytest.mark.asyncio


class _In(BaseModel):
    order_id: str = Field(description="订单号")


def _tool(name):
    return StructuredTool.from_function(
        func=lambda order_id: "ok", name=name, description=f"{name} 描述", args_schema=_In)


class StubClient:
    """get_tools(server_name) 按 server 返回;记录调用次数。"""

    def __init__(self, servers: dict[str, list]):
        self.servers = servers
        self.calls = 0

    async def get_tools(self, server_name=None):
        self.calls += 1
        return list(self.servers.get(server_name, []))


def _make(servers, ttl=30.0):
    registry = ToolRegistryV2()
    client = StubClient(servers)
    service = McpService(registry, {name: f"http://{name}:9000/mcp" for name in servers}, ttl)
    service._client = client
    return registry, service, client


async def test_sync_registers_mcp_tools():
    registry, service, _ = _make({"logistics": [_tool("logistics_tracker")]})
    await service.sync()
    rec = registry.get("logistics_tracker")
    assert rec is not None
    assert rec.source == "mcp" and rec.mcp_server == "logistics"
    assert rec.access == "read"  # 我侧规则:全只读,不看 Server 声明


async def test_sync_unregisters_vanished_tools():
    registry, service, _ = _make({"logistics": [_tool("logistics_tracker")]})
    await service.sync()
    service._client.servers["logistics"] = []
    await service.sync(force=True)
    assert registry.get("logistics_tracker") is None


async def test_sync_builtin_wins_on_name_collision():
    registry = ToolRegistryV2()
    registry.register(ToolRecord(_tool("query_x"), source="builtin", access="read", label="内置版"))
    service = McpService(registry, {"logistics": "http://x/mcp"})
    service._client = StubClient({"logistics": [_tool("query_x")]})
    await service.sync()
    rec = registry.get("query_x")
    assert rec.source == "builtin" and rec.mcp_server is None


async def test_ttl_skips_refetch():
    registry, service, client = _make({"logistics": [_tool("logistics_tracker")]}, ttl=60)
    await service.sync()
    calls = client.calls
    await service.sync()
    assert client.calls == calls  # TTL 内不重复拉


async def test_force_sync_bypasses_ttl():
    registry, service, client = _make({"logistics": [_tool("logistics_tracker")]}, ttl=60)
    await service.sync()
    await service.sync(force=True)
    assert client.calls == 2


async def test_server_down_degrades_without_raise(caplog):
    class DownClient:
        calls = 0

        async def get_tools(self, server_name=None):
            raise ConnectionError("server 不在线")

    registry = ToolRegistryV2()
    service = McpService(registry, {"logistics": "http://x/mcp"})
    service._client = DownClient()
    await service.sync()  # 不抛
    assert registry.names() == []


async def test_no_urls_is_noop():
    registry = ToolRegistryV2()
    service = McpService(registry, {})
    await service.sync()  # 不抛不拉
    assert registry.names() == []
