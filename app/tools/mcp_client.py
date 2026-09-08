"""ch08 MCP 接入:MultiServerMCPClient 封装,每轮聊天前 diff 同步(现问现拿)。

我侧权限规则:所有 MCP 工具 access=read(用途声明不可信);与内置重名时内置优先(注册表跳过)。
连接失败只 warn 降级,Agent 继续用内置工具;Server 侧重启/加工具,客户端不重启,下轮 sync 生效。
"""
import asyncio
import logging
import time

from langchain_mcp_adapters.client import MultiServerMCPClient

from app.tools.base import ToolRecord, ToolRegistryV2

logger = logging.getLogger(__name__)


class McpService:
    def __init__(self, registry: ToolRegistryV2, urls: dict[str, str],
                 ttl_seconds: float = 30.0) -> None:
        self.registry = registry
        self.urls = dict(urls)
        self.ttl_seconds = ttl_seconds
        self._client = (MultiServerMCPClient(
            {name: {"transport": "http", "url": url} for name, url in urls.items()}
        ) if urls else None)
        self._last_sync = 0.0

    def connected(self) -> bool:
        return bool(self.urls) and bool(
            [r for r in self.registry.records() if r.source == "mcp"])

    async def sync(self, force: bool = False) -> None:
        """拉取各 Server 工具清单并 diff 进注册表;TTL 内跳过(现问现拿,防抖不防新)。

        TTL 只在成功同步后起算:失败的发现下轮立即重试(Server 重启中不丢窗口)。
        """
        if self._client is None:
            return
        if not force and time.monotonic() - self._last_sync < self.ttl_seconds:
            return
        failed_servers: set[str] = set()
        for server_name in self.urls:
            tools = None
            for attempt in range(3):  # 立即重试:新 spawn 的 Server 握手偶发抖动
                try:
                    tools = await self._client.get_tools(server_name=server_name)
                    break
                except Exception:
                    # Server 不在线/重启中:降级为无该 Server 工具,不影响其余工具
                    if attempt == 2:
                        logger.warning("MCP Server %s 工具发现失败(含重试),本轮跳过",
                                       server_name, exc_info=True)
                        failed_servers.add(server_name)
                    else:
                        await asyncio.sleep(0.5)
            if tools is None:
                continue  # 失败保留既有注册(最后已知状态),不做删除性同步
            fresh_names = set()
            for tool in tools:
                fresh_names.add(tool.name)
                if self.registry.get(tool.name) is not None:
                    continue  # 内置优先:重名 MCP 工具跳过(注册时同样有兜底)
                self.registry.register(ToolRecord(
                    tool, source="mcp", access="read", mcp_server=server_name,
                    label=tool.name))
                logger.info("MCP 工具登记:%s(server=%s)", tool.name, server_name)
            for record in self.registry.records():
                if (record.source == "mcp" and record.mcp_server == server_name
                        and record.name not in fresh_names):
                    self.registry.unregister(record.name)
                    logger.info("MCP 工具下线:%s(server=%s)", record.name, server_name)
        if not failed_servers:
            self._last_sync = time.monotonic()
