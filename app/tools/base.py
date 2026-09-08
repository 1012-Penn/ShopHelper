"""ch08 工具注册中心:ToolRecord(名/描述/Schema 三件套 + 来源/读写属性)与统一注册表。

内置与 MCP 工具同表登记;bind_tools 直接产出 BaseTool 列表供模型绑定。
重名策略:内置先注册,MCP 版遇重名跳过并 warn(我们这侧的规则,不看对方声明)。
"""
import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


@dataclass
class ToolRecord:
    """注册表条目:tool 本体 + 来源/读写属性。用途描述与参数 Schema 由 BaseTool 自带。"""

    tool: BaseTool
    source: str                      # builtin / mcp
    access: str = "read"             # read / write(我们这侧的权限规则,不看工具自身声明)
    mcp_server: str | None = None
    label: str = ""                  # 前端徽章显示名;空则用工具名

    @property
    def name(self) -> str:
        return self.tool.name

    @property
    def json_schema(self) -> dict[str, Any]:
        """参数 JSON Schema:模型可见的入参契约,执行前统一按它校验。"""
        schema = self.tool.args_schema
        if schema is None:
            return {"type": "object", "properties": {}}
        return schema.model_json_schema()


@dataclass
class ToolContext:
    """注册期依赖注入面:插件与内置工具从这里拿基础设施,不碰应用全局。"""
    session_factory: Any = None
    service: Any = None              # RetrievalService(query_faq 用)
    kb: Any = None
    top_k: int = 3
    rewriter: Any = None
    reranker: Any = None


class ToolRegistryV2:
    def __init__(self) -> None:
        self._by_name: dict[str, ToolRecord] = {}

    def register(self, record: ToolRecord) -> bool:
        """登记;重名时内置优先(先到先得),MCP 版跳过 warn;非 MCP 重名视为程序错误。"""
        existing = self._by_name.get(record.name)
        if existing is not None:
            if record.source == "mcp":
                logger.warning("MCP 工具 %s 与已注册工具重名,跳过(内置优先)", record.name)
                return False
            raise ValueError(f"工具 {record.name} 重复注册")
        self._by_name[record.name] = record
        return True

    def unregister(self, name: str) -> None:
        self._by_name.pop(name, None)

    def get(self, name: str) -> ToolRecord | None:
        return self._by_name.get(name)

    def records(self) -> list[ToolRecord]:
        return list(self._by_name.values())

    def names(self) -> list[str]:
        return list(self._by_name.keys())

    def bind_tools(self) -> list[BaseTool]:
        return [r.tool for r in self._by_name.values()]

    def labels(self) -> dict[str, str]:
        return {r.name: (r.label or r.name) for r in self._by_name.values()}
