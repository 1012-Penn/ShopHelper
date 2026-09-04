"""工具注册表:统一执行入口,校验/超时/重试/错误都收敛为字符串回灌,不让请求 500。"""
import asyncio
import json
import logging

from pydantic import ValidationError

from app.tools.definitions import TOOL_LABELS

logger = logging.getLogger(__name__)


class ToolRegistry:
    TOOL_TIMEOUT_SECONDS = 10.0
    TOOL_RETRIES = 1

    def __init__(self, tools: list) -> None:
        self._by_name = {t.name: t for t in tools}

    @property
    def tools(self) -> list:
        return list(self._by_name.values())

    def has(self, name: str) -> bool:
        return name in self._by_name

    def labels(self) -> dict[str, str]:
        return {name: TOOL_LABELS.get(name, name) for name in self._by_name}

    async def execute(self, name: str, args_json: str) -> str:
        if name not in self._by_name:
            return f"错误:工具 {name} 未注册"
        try:
            args = json.loads(args_json or "{}")
        except json.JSONDecodeError:
            return f"错误:工具 {name} 的参数不是合法 JSON"
        if not isinstance(args, dict):
            return f"错误:工具 {name} 的参数必须是 JSON 对象"
        tool = self._by_name[name]
        last_error = ""
        for attempt in range(self.TOOL_RETRIES + 1):
            try:
                raw = await asyncio.wait_for(
                    asyncio.to_thread(tool.invoke, args), timeout=self.TOOL_TIMEOUT_SECONDS
                )
                return raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
            except asyncio.TimeoutError:
                last_error = f"错误:工具 {name} 执行超时({self.TOOL_TIMEOUT_SECONDS}s)"
            except ValidationError as exc:
                last_error = f"错误:工具 {name} 参数校验失败:{exc.errors()[0].get('msg', exc)}"
            except Exception as exc:
                last_error = f"错误:工具 {name} 执行失败:{exc}"
            logger.warning("%s(第 %d 次)", last_error, attempt + 1)
        return last_error
