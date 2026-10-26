"""工具执行可靠性包装：持久化写前状态，不修改既有 ToolEngine。"""
from __future__ import annotations

import json
import logging

from app.execution.context import current_run_id, current_run_owner
from app.execution.store import ExecutionStore

logger = logging.getLogger(__name__)


class ExecutionToolEngine:
    def __init__(self, delegate, store: ExecutionStore) -> None:
        self._delegate = delegate
        self._store = store

    def __getattr__(self, name):
        return getattr(self._delegate, name)

    async def execute(self, name: str, args_json: str, *, conversation_id: int | None = None,
                      tool_call_id: str | None = None, on_write_intercept=None,
                      confirmed: bool = False) -> str:
        run_id = current_run_id.get()
        owner_token = current_run_owner.get()
        record = self.registry.get(name)
        if run_id is None or record is None:
            return await self._delegate.execute(
                name, args_json, conversation_id=conversation_id,
                tool_call_id=tool_call_id, on_write_intercept=on_write_intercept,
                confirmed=confirmed)
        if owner_token is not None and not self._store.lease_owned_by(
                run_id, owner_token):
            return "错误:执行租约已失效,工具未执行"
        try:
            args = json.loads(args_json or "{}")
        except (json.JSONDecodeError, TypeError):
            args = None
        safe_args = args if isinstance(args, dict) else None
        call_id = tool_call_id or f"auto-{name}"
        try:
            self._store.start_tool_attempt(
                run_id,
                tool_call_id=call_id,
                tool_name=name,
                access=record.access,
                arguments=safe_args,
                idempotency_key=(safe_args or {}).get("confirm_key"),
                executes=confirmed,
            )
        except Exception:
            logger.error("工具 %s 写前尝试落库失败", name, exc_info=True)
            if record.access == "write" and confirmed:
                return "错误:写操作前置审计不可用,已拒绝执行,请稍后重试"

        try:
            result = await self._delegate.execute(
                name, args_json, conversation_id=conversation_id,
                tool_call_id=tool_call_id, on_write_intercept=on_write_intercept,
                confirmed=confirmed)
        except BaseException:
            if record.access == "write" and confirmed:
                try:
                    self._store.finish_tool_attempt(run_id, call_id, status="unknown")
                except Exception:
                    logger.error("工具 %s 未知状态落库失败", name, exc_info=True)
            raise

        if record.access == "write" and not confirmed:
            attempt_status = "intercepted"
        elif str(result).startswith("错误:"):
            attempt_status = "failed"
        else:
            attempt_status = "succeeded"
        try:
            self._store.finish_tool_attempt(
                run_id, call_id, status=attempt_status, result_text=str(result))
        except Exception:
            logger.error("工具 %s 后置尝试落库失败", name, exc_info=True)
            if record.access == "write" and confirmed:
                return "错误:写操作已发出但结果状态落库失败,请按确认幂等键核对"
        return result

    async def execute_confirmed(self, name: str, args_json: str, *,
                                conversation_id: int | None = None,
                                tool_call_id: str | None = None) -> str:
        return await self.execute(
            name, args_json, conversation_id=conversation_id,
            tool_call_id=tool_call_id, confirmed=True)
