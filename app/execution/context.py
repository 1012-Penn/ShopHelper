"""执行上下文变量：图节点的工具与消息写入共享同一租约 fencing。"""
from contextvars import ContextVar
from typing import Callable

current_run_id: ContextVar[str | None] = ContextVar(
    "shophelper_chat_run_id", default=None)
current_run_owner: ContextVar[str | None] = ContextVar(
    "shophelper_chat_run_owner", default=None)
current_lease_guard: ContextVar[Callable[[], bool] | None] = ContextVar(
    "shophelper_chat_lease_guard", default=None)
