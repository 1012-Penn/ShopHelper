"""执行任务生命周期：后台保活、取消与持久事件轮询。"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from uuid import uuid4

from app.execution.store import ExecutionStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExecutionPolicy:
    max_run_seconds: float = 120.0
    lease_seconds: int = 45
    poll_seconds: float = 0.02


class ExecutionService:
    def __init__(self, store: ExecutionStore, policy: ExecutionPolicy | None = None) -> None:
        self.store = store
        self.policy = policy or ExecutionPolicy()
        self.owner_token = uuid4().hex
        self._tasks: dict[str, asyncio.Task] = {}

    def track(self, run_id: str, task: asyncio.Task) -> None:
        self._tasks[run_id] = task
        task.add_done_callback(lambda _task: self._tasks.pop(run_id, None))

    def is_running(self, run_id: str) -> bool:
        task = self._tasks.get(run_id)
        return task is not None and not task.done()

    def cancel(self, run_id: str) -> bool:
        task = self._tasks.get(run_id)
        if task is None or task.done():
            return False
        task.cancel()
        return True

    async def shutdown(self) -> None:
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def events_after(self, run_id: str, after_id: int):
        return await asyncio.to_thread(self.store.read_events, run_id, after_id)

    async def status(self, run_id: str) -> dict:
        return await asyncio.to_thread(self.store.get_run, run_id)

    async def persist_event(self, run_id: str, event: dict,
                            owner_token: str | None = None) -> int:
        return await asyncio.to_thread(
            self.store.append_event, run_id, event,
            owner_token=owner_token or self.owner_token)

    async def mark_running(self, run_id: str) -> None:
        await asyncio.to_thread(
            self.store.mark_running, run_id, self.owner_token)

    async def finish(self, run_id: str, **kwargs) -> None:
        kwargs.setdefault("owner_token", self.owner_token)
        await asyncio.to_thread(self.store.finish_run, run_id, **kwargs)

    async def wait_terminal(self, run_id: str) -> dict:
        while True:
            status = await self.status(run_id)
            if status["status"] in {"completed", "failed", "cancelled", "interrupted"}:
                return status
            await asyncio.sleep(self.policy.poll_seconds)
