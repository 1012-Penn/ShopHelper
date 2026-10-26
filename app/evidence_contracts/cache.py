"""Run-local coalescing cache for equivalent read-only tool calls."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from app.evidence_contracts.extraction import subjects_from_arguments


@dataclass
class CacheStats:
    attempted: int = 0
    executed: int = 0
    deduplicated: int = 0
    errors: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "attempted": self.attempted,
            "executed": self.executed,
            "deduplicated": self.deduplicated,
            "errors": self.errors,
        }


@dataclass(frozen=True)
class ToolCacheResult:
    result: str
    deduplicated: bool


@dataclass
class _Entry:
    future: asyncio.Future[str]
    created: float
    ttl_seconds: int


def _canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)) or value is None:
        return str(value) if isinstance(value, (int, float)) and not isinstance(
            value, bool
        ) else value
    return str(value)


def call_cache_key(tool_name: str, arguments: dict[str, Any],
                   *, scope: dict[str, Any] | None = None) -> str:
    subjects = subjects_from_arguments(arguments)
    material = {
        "tool": tool_name,
        "arguments": _canonical(arguments),
        "subjects": sorted(subjects),
        "scope": _canonical(scope or {}),
    }
    return hashlib.sha256(
        json.dumps(material, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


class ToolResultCache:
    def __init__(self, *, dynamic_ttl_seconds: int = 300) -> None:
        self.dynamic_ttl_seconds = max(0, int(dynamic_ttl_seconds))
        self.stats = CacheStats()
        self._entries: dict[str, _Entry] = {}
        self._guard = asyncio.Lock()

    async def execute(self, tool_name: str, arguments: dict[str, Any], *,
                      read: bool, execute: Callable[[], Awaitable[str]],
                      scope: dict[str, Any] | None = None,
                      ttl_seconds: int | None = None) -> ToolCacheResult:
        self.stats.attempted += 1
        if not read:
            self.stats.executed += 1
            try:
                return ToolCacheResult(await execute(), deduplicated=False)
            except Exception:
                self.stats.errors += 1
                raise

        key = call_cache_key(tool_name, arguments, scope=scope)
        ttl = self.dynamic_ttl_seconds if ttl_seconds is None else max(0, ttl_seconds)
        owner = False
        async with self._guard:
            now = time.monotonic()
            entry = self._entries.get(key)
            if entry is None or now - entry.created >= entry.ttl_seconds:
                loop = asyncio.get_running_loop()
                entry = _Entry(loop.create_future(), now, ttl)
                self._entries[key] = entry
                owner = True
                self.stats.executed += 1
            else:
                self.stats.deduplicated += 1

        if owner:
            try:
                value = await execute()
                if not entry.future.done():
                    entry.future.set_result(value)
                return ToolCacheResult(value, deduplicated=False)
            except Exception as exc:
                self.stats.errors += 1
                if not entry.future.done():
                    entry.future.set_exception(exc)
                async with self._guard:
                    if self._entries.get(key) is entry:
                        self._entries.pop(key, None)
                raise
        try:
            value = await asyncio.shield(entry.future)
        except Exception:
            self.stats.errors += 1
            raise
        return ToolCacheResult(value, deduplicated=True)
