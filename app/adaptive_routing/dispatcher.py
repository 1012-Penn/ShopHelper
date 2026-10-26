"""Exact-once dispatch over independently injectable Agent executors."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any, Protocol

from app.adaptive_routing.models import (
    AdaptiveExecution,
    NormalizedResponse,
    RouteTarget,
    RoutingInput,
)
from app.adaptive_routing.policy import AdaptiveRouter


class RoutingExecutor(Protocol):
    async def __call__(self, routing_input: RoutingInput) -> Any: ...


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        result = dump()
        return result if isinstance(result, Mapping) else {}
    return {
        key: getattr(value, key)
        for key in (
            "final_reply", "reply", "answer", "message", "content",
            "conversation_id", "disposition", "facts", "evidence", "trace",
            "errors", "stop_reasons", "write_previews", "tokens",
        )
        if hasattr(value, key)
    }


def normalize_response(value: Any) -> NormalizedResponse:
    """Map observed single/multi result shapes without inventing runtime semantics."""

    if value is None:
        return NormalizedResponse()
    if isinstance(value, str):
        return NormalizedResponse(reply=value)
    row = _mapping(value)
    reply = next(
        (
            str(row[key])
            for key in ("final_reply", "reply", "answer", "message", "content")
            if row.get(key) is not None
        ),
        "",
    )
    errors = row.get("errors") or []
    stop_reasons = row.get("stop_reasons") or []
    previews = row.get("write_previews") or []
    trace = row.get("trace") or []
    try:
        tokens = int(row.get("tokens") or 0)
    except (TypeError, ValueError):
        tokens = 0
    conversation_id = row.get("conversation_id")
    try:
        conversation_id = int(conversation_id) if conversation_id is not None else None
    except (TypeError, ValueError):
        conversation_id = None
    return NormalizedResponse(
        reply=reply,
        conversation_id=conversation_id,
        disposition=str(row.get("disposition") or ""),
        facts=dict(row.get("facts") or {}),
        evidence=dict(row.get("evidence") or {}),
        trace=list(trace) if isinstance(trace, (list, tuple)) else [],
        errors=[str(item) for item in errors] if isinstance(errors, (list, tuple)) else [str(errors)],
        stop_reasons=(
            [str(item) for item in stop_reasons]
            if isinstance(stop_reasons, (list, tuple))
            else [str(stop_reasons)]
        ),
        write_previews=(
            [dict(item) for item in previews if isinstance(item, Mapping)]
            if isinstance(previews, (list, tuple))
            else []
        ),
        tokens=tokens,
    )


class RouteDispatcher:
    def __init__(self, *, router: AdaptiveRouter | None = None) -> None:
        self.router = router or AdaptiveRouter()

    async def dispatch(
        self,
        routing_input: RoutingInput,
        *,
        single_executor: RoutingExecutor,
        multi_executor: RoutingExecutor,
    ) -> AdaptiveExecution:
        """Route once, invoke exactly one executor, and never fall back to the other."""

        total_started = time.perf_counter()
        decision = await self.router.route_async(routing_input)
        routing_ms = int((time.perf_counter() - total_started) * 1000)
        selected: RouteTarget = decision.target
        executor = single_executor if selected == "single" else multi_executor
        error = ""
        raw: Any = None
        execution_started = time.perf_counter()
        try:
            raw = await executor(routing_input)
        except Exception as exc:
            error = f"{type(exc).__name__}:{exc}"
        execution_ms = int((time.perf_counter() - execution_started) * 1000)
        total_ms = int((time.perf_counter() - total_started) * 1000)
        return AdaptiveExecution(
            route=decision,
            selected_arm=selected,
            response=normalize_response(raw),
            raw_response=raw,
            error=error,
            routing_ms=routing_ms,
            execution_ms=execution_ms,
            total_ms=total_ms,
            routing_cost_usd=decision.routing_cost_usd,
            selected_executor_calls=1,
            other_executor_calls=0,
        )
