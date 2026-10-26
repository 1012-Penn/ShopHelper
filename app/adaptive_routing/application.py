"""Application-factory adapter for the opt-in adaptive routing endpoint."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import uuid4

from app.adaptive_routing.dispatcher import RouteDispatcher
from app.adaptive_routing.models import (
    AdaptiveExecution,
    HistoryTurn,
    RoutingInput,
)
from app.multi_agent.runtime import build_multi_agent_graph
from app.routers.chat import _execute_chat_run
from app.schemas import ChatRequest


class AppAdaptiveRoutingService:
    """Bind router executors to the existing app store/engine/model.

    The single path reuses the durable run implementation but returns its final
    snapshot as JSON instead of opening an SSE response.  The multi path keeps
    its existing process-local graph semantics and does not pretend to have the
    same recovery contract.
    """

    def __init__(self, app, *, dispatcher: RouteDispatcher | None = None) -> None:
        self.app = app
        self.dispatcher = dispatcher or RouteDispatcher()

    async def respond(
        self, routing_input: RoutingInput, *, session_id: int | None = None
    ) -> AdaptiveExecution:
        session_id = await self.app.state.store.resolve(session_id)
        history = await self.app.state.store.get_history(session_id)
        if not history:
            history = [
                item.model_dump() for item in routing_input.history
            ]
        visible = routing_input.model_copy(update={
            "history": [
                HistoryTurn(
                    role=item["role"],
                    content=str(item.get("content") or ""),
                )
                for item in history
                if item.get("role") in {"system", "user", "assistant", "tool"}
            ][-40:]
        })

        async def run_single(_input: RoutingInput):
            return await self._run_single(session_id, visible)

        async def run_multi(_input: RoutingInput):
            return await self._run_multi(session_id, visible)

        return await self.dispatcher.dispatch(
            visible,
            single_executor=run_single,
            multi_executor=run_multi,
        )

    async def _run_single(self, session_id: int, routing_input: RoutingInput):
        request = SimpleNamespace(app=self.app)
        body = ChatRequest(
            message=routing_input.question,
            session_id=session_id,
            client_request_id=f"adaptive-{uuid4().hex}",
        )
        reserved = await asyncio.to_thread(
            self.app.state.execution_store.reserve_run,
            conversation_id=session_id,
            client_request_id=body.client_request_id,
            request_payload={
                "message": body.message,
                "session_id": session_id,
                "resume": None,
            },
            owner_token=self.app.state.execution_service.owner_token,
            lease_seconds=self.app.state.execution_service.policy.lease_seconds,
        )
        await _execute_chat_run(request, reserved.run_id, session_id, body)
        status = await self.app.state.execution_service.status(reserved.run_id)
        events = await self.app.state.execution_service.events_after(reserved.run_id, 0)
        payloads = [item.event for item in events]
        result = status.get("result") or {}
        citations = [
            item for item in payloads if item.get("type") == "citations"
        ]
        return {
            "run_id": reserved.run_id,
            "status": status.get("status") or "",
            "final_reply": result.get("answer") or "",
            "conversation_id": session_id,
            "disposition": result.get("intent") or "",
            "evidence": (citations[-1].get("items") or []) if citations else [],
            "write_previews": [
                item for item in payloads if item.get("type") == "ticket_preview"
            ],
            "errors": [
                str(item.get("message") or item)
                for item in payloads if item.get("type") == "error"
            ],
            "usage": status.get("usage") or {},
            "tokens": int((status.get("usage") or {}).get("total_tokens") or 0),
            "trace": payloads,
            "stop_reasons": [
                str(event["reason"])
                for event in payloads
                if event.get("type") == "interrupted" and event.get("reason")
            ],
        }

    async def _run_multi(self, session_id: int, routing_input: RoutingInput):
        runtime = getattr(self.app.state, "multi_agent", None)
        if runtime is None:
            raise RuntimeError("multi-agent runtime unavailable")
        history = await self.app.state.store.get_history(session_id)
        graph = build_multi_agent_graph(runtime)
        state = await graph.ainvoke({
            "run_id": uuid4().hex,
            "task_id": uuid4().hex,
            "conversation_id": session_id,
            "user_message": routing_input.question,
            "conversation_history": history,
            "facts": {},
            "evidence": {},
            "errors": [],
            "stop_reasons": [],
            "trace": [],
            "role_histories": {},
        })
        reply = state.get("final_reply", "")
        await self.app.state.store.append(session_id, [
            {"role": "user", "content": routing_input.question},
            {"role": "assistant", "content": reply},
        ])
        return {
            "run_id": state.get("run_id"),
            "status": state.get("status", "success"),
            "final_reply": reply,
            "conversation_id": session_id,
            "facts": state.get("facts") or {},
            "evidence": state.get("evidence") or {},
            "errors": state.get("errors") or [],
            "stop_reasons": state.get("stop_reasons") or [],
            "trace": state.get("trace") or [],
            "write_previews": state.get("write_previews") or [],
            "tokens": int(state.get("tokens") or 0),
        }
