"""Optional standalone API adapter; not mounted by the production application."""

from __future__ import annotations

from typing import Protocol

from fastapi import APIRouter, FastAPI
from pydantic import BaseModel, ConfigDict, Field

from app.adaptive_routing.models import (
    AdaptiveExecution,
    HistoryTurn,
    RoutingInput,
    VisibleRunBudget,
)


class AdaptiveChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1)
    session_id: int | None = Field(default=None, ge=1)
    history: list[HistoryTurn] = Field(default_factory=list)
    visible_budget: VisibleRunBudget = Field(default_factory=VisibleRunBudget)


class SessionAwareAdaptiveService(Protocol):
    async def respond(
        self, routing_input: RoutingInput, *, session_id: int | None = None
    ) -> AdaptiveExecution: ...


def create_adaptive_routing_router(
    service: SessionAwareAdaptiveService,
) -> APIRouter:
    """Mount an opt-in endpoint whose service binds existing session/tool state."""

    router = APIRouter(tags=["adaptive-routing"])

    @router.post("/api/adaptive-routing/chat")
    async def chat(payload: AdaptiveChatRequest) -> dict:
        routing_input = RoutingInput(
            question=payload.message,
            history=payload.history,
            visible_budget=payload.visible_budget,
        )
        execution = await service.respond(
            routing_input, session_id=payload.session_id
        )
        # Raw objects may contain provider messages or recoverable runtime state.
        # The standalone adapter exposes the normalized audited contract only.
        return execution.model_dump(mode="json", exclude={"raw_response"})

    return router


def create_adaptive_routing_app(service: SessionAwareAdaptiveService) -> FastAPI:
    app = FastAPI(title="ShopHelper optional adaptive routing")
    app.include_router(create_adaptive_routing_router(service))
    return app
