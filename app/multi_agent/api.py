"""Optional FastAPI entry point; it is not mounted by the default application."""

from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.multi_agent.runtime import MultiAgentRunner, build_multi_agent_graph


class MultiAgentChatRequest(BaseModel):
    """Only user-visible input is accepted; evaluation gold is not part of the API."""

    message: str = Field(min_length=1, max_length=2000)
    conversation_history: list[dict] = Field(default_factory=list, max_length=40)
    task_id: str | None = Field(default=None, max_length=64)


class MultiAgentChatResponse(BaseModel):
    run_id: str
    task_id: str
    status: str
    final_reply: str
    facts: dict
    evidence: dict
    errors: list[str]
    stop_reasons: list[str]
    trace: list[dict]
    write_previews: list[dict]
    steps: int
    tokens: int


def create_multi_agent_router(runtime: MultiAgentRunner) -> APIRouter:
    """Create an optional router around one configured runtime.

    Production integration passes the existing `ToolEngine` and production models.
    Tests and evaluations pass the same contract with isolated fixtures.
    """

    router = APIRouter()
    graph = build_multi_agent_graph(runtime)

    @router.post("/api/multi-agent/chat", response_model=MultiAgentChatResponse)
    async def chat(payload: MultiAgentChatRequest) -> dict:
        if not payload.message.strip():
            raise HTTPException(status_code=422, detail="message must not be blank")
        run_id = uuid4().hex
        task_id = payload.task_id or uuid4().hex
        state = await graph.ainvoke({
            "run_id": run_id,
            "task_id": task_id,
            "user_message": payload.message.strip(),
            "conversation_history": payload.conversation_history,
            "facts": {},
            "evidence": {},
            "errors": [],
            "stop_reasons": [],
            "trace": [],
            "role_histories": {},
        })
        return {
            "run_id": run_id,
            "task_id": task_id,
            "status": state.get("status", "success"),
            "final_reply": state.get("final_reply", ""),
            "facts": state.get("facts", {}),
            "evidence": state.get("evidence", {}),
            "errors": state.get("errors", []),
            "stop_reasons": state.get("stop_reasons", []),
            "trace": state.get("trace", []),
            "write_previews": state.get("write_previews", []),
            "steps": int(state.get("steps", 0)),
            "tokens": int(state.get("tokens", 0)),
        }

    return router


def create_multi_agent_app(runtime: MultiAgentRunner) -> FastAPI:
    """Standalone app factory; the default application does not include it."""

    app = FastAPI(title="ShopHelper optional multi-agent service")
    app.include_router(create_multi_agent_router(runtime))
    return app
