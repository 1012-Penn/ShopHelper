"""Optional multi-Agent entry point; the default chat route remains unchanged."""

from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from app.multi_agent.runtime import MultiAgentRunner, build_multi_agent_graph


class MultiAgentChatRequest(BaseModel):
    """Only user-visible input is accepted; evaluation gold is not part of the API."""

    message: str = Field(min_length=1, max_length=2000)
    session_id: int | None = None
    conversation_history: list[dict] = Field(default_factory=list, max_length=40)
    task_id: str | None = Field(default=None, max_length=64)


class MultiAgentChatResponse(BaseModel):
    run_id: str
    task_id: str
    conversation_id: int | None
    status: str
    final_reply: str
    facts: dict
    evidence: dict
    errors: list[str]
    stop_reasons: list[str]
    trace: list[dict]
    write_previews: list[dict]
    action_artifacts: list[dict] | None = None
    steps: int
    tokens: int


def create_multi_agent_router(runtime: MultiAgentRunner | None = None) -> APIRouter:
    """Create an optional router around one configured runtime.

    Production integration passes the existing `ToolEngine` and production models.
    Tests and evaluations pass the same contract with isolated fixtures.
    """

    router = APIRouter()

    @router.post(
        "/api/multi-agent/chat",
        response_model=MultiAgentChatResponse,
        response_model_exclude_defaults=True,
    )
    async def chat(payload: MultiAgentChatRequest, request: Request) -> dict:
        if not payload.message.strip():
            raise HTTPException(status_code=422, detail="message must not be blank")
        active_runtime = runtime
        if active_runtime is None:
            active_runtime = getattr(request.app.state, "multi_agent", None)
        if active_runtime is None:
            raise HTTPException(status_code=503, detail="multi-agent runtime 未配置")

        # The app factory may swap runtime after constructing the router (tests and
        # embedding callers do); cache the compiled graph against that identity.
        graph = getattr(request.app.state, "_multi_agent_graph", None)
        if graph is None or getattr(
                request.app.state, "_multi_agent_graph_runtime", None
        ) is not active_runtime:
            graph = build_multi_agent_graph(active_runtime)
            request.app.state._multi_agent_graph = graph
            request.app.state._multi_agent_graph_runtime = active_runtime

        store = getattr(request.app.state, "store", None)
        if store is not None:
            conversation_id = await store.resolve(payload.session_id)
            conversation_history = await store.get_history(conversation_id)
        else:
            conversation_id = payload.session_id
            conversation_history = payload.conversation_history
        run_id = uuid4().hex
        task_id = payload.task_id or uuid4().hex
        state = await graph.ainvoke({
            "run_id": run_id,
            "task_id": task_id,
            "conversation_id": conversation_id,
            "user_message": payload.message.strip(),
            "conversation_history": conversation_history,
            "facts": {},
            "evidence": {},
            "errors": [],
            "stop_reasons": [],
            "trace": [],
            "role_histories": {},
        })
        reply = state.get("final_reply", "")
        if store is not None:
            await store.append(conversation_id, [
                {"role": "user", "content": payload.message.strip()},
                {"role": "assistant", "content": reply},
            ])
        return {
            "run_id": run_id,
            "task_id": task_id,
            "conversation_id": conversation_id,
            "status": state.get("status", "success"),
            "final_reply": reply,
            "facts": state.get("facts", {}),
            "evidence": state.get("evidence", {}),
            "errors": state.get("errors", []),
            "stop_reasons": state.get("stop_reasons", []),
            "trace": state.get("trace", []),
            "write_previews": state.get("write_previews", []),
            "action_artifacts": state.get("action_artifacts"),
            "steps": int(state.get("steps", 0)),
            "tokens": int(state.get("tokens", 0)),
        }

    return router


def create_multi_agent_app(runtime: MultiAgentRunner, *, store=None) -> FastAPI:
    """Standalone app factory; the default application does not include it."""

    app = FastAPI(title="ShopHelper optional multi-agent service")
    app.state.multi_agent = runtime
    if store is not None:
        app.state.store = store
    app.include_router(create_multi_agent_router(runtime))
    return app
