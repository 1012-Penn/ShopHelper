"""Serializable contracts for the optional multi-agent graph."""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field, model_validator


AgentName = Literal["retrieval", "action"]
Disposition = Literal["execute", "clarify", "refuse", "handoff"]


class AgentTask(BaseModel):
    task_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,31}$")
    agent: AgentName
    objective: str = Field(min_length=1, max_length=600)
    depends_on: list[str] = Field(default_factory=list, max_length=3)


class AgentPlan(BaseModel):
    """Planner-only output. It cannot change role prompts or tool whitelists."""

    disposition: Disposition = "execute"
    tasks: list[AgentTask] = Field(default_factory=list)
    clarification: str = Field(default="", max_length=500)
    reason: str = Field(default="", max_length=500)

    @model_validator(mode="after")
    def validate_dependencies(self) -> "AgentPlan":
        if len(self.tasks) > 3:
            raise ValueError("a plan may contain at most 3 tasks")
        if self.disposition != "execute" and self.tasks:
            raise ValueError("clarify/refuse/handoff plans cannot schedule tasks")
        if self.disposition == "clarify" and not self.clarification.strip():
            raise ValueError("clarify plans require a clarification question")

        seen: set[str] = set()
        retrieval_ids: set[str] = set()
        for task in self.tasks:
            if task.task_id in seen:
                raise ValueError(f"duplicate task id: {task.task_id}")
            unknown = set(task.depends_on) - seen
            if unknown:
                raise ValueError(f"unknown dependency: {sorted(unknown)}")
            if task.agent == "action" and not (set(task.depends_on) & retrieval_ids):
                raise ValueError("action task must depend on completed retrieval facts")
            seen.add(task.task_id)
            if task.agent == "retrieval":
                retrieval_ids.add(task.task_id)
        return self


class MultiAgentLimits(BaseModel):
    max_tasks: int = Field(default=3, ge=1, le=3)
    max_replans: int = Field(default=1, ge=0, le=1)
    max_steps: int = Field(default=12, ge=1)
    max_tokens: int = Field(default=6000, ge=1)
    max_seconds: float = Field(default=45.0, gt=0)
    child_timeout_seconds: float = Field(default=20.0, gt=0)
    max_agent_iterations: int = Field(default=3, ge=1, le=4)


class AgentOutcome(BaseModel):
    task_id: str
    agent: str
    objective: str
    status: Literal["success", "failed", "timeout"]
    draft: str = ""
    facts: dict[str, Any] = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    messages: list[dict[str, Any]] = Field(default_factory=list)
    write_preview: dict[str, Any] | None = None
    error: str = ""
    stop_reason: str = ""
    steps: int = 0
    tokens: int = 0


class TraceEvent(TypedDict, total=False):
    event: str
    run_id: str
    task_id: str
    agent_id: str
    detail: str
    elapsed_ms: int


class MultiAgentState(TypedDict, total=False):
    run_id: str
    task_id: str
    user_message: str
    conversation_history: list[dict[str, Any]]
    plan: dict[str, Any]
    facts: dict[str, Any]
    evidence: dict[str, Any]
    errors: list[str]
    stop_reasons: list[str]
    trace: Annotated[list[TraceEvent], lambda left, right: left + right]
    role_histories: dict[str, list[dict[str, Any]]]
    task_agents: dict[str, str]
    tool_calls: list[dict[str, Any]]
    write_previews: list[dict[str, Any]]
    replan_count: int
    needs_replan: bool
    disposition: str
    clarification: str
    final_reply: str
    status: str
    steps: int
    tokens: int
    started_at: float
    deadline: float
