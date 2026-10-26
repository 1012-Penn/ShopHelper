"""Strict visible-input and decision contracts for adaptive Agent routing."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

RouteTarget = Literal["single", "multi"]


class HistoryTurn(BaseModel):
    """One message that either Agent implementation is already allowed to see."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["system", "user", "assistant", "tool"]
    content: str


class VisibleRunBudget(BaseModel):
    """Budget visible before execution; no hidden provider or gold metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    remaining_seconds: float | None = Field(default=None, gt=0)
    remaining_cost_usd: float | None = Field(default=None, ge=0)
    max_steps: int | None = Field(default=None, ge=1)


class RoutingInput(BaseModel):
    """The complete router input.

    Gold labels, scenario IDs, expected tools, and scoring rules are deliberately
    outside this schema. Extra fields fail closed.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    question: str = Field(min_length=1)
    history: list[HistoryTurn] = Field(default_factory=list)
    visible_budget: VisibleRunBudget = Field(default_factory=VisibleRunBudget)


class RoutingFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    visible_order_ids: tuple[str, ...] = ()
    entity_count: int = Field(default=0, ge=0)
    cross_order_comparison: bool = False
    policy_order_dependency: bool = False
    correction: bool = False
    missing_slot: bool = False
    write_confirmation_risk: bool = False


class JudgeSignal(BaseModel):
    """Result of an optional bounded structured routing judgment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target: RouteTarget
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1, max_length=160)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)


class RoutingDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    target: RouteTarget
    reason_codes: tuple[str, ...]
    features: RoutingFeatures
    policy_version: str
    policy_sha256: str
    visible_budget: VisibleRunBudget
    judge_used: bool = False
    judge_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    judge_input_tokens: int = Field(default=0, ge=0)
    judge_output_tokens: int = Field(default=0, ge=0)
    routing_cost_usd: float = Field(default=0.0, ge=0.0)
    routing_latency_ms: int = Field(default=0, ge=0)


class NormalizedResponse(BaseModel):
    """Stable read view over the existing single/multi Agent return shapes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reply: str = ""
    run_id: str | None = None
    conversation_id: int | None = None
    status: str = ""
    disposition: str = ""
    facts: dict[str, Any] = Field(default_factory=dict)
    evidence: dict[str, Any] = Field(default_factory=dict)
    trace: list[Any] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    stop_reasons: list[str] = Field(default_factory=list)
    write_previews: list[dict[str, Any]] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)
    tokens: int = 0


class AdaptiveExecution(BaseModel):
    """One routed execution; raw output is retained for auditability."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    route: RoutingDecision
    selected_arm: RouteTarget
    response: NormalizedResponse
    raw_response: Any = None
    error: str = ""
    routing_ms: int = Field(default=0, ge=0)
    execution_ms: int = Field(default=0, ge=0)
    total_ms: int = Field(default=0, ge=0)
    routing_cost_usd: float = Field(default=0.0, ge=0.0)
    selected_executor_calls: int = Field(default=0, ge=0)
    other_executor_calls: int = Field(default=0, ge=0)
