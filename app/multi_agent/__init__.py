"""Optional multi-agent collaboration path.

The package is intentionally independent from the default chat graph. Callers opt in
through :func:`app.multi_agent.api.create_multi_agent_router` or the demo CLI.
"""

from app.multi_agent.models import AgentPlan, AgentTask, MultiAgentLimits
from app.multi_agent.runtime import (
    MultiAgentRunner,
    build_multi_agent_graph,
    build_multi_agent_runtime,
)

__all__ = [
    "AgentPlan",
    "AgentTask",
    "MultiAgentLimits",
    "MultiAgentRunner",
    "build_multi_agent_graph",
    "build_multi_agent_runtime",
]
