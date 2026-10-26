# Multi-Agent integration patch

Default behavior is unchanged. To expose the optional path from the existing app after
`app.state.engine` and `app.state.chat_model` are initialized:

```python
# app/main.py (integration only; not changed in this branch)
from app.multi_agent.api import create_multi_agent_router
from app.multi_agent.runtime import build_multi_agent_runtime

app.state.multi_agent = build_multi_agent_runtime(
    engine=app.state.engine,
    model=app.state.chat_model,
    limits=None,  # production defaults: 3 tasks, one replan, 45 s / 6000-token guard
)
app.include_router(create_multi_agent_router(app.state.multi_agent))
```

The endpoint is `POST /api/multi-agent/chat`. It returns the final reply, facts,
evidence IDs, task trace, errors, and `write_previews`. The existing
`POST /api/tickets/confirm` contract can consume each preview because the preview exposes
`conversation_id`, `description`, `ticket_type`, and `confirm_key`.

To run an isolated app instead:

```python
from app.multi_agent.api import create_multi_agent_app
app = create_multi_agent_app(runtime)
```

The caller must build `runtime` from the existing `ToolEngine` and chat model; tests pass
the same interface with fake read tools and a fake write sink.

Rollback: remove the optional router/runtime wiring above. No existing graph, config,
database schema, or default route depends on `app/multi_agent/`.

No change to `app/main.py`, `app/config.py`, or `app/graph/builder.py` is included in this
branch.
