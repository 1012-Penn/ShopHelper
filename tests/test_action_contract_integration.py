"""Application-level contracts for optional ticket action artifacts."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage
from sqlalchemy import select

from app.action_contracts import ActionArtifact, ActionSessionLedger
from app.models import Ticket


class _IntentModel:
    async def ainvoke(self, *_args, **_kwargs):
        return AIMessage(
            content=json.dumps({"intent": "其他", "confidence": 0.9}, ensure_ascii=False)
        )


@pytest.mark.asyncio
async def test_action_contract_runtime_is_default_off_and_opt_in(make_client):
    _, disabled_app = await make_client(_IntentModel())
    assert disabled_app.state.multi_agent.action_contracts_enabled is False

    _, enabled_app = await make_client(
        _IntentModel(),
        settings_overrides={"multi_agent_action_contracts_enabled": True},
    )
    assert enabled_app.state.multi_agent.action_contracts_enabled is True
    assert isinstance(enabled_app.state.action_ledger, ActionSessionLedger)


@pytest.mark.asyncio
async def test_multi_agent_http_response_exposes_action_artifacts(monkeypatch):
    from app.multi_agent import api as multi_api

    artifact = ActionArtifact(
        kind="ticket_preview",
        state="awaiting_confirmation",
        conversation_id=1,
        tool_name="create_ticket",
        confirm_key="contract-key-1",
    )

    class _Graph:
        async def ainvoke(self, _state):
            return {
                "final_reply": "预览已生成，等待确认。",
                "status": "success",
                "facts": {},
                "evidence": {},
                "errors": [],
                "stop_reasons": [],
                "trace": [],
                "write_previews": [artifact.preview or {}],
                "action_artifacts": [artifact.model_dump(mode="json")],
                "steps": 1,
                "tokens": 10,
            }

    monkeypatch.setattr(multi_api, "build_multi_agent_graph", lambda _runtime: _Graph())
    app = FastAPI()
    app.state.multi_agent = SimpleNamespace()
    app.include_router(multi_api.create_multi_agent_router())

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/api/multi-agent/chat",
            json={"message": "请预览售后工单", "session_id": 1},
        )

    assert response.status_code == 200, response.text
    assert response.json()["action_artifacts"][0]["state"] == "awaiting_confirmation"


@pytest.mark.asyncio
async def test_client_confirmation_updates_action_ledger_after_write(make_client):
    client, app = await make_client(
        _IntentModel(),
        settings_overrides={"multi_agent_action_contracts_enabled": True},
    )
    await app.state.store.resolve(777)
    ledger = ActionSessionLedger()
    app.state.action_ledger = ledger
    artifact = ledger.put_pending(
        ActionArtifact(
            kind="ticket_preview",
            state="awaiting_confirmation",
            conversation_id=777,
            tool_name="create_ticket",
            arguments={
                "conversation_id": 777,
                "description": "订单1001售后处理",
                "ticket_type": "售后",
            },
            confirm_key="confirm-action-1",
        )
    )[-1]

    response = await client.post(
        "/api/tickets/confirm",
        json={
            "conversation_id": 777,
            "tool_name": "create_ticket",
            "description": artifact.arguments["description"],
            "ticket_type": artifact.arguments["ticket_type"],
            "confirm_key": artifact.confirm_key,
        },
    )

    assert response.status_code == 200, response.text
    assert ledger.pending(777) is None
    confirmed = ledger.events()[-1]
    assert confirmed.state == "confirmed"
    assert confirmed.write_committed is True
    assert confirmed.ticket_no == response.json()["ticket_no"]


@pytest.mark.asyncio
async def test_client_cancel_updates_action_ledger_without_write(make_client):
    client, app = await make_client(
        _IntentModel(),
        settings_overrides={"multi_agent_action_contracts_enabled": True},
    )
    await app.state.store.resolve(778)
    ledger = ActionSessionLedger()
    app.state.action_ledger = ledger
    ledger.put_pending(
        ActionArtifact(
            kind="ticket_preview",
            state="awaiting_confirmation",
            conversation_id=778,
            tool_name="create_ticket",
            confirm_key="cancel-action-1",
        )
    )

    response = await client.post(
        "/api/tickets/cancel",
        json={"conversation_id": 778, "reason": "用户取消"},
    )

    assert response.status_code == 200, response.text
    assert ledger.pending(778) is None
    cancelled = ledger.events()[-1]
    assert cancelled.state == "cancelled"
    assert cancelled.write_committed is False


@pytest.mark.asyncio
async def test_cancelled_preview_key_cannot_be_replayed_into_a_ticket(make_client):
    client, app = await make_client(
        _IntentModel(),
        settings_overrides={"multi_agent_action_contracts_enabled": True},
    )
    await app.state.store.resolve(779)
    ledger = ActionSessionLedger()
    app.state.action_ledger = ledger
    preview = ledger.put_pending(
        ActionArtifact(
            kind="ticket_preview",
            state="awaiting_confirmation",
            conversation_id=779,
            tool_name="create_ticket",
            arguments={
                "conversation_id": 779,
                "description": "订单1001售后处理",
                "ticket_type": "售后",
            },
            confirm_key="cancelled-action-1",
        )
    )[-1]

    cancelled = await client.post(
        "/api/tickets/cancel",
        json={"conversation_id": 779, "reason": "用户取消"},
    )
    replay = await client.post(
        "/api/tickets/confirm",
        json={
            "conversation_id": 779,
            "tool_name": "create_ticket",
            "description": preview.arguments["description"],
            "ticket_type": preview.arguments["ticket_type"],
            "confirm_key": preview.confirm_key,
        },
    )

    assert cancelled.status_code == 200
    assert replay.status_code == 409
    with app.state.session_factory() as session:
        assert session.scalars(select(Ticket)).all() == []


@pytest.mark.asyncio
async def test_confirmation_fields_must_match_the_registered_preview(make_client):
    client, app = await make_client(
        _IntentModel(),
        settings_overrides={"multi_agent_action_contracts_enabled": True},
    )
    await app.state.store.resolve(780)
    ledger = ActionSessionLedger()
    app.state.action_ledger = ledger
    ledger.put_pending(
        ActionArtifact(
            kind="ticket_preview",
            state="awaiting_confirmation",
            conversation_id=780,
            tool_name="create_ticket",
            arguments={
                "conversation_id": 780,
                "description": "订单1001售后处理",
                "ticket_type": "售后",
            },
            confirm_key="tampered-action-1",
        )
    )

    response = await client.post(
        "/api/tickets/confirm",
        json={
            "conversation_id": 780,
            "tool_name": "create_ticket",
            "description": "被篡改的描述",
            "ticket_type": "投诉",
            "confirm_key": "tampered-action-1",
        },
    )

    assert response.status_code == 409
    assert ledger.pending(780) is not None
    with app.state.session_factory() as session:
        assert session.scalars(select(Ticket)).all() == []
