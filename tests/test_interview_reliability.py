"""CH11 可靠执行验收：真文件 SQLite、独立连接、真实子进程/SSE 边界。"""
import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import sessionmaker

from app.execution.models import ChatRun, ChatSessionLease
from app.execution.store import ExecutionStore, LeaseLost, RequestConflict, SessionBusy
from app.execution.tool_engine import ExecutionToolEngine, current_run_id
from app.context import build_layered
from app.history import estimate_tokens
from app.models import Base
from tests.helpers import GraphChatModel, post_chat_sse


def _factory(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'reliability.db'}",
        connect_args={"timeout": 30, "check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def test_same_client_request_id_returns_existing_run_without_second_lease(tmp_path):
    factory = _factory(tmp_path)
    store = ExecutionStore(factory)

    first = store.reserve_run(
        conversation_id=1,
        client_request_id="request-0001",
        request_payload={"message": "你好", "session_id": 1},
        owner_token="owner-a",
        lease_seconds=60,
    )
    second = store.reserve_run(
        conversation_id=1,
        client_request_id="request-0001",
        request_payload={"message": "你好", "session_id": 1},
        owner_token="owner-b",
        lease_seconds=60,
    )

    assert second.run_id == first.run_id
    assert first.created is True
    assert second.created is False
    with factory() as session:
        assert len(session.scalars(select(ChatRun)).all()) == 1
        assert len(session.scalars(select(ChatSessionLease)).all()) == 1


def test_events_have_replay_cursor_and_completed_run_records_usage(tmp_path):
    factory = _factory(tmp_path)
    store = ExecutionStore(factory)
    run = store.reserve_run(
        conversation_id=1,
        client_request_id="request-0002",
        request_payload={"message": "你好", "session_id": 1},
        owner_token="owner-a",
        lease_seconds=60,
    )

    first_id = store.append_event(run.run_id, {"type": "session", "session_id": 1})
    second_id = store.append_event(run.run_id, {"type": "token", "content": "答"})
    store.finish_run(
        run.run_id,
        status="completed",
        result={"answer": "答", "intent": "订单"},
        usage={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        duration_ms=12,
    )

    assert second_id > first_id
    assert [e.event["type"] for e in store.read_events(run.run_id)] == ["session", "token"]
    assert [e.event["type"] for e in store.read_events(run.run_id, after_id=first_id)] == ["token"]
    status = store.get_run(run.run_id)
    assert status["status"] == "completed"
    assert status["result"]["answer"] == "答"
    assert status["usage"] == {
        "input_tokens": 3,
        "output_tokens": 2,
        "total_tokens": 5,
    }
    assert status["duration_ms"] == 12
    with factory() as session:
        assert session.scalar(select(ChatSessionLease).where(
            ChatSessionLease.run_id == run.run_id)) is None


def test_reused_client_request_id_with_changed_payload_is_rejected(tmp_path):
    store = ExecutionStore(_factory(tmp_path))
    store.reserve_run(
        conversation_id=1,
        client_request_id="request-0003",
        request_payload={"message": "原来的问题", "session_id": 1},
        owner_token="owner-a",
        lease_seconds=60,
    )

    with pytest.raises(RequestConflict):
        store.reserve_run(
            conversation_id=1,
            client_request_id="request-0003",
            request_payload={"message": "被偷换的问题", "session_id": 1},
            owner_token="owner-b",
            lease_seconds=60,
        )


def test_same_session_second_request_conflicts_but_other_session_is_parallel(tmp_path):
    store = ExecutionStore(_factory(tmp_path))
    first = store.reserve_run(
        conversation_id=10,
        client_request_id="request-a",
        request_payload={"message": "第一问", "session_id": 10},
        owner_token="owner-a",
        lease_seconds=60,
    )

    with pytest.raises(SessionBusy) as conflict:
        store.reserve_run(
            conversation_id=10,
            client_request_id="request-b",
            request_payload={"message": "第二问", "session_id": 10},
            owner_token="owner-b",
            lease_seconds=60,
        )
    assert conflict.value.run_id == first.run_id

    other = store.reserve_run(
        conversation_id=11,
        client_request_id="request-c",
        request_payload={"message": "别的会话", "session_id": 11},
        owner_token="owner-b",
        lease_seconds=60,
    )
    assert other.created is True and other.run_id != first.run_id


def test_expired_lease_marks_old_run_interrupted_before_reusing_session(tmp_path):
    store = ExecutionStore(_factory(tmp_path))
    old = store.reserve_run(
        conversation_id=20,
        client_request_id="request-old",
        request_payload={"message": "崩溃前", "session_id": 20},
        owner_token="dead-owner",
        lease_seconds=-1,
    )

    new = store.reserve_run(
        conversation_id=20,
        client_request_id="request-new",
        request_payload={"message": "重启后", "session_id": 20},
        owner_token="new-owner",
        lease_seconds=60,
    )

    old_status = store.get_run(old.run_id)
    assert old_status["status"] == "interrupted"
    assert store.read_events(old.run_id)[-1].event == {
        "type": "interrupted",
        "reason": "lease_expired",
        "resume_safe": True,
    }
    assert new.run_id != old.run_id


def test_crash_after_write_start_is_marked_unknown_and_not_auto_resumable(tmp_path):
    store = ExecutionStore(_factory(tmp_path))
    old = store.reserve_run(
        conversation_id=30,
        client_request_id="request-write",
        request_payload={"message": "建工单", "session_id": 30},
        owner_token="dead-owner",
        lease_seconds=-1,
    )
    store.start_tool_attempt(
        old.run_id,
        tool_call_id="call-1",
        tool_name="create_ticket",
        access="write",
        arguments={"description": "模拟业务写入"},
        idempotency_key="confirm-key-0001",
    )

    store.recover_expired_lease(30)

    status = store.get_run(old.run_id)
    assert status["status"] == "interrupted"
    assert status["write_state"] == "unknown"
    interrupted = store.read_events(old.run_id)[-1].event
    assert interrupted == {
        "type": "interrupted",
        "reason": "write_outcome_unknown",
        "resume_safe": False,
    }


def test_only_current_owner_can_renew_session_lease(tmp_path):
    factory = _factory(tmp_path)
    store = ExecutionStore(factory)
    run = store.reserve_run(
        conversation_id=31,
        client_request_id="request-heartbeat",
        request_payload={"message": "长任务", "session_id": 31},
        owner_token="owner-a",
        lease_seconds=1,
    )
    with factory() as session:
        session.execute(update(ChatSessionLease).where(
            ChatSessionLease.run_id == run.run_id
        ).values(
            lease_expires_at=datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=2)
        ))
        session.commit()

    assert store.renew_lease(run.run_id, "wrong-owner", lease_seconds=10) is False
    assert store.renew_lease(run.run_id, "owner-a", lease_seconds=10) is True
    with pytest.raises(SessionBusy):
        store.reserve_run(
            conversation_id=31,
            client_request_id="request-other",
            request_payload={"message": "抢锁", "session_id": 31},
            owner_token="owner-b",
            lease_seconds=10,
        )


def test_stale_owner_cannot_append_or_finish_after_lease_takeover(tmp_path):
    store = ExecutionStore(_factory(tmp_path))
    old = store.reserve_run(
        conversation_id=34,
        client_request_id="stale-owner-old",
        request_payload={"message": "旧执行", "session_id": 34},
        owner_token="old-owner",
        lease_seconds=-1,
    )
    store.reserve_run(
        conversation_id=34,
        client_request_id="stale-owner-new",
        request_payload={"message": "新执行", "session_id": 34},
        owner_token="new-owner",
        lease_seconds=60,
    )

    with pytest.raises(LeaseLost):
        store.append_event(old.run_id, {"type": "token", "content": "迟到"},
                           owner_token="old-owner")
    with pytest.raises(LeaseLost):
        store.finish_run(old.run_id, status="completed", owner_token="old-owner")


async def test_tool_wrapper_records_read_and_keeps_intercepted_write_safe(make_client):
    _, app = await make_client(GraphChatModel('{"intent": "订单"}'))
    session_id = await app.state.store.resolve(32)
    run = app.state.execution_store.reserve_run(
        conversation_id=session_id,
        client_request_id="request-tool-wrapper",
        request_payload={"message": "工具", "session_id": session_id},
        owner_token="owner-a",
        lease_seconds=60,
    )
    engine = ExecutionToolEngine(app.state.engine, app.state.execution_store)
    token = current_run_id.set(run.run_id)
    try:
        preview_text = await engine.execute(
            "create_ticket",
            '{"conversation_id": 32, "description": "待确认", "ticket_type": "售后"}',
            conversation_id=session_id,
            tool_call_id="write-call-1",
        )
        assert "确认" in preview_text
        order_text = await engine.execute(
            "query_order",
            '{"order_id": "1001"}',
            conversation_id=session_id,
            tool_call_id="read-call-1",
        )
        assert "1001" in order_text
    finally:
        current_run_id.reset(token)

    status = app.state.execution_store.get_run(run.run_id)
    assert status["write_state"] == "none"
    with app.state.execution_store._factory() as session:
        from app.execution.models import ChatToolAttempt

        attempts = session.scalars(
            select(ChatToolAttempt).order_by(ChatToolAttempt.id)).all()
        assert [(a.tool_name, a.status) for a in attempts] == [
            ("create_ticket", "intercepted"),
            ("query_order", "succeeded"),
        ]


async def test_graph_tool_call_lands_attempt_under_its_run(make_client):
    model = GraphChatModel('{"intent": "订单"}', [
        ("tools", [{
            "name": "query_order",
            "args": '{"order_id": "1001"}',
            "id": "graph-call-1",
            "index": 0,
        }]),
        ("text", "订单 1001 已查询"),
    ])
    client, app = await make_client(model)
    events = await post_chat_sse(client, {
        "message": "查订单1001",
        "session_id": 33,
        "client_request_id": "request-graph-tool-1",
    })
    run_id = events[0]["run_id"]
    with app.state.execution_store._factory() as session:
        from app.execution.models import ChatToolAttempt

        attempt = session.scalars(
            select(ChatToolAttempt).where(ChatToolAttempt.run_id == run_id)).one()
        assert attempt.tool_name == "query_order"
        assert attempt.tool_call_id == "graph-call-1"
        assert attempt.status == "succeeded"


def test_summary_projection_is_counted_in_sliding_window_and_recent_revision_survives():
    summary = "早期地址是上海市。\n" + "历史摘要" * 200
    original_summary = summary
    l2_rows = [
        {"id": 1, "role": "tool", "content": "旧工具原文:订单已签收于去年"},
        {"id": 2, "role": "user", "content": "旧问题" * 80},
    ]
    l1_rows = [
        {"id": 3, "role": "user", "content": "我改口了,收货地址改成杭州市。"},
        {"id": 4, "role": "assistant", "content": "已按杭州市处理。"},
    ]
    sliding = 140

    layered = build_layered(l1_rows, l2_rows, summary, head_chars=60, sliding=sliding)
    injected = (
        estimate_tokens(layered["summary_text"])
        + sum(estimate_tokens(message.content or "") for message in layered["layer2_msgs"])
        + sum(estimate_tokens(message.content or "") for message in layered["layer1_msgs"])
    )

    assert summary == original_summary
    assert layered["summary_truncated"] is True
    assert layered["total_injected_tokens"] <= sliding
    assert injected <= sliding
    assert "杭州市" in layered["history_text"]
    assert "旧工具原文" not in layered["history_text"]


def test_multiple_compression_segments_are_only_projected_not_mutated():
    segments = [
        "第1段:用户地址在上海。" + "旧事实" * 80,
        "第2段:用户要求改到杭州。" + "补充" * 80,
        "第3段:客服已确认杭州。" + "结果" * 80,
    ]
    summary = "\n".join(segments)
    original = summary
    l1 = [{"id": 9, "role": "user", "content": "最终回复仍以杭州为准"}]

    layered = build_layered(l1, [], summary, head_chars=60, sliding=80)

    assert summary == original
    assert layered["summary_truncated"] is True
    assert layered["tokens_summary"] < layered["tokens_summary_full"]
    assert layered["total_injected_tokens"] <= 80
    assert "最终回复仍以杭州为准" in layered["history_text"]


def test_tight_budget_drops_stale_tool_and_old_address_keeps_latest_revision():
    summary = "历史地址:北京市。" + "旧摘要" * 50
    rows = [
        {"id": 1, "role": "tool", "content": "过期工具结果:地址北京市"},
        {"id": 2, "role": "user", "content": "旧地址是北京市。" * 20},
        {"id": 3, "role": "user", "content": "我改口,地址改成杭州市。"},
    ]
    sliding = estimate_tokens("我改口,地址改成杭州市。")

    layered = build_layered([], rows, summary, head_chars=60, sliding=sliding)
    injected_rows = [message.content for message in layered["layer2_msgs"]]

    assert injected_rows == ["我改口,地址改成杭州市。"]
    assert "杭州市" in layered["history_text"]
    assert "过期工具结果" not in layered["history_text"]
    assert layered["summary_text"] == ""


async def test_rag_step_store_is_wired_to_application_session_factory(make_client):
    client, app = await make_client(GraphChatModel('{"intent": "商品咨询"}'))
    assert app.state.rag_steps_store._factory is app.state.session_factory
    session_id = await app.state.store.resolve(410)
    app.state.rag_steps_store.insert_step(
        session_id=session_id,
        turn_user_msg_id="turn-rag-1",
        step={
            "round_index": 0,
            "phase": "retrieve",
            "query": "退货政策",
            "queries": ["退货政策"],
            "evidence_ids": [],
            "merged_evidence_ids": [],
            "latency_ms": 1,
        },
    )
    app.state.rag_steps_store.finalize_stop(session_id, "turn-rag-1", "first_pass_sufficient")
    with app.state.session_factory() as session:
        from app.models import RagLoopStep

        row = session.scalars(select(RagLoopStep)).one()
        assert row.conversation_id == session_id
        assert row.stop_reason == "first_pass_sufficient"


async def test_controlled_rag_graph_persists_steps_via_application_factory(make_client):
    from tests.helpers import seed_kb_chunk

    client, app = await make_client(
        GraphChatModel('{"intent": "商品咨询"}', [("text", "支持七天无理由退货。")]),
        settings_overrides={"retrieval_route": "controlled_loop"},
    )
    seed_kb_chunk(
        app,
        "knowledge/退货政策.md",
        "退货政策\n七天无理由",
        "支持七天无理由退货。",
        "售后政策",
    )

    events = await post_chat_sse(client, {
        "message": "退货政策",
        "session_id": 411,
        "client_request_id": "request-controlled-rag",
    })
    assert events[-1]["type"] == "done"
    with app.state.session_factory() as session:
        from app.models import RagLoopStep

        rows = session.scalars(select(RagLoopStep)).all()
        assert rows
        assert all(row.conversation_id == 411 for row in rows)
        assert any(row.stop_reason for row in rows)


async def test_sse_replay_does_not_resubmit_confirmed_write_and_confirm_key_dedupes(
        make_client):
    model = GraphChatModel('{"intent": "其他"}', [
        ("tools", [{
            "name": "create_ticket",
            "args": '{"conversation_id": 412, "description": "网络异常", "ticket_type": "售后"}',
            "id": "ticket-preview-call",
            "index": 0,
        }]),
        ("text", "请确认建单。"),
    ])
    client, app = await make_client(model)
    events = await post_chat_sse(client, {
        "message": "帮我建工单",
        "session_id": 412,
        "client_request_id": "request-write-replay",
    })
    preview = next(event for event in events if event["type"] == "ticket_preview")
    run_id = events[0]["run_id"]
    for _ in range(2):
        replay = await client.get(f"/api/chat/runs/{run_id}/events")
        assert any(event["type"] == "ticket_preview"
                   for _, event in _sse_ids_and_events(replay.text))
    with app.state.session_factory() as session:
        from app.models import Ticket

        assert session.query(Ticket).count() == 0

    payload = {
        "conversation_id": 412,
        "description": "网络异常",
        "ticket_type": "售后",
        "confirm_key": preview["confirm_key"],
    }
    first = (await client.post("/api/tickets/confirm", json=payload)).json()
    retry = (await client.post("/api/tickets/confirm", json=payload)).json()
    assert first["ticket_no"] == retry["ticket_no"]
    assert retry["deduplicated"] is True
    with app.state.session_factory() as session:
        from app.models import Ticket

        assert session.query(Ticket).count() == 1


async def test_chat_exposes_persistent_run_status_and_usage(make_client):
    client, app = await make_client(GraphChatModel('{"intent": "订单"}', [("text", "答")]))

    events = await post_chat_sse(client, {
        "message": "问",
        "session_id": 401,
        "client_request_id": "request-api-0001",
    })

    assert events[0]["type"] == "session"
    run_id = events[0]["run_id"]
    status = (await client.get(f"/api/chat/runs/{run_id}")).json()
    assert status["status"] == "completed"
    assert status["conversation_id"] == 401
    assert status["client_request_id"] == "request-api-0001"
    assert status["result"]["answer"] == "答"
    assert status["usage"]["total_tokens"] >= 0
    assert status["duration_ms"] >= 0
    assert status["started_at"] is not None
    assert status["finished_at"] is not None


async def test_same_client_request_id_replays_without_second_model_execution(make_client):
    model = GraphChatModel('{"intent": "订单"}', [("text", "只执行一次")])
    client, _ = await make_client(model)
    payload = {
        "message": "同一请求",
        "session_id": 402,
        "client_request_id": "request-api-0002",
    }

    first = await post_chat_sse(client, payload)
    second = await post_chat_sse(client, payload)

    assert first[0]["run_id"] == second[0]["run_id"]
    assert [e["type"] for e in first] == [e["type"] for e in second]
    assert "".join(e.get("content", "") for e in second if e["type"] == "token") == "只执行一次"
    assert len(model.prompts) == 1


def _sse_ids_and_events(raw: str):
    rows = []
    for block in raw.split("\n\n"):
        event_id = next((line[4:] for line in block.splitlines()
                         if line.startswith("id: ")), None)
        data = next((line[len("data: "):] for line in block.splitlines()
                     if line.startswith("data: ")), None)
        if event_id is not None and data is not None:
            import json
            rows.append((int(event_id), json.loads(data)))
    return rows


async def test_event_replay_accepts_last_event_id_cursor(make_client):
    client, app = await make_client(GraphChatModel(
        '{"intent": "订单"}', [("text", "第一段|第二段")]))
    events = await post_chat_sse(client, {"message": "游标", "session_id": 403})
    run_id = events[0]["run_id"]

    full = await client.get(f"/api/chat/runs/{run_id}/events")
    rows = _sse_ids_and_events(full.text)
    assert [event["type"] for _, event in rows] == [e["type"] for e in events]
    cursor = rows[0][0]

    resumed = await client.get(
        f"/api/chat/runs/{run_id}/events",
        headers={"Last-Event-ID": str(cursor)},
    )
    resumed_rows = _sse_ids_and_events(resumed.text)
    assert [event for _, event in resumed_rows] == [event for _, event in rows[1:]]


async def test_client_disconnect_does_not_cancel_run_or_repeat_model_execution(make_client):
    started = asyncio.Event()
    release = asyncio.Event()
    model = GraphChatModel(
        '{"intent": "订单"}', [("text", "断连后完成")], started=started, release=release)
    client, app = await make_client(model)
    payload = {
        "message": "断连测试",
        "session_id": 404,
        "client_request_id": "request-disconnect-01",
    }

    request_task = asyncio.create_task(post_chat_sse(client, payload))
    await asyncio.wait_for(started.wait(), timeout=1.0)
    request_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request_task
    release.set()

    run_id = None
    with app.state.execution_store._factory() as session:
        run = session.scalars(select(ChatRun).where(ChatRun.conversation_id == 404)).one()
        run_id = run.run_id
    status = await app.state.execution_service.wait_terminal(run_id)
    assert status["status"] == "completed"

    replay = await client.get(f"/api/chat/runs/{run_id}/events")
    events = [event for _, event in _sse_ids_and_events(replay.text)]
    assert "".join(e.get("content", "") for e in events if e["type"] == "token") == "断连后完成"
    assert len(model.prompts) == 1
    history = await app.state.store.get_history(404)
    assert [row["role"] for row in history] == ["user", "assistant"]


async def test_same_session_different_request_returns_409_while_run_active(make_client):
    started = asyncio.Event()
    release = asyncio.Event()
    model = GraphChatModel(
        '{"intent": "订单"}', [("text", "首轮完成")], started=started, release=release)
    client, _ = await make_client(model)
    first_task = asyncio.create_task(post_chat_sse(client, {
        "message": "首轮",
        "session_id": 405,
        "client_request_id": "request-session-a1",
    }))
    await asyncio.wait_for(started.wait(), timeout=1.0)

    conflict = await client.post("/api/chat", json={
        "message": "冲突请求",
        "session_id": 405,
        "client_request_id": "request-session-b1",
    })
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["message"] == "会话已有执行中的 run"

    release.set()
    events = await first_task
    assert events[-1]["type"] == "done"
    assert len(model.prompts) == 1


async def test_failed_run_records_error_usage_duration_and_no_chat_messages(make_client):
    class Exploding(GraphChatModel):
        async def astream(self, messages, config=None, **kwargs):
            self.prompts.append(list(messages))
            yield type("Chunk", (), {"text": "部分", "tool_call_chunks": []})()
            raise RuntimeError("故障注入")

    client, app = await make_client(Exploding('{"intent": "订单"}'))
    events = await post_chat_sse(client, {
        "message": "失败",
        "session_id": 406,
        "client_request_id": "request-failure-01",
    })
    assert events[-1]["type"] == "error"
    assert "故障注入" in events[-1]["message"]
    status = (await client.get(f"/api/chat/runs/{events[0]['run_id']}")).json()
    assert status["status"] == "failed"
    assert status["duration_ms"] >= 0
    assert status["usage"]["total_tokens"] >= 0
    assert await app.state.store.get_history(406) == []
    with app.state.session_factory() as session:
        from app.models import RequestUsage

        usage_rows = session.scalars(select(RequestUsage)).all()
        assert len(usage_rows) == 1
        assert usage_rows[0].conversation_id == 406


async def test_expired_read_only_run_can_be_explicitly_resumed(make_client):
    client, app = await make_client(GraphChatModel(
        '{"intent": "订单"}', [("text", "恢复成功")]))
    session_id = await app.state.store.resolve(407)
    old = app.state.execution_store.reserve_run(
        conversation_id=session_id,
        client_request_id="request-resume-read",
        request_payload={"message": "恢复只读", "session_id": session_id, "resume": None},
        owner_token="dead-owner",
        lease_seconds=-1,
    )
    app.state.execution_store.append_event(old.run_id, {
        "type": "session", "session_id": session_id, "run_id": old.run_id})

    response = await client.post(f"/api/chat/runs/{old.run_id}/resume")
    assert response.status_code == 200, response.text
    events = [event for _, event in _sse_ids_and_events(response.text)]
    assert events[-1]["type"] == "done"
    status = (await client.get(f"/api/chat/runs/{old.run_id}")).json()
    assert status["status"] == "completed"
    assert status["attempt"] == 2


async def test_unknown_write_outcome_cannot_be_resumed(make_client):
    client, app = await make_client(GraphChatModel('{"intent": "订单"}'))
    session_id = await app.state.store.resolve(408)
    old = app.state.execution_store.reserve_run(
        conversation_id=session_id,
        client_request_id="request-resume-write",
        request_payload={"message": "写入", "session_id": session_id, "resume": None},
        owner_token="dead-owner",
        lease_seconds=-1,
    )
    app.state.execution_store.start_tool_attempt(
        old.run_id,
        tool_call_id="write-call",
        tool_name="create_ticket",
        access="write",
        arguments={"description": "unknown"},
        idempotency_key="confirm-key-unknown",
    )
    app.state.execution_store.recover_expired_lease(session_id)

    response = await client.post(f"/api/chat/runs/{old.run_id}/resume")
    assert response.status_code == 409
    assert "禁止自动恢复" in response.json()["detail"]
    assert (await client.get(f"/api/chat/runs/{old.run_id}")).json()["status"] == "interrupted"


async def test_active_run_cancel_is_best_effort_and_persisted(make_client):
    started = asyncio.Event()
    release = asyncio.Event()
    client, app = await make_client(GraphChatModel(
        '{"intent": "订单"}', [("text", "不应完成")], started=started, release=release))
    request_task = asyncio.create_task(post_chat_sse(client, {
        "message": "取消",
        "session_id": 409,
        "client_request_id": "request-cancel-01",
    }))
    await asyncio.wait_for(started.wait(), timeout=1.0)
    with app.state.execution_store._factory() as session:
        run_id = session.scalars(select(ChatRun).where(ChatRun.conversation_id == 409)).one().run_id

    cancelled = await client.post(f"/api/chat/runs/{run_id}/cancel")
    assert cancelled.status_code == 202
    events = await request_task
    assert events[-1]["type"] == "error"
    status = (await client.get(f"/api/chat/runs/{run_id}")).json()
    assert status["status"] == "cancelled"
