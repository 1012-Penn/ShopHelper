"""ch12 工单确认幂等测试(ADR-3):「一次用户确认」稳定请求键 + DB 唯一约束。

场景:并发/双击同键只建一单、超时后同键重试返回同一工单、同内容两次独立确认建两单、
无键旧路径行为不变;另验证唯一约束在真实并发(文件库双连接)下兜底,以及预览帧下发确认键。
不依赖 tool_call_id,不用参数指纹合并合法提交。
"""
import asyncio
import json
import threading
import uuid

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from app.models import Base, Ticket
from tests.helpers import fake_chat

KEY_A = "confirm-key-aaaaaaaa"
KEY_B = "confirm-key-bbbbbbbb"


async def post_ticket(client, payload: dict):
    response = await client.post("/api/tickets", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


@pytest.fixture
async def client_app(make_client):
    client, app = await make_client(fake_chat("您好"))
    return client, app


@pytest.mark.asyncio
async def test_double_click_same_key_returns_same_ticket(client_app):
    client, app = client_app
    payload = {"description": "售后问题", "ticket_type": "售后", "confirm_key": KEY_A}
    first = await post_ticket(client, payload)
    second = await post_ticket(client, payload)  # 双击的第二下 / 客户端超时重试
    assert first["ticket_no"] == second["ticket_no"]
    assert second.get("deduplicated") is True
    with app.state.session_factory() as session:
        assert session.query(Ticket).count() == 1
        assert session.query(Ticket).one().confirm_key == KEY_A


@pytest.mark.asyncio
async def test_same_content_two_confirmations_create_two_tickets(client_app):
    client, app = client_app
    first = await post_ticket(client, {"description": "同样的描述", "ticket_type": "售后",
                                       "confirm_key": KEY_A})
    second = await post_ticket(client, {"description": "同样的描述", "ticket_type": "售后",
                                        "confirm_key": KEY_B})
    assert first["ticket_no"] != second["ticket_no"]  # 两次合法确认,不按参数指纹合并
    with app.state.session_factory() as session:
        assert session.query(Ticket).count() == 2


@pytest.mark.asyncio
async def test_legacy_requests_without_key_unchanged(client_app):
    client, app = client_app
    first = await post_ticket(client, {"description": "旧客户端", "ticket_type": "投诉"})
    second = await post_ticket(client, {"description": "旧客户端", "ticket_type": "投诉"})
    assert first["ticket_no"] != second["ticket_no"]
    with app.state.session_factory() as session:
        assert session.query(Ticket).count() == 2  # 无键=不参与去重(旧行为)


@pytest.mark.asyncio
async def test_confirm_endpoint_same_key_returns_same_ticket(client_app):
    client, app = client_app
    bootstrap = await post_ticket(client, {"description": "建会话", "ticket_type": "咨询"})
    with app.state.session_factory() as session:
        from app.models import Conversation

        conversation_id = session.query(Conversation.id).order_by(Conversation.id).first()[0]
    payload = {"conversation_id": conversation_id, "description": "确认流建单",
               "ticket_type": "售后", "confirm_key": KEY_A}
    first = (await client.post("/api/tickets/confirm", json=payload)).json()
    retry = (await client.post("/api/tickets/confirm", json=payload)).json()  # 超时后重试
    assert first["ticket_no"] == retry["ticket_no"] and retry.get("deduplicated") is True
    with app.state.session_factory() as session:
        assert session.query(Ticket).filter_by(confirm_key=KEY_A).count() == 1


@pytest.mark.asyncio
async def test_first_response_lost_then_replay_same_key(client_app):
    """首请求已建单但响应未达客户端(超时),客户端同键重放:同一工单、不重复建单。"""
    client, app = client_app
    payload = {"description": "重放", "ticket_type": "售后", "confirm_key": KEY_A}
    first = await post_ticket(client, payload)
    replay = await post_ticket(client, payload)
    assert first["ticket_no"] == replay["ticket_no"]
    assert replay.get("deduplicated") is True
    with app.state.session_factory() as session:
        assert session.query(Ticket).filter_by(confirm_key=KEY_A).count() == 1


def test_unique_constraint_guards_interleaved_checks(tmp_path):
    """确定性复现「两个请求都先查无、再各自插入」的交错(即并发双击的真实时间线):
    唯一约束只放行一个,输家拿 IntegrityError 后回查赢家 —— 这正是 create_ticket 的兜底路径。"""
    engine = create_engine(f"sqlite:///{tmp_path / 'tickets.db'}",
                           connect_args={"timeout": 30})
    Base.metadata.create_all(engine, tables=[Ticket.__table__])
    factory = sessionmaker(bind=engine, expire_on_commit=False)

    def precheck() -> None:
        with factory() as session:  # 两个请求在插入前都看到「无同键工单」
            assert session.execute(
                select(Ticket).where(Ticket.confirm_key == KEY_A)).scalar_one_or_none() is None

    precheck()  # 请求 A 的预查
    precheck()  # 请求 B 的预查

    with factory() as session:  # 请求 A 先提交
        session.add(Ticket(ticket_no="T20260101002", conversation_id=1,
                           description="并发", ticket_type="售后", confirm_key=KEY_A))
        session.commit()

    with factory() as session:  # 请求 B 后到:唯一约束拒绝
        session.add(Ticket(ticket_no="T20260101003", conversation_id=1,
                           description="并发", ticket_type="售后", confirm_key=KEY_A))
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()
    with factory() as session:  # 引擎同款回查:返回先建的那张
        winner = session.execute(
            select(Ticket).where(Ticket.confirm_key == KEY_A)).scalar_one()
        assert winner.ticket_no == "T20260101002"
    with factory() as session:
        assert session.query(Ticket).filter_by(confirm_key=KEY_A).count() == 1


def test_unique_constraint_holds_under_thread_race(tmp_path):
    """附:真线程竞争(非确定性,只断言「同键至多一单」这一不变量)。"""
    engine = create_engine(f"sqlite:///{tmp_path / 'race.db'}",
                           connect_args={"timeout": 30})
    Base.metadata.create_all(engine, tables=[Ticket.__table__])
    factory = sessionmaker(bind=engine, expire_on_commit=False)

    def attempt(no: str):
        try:
            with factory() as session:
                session.add(Ticket(ticket_no=no, conversation_id=1, description="竞争",
                                   ticket_type="售后", confirm_key=KEY_A))
                session.commit()
        except Exception:  # IntegrityError 或锁竞争都由不变量兜底
            try:
                with factory() as session:
                    session.execute(
                        select(Ticket).where(Ticket.confirm_key == KEY_A)).scalar_one_or_none()
            except Exception:
                pass

    threads = [threading.Thread(target=attempt, args=(f"T2026010101{n}",))
               for n in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    with factory() as session:
        assert session.query(Ticket).filter_by(confirm_key=KEY_A).count() <= 1


def test_preview_frame_carries_fresh_confirm_key():
    from app.graph.agent import _write_intercept

    frames: list[dict] = []
    state = {"session_id": 7}
    _write_intercept(frames.append, state)(None, {"ticket_type": "售后", "description": "x"})
    _write_intercept(frames.append, state)(None, {})
    assert frames[0]["type"] == "ticket_preview"
    assert len(frames[0]["confirm_key"]) == 32
    assert frames[1]["confirm_key"] != frames[0]["confirm_key"]  # 每次确认动作新键
