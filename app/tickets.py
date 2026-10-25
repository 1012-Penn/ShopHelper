"""ch12 工单幂等写入(ADR-3):同键命中必须核对「同一次提交」,不一致即冲突。

一次用户确认 = 一个 confirm_key;同键重复提交(双击/超时重试)只有在
会话、描述、工单类型三者都一致时才算重放,返回同一工单并标 deduplicated。
三者任一不一致说明是另一笔提交复用了键(或前端串键),必须拒绝(HTTP 409),
不得返回旧工单——否则会把 A 会话/内容的工单当成 B 的创建结果显示给用户。
预查命中与唯一约束冲突后的回查走同一个 `_admit` 比较,规则不可能分叉。
"""
from __future__ import annotations

import random
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.models import Ticket

CONFIRM_KEY_CONFLICT = "confirm_key_conflict"


class ConfirmKeyConflict(RuntimeError):
    """同一确认键的会话/描述/工单类型与已建工单不一致。"""


def new_ticket_no() -> str:
    return f"T{datetime.now().strftime('%Y%m%d')}{random.randint(0, 999):03d}"


def find_by_key(session, key: str):
    """按键回查;单独成函数便于测试模拟「预查未命中」的并发交错。"""
    return session.execute(select(Ticket).where(Ticket.confirm_key == key)).scalar_one_or_none()


def _same_submission(existing: Ticket, *, conversation_id: int, description: str,
                     ticket_type: str) -> bool:
    return (
        int(existing.conversation_id) == int(conversation_id)
        and (existing.description or "") == (description or "")
        and (existing.ticket_type or "") == (ticket_type or "")
    )


def _admit(existing: Ticket, *, conversation_id: int, description: str,
           ticket_type: str) -> dict:
    if not _same_submission(existing, conversation_id=conversation_id,
                            description=description, ticket_type=ticket_type):
        raise ConfirmKeyConflict(
            "同一确认键对应的会话/描述/工单类型与已建工单不一致,拒绝返回旧工单")
    return {"ticket_no": existing.ticket_no, "status": existing.status or "待处理",
            "deduplicated": True}


def create_ticket_idempotent(session_factory, *, conversation_id: int, description: str,
                             ticket_type: str, confirm_key: str | None = None) -> dict:
    """建单;带键时按键幂等。无键保持旧行为(不参与去重,向后兼容)。"""
    key = (confirm_key or "").strip() or None
    ticket_no = new_ticket_no()
    with session_factory() as session:
        if key is not None:
            existing = find_by_key(session, key)
            if existing is not None:
                return _admit(existing, conversation_id=conversation_id,
                              description=description, ticket_type=ticket_type)
        session.add(Ticket(
            ticket_no=ticket_no,
            conversation_id=conversation_id,
            description=description,
            ticket_type=ticket_type,
            confirm_key=key,
        ))
        try:
            session.commit()
        except IntegrityError:
            # 并发同键:唯一约束兜底;回查后仍走同一个 _admit 一致性规则
            if key is None:
                raise
            session.rollback()
            existing = find_by_key(session, key)
            if existing is None:
                raise
            return _admit(existing, conversation_id=conversation_id,
                          description=description, ticket_type=ticket_type)
    return {"ticket_no": ticket_no, "status": "待处理"}
