"""可靠执行所需 ORM；DDL 见 db/ch11-reliability-execution.sql。"""
from datetime import datetime

from sqlalchemy import (JSON, BigInteger, DateTime, ForeignKey, Integer, String, Text,
                        UniqueConstraint, func)
from sqlalchemy.orm import Mapped, mapped_column

from app.orm import Base

_BIGINT = BigInteger().with_variant(Integer, "sqlite")


class ChatRun(Base):
    __tablename__ = "chat_runs"
    __table_args__ = (
        UniqueConstraint("conversation_id", "client_request_id",
                         name="uk_chat_run_client_request"),
    )

    run_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    conversation_id: Mapped[int] = mapped_column(_BIGINT, nullable=False, index=True)
    client_request_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    request_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="queued")
    result_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    write_state: Mapped[str] = mapped_column(String(24), nullable=False, default="none")
    owner_token: Mapped[str] = mapped_column(String(64), nullable=False)
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    last_event_seq: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class ChatRunEvent(Base):
    __tablename__ = "chat_run_events"
    __table_args__ = (
        UniqueConstraint("run_id", "seq", name="uk_chat_run_event_seq"),
    )

    id: Mapped[int] = mapped_column(_BIGINT, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("chat_runs.run_id", ondelete="CASCADE"), nullable=False
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class ChatSessionLease(Base):
    __tablename__ = "chat_session_leases"

    conversation_id: Mapped[int] = mapped_column(_BIGINT, primary_key=True)
    run_id: Mapped[str] = mapped_column(String(36), nullable=False, unique=True)
    owner_token: Mapped[str] = mapped_column(String(64), nullable=False)
    lease_expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class ChatToolAttempt(Base):
    __tablename__ = "chat_tool_attempts"
    __table_args__ = (
        UniqueConstraint("run_id", "tool_call_id", name="uk_chat_tool_call"),
    )

    id: Mapped[int] = mapped_column(_BIGINT, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    tool_call_id: Mapped[str] = mapped_column(String(64), nullable=False)
    tool_name: Mapped[str] = mapped_column(String(128), nullable=False)
    access: Mapped[str] = mapped_column(String(8), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    idempotency_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    arguments: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    result_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
