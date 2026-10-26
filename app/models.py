"""ORM 模型,字段与 db/init.sql 一一对应。"""
from datetime import datetime

from sqlalchemy import (JSON, BigInteger, Boolean, DateTime, Enum, ForeignKey, Integer,
                        String, Text, UniqueConstraint, func)
from sqlalchemy.orm import Mapped, mapped_column

from app.orm import Base


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    )
    user_id: Mapped[str] = mapped_column(String(64), default="guest")
    status: Mapped[str] = mapped_column(
        Enum("进行中", "已转人工", "已结束", name="conv_status"), default="进行中"
    )
    # ch07 三列:summary 是分段梗概的拼接投影,两个锚点 id 划出三层边界(降级只挪 id、不搬数据)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary_upto_msg_id: Mapped[int | None] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), nullable=True
    )
    layer1_from_msg_id: Mapped[int | None] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class ConversationSummary(Base):
    """ch07 分段摘要,一段一行只追加:压完不回炉,一个事实只经历一次有损压缩。"""

    __tablename__ = "conversation_summaries"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    )
    conversation_id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), nullable=False
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    from_msg_id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), nullable=False
    )
    upto_msg_id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    __table_args__ = (UniqueConstraint("conversation_id", "seq", name="uk_conv_seq"),)


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    )
    conversation_id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), nullable=False
    )
    role: Mapped[str] = mapped_column(Enum("user", "assistant", "tool", name="msg_role"), nullable=False)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    tool_calls: Mapped[list | None] = mapped_column(JSON, nullable=True)
    tool_call_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Faq(Base):
    __tablename__ = "faq"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    )
    question: Mapped[str] = mapped_column(String(512), nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class Ticket(Base):
    __tablename__ = "tickets"

    ticket_no: Mapped[str] = mapped_column(String(32), primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), nullable=False
    )
    # ch12 幂等键(ADR-3):代表「一次用户确认」;唯一约束下同键重复提交返回同一工单。
    # 可空=旧客户端/旧路径无键,不参与去重。
    confirm_key: Mapped[str | None] = mapped_column(String(64), nullable=True, unique=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    ticket_type: Mapped[str] = mapped_column(Enum("售后", "投诉", "咨询", name="ticket_type"), nullable=False)
    status: Mapped[str] = mapped_column(Enum("待处理", "已处理", name="ticket_status"), default="待处理")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class RagLoopStep(Base):
    """ch12 受控 Agentic 检索步骤审计。

    仅可审计(ADR-1):事后查询每轮查询/子问题/证据ID/判定/停止原因;
    不提供断点续跑、回放或跨进程恢复——需要恢复能力时须另立设计并补杀进程恢复测试。
    """

    __tablename__ = "rag_loop_steps"
    __table_args__ = (
        UniqueConstraint("conversation_id", "turn_user_msg_id", "round_index", "phase",
                         name="uk_rag_step_round"),
    )

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    )
    conversation_id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), nullable=False
    )
    turn_user_msg_id: Mapped[str] = mapped_column(String(64), nullable=False)
    round_index: Mapped[int] = mapped_column(Integer, nullable=False)
    phase: Mapped[str] = mapped_column(String(16), nullable=False)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    queries: Mapped[list | None] = mapped_column(JSON, nullable=True)
    sub_queries: Mapped[list | None] = mapped_column(JSON, nullable=True)
    evidence_ids: Mapped[list | None] = mapped_column(JSON, nullable=True)
    new_evidence_ids: Mapped[list | None] = mapped_column(JSON, nullable=True)
    merged_evidence_ids: Mapped[list | None] = mapped_column(JSON, nullable=True)
    judge_answered: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    analyzer_action: Mapped[str | None] = mapped_column(String(24), nullable=True)
    conflicts: Mapped[list | None] = mapped_column(JSON, nullable=True)
    missing_aspects: Mapped[list | None] = mapped_column(JSON, nullable=True)
    stop_reason: Mapped[str | None] = mapped_column(String(32), nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"
    __table_args__ = (UniqueConstraint("review_id", name="uk_knowledge_review"),)

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    )
    review_id: Mapped[int | None] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        ForeignKey("review_queue.id", ondelete="SET NULL"), nullable=True
    )
    category: Mapped[str] = mapped_column(String(255), nullable=False)
    questions: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    section_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    content_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    is_key_clause: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), default=0
    )
    prev_chunk_id: Mapped[int | None] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), nullable=True
    )
    next_chunk_id: Mapped[int | None] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), nullable=True
    )
    vector_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    vectorize_status: Mapped[str] = mapped_column(
        Enum("pending", "done", name="vectorize_status"), default="pending"
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class QaStaging(Base):
    __tablename__ = "qa_extraction_staging"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    )
    batch_no: Mapped[str] = mapped_column(String(64), nullable=False)
    source_ref: Mapped[str | None] = mapped_column(String(255), nullable=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        Enum("extracted", "kept", "discarded", name="qa_staging_status"), default="extracted"
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class LowConfidenceQuestion(Base):
    """低置信度问题池(ch04):检索低置信 / 生成自评两入口,user_feedback 留枚举本章不写。"""

    __tablename__ = "low_confidence_questions"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    )
    conversation_id: Mapped[int | None] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), nullable=True
    )
    raw_question: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(
        Enum("retrieval_low_conf", "self_check", "user_feedback", name="lcq_source"), nullable=False
    )
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    matched_review_id: Mapped[int | None] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), ForeignKey("review_queue.id", ondelete="SET NULL"),
        nullable=True
    )
    retrieved_chunks: Mapped[list | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class ReviewQueue(Base):
    """去重后的知识缺口待审队列,一行代表一个待补 FAQ。"""

    __tablename__ = "review_queue"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    )
    normalized_question: Mapped[str] = mapped_column(String(512), nullable=False)
    ai_suggested_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    occurrence_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    review_status: Mapped[str] = mapped_column(
        Enum("待审", "通过", "驳回", name="review_status"), nullable=False, default="待审"
    )
    approved_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class EvalRun(Base):
    """自动评估流水线的一轮结果,用于按时间观察指标趋势。"""

    __tablename__ = "eval_runs"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    )
    triggered_by: Mapped[str] = mapped_column(
        Enum("定时", "手动", name="eval_triggered_by"), nullable=False, default="定时"
    )
    dataset_size: Mapped[int] = mapped_column(Integer, nullable=False)
    metrics: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class ToolAuditLog(Base):
    """ch08 工具调用审计:统一执行引擎每次调用落一条,被权限拒/被校验拦同样落。"""

    __tablename__ = "tool_audit_logs"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    )
    conversation_id: Mapped[int | None] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), nullable=True
    )
    tool_call_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    tool_name: Mapped[str] = mapped_column(String(128), nullable=False)
    tool_source: Mapped[str] = mapped_column(Enum("builtin", "mcp", name="tool_source"), nullable=False)
    mcp_server: Mapped[str | None] = mapped_column(String(64), nullable=True)
    arguments: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    result_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(
        Enum("成功", "失败", "超时", "校验拦下", "权限拒绝", name="tool_audit_status"), nullable=False
    )
    error_message: Mapped[str | None] = mapped_column(String(512), nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class RequestUsage(Base):
    """ch09：每次请求的本地 token/耗时副本，Langfuse 不可用时仍可统计。"""

    __tablename__ = "request_usage"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    )
    conversation_id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), nullable=False
    )
    trace_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    intent: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class FaithCase(Base):
    """忠实度编造个案台账(ch04):一题一行,seen_count 跨轮累加,复发退回未解决。"""

    __tablename__ = "faith_cases"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    )
    eval_id: Mapped[str] = mapped_column(String(16), nullable=False, unique=True)
    bucket: Mapped[str] = mapped_column(String(24), nullable=False)
    query: Mapped[str] = mapped_column(String(512), nullable=False)
    strategy: Mapped[str] = mapped_column(String(24), nullable=False, default="hybrid_rerank")
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[list | None] = mapped_column(JSON, nullable=True)
    judge_model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(
        Enum("未解决", "已解决", "无需解决", name="faith_status"), default="未解决"
    )
    seen_count: Mapped[int] = mapped_column(Integer, default=1)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    resolution: Mapped[str | None] = mapped_column(String(300), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class TopicClassification(Base):
    """ch10：微调分类器旁路批量归类结果。实时对话主链路不写这张表。"""

    __tablename__ = "topic_classifications"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    )
    question_id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        ForeignKey("low_confidence_questions.id"), unique=True, nullable=False,
    )
    labels: Mapped[list] = mapped_column(JSON, nullable=False)
    classified_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class AdminUser(Base):
    """ch14：后台管理员账号。角色以本表为准，口令只存 PBKDF2 哈希。"""

    __tablename__ = "admin_users"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    )
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(
        Enum("viewer", "operator", "admin", name="admin_role"),
        nullable=False, default="viewer",
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    failed_login_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    locked_until: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    session_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class AdminAuditLog(Base):
    """ch14：后台管理操作审计，append-only，不挂外键（审计写入不能被引用约束拦住）。"""

    __tablename__ = "admin_audit_logs"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    )
    admin_user_id: Mapped[int | None] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), nullable=True
    )
    username: Mapped[str] = mapped_column(String(64), nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    target_type: Mapped[str] = mapped_column(String(32), nullable=False)
    target_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    result: Mapped[str] = mapped_column(
        Enum("成功", "失败", name="admin_audit_result"), nullable=False, default="成功"
    )
    detail: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    ip: Mapped[str | None] = mapped_column(String(64), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


# 执行可靠性模型独立定义在 app/execution/models.py；这里只导出，保证共享同一个 metadata。
from app.execution.models import (  # noqa: E402
    ChatRun,
    ChatRunEvent,
    ChatSessionLease,
    ChatToolAttempt,
)
