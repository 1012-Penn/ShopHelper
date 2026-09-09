# app/schemas.py
from typing import Literal

from pydantic import BaseModel, Field


class OrderResume(BaseModel):
    """ch06 订单选择器无状态回传:点选后前端原样带回的槽位数据。
    intent 属于挂起上下文:resume 轮旁路了 intent 节点,由路由层直接注入 state。"""
    order_id: str = Field(pattern=r"^\d{3,6}$", description="点选的订单号")
    question: str = Field(min_length=1, max_length=300,
                          description="选择器帧带回的补全问题(限长防绕过消息长度闸)")
    intent: Literal["退款退货", "售后"] = Field(description="发起选择器时的意图")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, description="用户本轮输入")
    session_id: int | None = Field(default=None, description="会话 id,缺省则服务端新建")
    resume: OrderResume | None = Field(default=None,
                                       description="订单选择器点选回传,非空走子流程旁路")


class FeedbackRequest(BaseModel):
    session_id: int
    question: str = Field(min_length=1, max_length=2000)
    rating: Literal["up", "down"]


class ReviewApproveRequest(BaseModel):
    approved_answer: str = Field(min_length=1, max_length=10000)


class MessageItem(BaseModel):
    role: Literal["user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list | None = None
    tool_call_id: str | None = None


class HistoryResponse(BaseModel):
    session_id: int
    messages: list[MessageItem]


class ConversationItem(BaseModel):
    """ch07 侧栏列表项:新在前由排序保证,首问预览截 100 字,summarized=摘要投影已生成。"""
    id: int
    preview: str = ""
    summarized: bool = False
    updated_at: str = ""


class ConversationListResponse(BaseModel):
    items: list[ConversationItem]


class ConversationMessagesResponse(BaseModel):
    conversation_id: int
    messages: list[MessageItem]


class ExtractRequest(BaseModel):
    text: str = Field(min_length=1, description="售后描述原文")


IssueType = Literal["refund", "exchange", "repair", "logistics", "other"]


class AfterSaleExtraction(BaseModel):
    order_no: str | None = Field(default=None, description="订单号,用户描述里没有则为 null")
    issue_type: IssueType = Field(
        description="诉求类型:refund 退款、exchange 换货、repair 报修、logistics 物流问题、other 其他"
    )
    expected_resolution: str = Field(description="用户期望的处理方式,一句话概括")


class TicketRequest(BaseModel):
    session_id: int | None = Field(default=None, description="会话 id,缺省则新建")
    description: str = Field(min_length=1, description="工单问题描述")
    ticket_type: Literal["投诉", "售后", "咨询"] = Field(default="投诉", description="工单类型")


class TicketConfirmRequest(BaseModel):
    """ch08 建工单确认流:预览卡「确认提交」回传的载荷(前端原样回显,服务端零状态)。
    tool_name/arguments 支持 debug 写工具走同一确认通道(验收 6 制造写超时)。"""
    conversation_id: int = Field(description="会话 id")
    tool_name: str = Field(default="create_ticket", description="确认放行的写工具名")
    description: str = Field(default="", max_length=500, description="工单问题描述(create_ticket 用)")
    ticket_type: Literal["售后", "投诉", "咨询"] = Field(default="售后", description="工单类型")
    arguments: dict = Field(default_factory=dict, description="工具参数(非 create_ticket 时使用)")


class TicketCancelRequest(BaseModel):
    conversation_id: int = Field(description="会话 id")
    reason: str = Field(default="用户取消确认", description="取消原因,落审计")
