# app/schemas.py
from typing import Literal

from pydantic import BaseModel, Field


class OrderResume(BaseModel):
    """ch06 订单选择器无状态回传:点选后前端原样带回的槽位数据。
    intent 属于挂起上下文:resume 轮旁路了 intent 节点,由路由层直接注入 state。"""
    order_id: str = Field(pattern=r"^\d{3,6}$", description="点选的订单号")
    question: str = Field(min_length=1, description="选择器帧带回的补全问题")
    intent: Literal["退款退货", "售后"] = Field(description="发起选择器时的意图")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, description="用户本轮输入")
    session_id: int | None = Field(default=None, description="会话 id,缺省则服务端新建")
    resume: OrderResume | None = Field(default=None,
                                       description="订单选择器点选回传,非空走子流程旁路")


class MessageItem(BaseModel):
    role: Literal["user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list | None = None
    tool_call_id: str | None = None


class HistoryResponse(BaseModel):
    session_id: int
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
