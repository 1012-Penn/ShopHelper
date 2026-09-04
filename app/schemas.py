# app/schemas.py
from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, description="用户本轮输入")
    session_id: int | None = Field(default=None, description="会话 id,缺省则服务端新建")


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
