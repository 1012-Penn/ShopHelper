"""ch07 多会话只读端点:侧栏列表(新在前/首问预览/已摘要标记)与会话消息回载。"""
from fastapi import APIRouter, HTTPException, Request

from app.schemas import (ConversationItem, ConversationListResponse,
                         ConversationMessagesResponse, MessageItem)

router = APIRouter()


@router.get("/api/conversations")
async def list_conversations(request: Request) -> ConversationListResponse:
    store = request.app.state.store
    items = await store.list_conversations(user_id="guest")
    return ConversationListResponse(items=[ConversationItem.model_validate(i) for i in items])


@router.get("/api/conversations/{conversation_id}/messages")
async def conversation_messages(conversation_id: int, request: Request) -> ConversationMessagesResponse:
    store = request.app.state.store
    if not await store.exists(conversation_id):
        raise HTTPException(status_code=404, detail="会话不存在")
    return ConversationMessagesResponse(
        conversation_id=conversation_id,
        messages=[MessageItem.model_validate(m) for m in await store.get_history(conversation_id)],
    )
