"""ch08 建工单确认流 e2e:模型直调被拒+预览帧、确认落表、取消审计权限拒绝。"""
import json

import pytest

from tests.helpers import GraphChatModel, post_chat_sse

pytestmark = pytest.mark.asyncio


def _model(turns):
    return GraphChatModel('{"intent": "其他"}', turns)


async def test_model_direct_create_ticket_intercepted_with_preview_frame(make_client):
    """模型带齐信息直调 create_ticket:引擎拦截(不落 tickets),发预览帧,回灌等待确认指引。"""
    model = _model([
        ("tools", [{"name": "create_ticket", "args": '{"conversation_id": 1, "description": "耳机三天没到", "ticket_type": "售后"}', "id": "c1", "index": 0}]),
        ("text", "已为您发起工单确认,请在卡片上确认或取消。"),
    ])
    client, app = await make_client(model)
    events = await post_chat_sse(client, {"message": "帮我建个工单,耳机三天没到"})
    previews = [e for e in events if e["type"] == "ticket_preview"]
    assert len(previews) == 1
    assert previews[0]["ticket_type"] == "售后"
    assert previews[0]["description"] == "耳机三天没到"
    assert events[-1]["type"] == "done"
    # 工单未建(拦截),审计落「权限拒绝」
    with app.state.session_factory() as session:
        from app.models import Ticket, ToolAuditLog
        assert session.query(Ticket).count() == 0
        audit = session.query(ToolAuditLog).filter_by(tool_name="create_ticket").all()
        assert [a.status for a in audit] == ["权限拒绝"]


async def test_confirm_executes_and_returns_ticket_no(make_client, db_session_factory):
    """确认回传:confirmed 放行真实执行,tickets 落行,审计「成功」。"""
    model = _model([])
    client, app = await make_client(model)
    await app.state.store.resolve(1)  # 先建会话(confirm 端点校验 exists)
    resp = await client.post("/api/tickets/confirm", json={
        "conversation_id": 1, "description": "耳机三天没到", "ticket_type": "售后"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ticket_no"].startswith("T") and body["status"] == "待处理"
    with app.state.session_factory() as session:
        from app.models import Ticket, ToolAuditLog
        (ticket,) = session.query(Ticket).all()
        assert ticket.ticket_no == body["ticket_no"]
        (audit,) = session.query(ToolAuditLog).filter_by(tool_name="create_ticket",
                                                         status="成功").all()
        assert audit.conversation_id == 1


async def test_cancel_audits_permission_denied_without_ticket(make_client):
    client, app = await make_client(_model([]))
    await app.state.store.resolve(1)
    resp = await client.post("/api/tickets/cancel", json={"conversation_id": 1})
    assert resp.status_code == 200 and resp.json() == {"ok": True}
    with app.state.session_factory() as session:
        from app.models import Ticket, ToolAuditLog
        assert session.query(Ticket).count() == 0
        (audit,) = session.query(ToolAuditLog).filter_by(tool_name="create_ticket").all()
        assert audit.status == "权限拒绝" and "取消" in audit.error_message


async def test_confirm_404_unknown_conversation(make_client):
    client, app = await make_client(_model([]))
    resp = await client.post("/api/tickets/confirm", json={
        "conversation_id": 424242, "description": "x", "ticket_type": "售后"})
    assert resp.status_code == 404


async def test_mcp_unavailable_degrades_gracefully(make_client):
    """测试环境不接 MCP(sync 空实现):聊天照常,内置工具可用。"""
    model = _model([("text", "内置工具照常可用。")])
    client, app = await make_client(model)
    events = await post_chat_sse(client, {"message": "你好"})
    assert events[-1]["type"] == "done"


async def test_confirm_rejects_read_or_unknown_tools(make_client):
    """I-4 收口:确认通道仅放行已注册的写工具;读工具/未知工具 422。"""
    client, app = await make_client(_model([]))
    await app.state.store.resolve(1)
    resp = await client.post("/api/tickets/confirm", json={
        "conversation_id": 1, "tool_name": "query_faq",
        "arguments": {"keyword": "邮费"}})
    assert resp.status_code == 422
    resp = await client.post("/api/tickets/confirm", json={
        "conversation_id": 1, "tool_name": "not_a_tool", "arguments": {}})
    assert resp.status_code == 422


async def test_confirm_missing_description_422(make_client):
    client, app = await make_client(_model([]))
    await app.state.store.resolve(1)
    resp = await client.post("/api/tickets/confirm", json={
        "conversation_id": 1, "description": "", "ticket_type": "售后"})
    assert resp.status_code == 422
