"""/api/tickets:建工单按钮的唯一后端入口。"""
from app.models import Ticket
from tests.helpers import GraphChatModel, post_chat_sse


async def test_create_ticket_writes_table(make_client):
    client, app = await make_client(GraphChatModel())
    resp = await client.post("/api/tickets", json={"session_id": None,
                                                   "description": "商家一直不发货"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ticket_no"].startswith("T")
    with app.state.session_factory() as s:
        assert s.query(Ticket).count() == 1


async def test_create_ticket_unknown_session_404(make_client):
    client, _ = await make_client(GraphChatModel())
    resp = await client.post("/api/tickets", json={"session_id": 999, "description": "x"})
    assert resp.status_code == 404


async def test_create_ticket_default_type(make_client):
    client, app = await make_client(GraphChatModel())
    resp = await client.post("/api/tickets", json={"session_id": None, "description": "太差了"})
    assert resp.status_code == 200
    with app.state.session_factory() as s:
        row = s.query(Ticket).one()
        assert row.ticket_type == "投诉"  # 缺省工单类型
