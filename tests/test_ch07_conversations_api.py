"""ch07 多会话只读端点:新在前、首问预览、已摘要标记、messages 回载与 404。"""
import pytest

from tests.helpers import GraphChatModel, post_chat_sse

pytestmark = pytest.mark.asyncio


async def test_list_conversations_newest_first_with_preview(make_client):
    model = GraphChatModel('{"intent": "其他"}', [("text", "答。")] * 2)
    client, app = await make_client(model)
    first = (await post_chat_sse(client, {"message": "第一通会话的首问"}))[0]["session_id"]
    second = (await post_chat_sse(client, {"message": "第二通会话的首问"}))[0]["session_id"]
    resp = await client.get("/api/conversations")
    items = resp.json()["items"]
    assert [i["id"] for i in items][:2] == [second, first]  # 新在前
    assert items[0]["preview"].startswith("第二通")
    assert items[0]["summarized"] is False
    assert items[0]["updated_at"]


async def test_summarized_flag_and_messages_reload(make_client):
    model = GraphChatModel('{"intent": "其他"}', [("text", "答。")])
    client, app = await make_client(model)
    sid = (await post_chat_sse(client, {"message": "退货政策是什么"}))[0]["session_id"]
    rows = await app.state.store.get_rows(sid)
    await app.state.store.append_summary(sid, seq=1, from_id=rows[0]["id"],
                                         upto_id=rows[-1]["id"], content="梗概一段")
    resp = await client.get("/api/conversations")
    items = [i for i in resp.json()["items"] if i["id"] == sid]
    assert items and items[0]["summarized"] is True

    resp = await client.get(f"/api/conversations/{sid}/messages")
    body = resp.json()
    assert body["conversation_id"] == sid
    assert [(m["role"], m["content"]) for m in body["messages"]] == [
        ("user", "退货政策是什么"), ("assistant", "答。")]


async def test_messages_endpoint_404_unknown(make_client):
    client, app = await make_client(GraphChatModel('{"intent": "其他"}', []))
    resp = await client.get("/api/conversations/424242/messages")
    assert resp.status_code == 404
