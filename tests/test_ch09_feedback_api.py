from sqlalchemy import select

from app.models import LowConfidenceQuestion
from tests.helpers import fake_chat, post_chat_sse


async def test_thumbs_down_enters_backend_question_pool(make_client):
    client, app = await make_client(fake_chat("好的"))
    response = await client.post("/api/feedback", json={
        "session_id": 9, "question": "国际件运费怎么算", "rating": "down",
    })
    assert response.status_code == 200 and response.json()["accepted"] is True
    with app.state.session_factory() as session:
        row = session.scalar(select(LowConfidenceQuestion))
        assert row.source == "user_feedback" and row.raw_question == "国际件运费怎么算"
