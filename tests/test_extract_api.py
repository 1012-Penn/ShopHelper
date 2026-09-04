# tests/test_extract_api.py
from langchain_core.exceptions import OutputParserException

from app.schemas import AfterSaleExtraction
from tests.helpers import StubExtractModel, fake_chat


async def test_extract_success_contract(make_client):
    stub = StubExtractModel(
        result=AfterSaleExtraction(
            order_no="2026090112345",
            issue_type="exchange",
            expected_resolution="换一台新的",
        )
    )
    client, _ = await make_client(fake_chat("x"), extract_model=stub)
    resp = await client.post(
        "/api/extract", json={"text": "订单2026090112345屏幕碎了要换货"}
    )
    assert resp.status_code == 200
    assert resp.json() == {
        "order_no": "2026090112345",
        "issue_type": "exchange",
        "expected_resolution": "换一台新的",
    }


async def test_extract_parse_failure_422(make_client):
    stub = StubExtractModel(error=OutputParserException("模型输出不是合法 JSON"))
    client, _ = await make_client(fake_chat("x"), extract_model=stub)
    resp = await client.post("/api/extract", json={"text": "随便说说"})
    assert resp.status_code == 422
    assert "结构化抽取失败" in resp.json()["detail"]


async def test_extract_empty_text_422(make_client):
    client, _ = await make_client(fake_chat("x"))
    resp = await client.post("/api/extract", json={"text": ""})
    assert resp.status_code == 422
