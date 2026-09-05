# tests/test_schemas_prompts.py
import pytest
from pydantic import ValidationError

from app.prompts import SERVICE_PROMPT_TEMPLATE
from app.schemas import AfterSaleExtraction, ChatRequest


def test_service_prompt_renders():
    text = SERVICE_PROMPT_TEMPLATE.format()
    assert "小帮" in text
    assert "客服" in text
    assert "转人工" in text


def test_chat_request_rejects_empty_message():
    with pytest.raises(ValidationError):
        ChatRequest(message="")


def test_chat_request_session_id_optional():
    req = ChatRequest(message="你好")
    assert req.session_id is None


def test_extraction_allows_null_order_no():
    e = AfterSaleExtraction(order_no=None, issue_type="other", expected_resolution="进一步了解情况")
    assert e.order_no is None


def test_extraction_rejects_bad_issue_type():
    with pytest.raises(ValidationError):
        AfterSaleExtraction(order_no="1", issue_type="warranty", expected_resolution="x")


def test_service_prompt_contains_no_hallucination_rule():
    """ch02 留账:query_faq 查无结果时模型会把空结果编成答案(「满 99 包邮」)。
    ch03 起向量检索 + 本条硬约束双保险:prompt 必须明确「如实说明,严禁编造」。"""
    text = SERVICE_PROMPT_TEMPLATE.format()
    assert "如实" in text and "编造" in text
    assert "转人工" in text
