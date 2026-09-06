"""Reranker 客户端:MockTransport 契约 + 分数降序 + 异常收敛 RuntimeError + FakeReranker。"""
import json

import httpx
import pytest

from app.rerank import SiliconFlowReranker
from tests.helpers import FakeReranker


def _client(handler):
    return SiliconFlowReranker("https://api.siliconflow.cn/v1", "sk-test", "BAAI/bge-reranker-v2-m3",
                               transport=httpx.MockTransport(handler))


def test_rerank_sorted_desc_and_indices():
    def handler(request):
        assert json.loads(request.content)["model"] == "BAAI/bge-reranker-v2-m3"
        body = {"results": [
            {"index": 1, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.2},
        ]}
        return httpx.Response(200, json=body)

    out = _client(handler).rerank("q", ["a", "b"], top_n=2)
    assert out == [(1, 0.9), (0, 0.2)]


def test_rerank_non_200_raises():
    def handler(request):
        return httpx.Response(500, text="boom")

    with pytest.raises(RuntimeError, match="500"):
        _client(handler).rerank("q", ["a"])


def test_rerank_bad_payload_raises():
    def handler(request):
        return httpx.Response(200, json={"unexpected": []})

    with pytest.raises(RuntimeError, match="格式"):
        _client(handler).rerank("q", ["a"])


def test_fake_reranker_deterministic_overlap():
    r = FakeReranker()
    out = r.rerank("降噪耳机", ["无线降噪耳机 WH", "退货政策 邮费", "耳机 主动降噪 功能"], top_n=2)
    assert len(out) == 2
    assert out[0][1] >= out[1][1]
    assert out[0][0] in (0, 2)
