import httpx
import pytest

from app.embedding import SiliconFlowEmbedder, make_embedder
from tests.helpers import FakeEmbedding, cosine


def test_fake_embedding_synonym_paraphrase():
    fake = FakeEmbedding()
    v_postage, v_ship = fake.embed(["邮费是多少", "运费怎么算"])
    v_return, v_none = fake.embed(["退货政策", "今天天气不错"])
    assert cosine(v_postage, v_ship) > 0.9    # 近义词归一后完全同轴
    assert cosine(v_postage, v_return) < 0.7  # 不同话题自然拉开
    assert cosine(v_postage, v_none) == 0.0   # 无领域词 → 零向量


def test_fake_embedding_graduated_similarity():
    fake = FakeEmbedding()
    v_fee, = fake.embed(["运费"])
    v_mix, = fake.embed(["运费和退货的事"])
    assert 0.0 < cosine(v_fee, v_mix) < 1.0   # 单话题 vs 多话题:梯度相似


def test_embedder_splits_batches_and_parses():
    captured = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        body = json.loads(request.content)
        captured.append(body)
        return httpx.Response(
            200,
            json={"data": [{"index": i, "embedding": [0.1, 0.2]} for i in range(len(body["input"]))]},
        )

    client = SiliconFlowEmbedder(
        "https://api.siliconflow.cn/v1", "sk-test", "BAAI/bge-m3",
        batch_size=2, timeout=5, transport=httpx.MockTransport(handler),
    )
    vecs = client.embed([f"文本{i}" for i in range(5)])
    assert len(vecs) == 5 and all(len(v) == 2 for v in vecs)
    assert [len(b["input"]) for b in captured] == [2, 2, 1]  # 按 batch_size 切批
    assert all(b["model"] == "BAAI/bge-m3" for b in captured)


def test_embedder_preserves_order_across_batches():
    def handler(request: httpx.Request) -> httpx.Response:
        import json

        batch = json.loads(request.content)["input"]
        return httpx.Response(
            200,
            json={"data": [{"index": i, "embedding": [float(len(t))]} for i, t in enumerate(batch)]},
        )

    client = SiliconFlowEmbedder(
        "https://x/v1", "sk", "m", batch_size=2, transport=httpx.MockTransport(handler),
    )
    vecs = client.embed(["ab", "abcd", "abcdef"])
    assert vecs == [[2.0], [4.0], [6.0]]


def test_embedder_raises_on_http_error():
    def handler(request):
        return httpx.Response(401, json={"message": "invalid key"})

    client = SiliconFlowEmbedder(
        "https://x/v1", "bad", "BAAI/bge-m3", transport=httpx.MockTransport(handler),
    )
    with pytest.raises(RuntimeError, match="嵌入 API 调用失败"):
        client.embed(["hi"])


def test_make_embedder_from_settings():
    from app.config import Settings

    s = Settings(
        openai_api_key="sk-test", embedding_api_key="sk-emb", embedding_batch_size=8, _env_file=None,
    )
    embedder = make_embedder(s)
    assert isinstance(embedder, SiliconFlowEmbedder)
    assert embedder._batch_size == 8
