"""BGE-M3 嵌入客户端:硅基流动 OpenAI 兼容 /embeddings,httpx 直连(批量按 batch_size 切批)。"""
import math

import httpx

from app.config import Settings


def cosine(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


class SiliconFlowEmbedder:
    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str,
        batch_size: int = 16,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = httpx.Client(
            base_url=api_base.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            transport=transport,
        )
        self._model = model
        self._batch_size = batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量嵌入;返回顺序与入参一致。

        非 200 或返回格式/条数异常一律抛 RuntimeError——按 index 对齐错位会把向量
        upsert 到错误的 chunk id 上(静默的知识库错配),必须在客户端挡住。
        """
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            resp = self._client.post("/embeddings", json={"model": self._model, "input": batch})
            if resp.status_code != 200:
                raise RuntimeError(f"嵌入 API 调用失败:HTTP {resp.status_code} {resp.text[:200]}")
            try:
                data = sorted(resp.json()["data"], key=lambda d: d["index"])
            except (ValueError, KeyError, TypeError) as exc:
                raise RuntimeError(f"嵌入 API 返回格式异常:{exc}") from exc
            if len(data) != len(batch):
                raise RuntimeError(f"嵌入 API 返回条数不匹配:期望 {len(batch)},实得 {len(data)}")
            out.extend(d["embedding"] for d in data)
        return out


def make_embedder(settings: Settings) -> SiliconFlowEmbedder:
    return SiliconFlowEmbedder(
        settings.embedding_api_base,
        settings.embedding_api_key,
        settings.embedding_model,
        batch_size=settings.embedding_batch_size,
    )
