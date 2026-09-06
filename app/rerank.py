"""bge-reranker-v2-m3 重排客户端:硅基流动 /v1/rerank,httpx 直连(2026-09-06 真接口实测)。

POST /rerank {model, query, documents, top_n, return_documents:false}
→ 200 {"results": [{index, relevance_score}, …]} 已按分数降序;客户端重排序防上游顺序变更。
"""
import httpx

from app.config import Settings


class SiliconFlowReranker:
    def __init__(self, api_base: str, api_key: str, model: str,
                 timeout: float = 10.0, transport: httpx.BaseTransport | None = None) -> None:
        self._client = httpx.Client(
            base_url=api_base.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            transport=transport,
        )
        self._model = model

    def rerank(self, query: str, documents: list[str], top_n: int | None = None) -> list[tuple[int, float]]:
        """→ [(候选下标, relevance_score)] 分数降序;非 200 / 格式异常一律 RuntimeError。"""
        payload: dict = {"model": self._model, "query": query, "documents": documents,
                         "return_documents": False}
        if top_n is not None:
            payload["top_n"] = top_n
        resp = self._client.post("/rerank", json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"重排 API 调用失败:HTTP {resp.status_code} {resp.text[:200]}")
        try:
            results = resp.json()["results"]
            out = [(int(r["index"]), float(r["relevance_score"])) for r in results]
        except (ValueError, KeyError, TypeError) as exc:
            raise RuntimeError(f"重排 API 返回格式异常:{exc}") from exc
        out.sort(key=lambda x: -x[1])
        return out


def make_reranker(settings: Settings) -> SiliconFlowReranker | None:
    """key 缺省回落 embedding_api_key;两者皆空返回 None(调用方降级为不重排)。"""
    key = settings.rerank_api_key or settings.embedding_api_key
    if not key:
        return None
    return SiliconFlowReranker(settings.rerank_api_base, key, settings.rerank_model)
