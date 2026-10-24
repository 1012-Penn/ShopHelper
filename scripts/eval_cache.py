"""评测缓存:嵌入、重排、回答响应。

缓存键包含模型与输入文本;损坏缓存报 `CacheError` 并给出重建指引。
嵌入向量以 float32 base64 存储,避免 JSON 浮点列表体积过大。
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from array import array
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR = ROOT / "data" / "eval-caches"


class CacheError(RuntimeError):
    """缓存损坏或命名空间不匹配。"""


def _sha(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


class JsonCache:
    def __init__(self, path: Path, namespace: str) -> None:
        self.path = Path(path)
        self.namespace = namespace
        self.hits = 0
        self.misses = 0
        self._entries: dict[str, dict] = {}
        if self.path.exists():
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise CacheError(
                    f"缓存损坏:{self.path};删除后重跑可重建({exc})"
                ) from exc
            if payload.get("namespace") != namespace:
                raise CacheError(
                    f"缓存命名空间不匹配:{payload.get('namespace')} != {namespace}"
                )
            self._entries = dict(payload.get("entries") or {})

    def key(self, *parts: str) -> str:
        return _sha(*parts)

    def get(self, key: str) -> dict | None:
        value = self._entries.get(key)
        if value is None:
            self.misses += 1
            return None
        self.hits += 1
        return value

    def put(self, key: str, value: dict) -> None:
        self._entries[key] = value
        _atomic_write(self.path, {"namespace": self.namespace, "entries": self._entries})

    def sha256(self) -> str | None:
        return _sha(self.path.read_text(encoding="utf-8")) if self.path.exists() else None

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses, "entries": len(self._entries)}


def _encode_vector(vector: list[float]) -> str:
    return base64.b64encode(array("f", vector).tobytes()).decode("ascii")


def _decode_vector(payload: str) -> list[float]:
    raw = base64.b64decode(payload.encode("ascii"))
    values = array("f")
    values.frombytes(raw)
    return list(values)


class EmbeddingCache:
    def __init__(self, inner, path: Path, model: str) -> None:
        self.inner = inner
        self.model = model
        self.cache = JsonCache(Path(path), f"embedding:{model}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        keys = [_sha(self.model, text) for text in texts]
        out: list[list[float] | None] = [None] * len(texts)
        missing: list[int] = []
        for index, key in enumerate(keys):
            payload = self.cache.get(key)
            if payload is None:
                missing.append(index)
            else:
                out[index] = _decode_vector(payload["vector"])
        if missing:
            vectors = self.inner.embed([texts[index] for index in missing])
            for index, vector in zip(missing, vectors):
                out[index] = vector
                self.cache.put(keys[index], {"vector": _encode_vector(vector)})
        return [vector for vector in out if vector is not None]

    def stats(self) -> dict:
        return self.cache.stats()


class RerankCache:
    def __init__(self, inner, path: Path, model: str) -> None:
        self.inner = inner
        self.model = model
        self.cache = JsonCache(Path(path), f"rerank:{model}")

    def rerank(self, query: str, documents: list[str], top_n: int | None = None):
        key = _sha(
            self.model,
            query,
            json.dumps(documents, ensure_ascii=False),
            str(top_n),
        )
        payload = self.cache.get(key)
        if payload is not None:
            return [tuple(item) for item in payload["results"]]
        results = self.inner.rerank(query, documents, top_n=top_n)
        self.cache.put(key, {"results": [list(item) for item in results]})
        return results

    def stats(self) -> dict:
        return self.cache.stats()


class ResponseCache(JsonCache):
    def response_key(
        self,
        *,
        model: str,
        prompt_sha256: str,
        temperature: float,
        query: str,
        evidence_ids: list[int] | None = None,
    ) -> str:
        return self.key(
            model,
            prompt_sha256,
            str(temperature),
            query,
            json.dumps(evidence_ids or [], ensure_ascii=False),
        )
