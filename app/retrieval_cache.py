"""Bounded in-process caches for static retrieval model calls.

Only provider outputs whose complete input is immutable are cached: one embedding
per exact text, and one rerank response per exact query/document order/top_n.
Business state, generated answers, and evidence-gate decisions stay outside this
module. Keys include the provider model plus explicit corpus and retrieval-config
versions, so a rebuild cannot reuse stale outputs.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class _KeyConfig:
    model: str
    corpus_version: str
    config_version: str

    def key(self, kind: str, payload: dict[str, Any]) -> str:
        material = {
            "schema": "retrieval-static-cache-v1",
            "kind": kind,
            "model": self.model,
            "corpus_version": self.corpus_version,
            "config_version": self.config_version,
            "input": payload,
        }
        encoded = json.dumps(
            material,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass
class _CacheEntry:
    value: Any
    expires_at: float


@dataclass
class _Flight:
    event: threading.Event
    value: Any = None
    error: BaseException | None = None


class _QualifiedStaticCache:
    def __init__(
        self,
        *,
        kind: str,
        model: str,
        corpus_version: str,
        config_version: str,
        max_entries: int,
        ttl_seconds: float,
        clock: Callable[[], float],
    ) -> None:
        if not model or not corpus_version or not config_version:
            raise ValueError("model、corpus_version 与 config_version 均不能为空")
        if int(max_entries) < 1:
            raise ValueError("max_entries 必须 >= 1")
        if float(ttl_seconds) <= 0:
            raise ValueError("ttl_seconds 必须 > 0")
        self.kind = kind
        self.key_config = _KeyConfig(model, corpus_version, config_version)
        self.max_entries = int(max_entries)
        self.ttl_seconds = float(ttl_seconds)
        self._clock = clock
        self._lock = threading.RLock()
        self._entries: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._inflight: dict[str, _Flight] = {}
        self._hits = 0
        self._misses = 0
        self._coalesced = 0
        self._deduplicated = 0
        self._provider_calls = 0
        self._provider_items = 0
        self._evictions = 0
        self._expired = 0

    def begin(self, key: str) -> tuple[str, Any]:
        """Return hit, leader, or waiter; cache lookup and ownership are atomic."""
        with self._lock:
            now = self._clock()
            entry = self._entries.get(key)
            if entry is not None:
                if now <= entry.expires_at:
                    self._entries.move_to_end(key)
                    self._hits += 1
                    return "hit", entry.value
                self._entries.pop(key, None)
                self._expired += 1

            self._misses += 1
            flight = self._inflight.get(key)
            if flight is not None:
                self._coalesced += 1
                return "waiter", flight

            flight = _Flight(threading.Event())
            self._inflight[key] = flight
            return "leader", flight

    def note_deduplicated(self, count: int) -> None:
        if count:
            with self._lock:
                self._deduplicated += count

    def note_provider_call(self, item_count: int) -> None:
        with self._lock:
            self._provider_calls += 1
            self._provider_items += int(item_count)

    def finish(self, key: str, flight: _Flight, value: Any) -> None:
        with self._lock:
            self._entries[key] = _CacheEntry(
                value=value,
                expires_at=self._clock() + self.ttl_seconds,
            )
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
                self._evictions += 1
            self._inflight.pop(key, None)
            flight.value = value
        flight.event.set()

    def fail(self, key: str, flight: _Flight, error: BaseException) -> None:
        with self._lock:
            self._inflight.pop(key, None)
            flight.error = error
        flight.event.set()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            lookups = self._hits + self._misses
            return {
                "kind": self.kind,
                "model": self.key_config.model,
                "corpus_version": self.key_config.corpus_version,
                "config_version": self.key_config.config_version,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / lookups, 6) if lookups else 0.0,
                "coalesced": self._coalesced,
                "deduplicated_inputs": self._deduplicated,
                "provider_calls": self._provider_calls,
                "provider_items": self._provider_items,
                "evictions": self._evictions,
                "expired": self._expired,
                "entries": len(self._entries),
                "inflight": len(self._inflight),
                "capacity": self.max_entries,
                "ttl_seconds": self.ttl_seconds,
            }


class CachedEmbedder:
    """Transparent embedder wrapper; exact text inputs are cached individually."""

    def __init__(
        self,
        inner,
        *,
        model: str,
        corpus_version: str,
        config_version: str,
        max_entries: int = 512,
        ttl_seconds: float = 3600.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.inner = inner
        self._cache = _QualifiedStaticCache(
            kind="embedding",
            model=model,
            corpus_version=corpus_version,
            config_version=config_version,
            max_entries=max_entries,
            ttl_seconds=ttl_seconds,
            clock=clock,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        unique_texts = list(dict.fromkeys(texts))
        self._cache.note_deduplicated(len(texts) - len(unique_texts))
        keys = {
            text: self._cache.key_config.key("embedding", {"input": text})
            for text in unique_texts
        }
        resolved: dict[str, list[float]] = {}
        leaders: dict[str, tuple[str, _Flight]] = {}
        waiters: dict[str, _Flight] = {}

        for text in unique_texts:
            state, item = self._cache.begin(keys[text])
            if state == "hit":
                resolved[text] = item
            elif state == "leader":
                leaders[keys[text]] = (text, item)
            else:
                waiters[keys[text]] = item

        if leaders:
            leader_texts = [text for text, _ in leaders.values()]
            try:
                self._cache.note_provider_call(len(leader_texts))
                vectors = self.inner.embed(leader_texts)
                if len(vectors) != len(leader_texts):
                    raise RuntimeError(
                        f"嵌入客户端返回条数不匹配:期望 {len(leader_texts)},实得 {len(vectors)}"
                    )
            except BaseException as exc:
                for key, (_, flight) in leaders.items():
                    self._cache.fail(key, flight, exc)
                raise
            for (key, (text, flight)), vector in zip(leaders.items(), vectors):
                stored = list(vector)
                self._cache.finish(key, flight, stored)
                resolved[text] = stored

        for key, flight in waiters.items():
            flight.event.wait()
            if flight.error is not None:
                raise flight.error
            resolved[next(text for text in unique_texts if keys[text] == key)] = flight.value

        return [list(resolved[text]) for text in texts]

    def stats(self) -> dict[str, Any]:
        return self._cache.stats()


class CachedReranker:
    """Transparent reranker wrapper; the complete request is the cache key."""

    def __init__(
        self,
        inner,
        *,
        model: str,
        corpus_version: str,
        config_version: str,
        max_entries: int = 256,
        ttl_seconds: float = 3600.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.inner = inner
        self._cache = _QualifiedStaticCache(
            kind="rerank",
            model=model,
            corpus_version=corpus_version,
            config_version=config_version,
            max_entries=max_entries,
            ttl_seconds=ttl_seconds,
            clock=clock,
        )

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[tuple[int, float]]:
        key = self._cache.key_config.key(
            "rerank",
            {"query": query, "documents": documents, "top_n": top_n},
        )
        state, item = self._cache.begin(key)
        if state == "hit":
            return [tuple(row) for row in item]
        if state == "waiter":
            item.event.wait()
            if item.error is not None:
                raise item.error
            return [tuple(row) for row in item.value]

        try:
            self._cache.note_provider_call(len(documents))
            results = self.inner.rerank(query, documents, top_n=top_n)
            stored = [tuple(row) for row in results]
        except BaseException as exc:
            self._cache.fail(key, item, exc)
            raise
        self._cache.finish(key, item, stored)
        return [tuple(row) for row in stored]

    def stats(self) -> dict[str, Any]:
        return self._cache.stats()
