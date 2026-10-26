"""Production-path harness for E2E RAG quality R2.

The harness instruments the existing app without changing its graph or routing:
real ``create_app`` -> real ``POST /api/chat`` -> real graph/tools. External
calls are guarded by a private concurrent ledger; exact caches are namespaced
by split, case turn, prompt, tools and structured-output schema.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any, AsyncIterator

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from pydantic import BaseModel

from app.history import estimate_tokens
from scripts.e2e_quality_r2.budget import BudgetExceeded, ExactCallCache, R2Budget
from scripts.e2e_quality_r2.scoring import (
    assert_model_input_safe,
    resolve_citation_evidence,
    score_case,
)
from scripts.eval_cache import EmbeddingCache, RerankCache
from scripts.eval_cost import PricingTable
from scripts.eval_snapshot import tree_sha256

SELECTED_ENV_KEYS = {
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "EMBEDDING_API_KEY",
    "EMBEDDING_API_BASE",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIM",
    "RERANK_API_KEY",
    "RERANK_API_BASE",
    "RERANK_MODEL",
}


class RuntimeContractError(RuntimeError):
    """A private runtime copy or captured projection violates the R2 contract."""


@dataclass(frozen=True)
class RuntimePaths:
    snapshot: Path
    metadata_db: Path
    cache_dir: Path


def verify_snapshot_identity(snapshot: Path, expected_sha256: str) -> str:
    actual = tree_sha256(Path(snapshot))
    if actual != expected_sha256:
        raise RuntimeContractError(
            f"snapshot tree hash mismatch: expected {expected_sha256}, got {actual}"
        )
    return actual


def prepare_private_runtime(
    *,
    source_snapshot: Path,
    source_metadata_db: Path,
    expected_snapshot_sha256: str,
    destination: Path,
    source_cache_dir: Path | None = None,
) -> RuntimePaths:
    verify_snapshot_identity(source_snapshot, expected_snapshot_sha256)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    snapshot = destination / "snapshot.db"
    if snapshot.exists():
        verify_snapshot_identity(snapshot, expected_snapshot_sha256)
    else:
        shutil.copytree(
            source_snapshot,
            snapshot,
            ignore=shutil.ignore_patterns("LOCK"),
        )
        verify_snapshot_identity(snapshot, expected_snapshot_sha256)
    metadata_db = destination / "metadata.db"
    if not metadata_db.exists():
        shutil.copy2(source_metadata_db, metadata_db)
    cache_dir = destination / "caches"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if source_cache_dir is not None:
        for filename in ("embedding.json", "rerank.json"):
            source = Path(source_cache_dir) / filename
            target = cache_dir / filename
            if source.exists() and not target.exists():
                shutil.copy2(source, target)
    return RuntimePaths(snapshot=snapshot, metadata_db=metadata_db, cache_dir=cache_dir)


def _jsonable(value):
    if isinstance(value, BaseMessage):
        return _message_payload(value)
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    if isinstance(value, set):
        return sorted(_jsonable(child) for child in value)
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _message_payload(message) -> dict:
    if isinstance(message, BaseMessage):
        role = {
            "SystemMessage": "system",
            "HumanMessage": "user",
            "AIMessage": "assistant",
            "ToolMessage": "tool",
        }.get(type(message).__name__, type(message).__name__)
        return {
            "role": role,
            "content": _jsonable(message.content),
            "tool_calls": _jsonable(getattr(message, "tool_calls", None) or []),
            "tool_call_id": getattr(message, "tool_call_id", None),
            "name": getattr(message, "name", None),
        }
    if isinstance(message, dict):
        return {str(key): _jsonable(value) for key, value in message.items()}
    return {"role": "user", "content": str(message)}


def _projection(request) -> object:
    if isinstance(request, str):
        return {"input": request}
    if isinstance(request, (list, tuple)):
        return {"messages": [_message_payload(item) for item in request]}
    if isinstance(request, dict):
        if "messages" in request:
            return {
                **{key: _jsonable(value) for key, value in request.items()
                   if key != "messages"},
                "messages": [_message_payload(item) for item in request["messages"]],
            }
        return _jsonable(request)
    return {"input": _jsonable(request)}


def _request_text(request) -> str:
    return json.dumps(_projection(request), ensure_ascii=False, sort_keys=True)


def _usage(response) -> tuple[int, int, int]:
    usage = getattr(response, "usage_metadata", None)
    if not usage:
        metadata = getattr(response, "response_metadata", None) or {}
        usage = metadata.get("token_usage") or metadata.get("usage") or {}
    input_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
    details = usage.get("input_token_details") or {}
    cached = int(
        details.get("cache_read")
        or details.get("cached_tokens")
        or usage.get("prompt_cache_hit_tokens")
        or 0
    )
    return input_tokens, output_tokens, cached


def _tool_projection(tools) -> list[dict]:
    projection = []
    for tool in tools or []:
        schema = getattr(tool, "args_schema", None)
        if isinstance(schema, dict):
            schema_payload = _jsonable(schema)
        elif schema is not None and hasattr(schema, "model_json_schema"):
            schema_payload = schema.model_json_schema()
        else:
            schema_payload = None
        projection.append({
            "name": getattr(tool, "name", type(tool).__name__),
            "description": getattr(tool, "description", ""),
            "schema": schema_payload,
        })
    return projection


def json_structured_instruction(schema: object) -> str:
    schema_text = json.dumps(_jsonable(schema), ensure_ascii=False, sort_keys=True)
    return (
        "\n\n只输出一个 JSON 对象，不要 Markdown 代码块，不要解释。"
        f"必须严格符合这个 JSON Schema：{schema_text}"
    )


def _add_instruction(request, instruction: str):
    if isinstance(request, str):
        return request + instruction
    if isinstance(request, (list, tuple)):
        return [*request, HumanMessage(content=instruction)]
    return str(request) + instruction


def _retryable(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    status = getattr(exc, "status_code", None)
    if status == 429:
        return True
    if isinstance(status, int) and 500 <= status < 600:
        return True
    return any(token in text for token in (
        "429", "rate limit", "too many request", "timeout",
        "connection reset", "temporarily unavailable", "502", "503", "504",
    ))


class TraceCapture:
    """Per-case model/tool/graph captures. Gold data is never sent to model code."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.case: dict | None = None
        self.split = ""
        self.turn = 0
        self.arm = ""
        self.model_calls: list[dict] = []
        self.tool_calls: list[dict] = []
        self.graph_results: list[dict] = []
        self.technical_failures: list[dict] = []
        self.start_case(case=None, split="", turn=0)

    def start_case(self, *, case: dict | None, split: str, turn: int) -> None:
        with self._lock:
            self.case = case
            self.split = split
            self.turn = int(turn)

    def assert_safe(self, projection: object) -> None:
        with self._lock:
            case = self.case
        if case is not None:
            try:
                assert_model_input_safe(projection, case)
            except Exception as exc:
                raise RuntimeContractError(str(exc)) from exc

    def add_model_call(self, row: dict) -> None:
        with self._lock:
            self.model_calls.append({
                "case_id": (self.case or {}).get("id"),
                "split": self.split,
                "turn": self.turn,
                **row,
            })

    def add_tool_call(self, row: dict) -> None:
        with self._lock:
            self.tool_calls.append({
                "case_id": (self.case or {}).get("id"),
                "split": self.split,
                "turn": self.turn,
                **row,
            })

    def fail(self, kind: str, exc: BaseException, *, extra: dict | None = None) -> None:
        with self._lock:
            self.technical_failures.append({
                "case_id": (self.case or {}).get("id"),
                "split": self.split,
                "turn": self.turn,
                "kind": kind,
                "error": f"{type(exc).__name__}: {exc}"[:500],
                **(extra or {}),
            })

    def add_graph_result(self, result: dict) -> None:
        with self._lock:
            self.graph_results.append(result)

    def case_model_calls(self, case_id: str) -> list[dict]:
        with self._lock:
            return [dict(row) for row in self.model_calls if row["case_id"] == case_id]

    def case_tool_calls(self, case_id: str) -> list[dict]:
        with self._lock:
            return [dict(row) for row in self.tool_calls if row["case_id"] == case_id]

    def case_failures(self, case_id: str) -> list[dict]:
        with self._lock:
            return [dict(row) for row in self.technical_failures
                    if row["case_id"] == case_id]


class _CapturedCalls:
    """Shared call wrapper for chat and structured runnables."""

    def __init__(
        self,
        *,
        inner,
        model: str,
        pricing: PricingTable,
        budget: R2Budget,
        cache: ExactCallCache,
        capture: TraceCapture,
        namespace: str,
        kind: str,
        schema=None,
        tools: list[dict] | None = None,
    ) -> None:
        self.inner = inner
        self.model = model
        self.pricing = pricing
        self.budget = budget
        self.cache = cache
        self.capture = capture
        self.namespace = namespace
        self.kind = kind
        self.schema = schema
        self.tools = list(tools or [])

    def _estimate(self, request, schema=None, tools=None) -> float:
        text = _request_text({
            "request": request,
            "schema": schema or self.schema,
            "tools": tools or self.tools,
        })
        tokens = max(1, estimate_tokens(text))
        return self.pricing.cost(self.model, tokens, 512)

    def _context(self) -> dict:
        return {
            "case_id": (self.capture.case or {}).get("id"),
            "split": self.capture.split,
            "turn": self.capture.turn,
            "arm": self.capture.arm,
            "kind": self.kind,
            "model": self.model,
        }

    def _cache_key(self, request, schema=None, tools=None) -> str:
        return self.cache.key(
            namespace=self.namespace,
            model=self.model,
            request=_projection(request),
            schema=_jsonable(schema or self.schema),
            tools=tools if tools is not None else self.tools,
            split=self.capture.split,
            turn=self.capture.turn,
        )

    def _settle_success(self, attempt_id: str, response, *, cache_hit: bool = False) -> float:
        input_tokens, output_tokens, cached_tokens = _usage(response)
        actual = self.pricing.cost(
            self.model, input_tokens, output_tokens, cached_tokens
        )
        self.budget.settle(
            attempt_id,
            actual_usd=actual,
            status="cache_hit" if cache_hit else "succeeded",
            usage={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_hit_input_tokens": cached_tokens,
            },
        )
        return actual


class BudgetedChat(_CapturedCalls):
    def __init__(self, *, structured_mode: str = "production", **kwargs) -> None:
        super().__init__(**kwargs)
        self.structured_mode = structured_mode

    def _derive_kind(self, request) -> str:
        schema_name = type(self.schema).__name__ if self.schema is not None else ""
        mapping = {
            "ResolvedQuestion": "resolve",
            "ExpandedQueries": "expand",
            "QueryRewrite": "rewrite",
            "EvidenceVerdict": "evidence_judge",
            "AnswerVerdict": "answer_verifier",
            "AfterSaleExtraction": "extract",
            "NormalizedQuestionOutput": "flywheel_normalize",
            "DedupDecisionOutput": "flywheel_dedup",
        }
        if schema_name in mapping:
            return mapping[schema_name]
        text = _request_text(request)
        if "意图分类器" in text:
            return "intent"
        if "会话摘要器" in text:
            return "summary"
        if "证据审核员" in text:
            return "evidence_judge"
        if "问法澄清器" in text:
            return "resolve"
        return self.kind

    def _new(self, inner=None, *, schema=None, tools=None, kind=None):
        return BudgetedChat(
            inner=inner or self.inner,
            model=self.model,
            pricing=self.pricing,
            budget=self.budget,
            cache=self.cache,
            capture=self.capture,
            namespace=self.namespace,
            kind=kind or self.kind,
            schema=schema if schema is not None else self.schema,
            tools=tools if tools is not None else self.tools,
            structured_mode=self.structured_mode,
        )

    def bind(self, **kwargs):
        return self._new(inner=self.inner.bind(**kwargs))

    def bind_tools(self, tools, **kwargs):
        tools_projection = _tool_projection(tools)
        return self._new(
            inner=self.inner.bind_tools(tools, **kwargs),
            tools=tools_projection,
            kind="answer",
        )

    def with_structured_output(self, schema, **kwargs):
        schema_payload = (
            schema.model_json_schema()
            if hasattr(schema, "model_json_schema")
            else _jsonable(schema)
        )
        if self.structured_mode == "json-fallback":
            kind_by_schema = {
                "ResolvedQuestion": "resolve",
                "ExpandedQueries": "expand",
                "QueryRewrite": "rewrite",
                "EvidenceVerdict": "evidence_judge",
                "AnswerVerdict": "answer_verifier",
                "AfterSaleExtraction": "extract",
                "NormalizedQuestionOutput": "flywheel_normalize",
                "DedupDecisionOutput": "flywheel_dedup",
            }
            return BudgetedJsonStructured(
                inner=self.inner,
                model=self.model,
                pricing=self.pricing,
                budget=self.budget,
                cache=self.cache,
                capture=self.capture,
                namespace=self.namespace,
                kind=kind_by_schema.get(
                    getattr(schema, "__name__", type(schema).__name__),
                    "structured_json",
                ),
                schema=schema_payload,
                schema_class=schema,
            )
        inner = self.inner.with_structured_output(schema, include_raw=True, **kwargs)
        return BudgetedStructured(
            inner=inner,
            model=self.model,
            pricing=self.pricing,
            budget=self.budget,
            cache=self.cache,
            capture=self.capture,
            namespace=self.namespace,
            kind="structured",
            schema=schema_payload,
            schema_class=schema,
            tools=self.tools,
        )

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def invoke(self, request, config=None, **kwargs):
        projection = {
            "request": _projection(request),
            "schema": self.schema,
            "tools": self.tools,
        }
        self.capture.assert_safe(projection)
        kind = self._derive_kind(request)
        key = self._cache_key(request)
        cached = self.cache.get(key)
        if cached is not None:
            self.capture.add_model_call({
                "kind": kind,
                "model": self.model,
                "cache_hit": True,
                "input": projection,
                "output": cached.get("output"),
                "usage": cached.get("usage") or {},
            })
            return AIMessage(
                content=cached.get("content") or "",
                tool_calls=cached.get("tool_calls") or [],
            )
        last: BaseException | None = None
        for attempt in range(3):
            attempt_id = self.budget.reserve(
                {**self._context(), "kind": kind},
                self._estimate(request),
            )
            started = time.perf_counter()
            try:
                response = self.inner.invoke(request, config, **kwargs)
            except BaseException as exc:
                last = exc
                self.budget.settle(attempt_id, actual_usd=0.0, status="failed",
                                   error=f"{type(exc).__name__}: {exc}"[:500])
                if attempt < 2 and _retryable(exc):
                    time.sleep(min(2 ** attempt, 2))
                    continue
                self.capture.fail(kind, exc, extra={"retry": attempt})
                raise
            usage = _usage(response)
            self.budget.settle(
                attempt_id,
                actual_usd=self.pricing.cost(self.model, usage[0], usage[1], usage[2]),
                status="succeeded",
                usage={
                    "input_tokens": usage[0],
                    "output_tokens": usage[1],
                    "cache_hit_input_tokens": usage[2],
                },
            )
            output = {
                "content": str(getattr(response, "content", response)),
                "tool_calls": _jsonable(getattr(response, "tool_calls", None) or []),
                "response_metadata": _jsonable(
                    getattr(response, "response_metadata", None) or {}
                ),
            }
            self.cache.put(key, {
                "content": output["content"],
                "tool_calls": output["tool_calls"],
                "usage": {
                    "input_tokens": usage[0],
                    "output_tokens": usage[1],
                    "cache_hit_input_tokens": usage[2],
                },
                "output": output,
            })
            self.capture.add_model_call({
                "kind": kind,
                "model": self.model,
                "cache_hit": False,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                "input": projection,
                "output": output,
                "usage": {
                    "input_tokens": usage[0],
                    "output_tokens": usage[1],
                    "cache_hit_input_tokens": usage[2],
                },
            })
            return response
        raise last or RuntimeError("model call failed")

    async def ainvoke(self, request, config=None, **kwargs):
        projection = {
            "request": _projection(request),
            "schema": self.schema,
            "tools": self.tools,
        }
        self.capture.assert_safe(projection)
        kind = self._derive_kind(request)
        key = self._cache_key(request)
        cached = self.cache.get(key)
        if cached is not None:
            self.capture.add_model_call({
                "kind": kind,
                "model": self.model,
                "cache_hit": True,
                "input": projection,
                "output": cached.get("output"),
                "usage": cached.get("usage") or {},
            })
            return AIMessage(
                content=cached.get("content") or "",
                tool_calls=cached.get("tool_calls") or [],
            )
        last: BaseException | None = None
        for attempt in range(3):
            attempt_id = self.budget.reserve(
                {**self._context(), "kind": kind},
                self._estimate(request),
            )
            started = time.perf_counter()
            try:
                response = await self.inner.ainvoke(request, config, **kwargs)
            except BaseException as exc:
                last = exc
                self.budget.settle(attempt_id, actual_usd=0.0, status="failed",
                                   error=f"{type(exc).__name__}: {exc}"[:500])
                if attempt < 2 and _retryable(exc):
                    await asyncio.sleep(min(2 ** attempt, 2))
                    continue
                self.capture.fail(kind, exc, extra={"retry": attempt})
                raise
            usage = _usage(response)
            actual = self.pricing.cost(self.model, usage[0], usage[1], usage[2])
            self.budget.settle(
                attempt_id, actual_usd=actual, status="succeeded",
                usage={
                    "input_tokens": usage[0],
                    "output_tokens": usage[1],
                    "cache_hit_input_tokens": usage[2],
                },
            )
            output = {
                "content": str(getattr(response, "content", response)),
                "tool_calls": _jsonable(getattr(response, "tool_calls", None) or []),
                "response_metadata": _jsonable(
                    getattr(response, "response_metadata", None) or {}
                ),
            }
            self.cache.put(key, {
                "content": output["content"],
                "tool_calls": output["tool_calls"],
                "usage": {
                    "input_tokens": usage[0],
                    "output_tokens": usage[1],
                    "cache_hit_input_tokens": usage[2],
                },
                "output": output,
            })
            self.capture.add_model_call({
                "kind": kind,
                "model": self.model,
                "cache_hit": False,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                "input": projection,
                "output": output,
                "usage": {
                    "input_tokens": usage[0],
                    "output_tokens": usage[1],
                    "cache_hit_input_tokens": usage[2],
                },
            })
            return response
        raise last or RuntimeError("model call failed")

    async def astream(self, request, config=None, **kwargs) -> AsyncIterator[Any]:
        projection = {
            "request": _projection(request),
            "schema": self.schema,
            "tools": self.tools,
        }
        self.capture.assert_safe(projection)
        kind = self._derive_kind(request)
        key = self._cache_key(request)
        cached = self.cache.get(key)
        if cached is not None:
            self.capture.add_model_call({
                "kind": kind,
                "model": self.model,
                "cache_hit": True,
                "input": projection,
                "output": cached.get("output"),
                "usage": cached.get("usage") or {},
            })
            tool_chunks = [
                {
                    "name": call.get("name"),
                    "args": json.dumps(call.get("args") or {}, ensure_ascii=False),
                    "id": call.get("id"),
                    "index": index,
                }
                for index, call in enumerate(cached.get("tool_calls") or [])
            ]
            yield AIMessageChunk(
                content=cached.get("content") or "",
                tool_call_chunks=tool_chunks,
            )
            return

        last: BaseException | None = None
        for attempt in range(3):
            attempt_id = self.budget.reserve(
                {**self._context(), "kind": kind},
                self._estimate(request),
            )
            started = time.perf_counter()
            yielded = False
            chunks = []
            try:
                async for chunk in self.inner.astream(request, config, **kwargs):
                    yielded = True
                    chunks.append(chunk)
                    yield chunk
            except BaseException as exc:
                last = exc
                self.budget.settle(
                    attempt_id,
                    actual_usd=self._estimate(request) if yielded else 0.0,
                    status="failed_partial" if yielded else "failed",
                    error=f"{type(exc).__name__}: {exc}"[:500],
                )
                if not yielded and attempt < 2 and _retryable(exc):
                    await asyncio.sleep(min(2 ** attempt, 2))
                    continue
                self.capture.fail(
                    kind, exc, extra={"retry": attempt, "partial": yielded}
                )
                raise
            final = chunks[-1] if chunks else None
            usage = _usage(final) if final is not None else (0, 0, 0)
            if not usage[0]:
                usage = (estimate_tokens(_request_text(request)), 0, 0)
            content = "".join(str(getattr(chunk, "text", "") or "") for chunk in chunks)
            tool_call_chunks = []
            for chunk in chunks:
                tool_call_chunks.extend(
                    getattr(chunk, "tool_call_chunks", None) or []
                )
            tool_calls = _merge_tool_call_chunks(tool_call_chunks)
            if not usage[1]:
                usage = (usage[0], estimate_tokens(content), usage[2])
            actual = self.pricing.cost(self.model, usage[0], usage[1], usage[2])
            self.budget.settle(
                attempt_id, actual_usd=actual, status="succeeded",
                usage={
                    "input_tokens": usage[0],
                    "output_tokens": usage[1],
                    "cache_hit_input_tokens": usage[2],
                },
            )
            output = {
                "content": content,
                "tool_calls": tool_calls,
                "finish_reason": (
                    (getattr(final, "response_metadata", None) or {}).get(
                        "finish_reason"
                    )
                    if final is not None else None
                ),
            }
            self.cache.put(key, {
                "content": content,
                "tool_calls": tool_calls,
                "usage": {
                    "input_tokens": usage[0],
                    "output_tokens": usage[1],
                    "cache_hit_input_tokens": usage[2],
                },
                "output": output,
            })
            self.capture.add_model_call({
                "kind": kind,
                "model": self.model,
                "cache_hit": False,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                "input": projection,
                "output": output,
                "usage": {
                    "input_tokens": usage[0],
                    "output_tokens": usage[1],
                    "cache_hit_input_tokens": usage[2],
                },
            })
            return
        raise last or RuntimeError("streaming model call failed")


def _merge_tool_call_chunks(chunks: list[dict]) -> list[dict]:
    merged: dict[int, dict] = {}
    for chunk in chunks:
        slot = merged.setdefault(
            int(chunk.get("index", 0)),
            {"name": "", "args": "", "id": ""},
        )
        if chunk.get("name"):
            slot["name"] = chunk["name"]
        if chunk.get("args"):
            slot["args"] += chunk["args"]
        if chunk.get("id"):
            slot["id"] = chunk["id"]
    output = []
    for index in sorted(merged):
        slot = merged[index]
        try:
            args = json.loads(slot["args"] or "{}")
        except json.JSONDecodeError:
            args = {}
        output.append({"name": slot["name"], "args": args, "id": slot["id"]})
    return output


class BudgetedStructured(_CapturedCalls):
    def __init__(self, *, schema_class=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.schema_class = schema_class

    def _context(self) -> dict:
        return {
            "case_id": (self.capture.case or {}).get("id"),
            "split": self.capture.split,
            "turn": self.capture.turn,
            "arm": self.capture.arm,
            "kind": self.kind,
            "model": self.model,
        }

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def _parse(self, row):
        if not isinstance(row, dict):
            return row
        if row.get("parsing_error"):
            raise RuntimeError(f"structured output parse error: {row['parsing_error']}")
        return row.get("parsed")

    def _cache_response(self, row, raw) -> dict:
        parsed = self._parse(row)
        return {
            "parsed": _jsonable(parsed),
            "raw": {
                "content": str(getattr(raw, "content", "")),
                "tool_calls": _jsonable(getattr(raw, "tool_calls", None) or []),
            },
            "usage": _usage(raw),
        }

    def invoke(self, request, config=None, **kwargs):
        projection = {"request": _projection(request), "schema": self.schema}
        self.capture.assert_safe(projection)
        key = self._cache_key(request)
        cached = self.cache.get(key)
        if cached is not None:
            self.capture.add_model_call({
                "kind": self.kind,
                "model": self.model,
                "cache_hit": True,
                "input": projection,
                "output": cached,
            })
            parsed = cached.get("parsed")
            return self.schema_class.model_validate(parsed) if self.schema_class else parsed
        last = None
        for attempt in range(3):
            attempt_id = self.budget.reserve(
                {**self._context(), "kind": self.kind},
                self._estimate(request),
            )
            started = time.perf_counter()
            try:
                row = self.inner.invoke(request, config, **kwargs)
                parsed = self._parse(row)
            except BaseException as exc:
                last = exc
                self.budget.settle(attempt_id, actual_usd=0.0, status="failed",
                                   error=f"{type(exc).__name__}: {exc}"[:500])
                if attempt < 2 and _retryable(exc):
                    time.sleep(min(2 ** attempt, 2))
                    continue
                self.capture.fail(self.kind, exc, extra={"retry": attempt})
                raise
            raw = row.get("raw") if isinstance(row, dict) else row
            usage = _usage(raw)
            actual = self.pricing.cost(self.model, usage[0], usage[1], usage[2])
            self.budget.settle(
                attempt_id, actual_usd=actual, status="succeeded",
                usage={
                    "input_tokens": usage[0],
                    "output_tokens": usage[1],
                    "cache_hit_input_tokens": usage[2],
                },
            )
            payload = self._cache_response(row, raw)
            self.cache.put(key, payload)
            self.capture.add_model_call({
                "kind": self.kind,
                "model": self.model,
                "cache_hit": False,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                "input": projection,
                "output": payload,
            })
            return parsed
        raise last or RuntimeError("structured call failed")

    async def ainvoke(self, request, config=None, **kwargs):
        projection = {"request": _projection(request), "schema": self.schema}
        self.capture.assert_safe(projection)
        key = self._cache_key(request)
        cached = self.cache.get(key)
        if cached is not None:
            self.capture.add_model_call({
                "kind": self.kind,
                "model": self.model,
                "cache_hit": True,
                "input": projection,
                "output": cached,
            })
            parsed = cached.get("parsed")
            return self.schema_class.model_validate(parsed) if self.schema_class else parsed
        last = None
        for attempt in range(3):
            attempt_id = self.budget.reserve(
                {**self._context(), "kind": self.kind},
                self._estimate(request),
            )
            started = time.perf_counter()
            try:
                row = await self.inner.ainvoke(request, config, **kwargs)
                parsed = self._parse(row)
            except BaseException as exc:
                last = exc
                self.budget.settle(attempt_id, actual_usd=0.0, status="failed",
                                   error=f"{type(exc).__name__}: {exc}"[:500])
                if attempt < 2 and _retryable(exc):
                    await asyncio.sleep(min(2 ** attempt, 2))
                    continue
                self.capture.fail(self.kind, exc, extra={"retry": attempt})
                raise
            raw = row.get("raw") if isinstance(row, dict) else row
            usage = _usage(raw)
            actual = self.pricing.cost(self.model, usage[0], usage[1], usage[2])
            self.budget.settle(
                attempt_id, actual_usd=actual, status="succeeded",
                usage={
                    "input_tokens": usage[0],
                    "output_tokens": usage[1],
                    "cache_hit_input_tokens": usage[2],
                },
            )
            payload = self._cache_response(row, raw)
            self.cache.put(key, payload)
            self.capture.add_model_call({
                "kind": self.kind,
                "model": self.model,
                "cache_hit": False,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                "input": projection,
                "output": payload,
            })
            return parsed
        raise last or RuntimeError("structured call failed")


class BudgetedJsonStructured(_CapturedCalls):
    """DeepSeek thinking-mode compatible structured output.

    The model receives a JSON Schema instruction and returns plain text. This
    avoids unsupported ``tool_choice`` while preserving the production node's
    Pydantic contract. It is a diagnostic integration arm, not the default.
    """

    def __init__(self, *, schema_class=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.schema_class = schema_class
        self.instruction = json_structured_instruction(self.schema)

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def _request(self, request):
        return _add_instruction(request, self.instruction)

    def _parse(self, response):
        from scripts.eval_prompts import parse_json_object

        raw = str(getattr(response, "content", response))
        payload = parse_json_object(raw)
        return self.schema_class.model_validate(payload) if self.schema_class else payload

    def _cache_payload(self, response, parsed) -> dict:
        return {
            "content": str(getattr(response, "content", response)),
            "parsed": _jsonable(parsed),
            "usage": _usage(response),
        }

    def invoke(self, request, config=None, **kwargs):
        projection = {
            "request": _projection(request),
            "schema": self.schema,
            "format": "json-text-fallback",
        }
        self.capture.assert_safe(projection)
        key = self._cache_key(request)
        cached = self.cache.get(key)
        if cached is not None:
            self.capture.add_model_call({
                "kind": self.kind,
                "model": self.model,
                "cache_hit": True,
                "outcome": "succeeded",
                "input": projection,
                "output": cached,
            })
            payload = cached.get("parsed")
            return self.schema_class.model_validate(payload) if self.schema_class else payload
        current_request = self._request(request)
        last = None
        for attempt in range(3):
            attempt_id = self.budget.reserve(
                {**self._context(), "kind": self.kind},
                self._estimate(current_request),
            )
            started = time.perf_counter()
            try:
                response = self.inner.invoke(current_request, config, **kwargs)
                parsed = self._parse(response)
            except BaseException as exc:
                last = exc
                self.budget.settle(attempt_id, actual_usd=0.0, status="failed",
                                   error=f"{type(exc).__name__}: {exc}"[:500])
                if attempt < 2 and _retryable(exc):
                    time.sleep(min(2 ** attempt, 2))
                    continue
                self.capture.fail(self.kind, exc, extra={"retry": attempt})
                raise
            usage = _usage(response)
            self.budget.settle(
                attempt_id,
                actual_usd=self.pricing.cost(self.model, usage[0], usage[1], usage[2]),
                status="succeeded",
                usage={
                    "input_tokens": usage[0],
                    "output_tokens": usage[1],
                    "cache_hit_input_tokens": usage[2],
                },
            )
            payload = self._cache_payload(response, parsed)
            self.cache.put(key, payload)
            self.capture.add_model_call({
                "kind": self.kind,
                "model": self.model,
                "cache_hit": False,
                "outcome": "succeeded",
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                "input": projection,
                "output": payload,
            })
            return parsed
        raise last or RuntimeError("JSON structured call failed")

    async def ainvoke(self, request, config=None, **kwargs):
        projection = {
            "request": _projection(request),
            "schema": self.schema,
            "format": "json-text-fallback",
        }
        self.capture.assert_safe(projection)
        key = self._cache_key(request)
        cached = self.cache.get(key)
        if cached is not None:
            self.capture.add_model_call({
                "kind": self.kind,
                "model": self.model,
                "cache_hit": True,
                "outcome": "succeeded",
                "input": projection,
                "output": cached,
            })
            payload = cached.get("parsed")
            return self.schema_class.model_validate(payload) if self.schema_class else payload
        current_request = self._request(request)
        last = None
        for attempt in range(3):
            attempt_id = self.budget.reserve(
                {**self._context(), "kind": self.kind},
                self._estimate(current_request),
            )
            started = time.perf_counter()
            try:
                response = await self.inner.ainvoke(current_request, config, **kwargs)
                parsed = self._parse(response)
            except BaseException as exc:
                last = exc
                self.budget.settle(attempt_id, actual_usd=0.0, status="failed",
                                   error=f"{type(exc).__name__}: {exc}"[:500])
                if attempt < 2 and _retryable(exc):
                    await asyncio.sleep(min(2 ** attempt, 2))
                    continue
                self.capture.fail(self.kind, exc, extra={"retry": attempt})
                raise
            usage = _usage(response)
            self.budget.settle(
                attempt_id,
                actual_usd=self.pricing.cost(self.model, usage[0], usage[1], usage[2]),
                status="succeeded",
                usage={
                    "input_tokens": usage[0],
                    "output_tokens": usage[1],
                    "cache_hit_input_tokens": usage[2],
                },
            )
            payload = self._cache_payload(response, parsed)
            self.cache.put(key, payload)
            self.capture.add_model_call({
                "kind": self.kind,
                "model": self.model,
                "cache_hit": False,
                "outcome": "succeeded",
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                "input": projection,
                "output": payload,
            })
            return parsed
        raise last or RuntimeError("JSON structured call failed")


class MeteredEmbedder:
    def __init__(self, inner, *, model, pricing, budget, capture, kind):
        self.inner = inner
        self.model = model
        self.pricing = pricing
        self.budget = budget
        self.capture = capture
        self.kind = kind

    def _context(self):
        return {
            "case_id": (self.capture.case or {}).get("id"),
            "split": self.capture.split,
            "turn": self.capture.turn,
            "arm": self.capture.arm,
            "kind": self.kind,
            "model": self.model,
        }

    def embed(self, texts: list[str]):
        request = "\0".join(texts)
        self.capture.assert_safe({"request": request, "kind": self.kind})
        tokens = max(1, estimate_tokens(request))
        estimate = self.pricing.cost(self.model, tokens, 0)
        before = self.inner.stats() if hasattr(self.inner, "stats") else {}
        attempt_id = self.budget.reserve(self._context(), estimate)
        try:
            vectors = self.inner.embed(texts)
        except BaseException as exc:
            self.budget.settle(attempt_id, actual_usd=0.0, status="failed",
                               error=f"{type(exc).__name__}: {exc}"[:500])
            self.capture.fail(self.kind, exc)
            raise
        after = self.inner.stats() if hasattr(self.inner, "stats") else {}
        cache_hit = int(after.get("hits", 0)) > int(before.get("hits", 0))
        actual = 0.0 if cache_hit else estimate
        self.budget.settle(
            attempt_id, actual_usd=actual,
            status="cache_hit" if cache_hit else "succeeded",
            usage={"input_tokens": 0 if cache_hit else tokens, "output_tokens": 0},
        )
        self.capture.add_model_call({
            "kind": self.kind,
            "model": self.model,
            "cache_hit": cache_hit,
            "input": {"texts": texts},
            "output": {"vectors": len(vectors)},
            "usage": {"input_tokens": 0 if cache_hit else tokens},
        })
        return vectors


class MeteredReranker(MeteredEmbedder):
    def rerank(self, query: str, documents: list[str], top_n: int | None = None):
        request = query + "\0" + "\0".join(documents)
        self.capture.assert_safe({"request": request, "kind": self.kind})
        tokens = max(1, estimate_tokens(request))
        estimate = self.pricing.cost(self.model, tokens, 0)
        before = self.inner.stats() if hasattr(self.inner, "stats") else {}
        attempt_id = self.budget.reserve(self._context(), estimate)
        try:
            results = self.inner.rerank(query, documents, top_n=top_n)
        except BaseException as exc:
            self.budget.settle(attempt_id, actual_usd=0.0, status="failed",
                               error=f"{type(exc).__name__}: {exc}"[:500])
            self.capture.fail(self.kind, exc)
            raise
        after = self.inner.stats() if hasattr(self.inner, "stats") else {}
        cache_hit = int(after.get("hits", 0)) > int(before.get("hits", 0))
        actual = 0.0 if cache_hit else estimate
        self.budget.settle(
            attempt_id, actual_usd=actual,
            status="cache_hit" if cache_hit else "succeeded",
            usage={"input_tokens": 0 if cache_hit else tokens, "output_tokens": 0},
        )
        self.capture.add_model_call({
            "kind": self.kind,
            "model": self.model,
            "cache_hit": cache_hit,
            "input": {"query": query, "documents": documents},
            "output": {"results": _jsonable(results)},
            "usage": {"input_tokens": 0 if cache_hit else tokens},
        })
        return results


class GraphCaptureProxy:
    def __init__(self, inner, capture: TraceCapture) -> None:
        self.inner = inner
        self.capture = capture

    async def ainvoke(self, *args, **kwargs):
        result = await self.inner.ainvoke(*args, **kwargs)
        self.capture.add_graph_result(_jsonable(result))
        return result

    def __getattr__(self, name):
        return getattr(self.inner, name)


def _load_credential_env(path: Path) -> None:
    from dotenv import dotenv_values

    values = dotenv_values(Path(path))
    for name in SELECTED_ENV_KEYS:
        if values.get(name):
            os.environ[name] = str(values[name])


def _install_environment(paths: RuntimePaths, *, credential_env_file: Path) -> None:
    _load_credential_env(credential_env_file)
    os.environ["OPENAI_MODEL"] = "deepseek-flash"
    os.environ["DATABASE_URL"] = f"sqlite:///{paths.metadata_db.resolve()}"
    os.environ["MILVUS_DB_PATH"] = str(paths.snapshot.resolve())
    os.environ["LLM_MAX_RETRIES"] = "0"
    os.environ["LANGFUSE_ENABLED"] = "false"
    os.environ["MCP_LOGISTICS_URL"] = ""
    os.environ["MCP_AFTERSALES_URL"] = ""
    os.environ["RETRIEVAL_ROUTE"] = "hybrid_rerank"
    os.environ["ANSWER_VERIFICATION_ENABLED"] = "false"
    os.environ["RAG_GAP_TRIGGER_ENABLED"] = "false"
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeContractError("OPENAI_API_KEY missing from credential env file")
    if not os.environ.get("EMBEDDING_API_KEY"):
        raise RuntimeContractError("EMBEDDING_API_KEY missing from credential env file")


def build_instrumented_app(
    *,
    paths: RuntimePaths,
    credential_env_file: Path,
    pricing: PricingTable,
    budget: R2Budget,
    capture: TraceCapture,
    exact_cache: ExactCallCache,
    namespace: str,
    answer_verification_enabled: bool = False,
    structured_mode: str = "production",
):
    if structured_mode not in {"production", "json-fallback"}:
        raise RuntimeContractError(f"unsupported structured_mode: {structured_mode}")
    _install_environment(paths, credential_env_file=credential_env_file)
    from app.config import Settings
    from app.db import make_engine
    from app.execution import models as _execution_models  # noqa: F401 - register tables
    from app.models import Base

    settings = Settings(
        _env_file=None,
        answer_verification_enabled=answer_verification_enabled,
    )
    if settings.openai_model != "deepseek-flash":
        raise RuntimeContractError(
            f"business model must be deepseek-flash, got {settings.openai_model}"
        )
    Base.metadata.create_all(make_engine(settings))

    import app.embedding as embedding_module
    import app.llm as llm_module
    import app.main as main_module
    import app.rerank as rerank_module
    from langchain_openai import ChatOpenAI as RealChatOpenAI

    def chat_factory(*args, **kwargs):
        kwargs["max_retries"] = 0
        kwargs["stream_usage"] = True
        inner = RealChatOpenAI(*args, **kwargs)
        return BudgetedChat(
            inner=inner,
            model="deepseek-flash",
            pricing=pricing,
            budget=budget,
            cache=exact_cache,
            capture=capture,
            namespace=namespace,
            kind="llm",
            structured_mode=structured_mode,
        )

    llm_module.ChatOpenAI = chat_factory

    def embedder_factory(settings):
        from app.embedding import SiliconFlowEmbedder

        inner = SiliconFlowEmbedder(
            settings.embedding_api_base,
            settings.embedding_api_key,
            settings.embedding_model,
            batch_size=settings.embedding_batch_size,
        )
        cached = EmbeddingCache(
            inner,
            paths.cache_dir / "embedding.json",
            settings.embedding_model,
        )
        return MeteredEmbedder(
            cached,
            model=settings.embedding_model,
            pricing=pricing,
            budget=budget,
            capture=capture,
            kind="embedding",
        )

    def reranker_factory(settings):
        key = settings.rerank_api_key or settings.embedding_api_key
        if not key:
            return None
        from app.rerank import SiliconFlowReranker

        inner = SiliconFlowReranker(
            settings.rerank_api_base,
            key,
            settings.rerank_model,
        )
        cached = RerankCache(inner, paths.cache_dir / "rerank.json", settings.rerank_model)
        return MeteredReranker(
            cached,
            model=settings.rerank_model,
            pricing=pricing,
            budget=budget,
            capture=capture,
            kind="rerank",
        )

    embedding_module.make_embedder = embedder_factory
    rerank_module.make_reranker = reranker_factory

    app = main_module.create_app(settings)
    from app.tools.mcp_client import McpService

    # The HTTP route calls sync() every turn. Empty env URLs still create a
    # non-empty transport config, so bind an explicitly empty service for the
    # isolated run instead of generating failed network attempts.
    app.state.mcp_service = McpService(app.state.registry, {})
    capture.graph_results = []
    app.state.graph = GraphCaptureProxy(app.state.graph, capture)

    original_execute = app.state.engine.execute

    async def execute(self, name, args_json, **kwargs):
        result = await original_execute(name, args_json, **kwargs)
        call = {
            "name": name,
            "args": _safe_json(args_json) or {},
            "result": _safe_json(result) if _safe_json(result) is not None else result,
            "is_error": str(result).startswith("错误:"),
            "tool_call_id": kwargs.get("tool_call_id"),
        }
        capture.add_tool_call(call)
        return result

    app.state.engine.execute = MethodType(execute, app.state.engine)
    return app, settings


def _safe_json(text):
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    return value


def _parse_sse(raw: str) -> list[dict]:
    events = []
    for block in raw.split("\n\n"):
        data = next(
            (line[len("data: "):] for line in block.splitlines()
             if line.startswith("data: ")),
            None,
        )
        if data is None:
            continue
        try:
            events.append(json.loads(data))
        except json.JSONDecodeError:
            continue
    return events


async def post_turn(client, *, message: str, session_id: int | None, request_id: str):
    payload = {"message": message, "session_id": session_id,
               "client_request_id": request_id}
    started = time.perf_counter()
    raw = ""
    async with client.stream("POST", "/api/chat", json=payload) as response:
        response.raise_for_status()
        run_id = response.headers.get("x-run-id")
        async for chunk in response.aiter_text():
            raw += chunk
    latency_ms = (time.perf_counter() - started) * 1000
    return {
        "run_id": run_id,
        "latency_ms": latency_ms,
        "events": _parse_sse(raw),
        "raw_sse": raw,
    }


def _route_nodes(result: dict) -> list[str]:
    nodes = []
    for line in result.get("trace") or []:
        if isinstance(line, str) and line.startswith("node="):
            nodes.append(line.split("=", 1)[1].split()[0])
    return list(dict.fromkeys(nodes))


def _cost_for_case(records: list[dict], case_id: str, arm: str) -> dict:
    costs = {
        "generation_cost_usd": 0.0,
        "judge_cost_usd": 0.0,
        "embedding_cost_usd": 0.0,
        "rerank_cost_usd": 0.0,
        "other_cost_usd": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
    }
    for row in records:
        context = row.get("context") or {}
        if context.get("case_id") != case_id or context.get("arm") != arm:
            continue
        amount = float(row.get("actual_usd") or 0.0)
        kind = str(context.get("kind") or "")
        if kind in {"answer", "intent", "resolve", "expand", "rewrite", "summary",
                    "structured", "llm"}:
            costs["generation_cost_usd"] += amount
        elif kind in {"evidence_judge", "answer_verifier", "faithfulness_judge"}:
            costs["judge_cost_usd"] += amount
        elif kind == "embedding":
            costs["embedding_cost_usd"] += amount
        elif kind == "rerank":
            costs["rerank_cost_usd"] += amount
        else:
            costs["other_cost_usd"] += amount
        usage = row.get("usage") or {}
        costs["input_tokens"] += int(usage.get("input_tokens") or 0)
        costs["output_tokens"] += int(usage.get("output_tokens") or 0)
    return {key: round(value, 10) if isinstance(value, float) else value
            for key, value in costs.items()}


def infer_recorded_call_kind(call: dict) -> str:
    kind = str(call.get("kind") or "")
    if kind != "structured_json":
        return kind
    text = json.dumps(
        {
            "input": call.get("input"),
            "output": call.get("output"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    if "EvidenceVerdict" in text or "证据审核员" in text:
        return "evidence_judge"
    if "AnswerVerdict" in text or "最终答案的独立复核员" in text:
        return "answer_verifier"
    if "ResolvedQuestion" in text or "问法澄清器" in text:
        return "resolve"
    if "QueryRewrite" in text or "标准问法" in text:
        return "rewrite"
    if "NormalizedQuestionOutput" in text or "标准 FAQ 问题" in text:
        return "flywheel_normalize"
    if "DedupDecisionOutput" in text or "候选中的某一条是同一个意思" in text:
        return "flywheel_dedup"
    return "structured_json"


def reclassify_captured_costs(result: dict, pricing: PricingTable) -> dict:
    costs = {
        "generation_cost_usd": 0.0,
        "judge_cost_usd": 0.0,
        "embedding_cost_usd": 0.0,
        "rerank_cost_usd": 0.0,
        "other_cost_usd": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
    }
    for call in result.get("model_calls") or []:
        kind = infer_recorded_call_kind(call)
        usage = call.get("usage") or (call.get("output") or {}).get("usage") or {}
        if isinstance(usage, (list, tuple)):
            usage = {
                "input_tokens": usage[0] if len(usage) > 0 else 0,
                "output_tokens": usage[1] if len(usage) > 1 else 0,
                "cache_hit_input_tokens": usage[2] if len(usage) > 2 else 0,
            }
        input_tokens = int(usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        cached_tokens = int(usage.get("cache_hit_input_tokens") or 0)
        amount = 0.0 if call.get("cache_hit") else pricing.cost(
            str(call.get("model") or ""),
            input_tokens,
            output_tokens,
            cached_tokens,
        )
        if kind in {
            "answer", "intent", "resolve", "expand", "rewrite", "summary",
            "structured", "structured_json", "llm",
        }:
            costs["generation_cost_usd"] += amount
        elif kind in {"evidence_judge", "answer_verifier", "faithfulness_judge"}:
            costs["judge_cost_usd"] += amount
        elif kind == "embedding":
            costs["embedding_cost_usd"] += amount
        elif kind == "rerank":
            costs["rerank_cost_usd"] += amount
        else:
            costs["other_cost_usd"] += amount
        costs["input_tokens"] += input_tokens
        costs["output_tokens"] += output_tokens
    for key, value in costs.items():
        result[key] = round(value, 10) if isinstance(value, float) else value
    result["cost_source"] = "captured_calls_reclassified_v2"
    return result


async def run_case(
    app,
    client,
    case: dict,
    *,
    capture: TraceCapture,
    budget: R2Budget,
    output_root: Path,
) -> dict:
    case_id = str(case["id"])
    session_id = None
    turn_rows = []
    latest_result = None
    total_latency = 0.0
    for index, turn in enumerate(case["turns"]):
        capture.start_case(case=case, split=str(case["split"]), turn=index)
        before_graphs = len(capture.graph_results)
        response = await post_turn(
            client,
            message=str(turn["content"]),
            session_id=session_id,
            request_id=f"e2e-r2-{case_id}-{index}",
        )
        latest_result = capture.graph_results[-1] if len(capture.graph_results) > before_graphs else None
        if latest_result is None:
            raise RuntimeContractError(f"{case_id}: graph result was not captured")
        session_event = next(
            (event for event in response["events"] if event.get("type") == "session"),
            None,
        )
        if session_event is not None:
            session_id = int(session_event["session_id"])
        if any(event.get("type") == "error" for event in response["events"]):
            capture.fail(
                "chat_endpoint",
                RuntimeError("chat endpoint emitted error"),
                extra={"run_id": response.get("run_id")},
            )
        total_latency += float(response["latency_ms"])
        turn_rows.append({
            "turn": index,
            "message": turn["content"],
            "session_id": session_id,
            "run_id": response.get("run_id"),
            "latency_ms": round(float(response["latency_ms"]), 3),
            "events": response["events"],
        })

    tools = capture.case_tool_calls(case_id)
    model_calls = capture.case_model_calls(case_id)
    failures = capture.case_failures(case_id)
    state = latest_result or {}
    evidence = resolve_citation_evidence(state, tools)
    answer = str(state.get("final_reply") or "")
    from scripts.eval_checks import classify_behavior

    outcome = {
        "answer": answer,
        "behavior": classify_behavior(answer),
        "evidence": evidence,
        "tool_calls": tools,
        "technical_failures": failures,
        "route_nodes": _route_nodes(state),
    }
    score = score_case(case, outcome)
    costs = _cost_for_case(budget.records(), case_id, capture.arm)
    result = {
        "case_id": case_id,
        "split": case["split"],
        "source_group": case["source_group"],
        "turns": turn_rows,
        "answer": answer,
        "state": {
            key: _jsonable(state.get(key))
            for key in (
                "intent", "intent_confidence", "evidence", "retrieved_chunks",
                "gate_passed", "low_confidence", "trace", "rag_rounds",
                "rag_stop_reason",
            )
        },
        "evidence_map": _jsonable(evidence),
        "model_calls": model_calls,
        "tool_calls": tools,
        "technical_failures": failures,
        "score": score,
        "task_success": bool(score.get("task_success")),
        "review_required": bool(score.get("review_required")),
        "failure_labels": list(score.get("failure_labels") or []),
        "latency_ms": round(total_latency, 3),
        "input_tokens": costs["input_tokens"],
        "output_tokens": costs["output_tokens"],
        **{key: costs[key] for key in (
            "generation_cost_usd", "judge_cost_usd", "embedding_cost_usd",
            "rerank_cost_usd", "other_cost_usd",
        )},
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "cases.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
    review = {**result, "manual_review_status": "pending"}
    with (output_root / "manual-review.jsonl").open("a", encoding="utf-8") as stream:
        if result["review_required"] or result["failure_labels"]:
            stream.write(json.dumps(review, ensure_ascii=False, sort_keys=True) + "\n")
    return result


async def run_split(
    cases: list[dict],
    *,
    app,
    capture: TraceCapture,
    budget: R2Budget,
    output_root: Path,
) -> list[dict]:
    from httpx import ASGITransport, AsyncClient

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "cases.jsonl").write_text("", encoding="utf-8")
    (output_root / "manual-review.jsonl").write_text("", encoding="utf-8")
    results = []
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://e2e-r2",
        timeout=180.0,
    ) as client:
        for case in cases:
            results.append(await run_case(
                app, client, case,
                capture=capture,
                budget=budget,
                output_root=output_root,
            ))
    return results


def _nearest_rank(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def summarize_results(results: list[dict]) -> dict:
    failures: dict[str, int] = {}
    expected_actual = {}
    citation_statuses = {}
    for result in results:
        for label in result.get("failure_labels") or []:
            failures[label] = failures.get(label, 0) + 1
        score = result.get("score") or {}
        transition = (
            f"{score.get('expected_behavior', '')}->"
            f"{score.get('actual_behavior', '')}"
        )
        expected_actual[transition] = expected_actual.get(transition, 0) + 1
        status = str((score.get("citations") or {}).get("status") or "unknown")
        citation_statuses[status] = citation_statuses.get(status, 0) + 1
    latencies = [float(row.get("latency_ms") or 0.0) for row in results]
    fact_rows = [
        row for row in results
        if ((row.get("score") or {}).get("facts") or {}).get("rows")
    ]
    condition_rows = [
        row for row in results
        if int((((row.get("score") or {}).get("conditions") or {}).get("measured") or 0)) > 0
    ]
    entity_rows = [
        row for row in results
        if int((((row.get("score") or {}).get("entities") or {}).get("measured") or 0)) > 0
    ]
    tool_rows = [
        row for row in results
        if int((((row.get("score") or {}).get("tools") or {}).get("required") or 0)) > 0
    ]
    route_rows = [
        row for row in results
        if int((((row.get("score") or {}).get("route") or {}).get("required") or 0)) > 0
    ]
    return {
        "n": len(results),
        "success": sum(bool(row.get("task_success")) for row in results),
        "review_required": sum(
            bool(row.get("review_required") or (row.get("score") or {}).get("review_required"))
            for row in results
        ),
        "technical_failures": sum(
            bool((row.get("score") or {}).get("technical_failure")) for row in results
        ),
        "failure_labels": dict(sorted(failures.items())),
        "expected_actual_behavior": dict(sorted(expected_actual.items())),
        "facts": {
            "measured": len(fact_rows),
            "passed": sum(bool(row["score"]["facts"]["passed"]) for row in fact_rows),
        },
        "conditions": {
            "measured": len(condition_rows),
            "passed": sum(bool(row["score"]["conditions"]["passed"]) for row in condition_rows),
        },
        "entities": {
            "measured": len(entity_rows),
            "passed": sum(bool(row["score"]["entities"]["passed"]) for row in entity_rows),
        },
        "citations": dict(sorted(citation_statuses.items())),
        "tools": {
            "measured": len(tool_rows),
            "passed": sum(bool(row["score"]["tools"]["passed"]) for row in tool_rows),
        },
        "routes": {
            "measured": len(route_rows),
            "passed": sum(bool(row["score"]["route"]["passed"]) for row in route_rows),
        },
        "refusal": sum(
            (row.get("score") or {}).get("actual_behavior") == "refuse" for row in results
        ),
        "clarify": sum(
            (row.get("score") or {}).get("actual_behavior") == "clarify" for row in results
        ),
        "empty_answers": sum(not str(row.get("answer") or "").strip() for row in results),
        "generation_cost_usd": round(sum(
            float(row.get("generation_cost_usd") or 0.0) for row in results
        ), 10),
        "judge_cost_usd": round(sum(
            float(row.get("judge_cost_usd") or 0.0) for row in results
        ), 10),
        "embedding_cost_usd": round(sum(
            float(row.get("embedding_cost_usd") or 0.0) for row in results
        ), 10),
        "rerank_cost_usd": round(sum(
            float(row.get("rerank_cost_usd") or 0.0) for row in results
        ), 10),
        "other_cost_usd": round(sum(
            float(row.get("other_cost_usd") or 0.0) for row in results
        ), 10),
        "input_tokens": sum(int(row.get("input_tokens") or 0) for row in results),
        "output_tokens": sum(int(row.get("output_tokens") or 0) for row in results),
        "p50_latency_ms": _nearest_rank(latencies, 0.50),
        "p95_latency_ms": _nearest_rank(latencies, 0.95),
        "case_ids": [row["case_id"] for row in results if row.get("case_id")],
    }
