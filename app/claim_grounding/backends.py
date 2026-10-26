from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.claim_grounding.budget import ClaimBudgetLedger, usage_cost
from app.claim_grounding.models import (
    ClaimInput,
    GroundingResult,
    GroundingStatus,
    TokenUsage,
)

CLAIM_GROUNDING_PROMPT = """你是电商客服 claim-evidence 判定器。只使用候选证据判断一个 claim。
不得使用外部知识，不得因为证据没有提到就自动认定为矛盾。

输出状态:
- supported: 候选证据明确支持 claim 的主体、数值、条件、单位和时间口径。
- unsupported: 候选证据明确表达相反事实或冲突；必须给出有效证据 ID 和 conflict_codes。
- uncertain: 缺少必要证据、条件不明、指代不清、只能推测，或无法可靠区分支持与中立。

规则:
1. 不要使用 gold、期望行为或评分规则；这里没有这些信息。
2. 其他订单/型号的证据不能证明当前订单/型号。
3. “不支持”不等于“没有提及其它事实”；not_entailment 或没提到都只能是 uncertain。
4. 同义改写可以 supported；条件缺失默认 uncertain。
5. evidence_ids 只能从下面编号中原样选择。

【Claim】
{claim}
【结构化口径】
subject={subject}
model_number={model_number}
order_id={order_id}
as_of_date={as_of_date}
policy_version={policy_version}
unit={unit}
【候选证据】
{evidence}
只输出 JSON:
{{"status":"supported|unsupported|uncertain","reason":"...","evidence_ids":["..."],"conflict_codes":["..."]}}
"""


@dataclass(frozen=True)
class ProviderResponse:
    text: str
    usage: dict[str, int] = field(default_factory=dict)
    model_version: str = ""
    latency_ms: float = 0.0
    raw: Any = None


class ResponseCache(Protocol):
    def get(self, key: str) -> dict[str, Any] | None: ...
    def put(self, key: str, payload: dict[str, Any]) -> None: ...


class StructuredGroundingBackend:
    def __init__(
        self,
        *,
        caller: Callable[[str], ProviderResponse],
        ledger: ClaimBudgetLedger,
        pricing: dict[str, float],
        model: str = "deepseek-flash",
        prompt_version: str = "claim-grounding-api-v1",
        reserve_usd: float = 0.002,
        max_retries: int = 2,
        retry_sleep_s: float = 0.25,
        cache: ResponseCache | None = None,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.caller = caller
        self.ledger = ledger
        self.pricing = pricing
        self.model = model
        self.prompt_version = prompt_version
        self.reserve_usd = float(reserve_usd)
        self.max_retries = max(0, min(2, int(max_retries)))
        self.retry_sleep_s = max(0.0, float(retry_sleep_s))
        self.cache = cache
        self.event_sink = event_sink

    def build_prompt(self, claim: ClaimInput) -> str:
        evidence = "\n".join(
            f"[{item.evidence_id}] {item.text}" for item in claim.evidence
        )
        return CLAIM_GROUNDING_PROMPT.format(
            claim=claim.claim,
            subject=claim.subject,
            model_number=claim.model_number,
            order_id=claim.order_id,
            as_of_date=claim.as_of_date or "",
            policy_version=claim.policy_version,
            unit=claim.unit,
            evidence=evidence,
        )

    def _cache_key(self, prompt: str) -> str:
        return hashlib.sha256(
            f"{self.model}\0{self.prompt_version}\0{prompt}".encode()
        ).hexdigest()

    def _emit(self, payload: dict[str, Any]) -> None:
        if self.event_sink is not None:
            self.event_sink(payload)

    def _parse(
        self,
        response: ProviderResponse,
        claim: ClaimInput,
        *,
        cache_hit: bool,
        attempts: int,
    ) -> GroundingResult:
        usage = TokenUsage(
            input_tokens=max(0, int(response.usage.get("input_tokens") or 0)),
            output_tokens=max(0, int(response.usage.get("output_tokens") or 0)),
            cached_input_tokens=max(
                0, int(response.usage.get("cached_input_tokens") or 0)
            ),
        )
        try:
            payload = json.loads(response.text)
            if not isinstance(payload, dict):
                raise TypeError("JSON 顶层不是对象")
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            return GroundingResult(
                status="uncertain",
                reason=f"语义响应解析失败:{exc}",
                backend="structured_api",
                model_version=response.model_version or self.model,
                usage=usage,
                backend_called=True,
                cache_hit=cache_hit,
                latency_ms=response.latency_ms,
                attempts=attempts,
            )

        status = str(payload.get("status") or "")
        reason = str(payload.get("reason") or "").strip()
        evidence_ids = [str(item) for item in payload.get("evidence_ids") or []]
        conflict_codes = [str(item) for item in payload.get("conflict_codes") or []]
        allowed = claim.evidence_ids()
        if status not in {"supported", "unsupported", "uncertain"}:
            status = "uncertain"
            reason = f"未知状态;{reason}".strip(";")
        if not set(evidence_ids).issubset(allowed):
            status = "uncertain"
            reason = f"语义响应引用了上下文外 evidence:{evidence_ids};{reason}".strip(
                ";"
            )
        if status == "supported" and not evidence_ids:
            status = "uncertain"
            reason = f"supported 缺少有效引用;{reason}".strip(";")
        if status == "unsupported" and (not evidence_ids or not conflict_codes):
            status = "uncertain"
            reason = f"unsupported 缺少证据或矛盾码;{reason}".strip(";")
        return GroundingResult(
            status=status,  # type: ignore[arg-type]
            reason=reason,
            evidence_ids=list(dict.fromkeys(evidence_ids)),
            backend="structured_api",
            model_version=response.model_version or self.model,
            usage=usage,
            backend_called=True,
            cache_hit=cache_hit,
            latency_ms=response.latency_ms,
            attempts=attempts,
        )

    @staticmethod
    def _is_429(exc: BaseException) -> bool:
        status = getattr(exc, "status_code", None)
        response = getattr(exc, "response", None)
        if status is None and response is not None:
            status = getattr(response, "status_code", None)
        return status == 429 or " 429" in f" {exc} " or "rate limit" in str(exc).lower()

    def ground(self, claim: ClaimInput) -> GroundingResult:
        prompt = self.build_prompt(claim)
        key = self._cache_key(prompt)
        if self.cache is not None:
            cached = self.cache.get(key)
            if cached is not None:
                response = ProviderResponse(
                    text=str(cached.get("text") or ""),
                    usage=cached.get("usage") or {},
                    model_version=str(cached.get("model_version") or self.model),
                    latency_ms=float(cached.get("latency_ms") or 0.0),
                    raw=cached.get("raw"),
                )
                result = self._parse(response, claim, cache_hit=True, attempts=0)
                self._emit(
                    {
                        "event": "cache_hit",
                        "cache_key": key,
                        "prompt": prompt,
                        "result": result.model_dump(mode="json"),
                    }
                )
                return result

        last_error: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            reservation = self.ledger.reserve(
                self.reserve_usd,
                model=self.model,
                kind="claim_grounding",
                note=f"prompt_sha256={hashlib.sha256(prompt.encode()).hexdigest()}",
            )
            started = time.perf_counter()
            try:
                response = self.caller(prompt)
            except BaseException as exc:
                last_error = exc
                # A transport failure may still be billable; keep the conservative
                # reservation in the ledger rather than pretending it cost zero.
                self.ledger.settle(
                    reservation,
                    actual_usd=self.reserve_usd,
                    usage={},
                    note=(
                        f"failed_attempt:{type(exc).__name__};"
                        f"prompt_sha256={hashlib.sha256(prompt.encode()).hexdigest()}"
                    ),
                )
                self._emit(
                    {
                        "event": "backend_error",
                        "attempt": attempt + 1,
                        "prompt": prompt,
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:500],
                        "latency_ms": (time.perf_counter() - started) * 1000,
                    }
                )
                if not self._is_429(exc) or attempt >= self.max_retries:
                    raise
                if self.retry_sleep_s:
                    time.sleep(min(2.0, self.retry_sleep_s * (2**attempt)))
                continue

            usage = {
                "input_tokens": int(response.usage.get("input_tokens") or 0),
                "output_tokens": int(response.usage.get("output_tokens") or 0),
                "cached_input_tokens": int(
                    response.usage.get("cached_input_tokens") or 0
                ),
            }
            actual = usage_cost(usage, self.pricing)
            if usage["input_tokens"] + usage["output_tokens"] == 0:
                actual = self.reserve_usd
            self.ledger.settle(
                reservation,
                actual_usd=actual,
                usage=usage,
                note=f"prompt_sha256={hashlib.sha256(prompt.encode()).hexdigest()}",
            )
            result = self._parse(response, claim, cache_hit=False, attempts=attempt + 1)
            if self.cache is not None:
                self.cache.put(
                    key,
                    {
                        "text": response.text,
                        "usage": usage,
                        "model_version": response.model_version or self.model,
                        "latency_ms": response.latency_ms,
                        "raw": response.raw,
                    },
                )
            self._emit(
                {
                    "event": "backend_result",
                    "attempt": attempt + 1,
                    "prompt": prompt,
                    "raw_response": response.text,
                    "provider_model": response.model_version,
                    "usage": usage,
                    "cost_usd": actual,
                    "latency_ms": response.latency_ms,
                    "result": result.model_dump(mode="json"),
                }
            )
            return result
        raise RuntimeError(f"语义 backend 失败:{last_error}")


class NLIContract:
    """Conservative label mapping; non-entailment is never contradiction."""

    def __init__(
        self, entailment_threshold: float = 0.75, contradiction_threshold: float = 0.90
    ) -> None:
        self.entailment_threshold = float(entailment_threshold)
        self.contradiction_threshold = float(contradiction_threshold)

    def from_labels(self, labels: dict[str, float]) -> GroundingStatus:
        if float(labels.get("entailment") or 0.0) >= self.entailment_threshold:
            return "supported"
        return "uncertain"

    def from_three_labels(self, labels: dict[str, float]) -> GroundingStatus:
        contradiction = float(labels.get("contradiction") or 0.0)
        entailment = float(labels.get("entailment") or 0.0)
        if contradiction >= self.contradiction_threshold and contradiction > entailment:
            return "unsupported"
        if entailment >= self.entailment_threshold:
            return "supported"
        return "uncertain"
