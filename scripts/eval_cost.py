"""价格表、跨运行预算台账与调用前护栏。

`pricing.verified=false` 时拒绝任何付费调用;`CostTracker.guard` 在真实调用前
按估算 token 检查单次上限与累计 $10 台账。
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PRICING_PATH = ROOT / "config" / "pricing.yaml"
DEFAULT_LEDGER_PATH = ROOT / "data" / "ch11_budget.json"

_ATTEMPT_CONTEXT: ContextVar[dict] = ContextVar("ch16_attempt_context", default={})


def attempt_context(**values):
    """Attach non-sensitive run labels to API attempts for the CH16 ledger."""
    class _Context:
        def __enter__(self):
            self.token = _ATTEMPT_CONTEXT.set({**_ATTEMPT_CONTEXT.get(), **values})
            return self

        def __exit__(self, *_exc):
            _ATTEMPT_CONTEXT.reset(self.token)

    return _Context()


class BudgetExceeded(RuntimeError):
    """单次运行或累计台账超限。"""


class PricingError(RuntimeError):
    """价格表缺失/未核实/模型未知。"""


class PricingTable:
    def __init__(self, payload: dict, *, path: Path | None = None) -> None:
        self.payload = payload
        self.path = path
        self.version = str(payload.get("version") or "")
        self.currency = str(payload.get("currency") or "USD")
        self.verified = bool(payload.get("verified"))
        self.source_url = str(payload.get("source_url") or "")
        self.models = payload.get("models") or {}

    @classmethod
    def load(cls, path: Path = DEFAULT_PRICING_PATH) -> "PricingTable":
        try:
            payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise PricingError(f"无法读取价格表 {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise PricingError(f"价格表顶层必须是对象: {path}")
        return cls(payload, path=Path(path))

    def cost(self, model: str, input_tokens: int, output_tokens: int = 0,
             cache_hit_input_tokens: int = 0) -> float:
        if model not in self.models:
            raise PricingError(f"价格表缺少模型: {model}")
        entry = self.models[model] or {}
        input_price = float(entry.get("input_per_million") or 0.0)
        output_price = float(entry.get("output_per_million") or 0.0)
        cache_price = float(entry.get("input_cache_hit_per_million") or input_price)
        hit_tokens = min(max(int(cache_hit_input_tokens), 0), max(int(input_tokens), 0))
        miss_tokens = max(0, int(input_tokens) - hit_tokens)
        return ((miss_tokens / 1_000_000) * input_price
                + (hit_tokens / 1_000_000) * cache_price
                + (output_tokens / 1_000_000) * output_price)


class BudgetLedger:
    def __init__(self, path: Path = DEFAULT_LEDGER_PATH,
                 total_limit_usd: float | None = 10.0) -> None:
        self.path = Path(path)
        self.total_limit_usd = None if total_limit_usd is None else float(total_limit_usd)
        self._records: list[dict] = []
        if self.path.exists():
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
                self._records = list(payload.get("records") or [])
            except (OSError, json.JSONDecodeError) as exc:
                # 预算台账损坏时 fail closed,不能静默重置为 0。
                raise BudgetExceeded(f"预算台账损坏,拒绝继续: {self.path}: {exc}") from exc

    def total_usd(self) -> float:
        return sum(float(record.get("amount_usd") or 0.0) for record in self._records)

    def remaining_usd(self) -> float:
        if self.total_limit_usd is None:
            return float("inf")
        return self.total_limit_usd - self.total_usd()

    def record(self, amount_usd: float, note: str = "") -> None:
        amount = float(amount_usd)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        with lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                records: list[dict] = []
                if self.path.exists():
                    try:
                        payload = json.loads(self.path.read_text(encoding="utf-8"))
                        records = list(payload.get("records") or [])
                    except (OSError, json.JSONDecodeError) as exc:
                        raise BudgetExceeded(
                            f"预算台账损坏,拒绝继续: {self.path}: {exc}"
                        ) from exc
                total = sum(float(item.get("amount_usd") or 0.0) for item in records)
                if self.total_limit_usd is not None and total + amount > self.total_limit_usd:
                    raise BudgetExceeded(
                        f"累计预算超限:{total + amount:.4f} > {self.total_limit_usd:.4f}"
                    )
                records.append({
                    "amount_usd": amount,
                    "note": note,
                    "at": datetime.now(timezone.utc).isoformat(),
                })
                tmp = self.path.with_suffix(".json.tmp")
                payload = {"limit_usd": self.total_limit_usd, "records": records}
                tmp.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                os.replace(tmp, self.path)
                self._records = records
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _estimate_tokens(texts: list[str]) -> int:
    return max(1, sum(max(1, len(text) // 2) for text in texts))


class GuardedEmbedder:
    """把嵌入调用纳入 allow_paid/verified/台账;缓存命中不会走到这里。"""

    def __init__(self, inner, cost: "CostTracker", model: str,
                 journal: "CallResultJournal | None" = None) -> None:
        self.inner = inner
        self.cost = cost
        self.model = model
        self.journal = journal

    def embed(self, texts: list[str]) -> list[list[float]]:
        tokens = _estimate_tokens(texts)
        request_text = "\0".join(texts)
        self.cost.guard(tokens, self.model, kind="embedding", request_text=request_text)
        if self.journal is not None:
            cached = self.journal.get(self.model, request_text)
            if cached is not None:
                self.cost.mark_cache_hit()
                return cached["vectors"]
        try:
            vectors = self.inner.embed(texts)
        except BaseException as exc:
            self.cost.fail_attempt(exc)
            raise
        if self.journal is not None:
            self.journal.put(self.model, request_text, {"vectors": vectors})
        self.cost.add(self.model, tokens, 0, kind="embedding")
        return vectors


class GuardedReranker:
    def __init__(self, inner, cost: "CostTracker", model: str,
                 journal: "CallResultJournal | None" = None) -> None:
        self.inner = inner
        self.cost = cost
        self.model = model
        self.journal = journal

    def rerank(self, query: str, documents: list[str], top_n: int | None = None):
        tokens = _estimate_tokens([query, *documents])
        request_text = query + "\0" + "\0".join(documents)
        self.cost.guard(tokens, self.model, kind="rerank", request_text=request_text)
        if self.journal is not None:
            cached = self.journal.get(self.model, request_text)
            if cached is not None:
                self.cost.mark_cache_hit()
                self.cost.add(self.model, tokens, 0, kind="rerank")
                return [tuple(item) for item in cached["results"]]
        try:
            results = self.inner.rerank(query, documents, top_n=top_n)
        except BaseException as exc:
            self.cost.fail_attempt(exc)
            raise
        if self.journal is not None:
            self.journal.put(self.model, request_text, {"results": results})
        self.cost.add(self.model, tokens, 0, kind="rerank")
        return results


class GuardedEvidenceJudge:
    def __init__(self, inner, cost: "CostTracker", cache, model: str, prompt_sha: str) -> None:
        self.inner = inner
        self.cost = cost
        self.cache = cache
        self.model = model
        self.prompt_sha = prompt_sha
        self.calls = 0
        self.hits = 0

    def judge(self, query: str, documents: list[str]):
        from app.evidence import EvidenceVerdict

        key = None
        if self.cache is not None:
            key = self.cache.response_key(
                model=self.model, prompt_sha256=self.prompt_sha,
                temperature=0.0,
                query=query + "\0" + "\x1f".join(documents),
                evidence_ids=[],
            )
            payload = self.cache.get(key)
            if payload is not None:
                self.hits += 1
                return EvidenceVerdict(**payload)
        tokens = _estimate_tokens([query, *documents])
        request_text = query + "\0" + "\x1f".join(documents)
        self.cost.guard(tokens, self.model, kind="evidence_judge", request_text=request_text)
        try:
            verdict = self.inner.judge(query, documents)
        except BaseException as exc:
            self.cost.fail_attempt(exc)
            raise
        if getattr(getattr(self.inner, "_llm", None), "last_cache_hit", False):
            self.hits += 1
        else:
            self.calls += 1
        self.cost.add(self.model, tokens, 128, kind="evidence_judge")
        if self.cache is not None and key is not None:
            self.cache.put(key, {
                "answered": bool(verdict.answered),
                "supported_by": list(verdict.supported_by),
                "reason": verdict.reason,
            })
        return verdict


class GuardedFaithfulnessJudge:
    def __init__(self, inner, cost: "CostTracker", cache, model: str, prompt_sha: str) -> None:
        self.inner = inner
        self.cost = cost
        self.cache = cache
        self.model = model
        self.prompt_sha = prompt_sha
        self.calls = 0

    def invoke(self, prompt: str):
        from app.prompts import FaithfulnessVerdict

        key = None
        if self.cache is not None:
            key = self.cache.response_key(
                model=self.model, prompt_sha256=self.prompt_sha,
                temperature=0.0, query=prompt, evidence_ids=[],
            )
            payload = self.cache.get(key)
            if payload is not None:
                return FaithfulnessVerdict(**payload)
        tokens = _estimate_tokens([prompt])
        self.cost.guard(tokens, self.model, kind="faithfulness_judge", request_text=prompt)
        self.calls += 1
        try:
            verdict = self.inner.invoke(prompt)
        except BaseException as exc:
            self.cost.fail_attempt(exc)
            raise
        self.cost.add(self.model, tokens, 128, kind="faithfulness_judge")
        if self.cache is not None and key is not None:
            self.cache.put(key, {
                "fabricated": bool(verdict.fabricated),
                "reason": verdict.reason,
                "fabricated_claims": list(verdict.fabricated_claims),
            })
        return verdict


class GuardedGapAnalyzer:
    """ch12 缺口分析纳入守卫+缓存(同 GuardedEvidenceJudge 模式);缓存命中不进守卫不重复计费。"""

    def __init__(self, inner, cost: "CostTracker", cache, model: str, prompt_sha: str) -> None:
        self.inner = inner
        self.cost = cost
        self.cache = cache
        self.model = model
        self.prompt_sha = prompt_sha
        self.calls = 0
        self.hits = 0

    def analyze(self, query: str, evidence: str):
        from app.prompts import GapReport

        key = None
        if self.cache is not None:
            key = self.cache.response_key(
                model=self.model, prompt_sha256=self.prompt_sha,
                temperature=0.0, query=query + "\0" + evidence, evidence_ids=[],
            )
            payload = self.cache.get(key)
            if payload is not None:
                self.hits += 1
                return GapReport(**payload)
        tokens = _estimate_tokens([query, evidence])
        request_text = query + "\0" + evidence
        self.cost.guard(tokens, self.model, kind="gap_analyzer", request_text=request_text)
        try:
            report = self.inner.analyze(query, evidence)
        except BaseException as exc:
            self.cost.fail_attempt(exc)
            raise
        if getattr(getattr(self.inner, "_llm", None), "last_cache_hit", False):
            self.hits += 1
        else:
            self.calls += 1
        self.cost.add(self.model, tokens, 192, kind="gap_analyzer")
        if self.cache is not None and key is not None:
            self.cache.put(key, {
                "sufficient": bool(report.sufficient),
                "action": str(report.action),
                "missing_aspects": list(report.missing_aspects),
                "conflicts": list(report.conflicts),
                "sub_queries": list(report.sub_queries),
                "clarify_question": report.clarify_question,
                "reason": report.reason,
            })
        return report


class CostTracker:
    def __init__(
        self,
        pricing: PricingTable,
        ledger: BudgetLedger | None = None,
        max_cost_usd: float | None = 0.0,
        *,
        allow_paid: bool = False,
        attempt_ledger: "AttemptLedger | None" = None,
    ) -> None:
        self.pricing = pricing
        self.ledger = ledger or BudgetLedger()
        self.max_cost_usd = None if max_cost_usd is None else float(max_cost_usd)
        self.allow_paid = allow_paid
        self.attempt_ledger = attempt_ledger
        self.spent_usd = 0.0
        self.calls = 0
        self._pending_attempt_id: str | None = None
        self._pending_started_perf: float | None = None
        self._pending_kind: str | None = None
        self._latest_provider_usage: tuple[int, int, int] | None = None
        self._cached_replay_kind: str | None = None
        self.api_attempts = 0
        self.paid_api_attempts = 0
        self._pending_estimated_usd = 0.0

    def note_provider_usage(self, input_tokens: int, output_tokens: int,
                            cache_hit_input_tokens: int = 0) -> None:
        self._latest_provider_usage = (
            int(input_tokens), int(output_tokens), int(cache_hit_input_tokens)
        )

    def _ensure_allowed(self) -> None:
        if not self.allow_paid:
            raise BudgetExceeded("未允许付费调用(--allow-paid 未开启)")
        if not self.pricing.verified:
            raise BudgetExceeded("价格表未核实(verified=false),拒绝付费调用")

    def guard(self, estimated_input_tokens: int, model: str, *, kind: str = "api",
              request_text: str = "") -> None:
        self._ensure_allowed()
        if self._pending_attempt_id:
            raise BudgetExceeded("上一条 API 请求尚未闭合；停止以避免覆盖尝试账本")
        estimated = self.pricing.cost(model, estimated_input_tokens, 256)
        if self.max_cost_usd is not None and self.spent_usd + estimated > self.max_cost_usd:
            raise BudgetExceeded(
                f"单次运行预算不足:已花 ${self.spent_usd:.4f},"
                f"本次估算 ${estimated:.4f},上限 ${self.max_cost_usd:.4f}"
            )
        if (self.ledger.total_limit_usd is not None
                and self.ledger.total_usd() + estimated > self.ledger.total_limit_usd):
            raise BudgetExceeded(
                f"累计预算不足:台账 ${self.ledger.total_usd():.4f},"
                f"本次估算 ${estimated:.4f},上限 ${self.ledger.total_limit_usd:.4f}"
            )
        if self.attempt_ledger is not None:
            self._pending_attempt_id = self.attempt_ledger.begin(
                model=model, kind=kind, estimated_input_tokens=estimated_input_tokens,
                request_text=request_text, context=_ATTEMPT_CONTEXT.get(),
                estimated_usd=estimated,
                price_version=self.pricing.version,
            )
            self._pending_started_perf = time.perf_counter()
        self._pending_kind = kind
        self.api_attempts += 1
        self._pending_estimated_usd = estimated
        if estimated > 0:
            self.paid_api_attempts += 1

    def bind_request(self, request_text: str) -> None:
        if self.attempt_ledger is not None and self._pending_attempt_id:
            self.attempt_ledger.bind_request(self._pending_attempt_id, request_text)

    def fail_attempt(self, exc: BaseException) -> None:
        if self.attempt_ledger is not None and self._pending_attempt_id:
            self.attempt_ledger.finish(
                self._pending_attempt_id, status="failed", error=_safe_error(exc),
                retry_count=0, elapsed_ms=self._attempt_elapsed_ms(),
            )
            self._pending_attempt_id = None
            self._pending_started_perf = None
        self._latest_provider_usage = None
        self._pending_kind = None
        self._cached_replay_kind = None
        self._pending_estimated_usd = 0.0

    def mark_cache_hit(self) -> None:
        """Close a reserved attempt when the durable response journal served it."""
        if self.attempt_ledger is not None and self._pending_attempt_id:
            self.attempt_ledger.finish(
                self._pending_attempt_id, status="cache_hit", retry_count=0,
                cache_hit=True, elapsed_ms=0.0,
            )
        if self._pending_attempt_id:
            self._pending_attempt_id = None
            self._pending_started_perf = None
            self.api_attempts = max(0, self.api_attempts - 1)
            if self._pending_estimated_usd > 0:
                self.paid_api_attempts = max(0, self.paid_api_attempts - 1)
        self._cached_replay_kind = self._pending_kind
        self._pending_kind = None
        self._pending_estimated_usd = 0.0

    def record_provider_response(self, model: str, input_tokens: int,
                                 output_tokens: int, kind: str,
                                 cache_hit_input_tokens: int = 0) -> float:
        """Durably account for a model response before downstream parsing/caching."""
        self.note_provider_usage(input_tokens, output_tokens, cache_hit_input_tokens)
        amount = self.add(model, input_tokens, output_tokens, kind=kind)
        self._cached_replay_kind = kind
        return amount

    def _attempt_elapsed_ms(self) -> float | None:
        if self._pending_started_perf is None:
            return None
        return round((time.perf_counter() - self._pending_started_perf) * 1000, 3)

    def add(
        self, model: str, input_tokens: int, output_tokens: int, kind: str = "answer"
    ) -> float:
        self._ensure_allowed()
        if self._cached_replay_kind == kind:
            self._cached_replay_kind = None
            self._latest_provider_usage = None
            return 0.0
        usage_source = "estimated"
        cache_hit_input_tokens = 0
        if kind in {"answer", "evidence_judge", "gap_analyzer", "faithfulness_judge"} \
                and self._latest_provider_usage is not None:
            input_tokens, output_tokens, cache_hit_input_tokens = self._latest_provider_usage
            usage_source = "provider"
        self._latest_provider_usage = None
        amount = self.pricing.cost(model, input_tokens, output_tokens, cache_hit_input_tokens)
        self.spent_usd += amount
        self.calls += 1
        if self.max_cost_usd is not None and self.spent_usd > self.max_cost_usd:
            raise BudgetExceeded(
                f"单次运行超限:${self.spent_usd:.4f} > ${self.max_cost_usd:.4f}"
            )
        if (self.ledger.total_limit_usd is not None
                and self.ledger.total_usd() + amount > self.ledger.total_limit_usd):
            raise BudgetExceeded(
                f"累计台账超限:${self.ledger.total_usd() + amount:.4f} > "
                f"${self.ledger.total_limit_usd:.4f}"
            )
        self.ledger.record(amount, note=kind)
        if self.attempt_ledger is not None and self._pending_attempt_id:
            self.attempt_ledger.finish(
                self._pending_attempt_id, status="succeeded", amount_usd=amount,
                input_tokens=int(input_tokens), output_tokens=int(output_tokens),
                cache_hit_input_tokens=int(cache_hit_input_tokens),
                usage_source=usage_source, retry_count=0,
                elapsed_ms=self._attempt_elapsed_ms(),
            )
            self._pending_attempt_id = None
            self._pending_started_perf = None
        self._pending_kind = None
        self._pending_estimated_usd = 0.0
        return amount

    def snapshot(self) -> dict:
        return {
            "allow_paid": self.allow_paid,
            "pricing_verified": self.pricing.verified,
            "pricing_version": self.pricing.version,
            "pricing_currency": self.pricing.currency,
            "run_spent_usd": round(self.spent_usd, 6),
            "run_limit_usd": self.max_cost_usd,
            "ledger_total_usd": round(self.ledger.total_usd(), 6),
            "ledger_limit_usd": self.ledger.total_limit_usd,
            "calls": self.calls,
            "api_attempts": self.api_attempts,
            "paid_api_attempts": self.paid_api_attempts,
        }


def _safe_error(exc: BaseException) -> str:
    message = f"{type(exc).__name__}: {exc}"
    for name in ("OPENAI_API_KEY", "EMBEDDING_API_KEY", "RERANK_API_KEY"):
        secret = os.environ.get(name)
        if secret:
            message = message.replace(secret, "[redacted]")
    return message[:500]


class AttemptLedger:
    """Append-only attempt journal; pending attempts make resume fail closed."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_safe_to_resume()

    def _events(self) -> list[dict]:
        if not self.path.exists():
            return []
        rows = []
        for line_no, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise BudgetExceeded(f"调用账本损坏，拒绝续跑:{self.path}:{line_no}") from exc
        return rows

    def _ensure_safe_to_resume(self) -> None:
        state: dict[str, str] = {}
        for event in self._events():
            if event.get("event") == "started":
                state[str(event["attempt_id"])] = "pending"
            elif event.get("event") in {"succeeded", "failed", "cache_hit"}:
                state[str(event["attempt_id"])] = str(event["event"])
        pending = sorted(key for key, value in state.items() if value == "pending")
        if pending:
            raise BudgetExceeded(
                "存在未闭合的 API 尝试，结果可能已计费；为避免重复请求已停止，"
                f"请先审计 attempt_id={pending[0]}"
            )

    def _append(self, event: dict) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def begin(self, *, model: str, kind: str, estimated_input_tokens: int,
              request_text: str, context: dict, estimated_usd: float = 0.0,
              price_version: str = "") -> str:
        attempt_id = uuid.uuid4().hex
        self._append({
            "event": "started", "attempt_id": attempt_id,
            "at": datetime.now(timezone.utc).isoformat(), "model": model,
            "provider": "deepseek" if model.startswith("deepseek-") else "siliconflow",
            "kind": kind, "estimated_input_tokens": int(estimated_input_tokens),
            "estimated_usd": round(float(estimated_usd), 10),
            "price_version": price_version,
            "request_sha256": hashlib.sha256(request_text.encode("utf-8")).hexdigest()
            if request_text else None,
            "retry_count": 0, "cache_hit": False, **context,
        })
        return attempt_id

    def bind_request(self, attempt_id: str, request_text: str) -> None:
        self._append({
            "event": "request_bound", "attempt_id": attempt_id,
            "request_sha256": hashlib.sha256(request_text.encode("utf-8")).hexdigest(),
            "at": datetime.now(timezone.utc).isoformat(),
        })

    def finish(self, attempt_id: str, *, status: str, retry_count: int,
               amount_usd: float = 0.0, input_tokens: int | None = None,
               output_tokens: int | None = None, usage_source: str | None = None,
               cache_hit_input_tokens: int = 0,
               error: str | None = None, elapsed_ms: float | None = None,
               cache_hit: bool = False) -> None:
        self._append({
            "event": status, "attempt_id": attempt_id,
            "at": datetime.now(timezone.utc).isoformat(),
            "amount_usd": round(float(amount_usd), 10),
            "input_tokens": input_tokens, "output_tokens": output_tokens,
            "cache_hit_input_tokens": int(cache_hit_input_tokens),
            "usage_source": usage_source, "retry_count": int(retry_count),
            "elapsed_ms": elapsed_ms, "cache_hit": cache_hit, "error": error,
        })


class CallResultJournal:
    """Private, durable exact-request cache used to make process restarts safe."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._payload = self._read()

    @staticmethod
    def key(model: str, request: str) -> str:
        return hashlib.sha256((model + "\0" + request).encode("utf-8")).hexdigest()

    def _read(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("top-level value is not an object")
            return payload
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise BudgetExceeded(f"调用结果缓存损坏，拒绝继续:{self.path}:{exc}") from exc

    def get(self, model: str, request: str):
        return self._payload.get(self.key(model, request))

    def put(self, model: str, request: str, value: dict) -> None:
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        with lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                payload = self._read()
                payload[self.key(model, request)] = value
                tmp = self.path.with_suffix(self.path.suffix + ".tmp")
                tmp.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
                os.replace(tmp, self.path)
                self._payload = payload
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


class AuditedChat:
    """Small invoke adapter that records request hashes and provider token usage."""

    def __init__(self, inner, cost: CostTracker, journal: CallResultJournal,
                 model: str, kind: str) -> None:
        self.inner = inner
        self.cost = cost
        self.journal = journal
        self.model = model
        self.kind = kind
        self.last_cache_hit = False

    def invoke(self, prompt, **kwargs):
        request_text = prompt if isinstance(prompt, str) else json.dumps(
            prompt, ensure_ascii=False, sort_keys=True, default=str
        )
        self.last_cache_hit = False
        self.cost.bind_request(request_text)
        cached = self.journal.get(self.model, request_text)
        if cached is not None:
            from langchain_core.messages import AIMessage

            self.cost.mark_cache_hit()
            self.last_cache_hit = True
            return AIMessage(
                content=cached.get("content", ""),
                usage_metadata=cached.get("usage_metadata"),
            )
        try:
            response = self.inner.invoke(prompt, **kwargs)
        except BaseException as exc:
            self.cost.fail_attempt(exc)
            raise
        usage = getattr(response, "usage_metadata", None)
        if not usage:
            usage = (getattr(response, "response_metadata", None) or {}).get("token_usage") or {}
        input_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
        input_details = usage.get("input_token_details") or {}
        metadata = getattr(response, "response_metadata", None) or {}
        provider_usage = metadata.get("token_usage") or {}
        cached_input_tokens = int(
            input_details.get("cache_read") or input_details.get("cached_tokens")
            or provider_usage.get("prompt_cache_hit_tokens") or 0
        )
        if input_tokens or output_tokens:
            usage_metadata = {
                "input_tokens": input_tokens, "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }
            if cached_input_tokens:
                usage_metadata["input_token_details"] = {"cache_read": cached_input_tokens}
        else:
            usage_metadata = None
        content = getattr(response, "content", response)
        self.journal.put(self.model, request_text, {
            "content": content, "usage_metadata": usage_metadata,
        })
        if input_tokens or output_tokens:
            self.cost.record_provider_response(
                self.model, input_tokens, output_tokens, kind=self.kind,
                cache_hit_input_tokens=cached_input_tokens,
            )
        return response
