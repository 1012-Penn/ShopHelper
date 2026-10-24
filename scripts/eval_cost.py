"""价格表、跨运行预算台账与调用前护栏。

`pricing.verified=false` 时拒绝任何付费调用;`CostTracker.guard` 在真实调用前
按估算 token 检查单次上限与累计 $10 台账。
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PRICING_PATH = ROOT / "config" / "pricing.yaml"
DEFAULT_LEDGER_PATH = ROOT / "data" / "ch11_budget.json"


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

    def cost(self, model: str, input_tokens: int, output_tokens: int = 0) -> float:
        if model not in self.models:
            raise PricingError(f"价格表缺少模型: {model}")
        entry = self.models[model] or {}
        input_price = float(entry.get("input_per_million") or 0.0)
        output_price = float(entry.get("output_per_million") or 0.0)
        return (input_tokens / 1_000_000) * input_price + (output_tokens / 1_000_000) * output_price


class BudgetLedger:
    def __init__(self, path: Path = DEFAULT_LEDGER_PATH, total_limit_usd: float = 10.0) -> None:
        self.path = Path(path)
        self.total_limit_usd = float(total_limit_usd)
        self._records: list[dict] = []
        if self.path.exists():
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
                self._records = list(payload.get("records") or [])
            except (OSError, json.JSONDecodeError):
                self._records = []

    def total_usd(self) -> float:
        return sum(float(record.get("amount_usd") or 0.0) for record in self._records)

    def remaining_usd(self) -> float:
        return self.total_limit_usd - self.total_usd()

    def record(self, amount_usd: float, note: str = "") -> None:
        self._records.append({
            "amount_usd": float(amount_usd),
            "note": note,
            "at": datetime.now(timezone.utc).isoformat(),
        })
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        payload = {"limit_usd": self.total_limit_usd, "records": self._records}
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, self.path)


class CostTracker:
    def __init__(
        self,
        pricing: PricingTable,
        ledger: BudgetLedger | None = None,
        max_cost_usd: float = 0.0,
        *,
        allow_paid: bool = False,
    ) -> None:
        self.pricing = pricing
        self.ledger = ledger or BudgetLedger()
        self.max_cost_usd = float(max_cost_usd)
        self.allow_paid = allow_paid
        self.spent_usd = 0.0
        self.calls = 0

    def _ensure_allowed(self) -> None:
        if not self.allow_paid:
            raise BudgetExceeded("未允许付费调用(--allow-paid 未开启)")
        if not self.pricing.verified:
            raise BudgetExceeded("价格表未核实(verified=false),拒绝付费调用")

    def guard(self, estimated_input_tokens: int, model: str) -> None:
        self._ensure_allowed()
        estimated = self.pricing.cost(model, estimated_input_tokens, 256)
        if self.spent_usd + estimated > self.max_cost_usd:
            raise BudgetExceeded(
                f"单次运行预算不足:已花 ${self.spent_usd:.4f},"
                f"本次估算 ${estimated:.4f},上限 ${self.max_cost_usd:.4f}"
            )
        if self.ledger.total_usd() + estimated > self.ledger.total_limit_usd:
            raise BudgetExceeded(
                f"累计预算不足:台账 ${self.ledger.total_usd():.4f},"
                f"本次估算 ${estimated:.4f},上限 ${self.ledger.total_limit_usd:.4f}"
            )

    def add(
        self, model: str, input_tokens: int, output_tokens: int, kind: str = "answer"
    ) -> float:
        self._ensure_allowed()
        amount = self.pricing.cost(model, input_tokens, output_tokens)
        self.spent_usd += amount
        self.calls += 1
        if self.spent_usd > self.max_cost_usd:
            raise BudgetExceeded(
                f"单次运行超限:${self.spent_usd:.4f} > ${self.max_cost_usd:.4f}"
            )
        if self.ledger.total_usd() + amount > self.ledger.total_limit_usd:
            raise BudgetExceeded(
                f"累计台账超限:${self.ledger.total_usd() + amount:.4f} > "
                f"${self.ledger.total_limit_usd:.4f}"
            )
        self.ledger.record(amount, note=kind)
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
        }
