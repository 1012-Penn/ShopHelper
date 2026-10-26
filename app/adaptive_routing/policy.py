"""Frozen deterministic routing policy and optional bounded judge hook."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Protocol

from app.adaptive_routing.features import extract_routing_features
from app.adaptive_routing.models import (
    JudgeSignal,
    RoutingDecision,
    RoutingFeatures,
    RoutingInput,
)

POLICY_VERSION = "adaptive-routing-r2-v1"
JUDGE_MIN_CONFIDENCE = 0.75
POLICY_CONTRACT = {
    "version": POLICY_VERSION,
    "default": "single",
    "multi_rules": [
        "cross_order_comparison",
        "correction_with_multiple_policy_orders",
    ],
    "judge_band": "multiple_policy_orders_without_explicit_comparison_or_correction",
    "judge_min_confidence": JUDGE_MIN_CONFIDENCE,
    "write_risk_can_expand_permissions": False,
}


def policy_sha256() -> str:
    encoded = json.dumps(
        POLICY_CONTRACT, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class RoutingJudge(Protocol):
    async def assess(
        self, routing_input: RoutingInput, features: RoutingFeatures
    ) -> JudgeSignal: ...


class AdaptiveRouter:
    def __init__(
        self,
        *,
        judge: RoutingJudge | None = None,
        judge_min_confidence: float = JUDGE_MIN_CONFIDENCE,
    ) -> None:
        if not 0.0 <= judge_min_confidence <= 1.0:
            raise ValueError("judge_min_confidence must be between 0 and 1")
        self.judge = judge
        self.judge_min_confidence = judge_min_confidence

    def policy_signature(self) -> str:
        return policy_sha256()

    @staticmethod
    def _needs_judge(features: RoutingFeatures) -> bool:
        return (
            features.entity_count >= 2
            and features.policy_order_dependency
            and not features.cross_order_comparison
            and not features.correction
        )

    @staticmethod
    def _rule_decision(features: RoutingFeatures) -> tuple[str, list[str]]:
        reasons: list[str] = []
        if features.cross_order_comparison:
            reasons.append("cross_order_comparison")
        if (
            features.correction
            and features.policy_order_dependency
            and features.entity_count >= 2
        ):
            reasons.append("revision_with_order_dependency")
        if reasons:
            return "multi", reasons
        if features.missing_slot:
            reasons.append("missing_slot_clarify")
        if features.write_confirmation_risk:
            reasons.append("write_requires_confirmation")
        if features.correction:
            reasons.append("visible_revision")
        if features.policy_order_dependency:
            reasons.append("single_order_policy_lookup")
        elif features.entity_count:
            reasons.append("single_order_lookup")
        else:
            reasons.append("simple_policy")
        return "single", reasons

    def route(self, routing_input: RoutingInput) -> RoutingDecision:
        """Synchronous rules-only decision used by the frozen rule policy."""

        started = time.perf_counter()
        features = extract_routing_features(routing_input.question, routing_input.history)
        target, reasons = self._rule_decision(features)
        return RoutingDecision(
            target=target,
            reason_codes=tuple(reasons),
            features=features,
            policy_version=POLICY_VERSION,
            policy_sha256=policy_sha256(),
            visible_budget=routing_input.visible_budget,
            routing_latency_ms=int((time.perf_counter() - started) * 1000),
        )

    async def route_async(self, routing_input: RoutingInput) -> RoutingDecision:
        """Rules plus at most one optional judgment in the declared uncertain band."""

        started = time.perf_counter()
        base = self.route(routing_input)
        if self.judge is None or not self._needs_judge(base.features):
            return base.model_copy(
                update={
                    "routing_latency_ms": int((time.perf_counter() - started) * 1000)
                }
            )

        signal = await self.judge.assess(routing_input, base.features)
        target = base.target
        reasons = list(base.reason_codes)
        if signal.confidence >= self.judge_min_confidence:
            target = signal.target
            reasons.append(f"judge:{signal.reason}")
        else:
            reasons.append("judge_below_threshold")
        return base.model_copy(
            update={
                "target": target,
                "reason_codes": tuple(reasons),
                "judge_used": True,
                "judge_confidence": signal.confidence,
                "judge_input_tokens": signal.input_tokens,
                "judge_output_tokens": signal.output_tokens,
                "routing_cost_usd": signal.cost_usd,
                "routing_latency_ms": int((time.perf_counter() - started) * 1000),
            }
        )
