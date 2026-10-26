from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from app.answer_verifier import AnswerVerifier
from app.claim_grounding.answer_adapter import ClaimGroundingAnswerSemantic
from app.claim_grounding.backends import (
    NLIContract,
    ProviderResponse,
    StructuredGroundingBackend,
)
from app.claim_grounding.budget import BudgetExceeded, ClaimBudgetLedger
from app.claim_grounding.grounder import ClaimGrounder
from app.claim_grounding.models import ClaimInput, EvidenceCandidate


def test_runtime_factory_fails_closed_without_an_explicit_budget_path(tmp_path):
    from app.claim_grounding.runtime import build_claim_grounding_semantic
    from app.config import Settings

    settings = Settings(
        openai_api_key="sk-test",
        _env_file=None,
        claim_grounding_enabled=True,
        answer_verification_enabled=True,
        claim_grounding_ledger_path="",
    )
    with pytest.raises(ValueError, match="ledger_path"):
        build_claim_grounding_semantic(settings)


@pytest.mark.asyncio
async def test_application_factory_builds_explicit_claim_grounding_service(
        make_client, tmp_path):
    ledger_path = tmp_path / "claim-grounding.sqlite"
    _, app = await make_client(
        type("IntentModel", (), {
            "ainvoke": lambda self, *_args, **_kwargs: pytest.fail(
                "intent model must not run during factory construction"
            )
        })(),
        settings_overrides={
            "answer_verification_enabled": True,
            "claim_grounding_enabled": True,
            "claim_grounding_ledger_path": str(ledger_path),
        },
    )

    assert app.state.claim_grounding_semantic is not None
    assert ledger_path.exists()


class RecordingBackend:
    def __init__(self, status="uncertain", reason="语义不足"):
        self.status = status
        self.reason = reason
        self.calls = 0

    def ground(self, claim):
        self.calls += 1
        from app.claim_grounding.models import GroundingResult

        return GroundingResult(
            status=self.status,
            reason=self.reason,
            evidence_ids=[claim.evidence[0].evidence_id],
            backend="fake",
            model_version="fake-v1",
        )


class RaisingBackend:
    def __init__(self, exc):
        self.exc = exc
        self.calls = 0

    def ground(self, claim):
        self.calls += 1
        raise self.exc


class FakeCaller:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []
        self.calls = 0

    def __call__(self, prompt):
        self.prompts.append(prompt)
        self.calls += 1
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class Retry429(Exception):
    status_code = 429


def _claim(**overrides):
    payload = {
        "claim": "SH-E300 的售价是 299 元。",
        "subject": "商品售价",
        "model_number": "SH-E300",
        "unit": "元",
        "evidence": [
            EvidenceCandidate(
                evidence_id="E1",
                text="SH-E300 头戴式无线降噪耳机，售价 299 元。",
            )
        ],
    }
    payload.update(overrides)
    return ClaimInput(**payload)


def test_deterministic_numeric_conflict_cannot_be_overridden():
    backend = RecordingBackend(status="supported", reason="模型误判")
    result = ClaimGrounder(backend=backend).ground(
        _claim(claim="SH-E300 的售价是 399 元。")
    )

    assert result.status == "unsupported"
    assert result.backend == "deterministic"
    assert backend.calls == 0


def test_as_of_date_is_context_not_an_asserted_order_fact():
    result = ClaimGrounder(backend=None).ground(
        _claim(
            claim="该物流当日已经送达。",
            as_of_date="2026-10-15",
            evidence=[
                EvidenceCandidate(
                    evidence_id="L1",
                    text="物流快照：截至 2026-09-01 仍在转运中心。",
                )
            ],
        )
    )

    assert result.status == "uncertain"
    assert "日期与证据不一致" not in result.reason


def test_policy_boundary_is_derived_from_versioned_intervals():
    result = ClaimGrounder(backend=None).ground(
        _claim(
            claim="2026-10-01 签收应使用 v3 的 30 天退货窗口。",
            as_of_date="2026-10-01",
            policy_version="v3",
            unit="天",
            evidence=[
                EvidenceCandidate(
                    evidence_id="P1",
                    text=(
                        "v2 适用于 2026-06-01 至 2026-09-30，窗口为 15 天；"
                        "v3 自 2026-10-01 起，窗口为 30 天。"
                    ),
                )
            ],
        )
    )

    assert result.status != "unsupported"


def test_selective_grounding_skips_backend_when_surface_rule_is_supported():
    backend = RecordingBackend(status="uncertain")
    result = ClaimGrounder(backend=backend).ground(_claim())

    assert result.status == "supported"
    assert backend.calls == 0
    assert result.baseline_status == "supported"


def test_selective_grounding_calls_backend_only_after_uncertain_rule():
    backend = RecordingBackend(status="supported", reason="同义改写支持")
    result = ClaimGrounder(backend=backend).ground(
        _claim(
            claim="SH-E300 耳机采用当前目录中的标准报价。",
            evidence=[
                EvidenceCandidate(
                    evidence_id="E1",
                    text="SH-E300 头戴式无线降噪耳机，售价 299 元。",
                )
            ],
        )
    )

    assert result.status == "supported"
    assert backend.calls == 1
    assert result.backend_called is True
    assert result.baseline_status in {"uncertain", "supported"}


def test_backend_failure_and_timeout_remain_uncertain():
    backend = RaisingBackend(TimeoutError("timeout"))
    result = ClaimGrounder(backend=backend).ground(
        _claim(
            claim="SH-E300 耳机采用当前目录中的标准报价。",
            evidence=[
                EvidenceCandidate(
                    evidence_id="E1",
                    text="SH-E300 头戴式无线降噪耳机，售价 299 元。",
                )
            ],
        )
    )

    assert result.status == "uncertain"
    assert "timeout" in result.reason.lower()
    assert backend.calls == 1


def test_missing_evidence_and_invalid_semantic_citation_remain_uncertain():
    no_evidence = ClaimGrounder(backend=RecordingBackend(status="supported")).ground(
        _claim(evidence=[])
    )
    assert no_evidence.status == "uncertain"

    backend = StructuredGroundingBackend(
        caller=FakeCaller(
            [
                ProviderResponse(
                    text=json.dumps(
                        {
                            "status": "supported",
                            "reason": "支持",
                            "evidence_ids": ["NOT-IN-CONTEXT"],
                            "conflict_codes": [],
                        }
                    ),
                    usage={"input_tokens": 10, "output_tokens": 10},
                    model_version="deepseek-flash-test",
                    latency_ms=12.0,
                )
            ]
        ),
        ledger=ClaimBudgetLedger(":memory:", total_limit_usd=2.0),
        pricing={"input_per_million": 0.30, "output_per_million": 1.20},
        reserve_usd=0.0001,
    )
    result = backend.ground(
        _claim(
            claim="SH-E300 耳机采用当前目录中的标准报价。",
            evidence=[EvidenceCandidate(evidence_id="E1", text="等价但不同表面写法")],
        )
    )
    assert result.status == "uncertain"
    assert "引用" in result.reason or "evidence" in result.reason.lower()


def test_api_unsupported_requires_valid_citation_and_conflict_code():
    ledger = ClaimBudgetLedger(":memory:", total_limit_usd=2.0)
    backend = StructuredGroundingBackend(
        caller=FakeCaller(
            [
                ProviderResponse(
                    text=json.dumps(
                        {
                            "status": "unsupported",
                            "reason": "型号和价格冲突",
                            "evidence_ids": ["E1"],
                            "conflict_codes": ["numeric_mismatch"],
                        }
                    ),
                    usage={"input_tokens": 20, "output_tokens": 12},
                    model_version="deepseek-flash-test",
                    latency_ms=8.5,
                )
            ]
        ),
        ledger=ledger,
        pricing={"input_per_million": 0.30, "output_per_million": 1.20},
        reserve_usd=0.0001,
    )
    result = backend.ground(_claim(claim="SH-E300 的售价是 399 元。"))

    assert result.status == "unsupported"
    assert result.evidence_ids == ["E1"]
    assert result.usage.input_tokens == 20
    assert ledger.total_usd() < 0.0001


def test_binary_not_entailment_is_not_unsupported():
    contract = NLIContract()
    assert (
        contract.from_labels({"entailment": 0.91, "not_entailment": 0.09})
        == "supported"
    )
    assert (
        contract.from_labels({"entailment": 0.10, "not_entailment": 0.90})
        == "uncertain"
    )


def test_backend_retries_429_at_most_twice():
    ledger = ClaimBudgetLedger(":memory:", total_limit_usd=2.0)
    caller = FakeCaller(
        [
            Retry429("429 rate limit"),
            Retry429("429 rate limit"),
            ProviderResponse(
                text=json.dumps(
                    {
                        "status": "supported",
                        "reason": "证据支持",
                        "evidence_ids": ["E1"],
                        "conflict_codes": [],
                    }
                ),
                usage={"input_tokens": 10, "output_tokens": 5},
                model_version="deepseek-flash-test",
                latency_ms=4.0,
            ),
        ]
    )
    backend = StructuredGroundingBackend(
        caller=caller,
        ledger=ledger,
        pricing={"input_per_million": 0.30, "output_per_million": 1.20},
        reserve_usd=0.0001,
        max_retries=2,
        retry_sleep_s=0,
    )
    result = backend.ground(_claim())

    assert result.status == "supported"
    assert result.attempts == 3
    assert caller.calls == 3


def test_backend_does_not_retry_non_429():
    caller = FakeCaller([RuntimeError("bad request")])
    backend = StructuredGroundingBackend(
        caller=caller,
        ledger=ClaimBudgetLedger(":memory:", total_limit_usd=2.0),
        pricing={"input_per_million": 0.30, "output_per_million": 1.20},
        reserve_usd=0.0001,
        max_retries=2,
        retry_sleep_s=0,
    )
    try:
        backend.ground(_claim())
    except RuntimeError as exc:
        assert "bad request" in str(exc)
    else:
        raise AssertionError("non-429 error should not be retried")
    assert caller.calls == 1


def test_answer_verifier_semantic_adapter_returns_answer_verdict():
    backend = RecordingBackend(status="supported", reason="同义支持")
    adapter = ClaimGroundingAnswerSemantic(ClaimGrounder(backend=backend))
    verifier = AnswerVerifier(adapter)
    result = verifier.verify(
        "该耳机采用当前目录中的标准报价 [1]。",
        [
            {
                "n": 1,
                "section_path": "商品",
                "question": "SH-E300 售价",
                "answer": "SH-E300 售价 299 元。",
            }
        ],
        query="SH-E300 多少钱？",
    )

    assert result.status == "pass"
    assert backend.calls == 1


def test_budget_limit_fails_closed_and_reservations_can_run_concurrently(tmp_path):
    path = tmp_path / "claim-budget.sqlite"
    ledger = ClaimBudgetLedger(path, total_limit_usd=0.01)
    first = ledger.reserve(0.006, model="deepseek-flash", kind="claim_grounding")
    try:
        ledger.reserve(0.006, model="deepseek-flash", kind="claim_grounding")
    except BudgetExceeded:
        pass
    else:
        raise AssertionError("concurrent reservations exceeded the hard cap")
    ledger.settle(
        first, actual_usd=0.002, usage={"input_tokens": 100, "output_tokens": 20}
    )

    second = ledger.reserve(0.006, model="deepseek-flash", kind="claim_grounding")
    ledger.settle(
        second, actual_usd=0.003, usage={"input_tokens": 120, "output_tokens": 30}
    )
    assert ledger.total_usd() == 0.005


def test_budget_atomically_rejects_one_of_two_concurrent_reservations(tmp_path):
    ledger = ClaimBudgetLedger(
        tmp_path / "concurrent.sqlite",
        total_limit_usd=0.01,
    )

    def reserve():
        try:
            return ledger.reserve(
                0.006,
                model="deepseek-flash",
                kind="claim_grounding",
            )
        except BudgetExceeded:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        values = list(pool.map(lambda _: reserve(), range(2)))
    assert sum(value is not None for value in values) == 1
    assert ledger.total_usd() == 0.006


def test_prompt_contains_candidates_but_not_gold_fields():
    backend = StructuredGroundingBackend(
        caller=FakeCaller([]),
        ledger=ClaimBudgetLedger(":memory:", total_limit_usd=2.0),
        pricing={"input_per_million": 0.30, "output_per_million": 1.20},
    )
    prompt = backend.build_prompt(_claim())

    assert "[E1] SH-E300 头戴式无线降噪耳机，售价 299 元。" in prompt
    assert "gold_reason" not in prompt
    assert '"gold"' not in prompt
