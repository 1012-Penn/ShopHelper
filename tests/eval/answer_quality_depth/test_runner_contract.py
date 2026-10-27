from __future__ import annotations

import hashlib
import json

import pytest

from scripts.answer_quality_depth.runner import (
    ResponseJournal,
    evidence_projection,
    generation_was_truncated,
    judge_response_text,
    parse_judge_payload,
    request_key,
    run_config_hash,
)


def test_response_journal_separates_generation_and_judge_and_arms(tmp_path) -> None:
    journal = ResponseJournal(tmp_path / "responses.sqlite3")
    journal.put("answer", "case:dev:arm:slate", {"text": "x"})
    journal.put("judge", "case:dev:arm:slate", {"status": "correct"})

    assert journal.get("answer", "case:dev:arm:slate") == {"text": "x"}
    assert journal.get("judge", "case:dev:arm:slate") == {"status": "correct"}
    assert journal.get("answer", "case:dev:arm:baseline") is None


def test_evidence_projection_is_balanced_and_contains_no_gold() -> None:
    payload = evidence_projection(
        case={
            "id": "C1",
            "query": "Q",
            "bucket": "multi_hop",
            "required_evidence": [{"quotes": ["gold"]}],
        },
        baseline=[{"chunk_id": 1, "text": "base", "score": 0.9}],
        slate=[{"chunk_id": 2, "text": "slate", "score": 0.8}],
        top_k=1,
    )

    assert payload["case"]["query"] == "Q"
    assert len(payload["arms"]["baseline"]) == len(payload["arms"]["slate"]) == 1
    assert "gold" not in json.dumps(payload, ensure_ascii=False)
    assert "bucket" not in json.dumps(payload, ensure_ascii=False)


def test_run_config_hash_changes_when_judge_prompt_changes() -> None:
    base = {
        "model": "deepseek-flash",
        "dataset_sha256": {"dev": "a"},
        "generation_prompt_sha256": "b",
        "judge_prompt_sha256": "c",
        "top_k": 10,
    }
    changed = {**base, "judge_prompt_sha256": "d"}

    assert run_config_hash(base) != run_config_hash(changed)
    assert (
        run_config_hash(base)
        == hashlib.sha256(
            json.dumps(base, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
    )


def test_judge_payload_is_strict_and_contains_no_method_identity() -> None:
    payload = parse_judge_payload(
        json.dumps(
            {
                "disposition": "answer",
                "answer_status": "correct",
                "grounding": "supported",
                "completeness": 2,
                "citation_support": "supported",
                "unsupported_claims": [],
                "missing_requirements": [],
                "rationale": "证据直接给出价格。",
            },
            ensure_ascii=False,
        )
    )

    assert payload["answer_status"] == "correct"

    wrapped = parse_judge_payload(
        json.dumps(
            {
                "type": "chunk",
                "disposition": "answer",
                "answer_status": "correct",
                "grounding": "supported",
                "completeness": 2,
                "citation_support": "supported",
                "unsupported_claims": [],
                "missing_requirements": [],
                "rationale": "证据直接给出价格。",
            },
            ensure_ascii=False,
        )
    )
    assert "type" not in wrapped

    with pytest.raises(ValueError):
        parse_judge_payload(
            '{"disposition":"answer","answer_status":"correct","grounding":"supported",'
            '"completeness":2,"citation_support":"supported","arm":"slate",'
            '"unsupported_claims":[],"missing_requirements":[],"rationale":"x"}'
        )


def test_request_key_depends_on_evidence_and_prompt_hash() -> None:
    first = request_key(
        phase="answer",
        model="deepseek-flash",
        prompt_sha256="p",
        case_id="C1",
        arm="baseline",
        evidence=[{"chunk_id": 1}],
    )
    second = request_key(
        phase="answer",
        model="deepseek-flash",
        prompt_sha256="p",
        case_id="C1",
        arm="baseline",
        evidence=[{"chunk_id": 2}],
    )
    third = request_key(
        phase="judge",
        model="deepseek-flash",
        prompt_sha256="p",
        case_id="C1",
        arm="baseline",
        evidence=[{"chunk_id": 1}],
    )

    assert first != second
    assert first != third
    assert first != request_key(
        phase="answer",
        model="deepseek-flash",
        prompt_sha256="p",
        case_id="C1",
        arm="baseline",
        evidence=[{"chunk_id": 1}],
        variant="non_thinking",
    )


def test_judge_text_falls_back_to_reasoning_when_content_is_empty() -> None:
    raw = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "reasoning_content": (
                        '分析结束。{"disposition":"refuse","answer_status":"not_answerable",'
                        '"grounding":"unsupported","completeness":0,'
                        '"citation_support":"not_applicable","unsupported_claims":[],'
                        '"missing_requirements":[],"rationale":"证据不足。"}'
                    ),
                }
            }
        ]
    }

    payload = parse_judge_payload(judge_response_text("", raw))

    assert payload["disposition"] == "refuse"


def test_generation_truncation_is_detected_from_raw_finish_reason() -> None:
    assert generation_was_truncated(
        {
            "answer": "",
            "generation": {"raw": {"choices": [{"finish_reason": "length"}]}},
        }
    )
    assert generation_was_truncated(
        {
            "answer": "ok",
            "generation": {"raw": {"choices": [{"finish_reason": "length"}]}},
        }
    )
