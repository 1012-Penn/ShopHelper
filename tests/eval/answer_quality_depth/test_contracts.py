from __future__ import annotations

import json

from scripts.answer_quality_depth.data import (
    ARM_BASELINE,
    ARM_SLATE,
    build_generation_messages,
    build_judge_prompt,
    case_projection,
    load_depth_cases,
)


def _case() -> dict:
    return {
        "id": "DEV-SINGLE-01",
        "split": "dev",
        "bucket": "single_hop",
        "query": "SH-E300 多少钱？",
        "expected_behavior": "answer",
        "gold_facts": [
            {
                "fact_id": "CATALOG-SH-E300-PRICE",
                "value": 299,
                "unit": "元",
            }
        ],
        "required_conditions": [],
        "forbidden_claims": [],
        "required_evidence": [{"quotes": ["SH-E300 售价 299 元"]}],
        "conversation_context": [],
    }


def _evidence() -> list[dict]:
    return [
        {
            "chunk_id": 167,
            "section_path": "商品目录.md > 商品条目 > SH-E300",
            "text": "SH-E300 头戴式无线降噪耳机，售价 299 元。",
            "score": 0.92,
        }
    ]


def test_model_projection_contains_no_labels_or_gold() -> None:
    projected = case_projection(_case())
    encoded = json.dumps(projected, ensure_ascii=False)

    assert projected == {
        "query": "SH-E300 多少钱？",
        "conversation_context": [],
    }
    for forbidden in (
        "expected_behavior",
        "gold_facts",
        "required_conditions",
        "forbidden_claims",
        "required_evidence",
        "bucket",
    ):
        assert forbidden not in encoded


def test_generation_arms_differ_only_in_the_evidence_block() -> None:
    baseline = build_generation_messages(_case(), _evidence(), arm=ARM_BASELINE)
    slate = build_generation_messages(
        _case(),
        [
            {
                **_evidence()[0],
                "chunk_id": 168,
                "text": "SH-E300 售价 299 元；配件另售。",
            }
        ],
        arm=ARM_SLATE,
    )

    assert baseline[0] == slate[0]
    assert baseline[1]["role"] == slate[1]["role"] == "user"
    left = baseline[1]["content"].replace(
        "[1] 商品目录.md > 商品条目 > SH-E300\n"
        "SH-E300 头戴式无线降噪耳机，售价 299 元。",
        "<EVIDENCE>",
    )
    right = slate[1]["content"].replace(
        "[1] 商品目录.md > 商品条目 > SH-E300\nSH-E300 售价 299 元；配件另售。",
        "<EVIDENCE>",
    )
    assert left == right


def test_judge_prompt_is_anonymous_and_has_no_labels_or_arm() -> None:
    prompt = build_judge_prompt(_case(), _evidence(), "SH-E300 售价 299 元 [1]。")
    encoded = prompt.casefold()

    assert "baseline" not in encoded
    assert "slate" not in encoded
    assert "候选答案甲" in prompt
    assert "gold_facts" not in encoded
    assert "expected_behavior" not in encoded
    assert "DEV-SINGLE-01" not in prompt


def test_depth_dataset_uses_the_frozen_170_scope() -> None:
    cases = load_depth_cases()

    assert [
        len(cases[name]) for name in ("dev", "frozen-v1", "retrieval-slate-holdout-v1")
    ] == [
        60,
        90,
        20,
    ]
    assert (
        sum(
            bool(case.get("required_evidence"))
            for rows in cases.values()
            for case in rows
        )
        == 128
    )
    assert (
        sum(
            not bool(case.get("required_evidence"))
            for rows in cases.values()
            for case in rows
        )
        == 42
    )
