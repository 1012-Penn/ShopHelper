from __future__ import annotations

from scripts.answer_quality_depth.evidence import (
    evidence_rows_from_ids,
    select_from_saved_rerank,
)


def _rows():
    return {
        1: {
            "chunk_id": 1,
            "section_path": "政策版本说明.md > v2 版十五天无理由",
            "text": "签收日期为 2026-06-01 或之后的订单适用 v2，15 天内可退。",
        },
        2: {
            "chunk_id": 2,
            "section_path": "退货政策.md > 不予退货情形",
            "text": "定制类商品仅在有质量问题时可退。",
        },
        3: {
            "chunk_id": 3,
            "section_path": "商品目录.md > SH-E300",
            "text": "SH-E300 售价 299 元。",
        },
    }


def test_saved_rerank_uses_query_facets_for_slate_only() -> None:
    payload = select_from_saved_rerank(
        case_id="C1",
        query="定制键盘签收 3 天，没有质量问题，能无理由退吗？",
        artifact={
            "case_id": "C1",
            "candidate_ids": [3, 1, 2],
            "ranked": [[0, 0.9], [1, 0.8], [2, 0.7]],
        },
        rows=_rows(),
        top_k=2,
    )

    assert [row["chunk_id"] for row in payload["baseline"]] == [3, 1]
    assert next(row["chunk_id"] for row in payload["slate"]) == 2
    assert payload["selection_facets"]


def test_evidence_rows_are_bounded_and_have_no_evaluation_labels() -> None:
    rows = evidence_rows_from_ids(
        [3, 1, 2],
        [0.9, 0.8, 0.7],
        _rows(),
        top_k=2,
    )

    assert len(rows) == 2
    assert rows[0]["chunk_id"] == 3
    assert "expected_behavior" not in rows[0]
    assert "gold_facts" not in rows[0]
