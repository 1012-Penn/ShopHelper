"""ch11 评测数据集 v2:schema、registry、overlay、稳定哈希与校验。

全部使用真实 regression 文件、真实 knowledge 文档与真实 FactBook,不 mock。
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.corpus_facts import load_factbook
from scripts.eval_dataset import (
    BUCKETS,
    EXPECTED_FROZEN,
    SCHEMA_VERSION,
    EvalDataError,
    apply_overlay,
    bucket_counts,
    dataset_sha256,
    load_cases_from_path,
    load_registry,
    load_split,
    split_sha256,
    validate_cases,
)

ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = ROOT / "knowledge"
CH04_PATH = ROOT / "tests" / "eval" / "ch04_eval_set.jsonl"
CH09_PATH = ROOT / "tests" / "eval" / "ch09_junk_set.jsonl"


def _case(case_id: str = "T01", **overrides) -> dict:
    """一条除目标规则外都合法的 v2 用例。"""
    case = {
        "schema_version": SCHEMA_VERSION,
        "id": case_id,
        "split": "dev",
        "bucket": "single_hop",
        "label_version": "v2",
        "query": f"查询 {case_id}",
        "expected_behavior": "answer",
        "answerable": True,
        "evidence_state": "sufficient",
        "gold_facts": [{"fact_id": "CATALOG-SH-E300-PRICE", "value": 299}],
    }
    case.update(overrides)
    return case


def test_constants_match_contract():
    assert SCHEMA_VERSION == 2
    assert BUCKETS == (
        "single_hop",
        "multi_hop",
        "multi_constraint",
        "entity_mismatch",
        "policy_conflict",
        "hard_negative",
        "clarify",
        "ood",
    )
    assert EXPECTED_FROZEN == {
        "single_hop": 18,
        "multi_hop": 16,
        "multi_constraint": 14,
        "entity_mismatch": 10,
        "policy_conflict": 8,
        "hard_negative": 10,
        "clarify": 8,
        "ood": 6,
    }
    assert sum(EXPECTED_FROZEN.values()) == 90


def test_schema_rejects_refuse_without_reason():
    bad = _case(
        expected_behavior="refuse",
        answerable=False,
        evidence_state="insufficient",
        expected_refusal_reason="",
        gold_facts=[],
    )
    with pytest.raises(EvalDataError) as excinfo:
        validate_cases([bad], split="dev")
    assert "expected_refusal_reason" in str(excinfo.value)


def test_schema_rejects_refuse_when_answerable_and_state_wrong():
    bad = _case(
        expected_behavior="refuse",
        answerable=True,
        evidence_state="sufficient",
        expected_refusal_reason="知识库无依据",
        gold_facts=[],
    )
    with pytest.raises(EvalDataError) as excinfo:
        validate_cases([bad], split="dev")
    message = str(excinfo.value)
    assert "answerable=false" in message
    assert "insufficient" in message


def test_schema_rejects_clarify_without_slots():
    bad = _case(
        expected_behavior="clarify",
        answerable=True,
        clarifying_slots=[],
        gold_facts=[],
    )
    with pytest.raises(EvalDataError) as excinfo:
        validate_cases([bad], split="dev")
    assert "clarifying_slots" in str(excinfo.value)


def test_schema_rejects_answer_with_conditions_without_conditions():
    bad = _case(expected_behavior="answer_with_conditions")
    with pytest.raises(EvalDataError) as excinfo:
        validate_cases([bad], split="dev")
    assert "answer_with_conditions" in str(excinfo.value)
    assert "required_conditions" in str(excinfo.value)


def test_schema_rejects_non_refuse_when_not_answerable():
    bad = _case(expected_behavior="answer", answerable=False)
    with pytest.raises(EvalDataError) as excinfo:
        validate_cases([bad], split="dev")
    assert "非 refuse" in str(excinfo.value)


def test_new_split_rejects_unknown_top_level_field():
    bad = _case(required_evidenc=[])
    with pytest.raises(EvalDataError) as excinfo:
        validate_cases([bad], split="dev")
    assert "未知字段" in str(excinfo.value)
    assert "required_evidenc" in str(excinfo.value)


def test_schema_rejects_policy_conflict_without_as_of_date():
    bad = _case(
        bucket="policy_conflict",
        expected_behavior="answer_with_conditions",
        evidence_state="conflicting",
        as_of_date=None,
        gold_facts=[{"fact_id": "CATALOG-SH-E300-PRICE", "value": 299}],
    )
    with pytest.raises(EvalDataError) as excinfo:
        validate_cases([bad], split="dev")
    assert "as_of_date" in str(excinfo.value)


def test_schema_rejects_verify_then_answer_without_conditions():
    bad = _case(
        expected_behavior="verify_then_answer",
        required_conditions=[],
        forbidden_claims=[],
    )
    with pytest.raises(EvalDataError) as excinfo:
        validate_cases([bad], split="dev")
    message = str(excinfo.value)
    assert "required_conditions" in message
    assert "forbidden_claims" in message


def test_schema_rejects_answer_with_non_sufficient_evidence():
    bad = _case(expected_behavior="answer", evidence_state="partial")
    with pytest.raises(EvalDataError) as excinfo:
        validate_cases([bad], split="dev")
    assert "evidence_state" in str(excinfo.value)


def test_overlay_defaults_and_explicit_entries_merge():
    source = [
        {"id": "C01", "bucket": "C_colloquial", "query": "q1", "answerable": True,
         "notes": "原始备注"}
    ]
    overlay = {
        "defaults": {
            "expected_behavior": "answer",
            "source": "synthetic",
            "notes": "默认备注",
        },
        "cases": {
            "C01": {
                "expected_behavior": "verify_then_answer",
                "required_conditions": ["需检测"],
                "forbidden_claims": ["一定可以免费退"],
            }
        },
    }

    merged = apply_overlay(source, overlay)

    assert merged[0]["expected_behavior"] == "verify_then_answer"
    assert merged[0]["source"] == "synthetic"
    # original 字段优先于 defaults,显式 cases 条目优先于 original
    assert merged[0]["notes"] == "原始备注"
    assert merged[0]["required_conditions"] == ["需检测"]
    assert merged[0]["forbidden_claims"] == ["一定可以免费退"]
    # 深拷贝:源数据不被修改
    assert source[0] == {
        "id": "C01", "bucket": "C_colloquial", "query": "q1",
        "answerable": True, "notes": "原始备注",
    }
    assert merged[0] is not source[0]

    untouched = apply_overlay(source, None)
    assert untouched == source
    assert untouched[0] is not source[0]


def test_overlay_rejects_unknown_case_id():
    source = [{"id": "C01", "query": "q1"}]
    overlay = {"defaults": {}, "cases": {"C99": {"notes": "x"}}}
    with pytest.raises(EvalDataError) as excinfo:
        apply_overlay(source, overlay)
    assert "C99" in str(excinfo.value)


def test_split_sha256_ignores_review_metadata():
    first = [
        {"id": "B", "bucket": "single_hop", "query": "b"},
        {"id": "A", "bucket": "single_hop", "query": "a"},
    ]
    reordered = [copy.deepcopy(first[1]), copy.deepcopy(first[0])]
    assert split_sha256(first) == split_sha256(reordered)

    reviewed = copy.deepcopy(first)
    reviewed[0]["annotator"] = "claude-code"
    reviewed[0]["reviewer"] = "codex"
    reviewed[0]["review_status"] = "frozen"
    assert split_sha256(reviewed) == split_sha256(first)

    changed = copy.deepcopy(first)
    changed[0]["query"] = "b-changed"
    assert split_sha256(changed) != split_sha256(first)


def test_dataset_sha256_hashes_raw_file_bytes(tmp_path: Path):
    path = tmp_path / "cases.jsonl"
    payload = '{"id": "A", "query": "问题"}\n'.encode("utf-8")
    path.write_bytes(payload)
    assert dataset_sha256(path) == hashlib.sha256(payload).hexdigest()


def test_duplicate_ids_and_queries_rejected():
    duplicated_ids = [_case("X", query="q1"), _case("X", query="q2")]
    with pytest.raises(EvalDataError) as excinfo:
        validate_cases(duplicated_ids, split="dev")
    assert "id 重复" in str(excinfo.value)

    duplicated_queries = [_case("X", query="同一条"), _case("Y", query="同一条")]
    with pytest.raises(EvalDataError) as excinfo:
        validate_cases(duplicated_queries, split="dev")
    assert "query 重复" in str(excinfo.value)


def test_new_split_answer_cases_require_gold_facts():
    bad = _case(gold_facts=[])
    with pytest.raises(EvalDataError) as excinfo:
        validate_cases([bad], split="dev")
    assert "gold_facts" in str(excinfo.value)

    # regression-v1 显式豁免
    legacy = _case("A01", split="regression-v1", gold_facts=[])
    validate_cases([legacy], split="regression-v1")


def test_requires_tools_must_be_covered_by_fixtures():
    bad = _case(
        gold_facts=[{"fact_id": "CATALOG-SH-E300-PRICE", "value": 299}],
        requires_tools=["create_ticket"],
        tool_fixtures=[],
    )
    with pytest.raises(EvalDataError) as excinfo:
        validate_cases([bad], split="dev")
    assert "create_ticket" in str(excinfo.value)


def test_registry_registers_only_regression_v1():
    registry = load_registry()
    splits = registry["splits"]
    assert set(splits) >= {"regression-v1"}
    entry = splits["regression-v1"]
    assert entry["kind"] == "composed"
    assert entry["n"] == 60
    assert entry["label_version"] == "v2-overlay"
    assert [item["path"] for item in entry["cases"]] == [
        "tests/eval/ch04_eval_set.jsonl",
        "tests/eval/ch09_junk_set.jsonl",
    ]
    assert all(item["overlay"] is None for item in entry["cases"])


def test_regression_v1_composes_old_files_with_v2_defaults():
    cases = load_split("regression-v1")

    assert len(cases) == 60
    assert {case["split"] for case in cases} == {"regression-v1"}
    assert {case["label_version"] for case in cases} == {"v2-overlay"}
    assert all(case["schema_version"] == SCHEMA_VERSION for case in cases)
    assert len({case["id"] for case in cases}) == 60

    for case in cases:
        expected_behavior = "answer" if case["answerable"] else "refuse"
        expected_state = "sufficient" if case["answerable"] else "insufficient"
        assert case["expected_behavior"] == expected_behavior, case["id"]
        assert case["evidence_state"] == expected_state, case["id"]
        assert case["gold_facts"] == [], case["id"]
        if not case["answerable"]:
            assert case["expected_refusal_reason"], case["id"]

    a01 = next(case for case in cases if case["id"] == "A01")
    # 旧字段全部保留,不被 v2 默认值覆盖
    assert a01["expect_doc"] == ["退货政策.md"]
    assert a01["notes"] == "运费说明"
    assert a01["required_evidence"][0]["doc"] == "退货政策.md"

    j01 = next(case for case in cases if case["id"] == "J01")
    assert "保修信息" in j01["expected_refusal_reason"]
    j02 = next(case for case in cases if case["id"] == "J02")
    assert j02["expected_refusal_reason"] == "家电类纯噪声"

    # 源 JSONL 一个字节都不改
    raw = load_cases_from_path(CH04_PATH)
    assert all("schema_version" not in row for row in raw)
    assert all("expected_behavior" not in row for row in raw)


def test_regression_v1_validates_against_real_documents():
    cases = load_split("regression-v1")
    validate_cases(cases, split="regression-v1", docs_dir=KNOWLEDGE_DIR)


def test_regression_v1_hash_is_stable_across_loads():
    first = split_sha256(load_split("regression-v1"))
    second = split_sha256(load_split("regression-v1"))
    assert first == second
    assert len(first) == 64


def test_gold_facts_are_checked_against_factbook():
    book = load_factbook()
    fact = book.fact("CATALOG-SH-E300-PRICE")
    gold = {"fact_id": fact.fact_id, "key": fact.key, "value": fact.value, "unit": fact.unit}
    good = _case(gold_facts=[gold])
    validate_cases([good], split="dev", factbook=book)

    missing = _case(gold_facts=[{"fact_id": "NO-SUCH-FACT", "value": 1}])
    with pytest.raises(EvalDataError) as excinfo:
        validate_cases([missing], split="dev", factbook=book)
    assert "NO-SUCH-FACT" in str(excinfo.value)

    wrong_value = copy.deepcopy(good)
    wrong_value["gold_facts"][0]["value"] = fact.value + 12345 if isinstance(
        fact.value, (int, float)) else "错值"
    with pytest.raises(EvalDataError) as excinfo:
        validate_cases([wrong_value], split="dev", factbook=book)
    assert "gold_fact" in str(excinfo.value)

    wrong_unit = copy.deepcopy(good)
    wrong_unit["gold_facts"][0]["unit"] = "不存在的单位"
    with pytest.raises(EvalDataError) as excinfo:
        validate_cases([wrong_unit], split="dev", factbook=book)
    assert "unit" in str(excinfo.value)


def test_gold_fact_as_of_date_outside_effective_range_rejected():
    book = load_factbook()
    fact = next(
        item for item in book.facts.values()
        if item.effective_from is not None and item.effective_to is not None
    )
    gold = {"fact_id": fact.fact_id, "key": fact.key, "value": fact.value, "unit": fact.unit}
    bad = _case(gold_facts=[gold], as_of_date="2000-01-01")
    with pytest.raises(EvalDataError) as excinfo:
        validate_cases([bad], split="dev", factbook=book)
    assert "as_of_date" in str(excinfo.value)

    inside = _case(gold_facts=[gold], as_of_date=fact.effective_from.isoformat())
    validate_cases([inside], split="dev", factbook=book)


def test_evidence_quote_must_be_verbatim():
    bad = _case(
        required_evidence=[{
            "doc": "退货政策.md",
            "section": "邮费与运费",
            "quotes": ["这句引用并不存在于知识库"],
        }]
    )
    with pytest.raises(EvalDataError) as excinfo:
        validate_cases([bad], split="dev", docs_dir=KNOWLEDGE_DIR)
    assert "quote" in str(excinfo.value)


def test_cross_split_id_collision():
    from scripts.eval_dataset import cross_split_issues

    assert cross_split_issues({"dev": [{"id": "D01"}], "frozen-v1": [{"id": "D01"}]})
    assert cross_split_issues({"dev": [{"id": "D01"}], "frozen-v1": [{"id": "F01"}]}) == []
    # 只有单边存在时跳过交叉检查
    assert cross_split_issues({"dev": [{"id": "D01"}]}) == []


def test_bucket_and_behavior_counts():
    cases = [
        _case("A", bucket="single_hop"),
        _case("B", bucket="single_hop", expected_behavior="answer_with_conditions"),
        _case("C", bucket="ood", expected_behavior="refuse",
              answerable=False, evidence_state="insufficient",
              expected_refusal_reason="知识库无依据", gold_facts=[]),
    ]
    assert bucket_counts(cases) == {"single_hop": 2, "ood": 1}
    from scripts.eval_dataset import behavior_counts

    assert behavior_counts(cases) == {"answer": 1, "answer_with_conditions": 1, "refuse": 1}


def test_validate_cli_all_passes():
    from scripts import validate_eval_data

    code = validate_eval_data.main(
        ["--split", "all", "--check-evidence", "--check-facts", "--check-buckets"]
    )
    assert code == 0


def test_validate_cli_unregistered_split_exits_4():
    from scripts import validate_eval_data

    assert validate_eval_data.main(["--split", "does-not-exist"]) == 4
