import json
import hashlib
from collections import Counter
from pathlib import Path


SCENARIOS = Path(__file__).with_name("scenarios.jsonl")
SEMANTIC_RUBRIC = Path(__file__).with_name("semantic-rubric-v2.json")
EXPECTED_COUNTS = {
    "policy_qa": 14,
    "order_lookup": 12,
    "refund_guidance": 18,
    "multi_turn_revision": 12,
    "refusal_handoff": 10,
    "fault_injection": 6,
}
ROW_FIELDS = {
    "id", "category", "comparison_scope", "allowed_tools", "forbidden_actions",
    "turns", "gold", "fault",
}


def read_scenarios():
    return [json.loads(line) for line in SCENARIOS.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def test_ch15_scenario_set_is_fixed_and_complete():
    scenarios = read_scenarios()

    assert len(scenarios) == 72
    assert len({row["id"] for row in scenarios}) == 72
    assert Counter(row["category"] for row in scenarios) == EXPECTED_COUNTS


def test_every_scenario_has_precommitted_gold_actions_and_database_assertions():
    for row in read_scenarios():
        assert ROW_FIELDS <= row.keys(), row.get("id")
        assert row["comparison_scope"] in {"paired_refund", "single_agent"}
        assert row["allowed_tools"] is not None
        assert row["forbidden_actions"]
        assert row["turns"]

        arms = {"fixed", "autonomous"} if row["comparison_scope"] == "paired_refund" else {"autonomous"}
        assert set(row["gold"]) == arms, row["id"]
        for arm in arms:
            gold = row["gold"][arm]
            assert gold["outcome"] in {"answer", "clarify", "refuse", "handoff"}
            assert isinstance(gold["expected_tools"], list)
            assert isinstance(gold["expected_calls"], list)
            assert "tickets" in gold["db_assertions"]
            expected_tools = [name for turn in row["turns"]
                              for name in turn["expected"][arm]["tools"]]
            expected_calls = [call for turn in row["turns"]
                              for call in turn["expected"][arm]["calls"]]
            assert gold["expected_tools"] == expected_tools, row["id"]
            assert gold["expected_calls"] == expected_calls, row["id"]
            assert set(expected_tools) <= set(row["allowed_tools"]), row["id"]

        for turn in row["turns"]:
            assert turn["message"].strip()
            assert set(turn["actors"]) == arms
            assert set(turn["intents"]) == arms
            assert set(turn["expected"]) == arms
            for actor in turn["actors"].values():
                assert "tool_calls" in actor and "reply" in actor


def test_semantic_rubric_is_bound_to_the_frozen_manifest_and_known_cases():
    rubric = json.loads(SEMANTIC_RUBRIC.read_text(encoding="utf-8"))
    manifest_hash = hashlib.sha256(SCENARIOS.read_bytes()).hexdigest()
    scenario_ids = {row["id"] for row in read_scenarios()}

    assert rubric["applies_to_scenario_manifest_sha256"] == manifest_hash
    assert set(rubric["case_corrections"]) <= scenario_ids
    assert rubric["name"] == "semantic_task_v2"
