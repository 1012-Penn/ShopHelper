"""评估集 schema 校验:字段齐全、意图标签在枚举内、验收 1 的来回切用例存在。"""
import json
from pathlib import Path

from app.graph.entry import INTENTS

PATH = Path("tests/eval/router_samples.jsonl")


def _rows():
    return [json.loads(line) for line in PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_rows_schema():
    for row in _rows():
        assert row["id"] and row["tag"]
        for t in row["turns"]:
            assert t["user"] and t["expect_intent"] in INTENTS
            if "expect_resolved" in t:
                assert t["expect_resolved"]


def test_covers_switch_and_other():
    rows = _rows()
    assert any(r["tag"] == "多轮切换" for r in rows)
    assert any(t["expect_intent"] == "其他" for r in rows for t in r["turns"])
    # 验收 1 的来回切:同对话内物流→退款退货→物流
    assert any([t["expect_intent"] for t in r["turns"]][:3] == ["物流", "退款退货", "物流"]
               for r in rows)
