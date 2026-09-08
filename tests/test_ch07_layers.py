"""ch07 分层切分/渲染/组装/级联:锚点数学、半压三规则、trim_messages 装填、背景块、工具截断。"""
from langchain_core.messages import AIMessage, HumanMessage

from app.context import (build_background, build_layered, cascade_layer1,
                         layer2_tokens, render_history_text, render_layer2,
                         split_layers, truncate_tool_result)
from app.history import estimate_tokens, rows_to_messages


def _rows(*pairs):
    """(user, assistant) 对 → id 从 100 起的连续行。"""
    out, i = [], 100
    for u, a in pairs:
        out.append({"id": i, "role": "user", "content": u})
        i += 1
        out.append({"id": i, "role": "assistant", "content": a})
        i += 1
    return out


# ---- 切分 ----

def test_split_layers_null_anchors_all_layer1():
    rows = _rows(("u", "a"))
    l1, l2 = split_layers(rows, None, None)
    assert len(l1) == 2 and l2 == []


def test_split_layers_band_boundaries():
    rows = _rows(("u1", "a1"), ("u2", "a2"), ("u3", "a3"))  # id 100..105
    l1, l2 = split_layers(rows, summary_upto=101, layer1_from=103)
    assert [r["id"] for r in l2] == [102, 103]
    assert [r["id"] for r in l1] == [104, 105]


def test_split_layers_summary_only_no_layer1_anchor():
    rows = _rows(("u1", "a1"), ("u2", "a2"))
    l1, l2 = split_layers(rows, summary_upto=101, layer1_from=None)
    assert l2 == [] and len(l1) == 2


# ---- 层 2 半压渲染 ----

def test_render_layer2_rules():
    rows = [{"id": 1, "role": "user", "content": "原始用户话" * 30},
            {"id": 2, "role": "assistant", "content": "客服长答复" * 40},
            {"id": 3, "role": "tool", "content": "大块JSON"},
            {"id": 4, "role": "assistant", "content": None,
             "tool_calls": [{"name": "q", "args": {}, "id": "1"}]}]
    msgs = render_layer2(rows, head_chars=60)
    assert isinstance(msgs[0], HumanMessage)
    assert msgs[0].content == "原始用户话" * 30                      # user 原文不动
    assert msgs[1].content.endswith("…") and len(msgs[1].content) <= 61  # assistant 截头
    assert msgs[2].content == "[工具结果已省略]"                      # tool 一行标识
    assert len(msgs) == 3                                            # 纯工具调用行跳过


def test_layer2_tokens_counts_rendered_not_raw():
    rows = [{"id": 1, "role": "user", "content": "短问"},
            {"id": 2, "role": "assistant", "content": "很长的答复" * 100}]
    raw = sum(estimate_tokens(r["content"]) for r in rows)
    assert layer2_tokens(rows, head_chars=60) < raw
    assert layer2_tokens(rows, head_chars=60) <= estimate_tokens("短问") + 61


# ---- 级联降级 ----

def _long_history(rounds: int, pad: int = 40):
    # 每轮 ≈ (10+4×pad)+(10+4×pad) token;pad=40 时一轮 340 token
    return _rows(*[(f"用户第{i}轮的问题描述" + "补充细节" * pad,
                    f"客服第{i}轮的答复内容" + "展开说明" * pad) for i in range(rounds)])


def test_cascade_moves_anchor_to_turn_boundary():
    rows = _long_history(10)                              # 共约 3400 token
    new_anchor, t0, t1 = cascade_layer1(rows, None, 600)  # 预算装 1 轮(340),装不下 2 轮
    assert new_anchor is not None and t1 <= 600 < t0
    kept = [r for r in rows if r["id"] > new_anchor]
    assert kept[0]["role"] == "user"                      # 整轮边界切入
    assert int(rows_to_messages(kept)[0].id) == new_anchor + 1


def test_cascade_noop_within_budget():
    rows = _rows(("u", "a"))
    anchor, t0, t1 = cascade_layer1(rows, None, 10000)
    assert anchor is None and t0 == t1


def test_cascade_keeps_last_round_when_even_one_does_not_fit():
    rows = _rows(("超长问题" * 200, "超长答复" * 200))     # 一轮 1600 token,预算 50
    new_anchor, _, t1 = cascade_layer1(rows, None, 50)
    # trim_messages 拒绝返回空:整轮保留(超预算也不清空上下文),锚点退到首轮边界
    assert t1 == 1600 and new_anchor == rows[0]["id"] - 1
    assert len([r for r in rows if r["id"] > new_anchor]) == 2


def test_cascade_redegrades_from_existing_anchor():
    rows = _long_history(12)
    first_anchor, _, _ = cascade_layer1(rows, None, 900)
    second_anchor, _, t1 = cascade_layer1(rows, first_anchor, 400)
    # 降级把更老的内容从层 1 挪进层 2:边界向新方向单调右移
    assert second_anchor is not None and second_anchor > first_anchor and t1 <= 400
    kept = [r for r in rows if r["id"] > second_anchor]
    assert kept[0]["role"] == "user"


# ---- 组装 ----

def test_build_layered_shapes_and_no_drop_within_budget():
    l1 = _rows(("u1", "a1"))
    l2 = _rows(("u2", "a2"))
    layered = build_layered(l1, l2, "梗概", head_chars=60, sliding=10000)
    assert layered["dropped"] == 0
    assert len(layered["layer1_msgs"]) == 2 and len(layered["layer2_msgs"]) == 2
    assert layered["layer1_msgs"][0].id == "100"
    assert layered["summary_text"] == "梗概"
    assert "【早期对话梗概】梗概" in layered["history_text"]
    assert "u1" in layered["history_text"] and "a2" in layered["history_text"]


def test_build_layered_drops_oldest_l2_on_overflow():
    l1 = _rows(("u1", "a1"))
    l2 = _rows(*[(f"问题{i}" * 50, f"答复{i}" * 50) for i in range(8)])
    layered = build_layered(l1, l2, "", head_chars=60, sliding=400)
    assert layered["dropped"] > 0
    total = sum(estimate_tokens(m.content or "")
                for m in layered["layer2_msgs"] + layered["layer1_msgs"])
    assert total <= 400
    assert len(layered["layer2_msgs"]) < 16


def test_render_history_text_empty_everything():
    assert "(无)" in render_history_text(None, [], [], 60)


# ---- 背景块 ----

def test_background_combines_summary_and_evidence():
    bg = build_background("梗概内容",
                          [{"n": 1, "section_path": "p", "question": "q", "answer": "a"}],
                          None, "")
    assert "【历史梗概】梗概内容" in bg
    assert "【参考知识】" in bg and "[1]" in bg
    assert "SystemMessage" not in bg  # 梗概/证据绝不以 system 形态存在(文本块)


def test_background_with_order_and_instruction():
    bg = build_background("", [], {"order_id": "1001", "product": "无线耳机"},
                          "只答这一单能不能退")
    assert "【订单数据】" in bg and "1001" in bg and "【指令】只答这一单能不能退" in bg


def test_background_none_when_all_empty():
    assert build_background("", [], None, "") is None


# ---- 工具结果截断 ----

def test_truncate_tool_result_bounds_tokens():
    big = "数" * 5000
    out = truncate_tool_result(big, 200)
    assert estimate_tokens(out) <= 210
    assert out.endswith(")")


def test_truncate_tool_result_passthrough_small():
    assert truncate_tool_result("短结果", 200) == "短结果"


def test_rows_to_messages_carries_ids():
    msgs = rows_to_messages(_rows(("u", "a")))
    assert msgs[0].id == "100" and isinstance(msgs[0], HumanMessage)
    assert msgs[1].id == "101" and isinstance(msgs[1], AIMessage)
