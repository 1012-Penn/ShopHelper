"""意图宽松解析(裸 JSON 契约)与三件套 prompt 常量。"""
from app.graph.entry import INTENTS, parse_intent
from app.prompts import EXPAND_PROMPT, INTENT_PROMPT, RESOLVE_PROMPT, ExpandedQueries, ResolvedQuestion


def test_intents_eight_with_other():
    assert INTENTS == ("物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊", "其他")


def test_parse_intent_normal():
    intent, conf, bad = parse_intent('{"intent": "退款退货", "confidence": 0.92}')
    assert (intent, conf, bad) == ("退款退货", 0.92, False)


def test_parse_intent_in_code_fence():
    intent, conf, bad = parse_intent('```json\n{"intent": "物流", "confidence": 0.8}\n```')
    assert (intent, conf, bad) == ("物流", 0.8, False)


def test_parse_intent_confidence_missing_or_bad_defaults_zero():
    assert parse_intent('{"intent": "投诉"}') == ("投诉", 0.0, False)
    assert parse_intent('{"intent": "闲聊", "confidence": "很高"}') == ("闲聊", 0.0, False)
    assert parse_intent('{"intent": "闲聊", "confidence": 1.7}')[1] == 1.0  # 截断到 [0,1]


def test_parse_intent_garbage_and_out_of_enum_go_other():
    assert parse_intent("我说不好") == ("其他", 0.0, True)
    assert parse_intent('{"intent": "聊天气", "confidence": 0.5}') == ("其他", 0.0, True)
    assert parse_intent("") == ("其他", 0.0, True)


def test_resolve_prompt_contract():
    # 占位符只有 history/query;渲染后两段输入都在
    assert set(RESOLVE_PROMPT.input_variables) == {"history", "query"}
    rendered = RESOLVE_PROMPT.format(history="用户:耳机能退吗", query="它能退吗")
    assert "耳机能退吗" in rendered and "它能退吗" in rendered


def test_expand_prompt_contract():
    assert set(EXPAND_PROMPT.input_variables) == {"query"}
    assert ExpandedQueries(queries=["a", "b"]).queries == ["a", "b"]
    assert ResolvedQuestion(resolved="x", changed=False).changed is False
    # 四件套实质:枚举八项 + JSON 两字段 + 边界样例 + 其他口径,关键词齐全
    for kw in ("物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊", "其他",
               "confidence", "其他", "样例"):
        assert kw in INTENT_PROMPT
