"""订单目录:固定三单、未命中话术、单号抽取正则;query_order 稳定化契约。"""
import json

from app.orders import ORDER_CATALOG, extract_order_id, get_order, list_orders


def test_catalog_three_fixed_orders():
    assert len(ORDER_CATALOG) == 3
    assert list_orders()[0]["order_id"] == "1001"
    assert list_orders()[0]["product"] == "无线耳机"
    assert list_orders()[1]["status"] == "已发货"


def test_get_order_hit_returns_copy_without_mutation():
    o = get_order("1002")
    o["product"] = "改坏"
    assert get_order("1002")["product"] == "机械键盘"


def test_get_order_miss_returns_error_dict():
    o = get_order("9999")
    assert o["error"] == "未找到该订单"
    assert o["order_id"] == "9999"


def test_extract_contextual_first():
    assert extract_order_id("订单1001的物流到哪了") == "1001"
    assert extract_order_id("order 1002 什么状态") == "1002"


def test_extract_bare_fallback_4_to_6_digits():
    assert extract_order_id("1001能退吗") == "1001"
    assert extract_order_id("SH-E300 多少钱") is None      # 3 位不误伤
    assert extract_order_id("普通一句话没有数字") is None
    assert extract_order_id("123456789太长不抓") is None    # 9 连位不带边界


async def test_query_order_tool_stable():
    from tests.conftest import make_session_factory
    from app.tools.definitions import build_tools

    tools = {t.name: t for t in build_tools(make_session_factory(),
                                            embedder=object(), vectors=object())}
    data = json.loads(tools["query_order"].invoke({"order_id": "1001"}))
    assert data["product"] == "无线耳机" and data["amount"] == 299.0
    assert "未找到" in json.loads(tools["query_order"].invoke({"order_id": "424242"}))["error"]
