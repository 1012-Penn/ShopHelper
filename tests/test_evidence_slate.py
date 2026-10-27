from app.evidence_slate import EvidenceDocument, select_evidence_slate


def _doc(chunk_id: int, section_path: str, text: str) -> EvidenceDocument:
    return EvidenceDocument(chunk_id=chunk_id, section_path=section_path, text=text)


def test_product_and_shipping_query_reserves_entity_and_policy_evidence():
    documents = {
        1: _doc(1, "商品目录.md > 商品目录 > 核心参数对比", "SH-E300 299 元"),
        2: _doc(2, "商品目录.md > 商品目录 > 商品条目 > SH-E300 耳机",
                "SH-E300 头戴式耳机,售价 299 元。"),
        3: _doc(3, "商品FAQ.md > 商品FAQ > 交易 > 可以合并订单或拆单吗?",
                "同地址未发货订单可尝试合并,按总额计算包邮门槛。"),
        4: _doc(4, "会员与优惠.md > 会员与优惠 > 包邮门槛与合并订单",
                "单笔订单商品金额满 99 元包邮。"),
    }

    slate = select_evidence_slate(
        "SH-E300 售价 299 元，这一单包邮吗？",
        ranked_ids=[1, 3, 2, 4],
        documents=documents,
        top_k=2,
    )

    assert set(slate.ordered_ids) == {2, 4}
    assert slate.facets == ["product_entry", "free_shipping"]


def test_product_price_plus_shipping_question_uses_threshold_rule_not_generic_fee():
    documents = {
        1: _doc(1, "商品目录.md > 商品目录 > 商品条目 > SH-E200 耳机",
                "SH-E200 售价 159 元。"),
        2: _doc(2, "退货政策.md > 退货政策 > 邮费与运费",
                "普通订单邮费 8 元,单笔满 99 元包邮。"),
        3: _doc(3, "会员与优惠.md > 会员与优惠 > 包邮门槛与合并订单",
                "单笔订单商品金额满 99 元包邮。"),
    }

    slate = select_evidence_slate(
        "SH-E200 售价 159 元，这一单需要付运费吗？",
        ranked_ids=[2, 1, 3],
        documents=documents,
        top_k=2,
    )

    assert set(slate.ordered_ids) == {1, 3}
    assert slate.facets == ["product_entry", "free_shipping"]


def test_remote_shipping_query_keeps_generic_threshold_and_regional_supplement():
    documents = {
        1: _doc(
            1,
            "会员与优惠.md > 会员与优惠 > 包邮门槛与合并订单",
            "单笔订单商品金额满 99 元包邮。",
        ),
        2: _doc(
            2,
            "会员与优惠.md > 会员与优惠 > 偏远地区补运",
            "西藏等偏远地区即使满足包邮门槛，仍需补 5 元运费。",
        ),
        3: _doc(3, "商品FAQ.md > 商品FAQ > 物流时效", "现货 48 小时内发出。"),
        4: _doc(4, "退货政策.md > 退货政策 > 邮费与运费", "普通退货运费由买家承担。"),
    }

    slate = select_evidence_slate(
        "西藏订单满 99 元包邮后，还要补运费吗？",
        ranked_ids=[3, 1, 2, 4],
        documents=documents,
        top_k=3,
    )

    assert slate.ordered_ids == [1, 2, 3]
    assert slate.facets == ["free_shipping", "remote_shipping"]


def test_region_rate_query_prefers_region_table_over_generic_shipping_rule():
    documents = {
        1: _doc(
            1,
            "订单与支付.md > 订单与支付 > 订单状态总览",
            "| ORD-1003 | 149 元 | 普通会员 | 华南-广州 | 已发货 |",
        ),
        2: _doc(
            2,
            "退货政策.md > 退货政策 > 邮费与运费",
            "普通订单邮费 8 元，单笔满 99 元包邮。",
        ),
        3: _doc(
            3,
            "售后手册.md > 售后手册 > 运费标准",
            "以下为各区域运费标准明细，首重按 1kg 计。",
        ),
        4: _doc(
            4,
            "售后手册.md > 售后手册 > 运费标准",
            "| 区域 | 首重运费 | 续重运费 |\n| 华南 | 8 元 | 2 元/kg |",
        ),
    }

    slate = select_evidence_slate(
        "ORD-1003 在广东，普通快递首重运费多少？",
        ranked_ids=[1, 2, 3, 4],
        documents=documents,
        top_k=2,
    )

    assert slate.ordered_ids == [1, 4]
    assert slate.facets == ["order_summary", "regional_shipping"]


def test_refund_query_reserves_order_summary_payment_and_refund_rule():
    documents = {
        10: _doc(10, "订单与支付.md > 订单与支付 > 订单状态总览",
                 "| ORD-1011 | 79 元 | 普通会员 | 云南-昆明 | 已退款 |"),
        11: _doc(11, "订单与支付.md > 订单与支付 > 订单明细 > ORD-1011 已退款订单",
                 "ORD-1011 已退款。支付渠道为微信支付。"),
        12: _doc(12, "退货政策.md > 退货政策 > 退款时效",
                 "退款原路返回,微信/支付宝一般 1-3 个工作日到账。"),
        13: _doc(13, "订单与支付.md > 订单与支付 > 订单明细 > ORD-1007 已退款订单",
                 "ORD-1007 已退款。支付渠道为支付宝。"),
    }

    slate = select_evidence_slate(
        "ORD-1011 已退款，钱退到哪个渠道？",
        ranked_ids=[13, 12, 11, 10],
        documents=documents,
        top_k=3,
    )

    assert set(slate.ordered_ids) == {10, 11, 12}
    assert slate.facets == ["order_summary", "order_payment", "refund_rule"]


def test_date_policy_question_reserves_both_version_branches():
    documents = {
        20: _doc(20, "政策版本说明.md > 政策版本说明 > 版本时间线",
                 "v2 自 2026-06-01 起生效。"),
        21: _doc(21, "政策版本说明.md > 政策版本说明 > v2 版十五天无理由",
                 "2026-06-01 或之后签收适用 v2,支持 15 天无理由退货。"),
        22: _doc(22, "政策版本说明.md > 政策版本说明 > v1 版七天无理由",
                 "早于 2026-06-01 签收适用 v1,支持 7 天无理由退货。"),
    }

    slate = select_evidence_slate(
        "2026-06-05 签收的订单，无理由退货窗口是几天？",
        ranked_ids=[20, 21, 22],
        documents=documents,
        top_k=2,
    )

    assert set(slate.ordered_ids) == {21, 22}
    assert slate.facets == ["v2_window", "v1_window"]


def test_return_condition_and_shipping_are_selected_from_different_sections():
    documents = {
        30: _doc(30, "退货政策.md > 退货政策 > 邮费与运费",
                 "非质量问题退货,运费由买家承担。"),
        31: _doc(31, "商品FAQ.md > 商品FAQ > 售后 > 商品有质量问题怎么办?",
                 "质量问题可凭照片凭证免费退货。"),
        32: _doc(32, "退货政策.md > 退货政策 > 七天无理由退货",
                 "商品必须保持完好,吊牌、包装齐全。"),
    }

    slate = select_evidence_slate(
        "非质量问题退货，商品完好，寄回运费谁出？",
        ranked_ids=[31, 30, 32],
        documents=documents,
        top_k=2,
    )

    assert set(slate.ordered_ids) == {30, 32}
    assert slate.facets == ["shipping", "return_condition"]


def test_quality_return_after_deadline_adds_policy_and_detection_rule():
    documents = {
        1: _doc(1, "商品目录.md > 商品目录 > 商品条目 > SH-K104 键盘",
                "SH-K104 机械键盘。"),
        2: _doc(2, "政策版本说明.md > 政策版本说明 > v2 版十五天无理由",
                "签收 15 天内支持无理由退货。"),
        3: _doc(3, "政策版本说明.md > 政策版本说明 > v1 版七天无理由",
                "签收 7 天内支持无理由退货。"),
        4: _doc(4, "售后核验指引.md > 售后核验指引 > 质量问题认定原则",
                "是否质量问题以检测结果为准。"),
    }

    slate = select_evidence_slate(
        "SH-K104 签收 20 天后发现质量问题，还能退吗？",
        ranked_ids=[1, 2, 3, 4],
        documents=documents,
        top_k=4,
    )

    assert set(slate.ordered_ids) == {1, 2, 3, 4}
    assert "quality_rule" in slate.facets


def test_version_choice_question_includes_both_concrete_windows():
    documents = {
        1: _doc(1, "政策版本说明.md > 政策版本说明 > 新旧版本适用判定",
                "按签收日期选择 v1 或 v2。"),
        2: _doc(2, "政策版本说明.md > 政策版本说明 > 冲突处理",
                "同一订单只适用一个版本。"),
        3: _doc(3, "政策版本说明.md > 政策版本说明 > v2 版十五天无理由",
                "v2 支持 15 天无理由退货。"),
        4: _doc(4, "政策版本说明.md > 政策版本说明 > v1 版七天无理由",
                "v1 支持 7 天无理由退货。"),
    }

    slate = select_evidence_slate(
        "ORD-1008 签收后适用哪版无理由退货窗口？",
        ranked_ids=[3, 4, 1, 2],
        documents=documents,
        top_k=4,
    )

    assert set(slate.ordered_ids) == {1, 2, 3, 4}


def test_unstructured_query_keeps_rerank_order():
    documents = {
        40: _doc(40, "商品目录.md > 商品目录 > 商品条目 > SH-B7 手环",
                 "SH-B7 支持血氧监测。"),
        41: _doc(41, "商品目录.md > 商品目录 > 核心参数对比",
                 "SH-B7 采用 AMOLED 屏。"),
    }

    slate = select_evidence_slate(
        "这款设备支持血氧监测吗？",
        ranked_ids=[41, 40],
        documents=documents,
        top_k=2,
    )

    assert slate.ordered_ids == [41, 40]
    assert slate.facets == []


def test_unmatched_required_facet_falls_back_without_dropping_candidates():
    documents = {
        50: _doc(50, "会员与优惠.md > 会员与优惠 > 优惠叠加规则",
                 "会员 95 折可与满减券叠加。"),
        51: _doc(51, "会员与优惠.md > 会员与优惠 > 95 折适用范围",
                 "95 折适用于在售正价商品。"),
    }

    slate = select_evidence_slate(
        "会员 95 折能和满减券一起用吗？",
        ranked_ids=[51, 50],
        documents=documents,
        top_k=2,
    )

    assert slate.ordered_ids == [50, 51]
    assert "coupon_limit" not in slate.facets
