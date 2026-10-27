from app.rewrite import QueryRewriter


class _EntitySwapLLM:
    def invoke(self, prompt):
        return {
            "normalized": "订单未发货时能否直接取消",
            "synonyms": ["ORD-1002 未发货怎么取消", "未发货订单取消方法"],
        }


class _SafeSynonymLLM:
    def invoke(self, prompt):
        return {
            "normalized": "ORD-1002 未发货时怎么取消",
            "synonyms": ["ORD-1002 未发货订单取消流程", "订单取消流程"],
        }


def test_query_rewriter_falls_back_when_normalization_drops_order_id():
    decision = QueryRewriter(_EntitySwapLLM()).rewrite_guarded(
        "ORD-1002 还没发货，能直接取消吗？"
    )

    assert decision.rewrite.normalized == "ORD-1002 还没发货，能直接取消吗？"
    assert decision.rewrite.synonyms == []
    assert decision.dropped_synonyms == (
        "未发货订单取消方法",
    )


def test_query_rewriter_keeps_only_anchor_preserving_synonyms():
    decision = QueryRewriter(_SafeSynonymLLM()).rewrite_guarded(
        "ORD-1002 还没发货，能直接取消吗？"
    )

    assert decision.rewrite.normalized == "ORD-1002 未发货时怎么取消"
    assert decision.rewrite.synonyms == ["ORD-1002 未发货订单取消流程"]
    assert decision.dropped_synonyms == ("订单取消流程",)
