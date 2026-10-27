"""Factory-level rollback contract for protected query rewriting."""

from __future__ import annotations

from app.config import Settings
from app.rewrite import QueryRewrite, make_rewriter


class _BadRewrite:
    def invoke(self, _prompt):
        return QueryRewrite(
            normalized="订单未发货能否取消",
            synonyms=["未发货订单取消方法"],
        )


def _settings(*, guard: bool) -> Settings:
    return Settings(
        openai_api_key="sk-test",
        embedding_api_key="sk-test",
        _env_file=None,
        query_guard_enabled=guard,
    )


def test_protected_rewrite_is_explicitly_configurable(monkeypatch):
    import app.rewrite as rewrite_module

    monkeypatch.setattr(
        rewrite_module,
        "make_structured_extractor",
        lambda _settings, _schema: _BadRewrite(),
    )

    protected = make_rewriter(_settings(guard=True)).rewrite(
        "ORD-1002 还没发货，能直接取消吗？"
    )
    legacy = make_rewriter(_settings(guard=False)).rewrite(
        "ORD-1002 还没发货，能直接取消吗？"
    )

    assert protected.normalized == "ORD-1002 还没发货，能直接取消吗？"
    assert protected.synonyms == []
    assert legacy.normalized == "订单未发货能否取消"
