"""ch09 可观测性：Langfuse 可选接入与本地 token 用量底座。"""
import os
from types import SimpleNamespace

import pytest
from langchain_core.callbacks import BaseCallbackHandler

from app.config import Settings
from app.observability import ObservabilityService, UsageCallback
from app.store import UsageStore


class _Runnable:
    def __init__(self):
        self.configs = []

    def with_config(self, config):
        self.configs.append(config)
        return self


def _settings(**overrides) -> Settings:
    return Settings(openai_api_key="sk-test", _env_file=None, **overrides)


def test_disabled_langfuse_binds_graph_without_changing_runnable():
    """缺 key/开关时若仍向 Graph 注入 callback，会让正常聊天依赖可观测服务。"""
    graph = _Runnable()

    bound = ObservabilityService(_settings(), usage_store=None).bind_graph(graph)

    assert bound is graph
    assert graph.configs == []


def test_enabled_langfuse_constructs_one_callback_handler(monkeypatch):
    """重复绑定时若重新构造 handler，会造成重复 trace/callback 开销。"""
    import app.observability as observability

    created = []

    class FakeHandler:
        def __init__(self, **kwargs):
            created.append(kwargs)

    monkeypatch.setattr(observability, "CallbackHandler", FakeHandler)
    service = ObservabilityService(_settings(
        langfuse_enabled=True,
        langfuse_public_key="public",
        langfuse_secret_key="secret",
    ), usage_store=None)

    service.bind_graph(_Runnable())
    service.bind_graph(_Runnable())

    assert len(created) == 1


def test_bind_graph_configures_same_compiled_graph_only_once(monkeypatch):
    """同一编译图重复绑定会叠加 callback，并生成多个 runnable wrapper。"""
    import app.observability as observability

    class FakeHandler:
        def __init__(self, **_kwargs):
            pass

    class BindingRunnable(_Runnable):
        def with_config(self, config):
            self.configs.append(config)
            return object()

    monkeypatch.setattr(observability, "CallbackHandler", FakeHandler)
    monkeypatch.setattr(observability, "Langfuse", None)
    service = ObservabilityService(_settings(
        langfuse_enabled=True,
        langfuse_public_key="public",
        langfuse_secret_key="secret",
    ), usage_store=None)
    graph = BindingRunnable()

    first = service.bind_graph(graph)
    second = service.bind_graph(graph)

    assert second is first
    assert len(graph.configs) == 1


def test_request_metadata_includes_conversation_and_final_intent():
    """遗漏会话或意图会使 Langfuse 无法按客服场景筛选一次请求。"""
    trace = ObservabilityService(_settings(), usage_store=None).start_request(42, "能退货吗")

    metadata = trace.metadata(intent="退款退货", entry_route="retrieve")

    assert metadata == {
        "session_id": "42",
        "conversation_id": "42",
        "source": "chat",
        "intent": "退款退货",
        "entry_route": "retrieve",
    }


def test_request_metadata_normalizes_every_value_to_short_string():
    """直接透传对象或长文本会违反 Langfuse metadata 的稳定筛选契约。"""
    trace = ObservabilityService(_settings(), usage_store=None).start_request(42, "能退货吗")
    long_intent = "退款" * 100

    metadata = trace.metadata(intent=long_intent, entry_route=7)

    assert metadata["intent"] == long_intent[:128]
    assert metadata["entry_route"] == "7"
    assert all(isinstance(value, str) and len(value) <= 128 for value in metadata.values())


def test_request_trace_suppresses_langfuse_context_exit_failure():
    """Langfuse context 在退出时失败也不能把已完成的聊天改成异常。"""
    trace = ObservabilityService(_settings(), usage_store=None).start_request(42, "能退货吗")

    class ExitFailureContext:
        def __enter__(self):
            return SimpleNamespace()

        def __exit__(self, *_args):
            raise RuntimeError("langfuse exit failed")

    trace._observation_context = ExitFailureContext()

    with trace.activate():
        activated = True

    assert activated is True


def test_usage_callback_accumulates_llm_output_and_message_metadata():
    """只识别一种 LangChain 用量结构会把部分模型调用错误记成零成本。"""
    callback = UsageCallback()
    callback.on_llm_end(SimpleNamespace(llm_output={
        "token_usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}
    }))
    callback.on_llm_end(SimpleNamespace(
        llm_output={},
        generations=[[SimpleNamespace(message=SimpleNamespace(usage_metadata={
            "input_tokens": 7, "output_tokens": 11, "total_tokens": 18,
        }))]],
    ))

    assert callback.snapshot() == {
        "input_tokens": 10,
        "output_tokens": 16,
        "total_tokens": 26,
    }


def test_request_usage_callbacks_do_not_share_token_counters():
    """把 callback 放在 service 上会让后一轮请求带上前一轮的 token。"""
    service = ObservabilityService(_settings(), usage_store=None)
    first = service.start_request(1, "第一问")
    second = service.start_request(2, "第二问")

    first.usage_callback.on_llm_end(SimpleNamespace(llm_output={
        "token_usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}
    }))

    assert first.usage_callback.snapshot()["total_tokens"] == 5
    assert second.usage_callback.snapshot()["total_tokens"] == 0


def test_request_usage_callback_implements_langchain_callback_protocol():
    """未继承 LangChain callback 基类会在 Graph invocation 前中断 SSE。"""
    trace = ObservabilityService(_settings(), usage_store=None).start_request(1, "问题")

    assert isinstance(trace.usage_callback, BaseCallbackHandler)


@pytest.mark.asyncio
async def test_usage_store_persists_and_aggregates_intent_tokens(db_session_factory):
    """不按 intent 聚合会令后续管理端无法呈现真实的意图成本。"""
    store = UsageStore(db_session_factory)

    await store.record(conversation_id=9, trace_id="trace-1", intent="退款退货",
                       input_tokens=4, output_tokens=6, total_tokens=10, duration_ms=12)
    await store.record(conversation_id=9, trace_id="trace-2", intent="退款退货",
                       input_tokens=5, output_tokens=10, total_tokens=15, duration_ms=24)

    assert await store.aggregate_by_intent() == [{
        "intent": "退款退货", "request_count": 2, "total_tokens": 25,
        "avg_tokens": 12.5,
    }]


def test_real_sdk_handler_constructs_when_installed(monkeypatch):
    """装了 SDK 时必须能真构造 handler:v3 不收凭据参数,曾被 no-op 分支掩盖成真机必挂。"""
    pytest.importorskip("langfuse")
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)

    settings = _settings(langfuse_enabled=True, langfuse_public_key="pk-lf-test",
                         langfuse_secret_key="sk-lf-test", langfuse_host="http://127.0.0.1:3000")
    service = ObservabilityService(settings, usage_store=None)
    service._ensure_handler()

    assert service.callback_handler is not None
    assert os.environ.get("LANGFUSE_PUBLIC_KEY") == "pk-lf-test"
