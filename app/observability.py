"""ch09：Langfuse 的可选接线与本地 token 用量统计。"""
import asyncio
import inspect
import logging
from contextlib import contextmanager

from langchain_core.callbacks import BaseCallbackHandler

try:  # SDK 是可选依赖：部署未安装时客服主流程仍可运行。
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler
except ImportError:  # pragma: no cover - 覆盖由未安装 SDK 的运行环境验证
    Langfuse = None
    CallbackHandler = None

logger = logging.getLogger(__name__)


class UsageCallback(BaseCallbackHandler):
    """兼容 LangChain 两种公开 token metadata 形态的累计 callback。"""

    def __init__(self) -> None:
        self._input_tokens = 0
        self._output_tokens = 0
        self._total_tokens = 0

    def on_llm_end(self, response, **_kwargs) -> None:
        usage = (getattr(response, "llm_output", None) or {}).get("token_usage")
        if usage is None:
            usage = self._message_usage(response)
        if not usage:
            return
        self._input_tokens += int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
        self._output_tokens += int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
        self._total_tokens += int(usage.get("total_tokens", 0) or 0)

    @staticmethod
    def _message_usage(response) -> dict:
        for generation_group in getattr(response, "generations", []) or []:
            for generation in generation_group:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None)
                if usage:
                    return usage
        return {}

    def snapshot(self) -> dict:
        return {
            "input_tokens": self._input_tokens,
            "output_tokens": self._output_tokens,
            "total_tokens": self._total_tokens,
        }


class RequestTrace:
    """一轮聊天的根 observation；所有 SDK 故障在这里终止，不向聊天路径传播。"""

    def __init__(self, service, conversation_id: int, question: str, observation=None) -> None:
        self._service = service
        self._conversation_id = conversation_id
        self._question = question
        self._observation_context = observation
        self._observation = None
        self.usage_callback = UsageCallback()

    def metadata(self, intent: str | None = None, entry_route: str | None = None) -> dict:
        metadata = {
            "session_id": str(self._conversation_id),
            "conversation_id": str(self._conversation_id),
            "source": "chat",
        }
        if intent is not None:
            metadata["intent"] = intent
        if entry_route is not None:
            metadata["entry_route"] = entry_route
        return metadata

    @contextmanager
    def activate(self):
        """把根 observation 设为当前上下文，让 Graph callback 自动成为它的子节点。"""
        if self._observation_context is None:
            yield self
            return
        try:
            with self._observation_context as observation:
                self._observation = observation
                yield self
        except Exception:
            logger.warning("[observability] Langfuse 根 observation 不可用，继续聊天", exc_info=True)
            yield self

    def finish(self, *, intent: str, answer: str, duration_ms: int) -> None:
        metadata = self.metadata(intent=intent)
        if self._observation is not None:
            try:
                self._observation.update(metadata=metadata, output={"answer": answer})
            except Exception:
                logger.warning("[observability] 更新 Langfuse observation 失败", exc_info=True)
        self._record_usage(intent=intent, duration_ms=duration_ms)
        self._service.flush()

    def observe_retrieval(self, query: str, snapshot: list[dict]):
        """供检索节点调用；未启用时是 no-op。"""
        if self._service.client is None:
            return None
        try:
            return self._service.client.start_as_current_observation(
                name="retrieval", as_type="retriever", input={"query": query}, output={"snapshot": snapshot}
            )
        except Exception:
            logger.warning("[observability] 写入检索 observation 失败", exc_info=True)
            return None

    def _record_usage(self, *, intent: str, duration_ms: int) -> None:
        if self._service.usage_store is None:
            return
        usage = self.usage_callback.snapshot()
        try:
            recorded = self._service.usage_store.record(
                conversation_id=self._conversation_id,
                trace_id=getattr(self._observation, "trace_id", None),
                intent=intent,
                duration_ms=duration_ms,
                **usage,
            )
            if inspect.isawaitable(recorded):
                asyncio.get_running_loop().create_task(recorded)
        except Exception:
            logger.warning("[observability] 写入本地用量失败", exc_info=True)


class ObservabilityService:
    """创建一次 callback，按请求创建轻量根 trace；所有失败均降级为 no-op。"""

    def __init__(self, settings, usage_store) -> None:
        self.settings = settings
        self.usage_store = usage_store
        self.client = None
        self.callback_handler = None
        self._enabled = bool(settings.langfuse_enabled and settings.langfuse_public_key
                             and settings.langfuse_secret_key)

    def _ensure_handler(self) -> None:
        if not self._enabled or self.callback_handler is not None:
            return
        if CallbackHandler is None:
            logger.warning("[observability] Langfuse SDK 未安装，禁用 tracing")
            return
        try:
            if Langfuse is not None and self.client is None:
                self.client = Langfuse(
                    public_key=self.settings.langfuse_public_key,
                    secret_key=self.settings.langfuse_secret_key,
                    host=self.settings.langfuse_host,
                    release=self.settings.langfuse_release,
                    environment=self.settings.langfuse_tracing_environment,
                )
            self.callback_handler = CallbackHandler(
                public_key=self.settings.langfuse_public_key,
                secret_key=self.settings.langfuse_secret_key,
                host=self.settings.langfuse_host,
            )
        except Exception:
            logger.warning("[observability] Langfuse 初始化失败，禁用 tracing", exc_info=True)
            self.client = None
            self.callback_handler = None

    def bind_graph(self, compiled_graph):
        """仅启用时向已编译 Graph 注入一次默认 callback。"""
        self._ensure_handler()
        if self.callback_handler is None:
            return compiled_graph
        try:
            return compiled_graph.with_config({"callbacks": [self.callback_handler]})
        except Exception:
            logger.warning("[observability] 绑定 Graph callback 失败，继续使用原图", exc_info=True)
            return compiled_graph

    def start_request(self, conversation_id: int, question: str) -> RequestTrace:
        self._ensure_handler()
        trace = RequestTrace(self, conversation_id, question)
        if self.client is not None:
            try:
                trace._observation_context = self.client.start_as_current_observation(
                    name="chat_request", as_type="chain", input={"question": question},
                    metadata=trace.metadata(),
                )
            except Exception:
                logger.warning("[observability] 创建 Langfuse 根 observation 失败", exc_info=True)
        return trace

    def flush(self) -> None:
        if self.client is None:
            return
        try:
            self.client.flush()
        except Exception:
            logger.warning("[observability] Langfuse flush 失败", exc_info=True)
