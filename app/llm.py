"""Chat model factories and structured-output compatibility.

DeepSeek thinking models reject ``tool_choice``.  The JSON extraction path keeps
the same Pydantic validation contract without changing ordinary chat streaming.
"""
from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.config import Settings


def make_chat_model(settings: Settings) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        temperature=settings.chat_temperature,
        max_tokens=settings.max_output_tokens,
        max_retries=settings.llm_max_retries,  # 上游偶发 429/1305 全局限流,亚秒退避扛不住
    )


def make_extract_model(settings: Settings, model: str | None = None) -> ChatOpenAI:
    """model 覆盖参数(ch06):意图降级路给小模型传 intent_small_model 时用。"""
    return ChatOpenAI(
        model=model or settings.openai_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        temperature=settings.extract_temperature,
        max_tokens=settings.max_output_tokens,
        max_retries=settings.llm_max_retries,
    )


class JsonStructuredOutput:
    """Prompt for JSON, parse it, then enforce the supplied Pydantic schema."""

    def __init__(self, model, schema: type[BaseModel]) -> None:
        self.model = model
        self.schema = schema
        schema_json = json.dumps(
            schema.model_json_schema(), ensure_ascii=False, sort_keys=True
        )
        self.instruction = (
            "只输出一个 JSON 对象，不要代码围栏或额外说明。"
            f"必须满足以下 JSON Schema：\n{schema_json}"
        )

    def _request(self, request):
        if isinstance(request, str):
            return f"{request}\n\n{self.instruction}"
        if isinstance(request, BaseMessage):
            return [request, SystemMessage(content=self.instruction)]
        if isinstance(request, (list, tuple)):
            return [*request, SystemMessage(content=self.instruction)]
        converter = getattr(request, "to_messages", None)
        if callable(converter):
            return [*converter(), SystemMessage(content=self.instruction)]
        return [HumanMessage(content=f"{request}\n\n{self.instruction}")]

    @staticmethod
    def _parse_json(raw: Any) -> dict:
        text = str(raw).strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                raise ValueError("structured output did not contain a JSON object")
            payload = json.loads(match.group(0))
        if not isinstance(payload, dict):
            raise TypeError("structured output JSON must be an object")
        return payload

    def _validate(self, response):
        return self.schema.model_validate(
            self._parse_json(getattr(response, "content", response))
        )

    def invoke(self, request, config=None, **kwargs):
        response = self.model.invoke(self._request(request), config=config, **kwargs)
        return self._validate(response)

    async def ainvoke(self, request, config=None, **kwargs):
        response = await self.model.ainvoke(self._request(request), config=config, **kwargs)
        return self._validate(response)


def make_json_structured_output(model, schema: type[BaseModel]) -> JsonStructuredOutput:
    return JsonStructuredOutput(model, schema)


def make_structured_extractor(
    settings: Settings,
    schema: type[BaseModel],
    *,
    model: str | None = None,
    base_model=None,
):
    """Build a structure extractor while preserving ordinary model streaming."""

    raw = base_model or make_extract_model(settings, model=model)
    selected_model = model or settings.openai_model
    mode = settings.structured_output_mode
    use_json = mode == "json" or (
        mode == "auto" and "deepseek" in selected_model.casefold()
    )
    if use_json:
        return make_json_structured_output(raw, schema)
    return raw.with_structured_output(schema, method="function_calling")
