from pydantic import BaseModel

from app.config import Settings
from app.llm import (
    JsonStructuredOutput,
    make_json_structured_output,
    make_structured_extractor,
)


class AnswerSchema(BaseModel):
    answer: str
    confidence: float


class FakeStructuredModel:
    def __init__(self) -> None:
        self.calls = []

    def invoke(self, request, config=None, **kwargs):
        self.calls.append((request, config))
        return type("Response", (), {
            "content": '{"answer":"ok","confidence":0.9}',
        })()


def test_json_structured_output_adds_schema_instruction_and_validates_result():
    model = FakeStructuredModel()
    structured = make_json_structured_output(model, AnswerSchema)

    result = structured.invoke("original prompt", config={"run": "test"})

    request, config = model.calls[0]
    assert config == {"run": "test"}
    assert "original prompt" in request
    assert AnswerSchema.model_json_schema()["title"] in request
    assert result == AnswerSchema(answer="ok", confidence=0.9)


def test_auto_mode_uses_json_fallback_for_deepseek_thinking_models():
    class RawModel:
        def with_structured_output(self, *_args, **_kwargs):
            raise AssertionError("tool_choice path must not be selected for DeepSeek")

    settings = Settings(
        openai_api_key="sk-test",
        _env_file=None,
        openai_model="deepseek-flash",
        structured_output_mode="auto",
    )
    extractor = make_structured_extractor(
        settings, AnswerSchema, base_model=RawModel()
    )
    assert isinstance(extractor, JsonStructuredOutput)
