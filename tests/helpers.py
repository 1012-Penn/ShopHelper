"""测试替身与 SSE 解析助手——与实现代码无关,只面向 HTTP 契约。"""
import json

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage


def fake_chat(*reply_texts: str) -> GenericFakeChatModel:
    """每个回复一条 AIMessage;调用顺序消费。**每个测试新建**,迭代器一次性。"""
    return GenericFakeChatModel(messages=iter(AIMessage(content=t) for t in reply_texts))


class StubExtractModel:
    """模拟 with_structured_output 产物:result 与 error 二选一。"""

    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error

    async def ainvoke(self, text):
        if self.error is not None:
            raise self.error
        return self.result


def parse_sse(raw: str) -> list[dict]:
    return [
        json.loads(line[len("data: "):])
        for line in raw.split("\n\n")
        if line.startswith("data: ")
    ]


async def post_chat_sse(client, payload: dict) -> list[dict]:
    async with client.stream("POST", "/api/chat", json=payload) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        raw = ""
        async for chunk in resp.aiter_text():
            raw += chunk
    return parse_sse(raw)
