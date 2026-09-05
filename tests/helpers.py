"""测试替身与 SSE 解析助手——与实现代码无关,只面向 HTTP 契约。"""
import hashlib
import json
import math

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage


class FakeChatWithTools(GenericFakeChatModel):
    """GenericFakeChatModel 不实现 bind_tools(基类 NotImplementedError);替身直接返回自身。"""

    def bind_tools(self, tools, **kwargs):
        return self


def fake_chat(*reply_texts: str) -> GenericFakeChatModel:
    """每个回复一条 AIMessage;调用顺序消费。**每个测试新建**,迭代器一次性。"""
    return FakeChatWithTools(messages=iter(AIMessage(content=t) for t in reply_texts))


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


def cosine(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


class FakeEmbedding:
    """确定性假嵌入:近义词归一 → 只保留领域词(词库锚定,长词优先)→ 每词一个稳定伪随机向量求和归一。
    「邮费/快递费/寄费」都归一到「运费」,换说法 cos=1.0;异话题仅靠随机向量近似正交自然拉开;
    多话题文本按词叠加,得到可控的梯度相似度。无领域词的文本返回零向量(与任何向量 cos=0)。"""

    dim = 64
    SYNONYMS = {"邮费": "运费", "快递费": "运费", "寄费": "运费", "送货": "发货"}
    LEXICON = sorted(
        ["七天无理由", "运费", "退货", "退款", "发货", "物流", "发票", "会员", "付款", "支付", "质量"],
        key=len,
        reverse=True,
    )

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in self._tokens(text):
            for j, v in enumerate(self._token_vec(tok)):
                vec[j] += v
        norm = math.sqrt(sum(a * a for a in vec))
        if norm == 0:
            return vec
        return [a / norm for a in vec]

    def _tokens(self, text: str) -> list[str]:
        canon = text
        for src, dst in self.SYNONYMS.items():
            canon = canon.replace(src, dst)
        toks: list[str] = []
        i = 0
        while i < len(canon):
            for word in self.LEXICON:
                if canon.startswith(word, i):
                    toks.append(word)
                    i += len(word)
                    break
            else:
                i += 1
        return toks

    def _token_vec(self, tok: str) -> list[float]:
        digest = hashlib.sha512(tok.encode()).digest()  # 64 字节 = 64 维,字节独立近似正交
        return [b / 255.0 - 0.5 for b in digest]
