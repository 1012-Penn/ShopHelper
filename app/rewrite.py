"""Query 理解(ch04):口语模糊问法 LLM 改写归一 + 检索侧同义词扩展。只在检索侧,不动入库侧。"""
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel

from app.config import Settings
from app.llm import make_extract_model

# 模板无花括号文案,只有 {query} 占位
REWRITE_PROMPT = PromptTemplate.from_template(
    "把用户的客服问法改写成知识库里的标准问法,不改意思,不新增假设;再给最多 3 个同义问法。\n"
    "用户问法:{query}"
)


class QueryRewrite(BaseModel):
    normalized: str
    synonyms: list[str] = []


class QueryRewriter:
    def __init__(self, llm) -> None:
        self._llm = llm  # with_structured_output(QueryRewrite) 的产物,测试可注替身

    def rewrite(self, query: str) -> QueryRewrite:
        """失败兜底原句:改写是增强,不是依赖——上游挂了检索照常走。"""
        try:
            result = self._llm.invoke(REWRITE_PROMPT.format(query=query))
            if isinstance(result, dict):
                result = QueryRewrite(**result)
            normalized = (result.normalized or "").strip()
            if not normalized:
                return QueryRewrite(normalized=query, synonyms=[])
            synonyms = [s.strip() for s in result.synonyms if s and s.strip()][:3]
            return QueryRewrite(normalized=normalized, synonyms=synonyms)
        except Exception:
            return QueryRewrite(normalized=query, synonyms=[])


def bm25_query_text(qr: QueryRewrite) -> str:
    """BM25 词面扩展:归一问法 + 同义词拼接(dense 只吃 normalized)。"""
    return " ".join([qr.normalized, *qr.synonyms])


def make_rewriter(settings: Settings) -> QueryRewriter:
    llm = make_extract_model(settings).with_structured_output(QueryRewrite, method="function_calling")
    return QueryRewriter(llm)
