"""Query 理解(ch04):口语模糊问法 LLM 改写归一 + 检索侧同义词扩展。只在检索侧,不动入库侧。"""
from dataclasses import dataclass

from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel

from app.config import Settings
from app.llm import make_structured_extractor
from app.query_guard import QueryGuardDecision, audit_query_rewrite

# 模板无花括号文案,只有 {query} 占位
REWRITE_PROMPT = PromptTemplate.from_template(
    "把用户的客服问法改写成知识库里的标准问法,不改意思,不新增假设;再给最多 3 个同义问法。\n"
    "用户问法:{query}"
)
SAFE_REWRITE_PROMPT = PromptTemplate.from_template(
    "把用户的客服问法改写成知识库里的标准问法,不改意思,不新增假设;再给最多 3 个同义问法。\n"
    "必须保留原句中的每个型号、订单号、日期、政策版本、数字与单位、否定、例外和改口约束;"
    "同义问法也必须遵守这些约束。无法安全归一或扩展时,normalized 原样返回且 synonyms 为空。\n"
    "用户问法:{query}"
)


class QueryRewrite(BaseModel):
    normalized: str
    synonyms: list[str] = []


@dataclass(frozen=True)
class RewriteDecision:
    rewrite: QueryRewrite
    audit: QueryGuardDecision
    dropped_synonyms: tuple[str, ...] = ()


class QueryRewriter:
    def __init__(self, llm, *, protect: bool = True) -> None:
        self._llm = llm  # with_structured_output(QueryRewrite) 的产物,测试可注替身
        self._protect = bool(protect)

    def rewrite(self, query: str) -> QueryRewrite:
        """失败兜底原句:改写是增强,不是依赖——上游挂了检索照常走。"""
        return self.rewrite_guarded(query).rewrite

    def rewrite_guarded(self, query: str) -> RewriteDecision:
        """Return the rewrite plus the exact guard decision used for fallback."""
        try:
            prompt = (SAFE_REWRITE_PROMPT if self._protect else REWRITE_PROMPT).format(
                query=query
            )
            result = self._llm.invoke(prompt)
            if isinstance(result, dict):
                result = QueryRewrite(**result)
            normalized = (result.normalized or "").strip()
            if not normalized:
                audit = QueryGuardDecision(
                    False, query, ("empty_normalized",), (), ()
                )
                return RewriteDecision(QueryRewrite(normalized=query, synonyms=[]), audit)
            synonyms = [s.strip() for s in result.synonyms if s and s.strip()][:3]
            if not self._protect:
                audit = audit_query_rewrite(query, normalized)
                return RewriteDecision(
                    QueryRewrite(normalized=normalized, synonyms=synonyms),
                    QueryGuardDecision(True, normalized, (), audit.input_anchors,
                                       audit.output_anchors),
                )
            normalized_audit = audit_query_rewrite(query, normalized)
            accepted_synonyms: list[str] = []
            dropped_synonyms: list[str] = []
            for synonym in synonyms:
                synonym_audit = audit_query_rewrite(query, synonym)
                if synonym_audit.accepted:
                    accepted_synonyms.append(synonym)
                else:
                    dropped_synonyms.append(synonym)
            if not normalized_audit.accepted:
                return RewriteDecision(
                    QueryRewrite(normalized=query, synonyms=[]),
                    normalized_audit,
                    tuple(dropped_synonyms),
                )
            return RewriteDecision(
                QueryRewrite(normalized=normalized, synonyms=accepted_synonyms),
                normalized_audit,
                tuple(dropped_synonyms),
            )
        except Exception:
            audit = QueryGuardDecision(
                False, query, ("upstream_error",), (), ()
            )
            return RewriteDecision(QueryRewrite(normalized=query, synonyms=[]), audit)


def bm25_query_text(qr: QueryRewrite) -> str:
    """BM25 词面扩展:归一问法 + 同义词拼接(dense 只吃 normalized)。"""
    return " ".join([qr.normalized, *qr.synonyms])


def make_rewriter(settings: Settings) -> QueryRewriter:
    llm = make_structured_extractor(settings, QueryRewrite)
    return QueryRewriter(llm)
