"""证据蕴含校验（ch09 v2）：用 LLM 判断检索证据是否真的回答了用户问题。

分数闸只能看“像不像”，分不清「同域硬负例」和「低分但可答」；这里用一次
结构化判断补上语义层。调用失败时由 RetrievalService 回退到分数闸。
"""
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel

from app.config import Settings
from app.llm import make_structured_extractor

EVIDENCE_JUDGE_PROMPT = PromptTemplate.from_template(
    "你是证据审核员。只根据给定证据判断:证据是否明确回答了用户问题。\n"
    "规则:\n"
    "1. 证据直接写出答案,或可由证据中的事实直接计算/推出,才算 answered=true。\n"
    "2. 问题描述的场景落在证据规则内(例如七天内、商品完好),证据给出了该规则,判 true;\n"
    "   不要因为问题没有逐字重复规则就判 false。\n"
    "3. 多约束问题:每个约束都能在证据里找到依据,判 true;缺少任一约束,判 false。\n"
    "4. 证据只是话题相关、需要外部常识或编造才能回答,判 false;\n"
    "   不要因为问题中的某个词出现在证据里就判 true。\n"
    "5. 问题点名了具体商品/型号/订单时,证据必须明确提到该对象;\n"
    "   只有通用规则、没有该对象信息,判 false。\n"
    "示例:\n"
    "问题:这个能便宜点吗? 证据:会员享 9 折。→ true(会员折扣就是优惠)。\n"
    "问题:会员买 A 商品多少钱? 证据:会员享 9 折;A 售价 100 元。→ true(可计算)。\n"
    "问题:X 支持无线充电吗? 证据:只写了蓝牙和续航,没有充电方式。→ false。\n"
    "问题:A 商品保修多久? 证据:只有通用保修规则,没有提到 A。→ false(缺少该商品信息)。\n"
    "【用户问题】{query}\n【证据】\n{evidence}"
)


class EvidenceVerdict(BaseModel):
    answered: bool
    supported_by: list[int] = []
    reason: str = ""


class EvidenceJudge:
    def __init__(self, llm) -> None:
        self._llm = llm  # with_structured_output(EvidenceVerdict) 的产物,测试可注替身

    def judge(self, query: str, documents: list[str]) -> EvidenceVerdict:
        evidence = "\n".join(f"[{i + 1}] {doc}" for i, doc in enumerate(documents))
        result = self._llm.invoke(EVIDENCE_JUDGE_PROMPT.format(query=query, evidence=evidence))
        if isinstance(result, dict):
            result = EvidenceVerdict(**result)
        return result


def make_evidence_judge(settings: Settings) -> EvidenceJudge:
    llm = make_structured_extractor(settings, EvidenceVerdict)
    return EvidenceJudge(llm)
