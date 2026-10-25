"""证据蕴含判断器：prompt 构造、结构化结果、dict 兼容。"""
from app.evidence import EvidenceJudge, EvidenceVerdict


class StubLLM:
    def __init__(self, result):
        self.result = result
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return self.result


def test_judge_builds_numbered_evidence_and_returns_verdict():
    stub = StubLLM(EvidenceVerdict(answered=True, supported_by=[2], reason="证据写了"))
    verdict = EvidenceJudge(stub).judge("支持无线充电吗", ["只写蓝牙", "写了无线充电"])

    assert verdict.answered is True
    assert verdict.supported_by == [2]
    prompt = stub.prompts[0]
    assert "支持无线充电吗" in prompt
    assert "[1] 只写蓝牙" in prompt
    assert "[2] 写了无线充电" in prompt


def test_judge_accepts_dict_result():
    stub = StubLLM({"answered": False, "supported_by": [], "reason": "证据只写了蓝牙"})
    verdict = EvidenceJudge(stub).judge("支持无线充电吗", ["只写蓝牙"])

    assert verdict.answered is False
    assert verdict.reason == "证据只写了蓝牙"
