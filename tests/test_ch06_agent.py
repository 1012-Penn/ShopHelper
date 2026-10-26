"""agent 子流程窄化:订单数据+政策证据注入、refund_form action、n_offset 续排。"""
import json

from app.graph.agent import make_agent_node
from app.answer_verifier import VerificationResult
from tests.helpers import ChunkStub, GraphChatModel, graph_state


def _state(**kw):
    """子流程轮默认底座:refund_flow=True + 订单1001;显式传参可覆盖。"""
    kw.setdefault("intent", "退款退货")
    kw.setdefault("resolved_message", "订单1001的无线耳机能退吗")
    kw.setdefault("refund_flow", True)
    kw.setdefault("order", {"order_id": "1001", "product": "无线耳机", "amount": 299.0,
                            "status": "已签收", "created_at": "2026-08-30"})
    return graph_state(**kw)


def _settings(**overrides):
    from app.config import Settings

    return Settings(openai_api_key="sk-test", _env_file=None,
                    max_agent_steps=4, agent_token_budget=2000, **overrides)


class _Registry:
    tools = []

    @property
    def registry(self):
        return self  # ch08 引擎形态:engine.registry 即注册表

    def bind_tools(self):
        return self.tools

    def labels(self):
        return {}

    async def execute(self, name, args_json, **kwargs):
        return ""


_EVIDENCE2 = [{"n": 1, "chunk_id": 11, "question": "a", "answer": "a",
               "category": "退货政策", "section_path": "a"},
              {"n": 2, "chunk_id": 12, "question": "b", "answer": "b",
               "category": "退货政策", "section_path": "b"}]


async def test_narrowing_injects_order_and_instruction():
    model = GraphChatModel(intent_reply="{}", turns=[("text", "这一单可以退。")])
    node = make_agent_node(model, _Registry(), _settings(), pool=None)
    out = await node(_state(evidence=[{"n": 1, "chunk_id": 11, "question": "退货条件",
                                       "answer": "签收后7天内", "category": "退货政策",
                                       "section_path": "退货政策.md"}]), writer=None)
    msgs = model.prompts[0]
    assert "订单数据" not in msgs[0].content  # ch07:system 纯静态,窄化材料进背景块
    bg = msgs[-1].content
    assert "【订单数据】" in bg and "无线耳机" in bg
    assert "能不能退" in bg and "1001" in bg
    assert "【参考知识】" in bg and "退货条件" in bg  # 证据块进背景块
    # I-1 回归:{order_id} 占位符必须已 format,字面量不得泄漏
    assert "判断这一单(订单1001)能不能退" in bg and "{order_id}" not in bg
    assert out["final_reply"] == "这一单可以退。"


async def test_aftersale_narrowing_wording():
    model = GraphChatModel(intent_reply="{}", turns=[("text", "可以换新。")])
    node = make_agent_node(model, _Registry(), _settings(), pool=None)
    await node(_state(intent="售后"), writer=None)
    assert "修/换/退" in model.prompts[0][-1].content


async def test_non_refund_converge_emits_refund_form_action():
    model = GraphChatModel(intent_reply="{}", turns=[("text", "这一单可以退。")])
    node = make_agent_node(model, _Registry(), _settings(), pool=None)
    out = await node(_state(), writer=None)
    assert out["suggested_actions"] == ["refund_form"]


async def test_refusal_keeps_transfer_actions():
    from app.guard import REFUSAL_MARKER

    model = GraphChatModel(intent_reply="{}",
                           turns=[("text", REFUSAL_MARKER + "证据不足,建议转人工。")])
    node = make_agent_node(model, _Registry(), _settings(), pool=None)
    out = await node(_state(), writer=None)
    assert out["suggested_actions"] == ["transfer_human", "create_ticket"]


async def test_non_refund_flow_turn_gets_no_refund_form():
    model = GraphChatModel(intent_reply="{}", turns=[("text", "无线耳机 299 元。")])
    node = make_agent_node(model, _Registry(), _settings(), pool=None)
    out = await node(_state(refund_flow=False, order={}, intent="商品咨询"), writer=None)
    assert out["suggested_actions"] == []


async def test_query_faq_numbering_continues_after_evidence():
    """子流程证据已有 n=1..2 时,agent 内 query_faq 的引用编号从 3 续排(防撞号)。"""
    tool_chunks = [{"name": "query_faq", "args": '{"keyword": "退货"}', "id": "t1", "index": 0}]
    model = GraphChatModel(intent_reply="{}", turns=[
        ("tools", tool_chunks),
        ("text", "结合政策这一单可以退[3]。"),
    ])
    frames = []

    class Reg(_Registry):
        tools = ["query_faq"]

        def labels(self):
            return {"query_faq": "FAQ 检索"}

        async def execute(self, name, args_json, **kwargs):
            return json.dumps({"items": [{"n": 1, "chunk_id": 99, "question": "运费",
                                          "answer": "8 元", "category": "退货政策",
                                          "section_path": "p"}],
                               "low_confidence": False, "reason": "",
                               "filter_fallback": False, "degraded": False},
                              ensure_ascii=False)

    node = make_agent_node(model, Reg(), _settings(), pool=None)
    await node(_state(evidence=_EVIDENCE2), writer=frames.append)
    cite = [f for f in frames if f["type"] == "citations"][-1]
    assert [it["n"] for it in cite["items"]] == [1, 2, 3]  # 既有证据 + 续排的工具证据


async def test_unknown_order_gets_no_refund_form():
    """M-4 回归:订单未命中(error dict)时不发 refund_form。"""
    model = GraphChatModel(intent_reply="{}", turns=[("text", "这一单可以退。")])
    node = make_agent_node(model, _Registry(), _settings(), pool=None)
    out = await node(_state(order={"order_id": "9999", "error": "未找到该订单"}), writer=None)
    assert out["suggested_actions"] == ["refund_form"]  # agent 侧照常建议,由 log 层裁剪

    from app.graph.logging_node import make_log_node
    from tests.helpers import graph_state
    frames = []

    class _Store:
        async def append(self, sid, msgs):
            pass

    class _Pool:
        def insert(self, *a):
            pass

    st = graph_state(final_reply="这一单可以退。", intent="退款退货", refund_flow=True,
                     suggested_actions=["refund_form"],
                     order={"order_id": "9999", "error": "未找到该订单"},
                     messages=[{"role": "user", "content": "x"}], trace=[])
    await make_log_node(_Store(), _Pool())(st, writer=frames.append)
    assert frames == []  # log 层裁剪:未知订单无 actions 帧


async def test_answer_verification_off_preserves_token_streaming_contract():
    model = GraphChatModel(intent_reply="{}", turns=[("text", "这一|单可以退。")])
    node = make_agent_node(model, _Registry(), _settings(), pool=None)
    frames = []
    out = await node(_state(evidence=_EVIDENCE2, refund_flow=False, intent="商品咨询"),
                     writer=frames.append)
    assert [f["content"] for f in frames if f["type"] == "token"] == ["这一", "单可以退。"]
    assert out["final_reply"] == "这一单可以退。"


async def test_cutoff_does_not_persist_existing_assistant_text_twice():
    class TextAndToolModel(GraphChatModel):
        async def astream(self, messages, config=None, **kwargs):
            self.prompts.append(list(messages))
            yield ChunkStub(
                text="正在查询订单。",
                tool_call_chunks=[{
                    "name": "query_order",
                    "args": '{"order_id": "1001"}',
                    "id": "cutoff-tool",
                    "index": 0,
                }],
            )

    model = TextAndToolModel()
    settings = _settings()
    settings.max_agent_steps = 1
    node = make_agent_node(
        model, _Registry(), settings, pool=None)
    out = await node(_state(refund_flow=False, intent="商品咨询"), writer=None)

    assert out["final_reply"] == "正在查询订单。"
    assert [
        message.content for message in out["messages"]
        if message.__class__.__name__ == "AIMessage" and message.content
    ] == ["正在查询订单。"]


async def test_answer_verification_buffers_bad_candidate_until_checked():
    class PassingVerifier:
        max_revisions = 1

        def verify(self, answer, evidence, *, query=""):
            return VerificationResult("pass", "ok", [], answer)

    model = GraphChatModel(intent_reply="{}", turns=[("text", "错误|候选。")])
    node = make_agent_node(
        model, _Registry(),
        _settings(answer_verification_enabled=True), pool=None,
        answer_verifier=PassingVerifier(),
    )
    frames = []
    out = await node(_state(evidence=_EVIDENCE2, refund_flow=False, intent="商品咨询"),
                     writer=frames.append)
    assert [f["content"] for f in frames if f["type"] == "token"] == ["错误候选。"]
    assert out["final_reply"] == "错误候选。"
    assert any("answer_verification=pass" in item for item in out["trace"])


async def test_answer_verification_revises_once_then_releases_verified_text():
    class RevisingVerifier:
        max_revisions = 1

        def __init__(self):
            self.answers = []

        def verify(self, answer, evidence, *, query=""):
            self.answers.append(answer)
            if answer == "错误候选。":
                return VerificationResult("fail", "实体错误", ["订单日期错误"], answer)
            return VerificationResult("pass", "ok", [], answer)

        def safe_result(self, checked, *, revision_calls):
            return VerificationResult("refuse", checked.reason, checked.issues, "拒绝", revision_calls)

    verifier = RevisingVerifier()
    model = GraphChatModel(intent_reply="修订后的正确答案。",
                           turns=[("text", "错误|候选。")])
    node = make_agent_node(
        model, _Registry(),
        _settings(answer_verification_enabled=True), pool=None,
        answer_verifier=verifier,
    )
    frames = []
    out = await node(_state(evidence=_EVIDENCE2, refund_flow=False, intent="商品咨询"),
                     writer=frames.append)
    assert verifier.answers == ["错误候选。", "修订后的正确答案。"]
    assert [f["content"] for f in frames if f["type"] == "token"] == ["修订后的正确答案。"]
    assert out["final_reply"] == "修订后的正确答案。"
    assert any("revision_calls=1" in item for item in out["trace"])


async def test_answer_verification_failure_emits_only_safe_refusal():
    class FailingVerifier:
        max_revisions = 1

        def verify(self, answer, evidence, *, query=""):
            return VerificationResult("fail", "实体错误", ["订单日期错误"], answer)

        def safe_result(self, checked, *, revision_calls):
            return VerificationResult(
                "refuse", checked.reason, checked.issues,
                "【无法回答】最终答案未通过复核，请转人工。", revision_calls,
            )

    model = GraphChatModel(intent_reply="仍错误。", turns=[("text", "错误|候选。")])
    node = make_agent_node(
        model, _Registry(),
        _settings(answer_verification_enabled=True), pool=None,
        answer_verifier=FailingVerifier(),
    )
    frames = []
    out = await node(_state(evidence=_EVIDENCE2, refund_flow=False, intent="商品咨询"),
                     writer=frames.append)
    tokens = [f["content"] for f in frames if f["type"] == "token"]
    assert tokens == ["【无法回答】最终答案未通过复核，请转人工。"]
    assert "错误候选" not in "".join(tokens)
    assert out["suggested_actions"] == ["transfer_human", "create_ticket"]
