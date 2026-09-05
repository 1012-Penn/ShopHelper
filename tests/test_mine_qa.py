from app.chunking import Chunk
from app.kb import KnowledgeBaseStore
from scripts.mine_qa import _conversation_text, mine
from tests.helpers import FakeEmbedding, StubMineModel


def _seed_conversations(db_session_factory):
    """三通对话:发票(与库内重复)、海外运费(新知)、大件运费(与批内新知同词面)。"""
    from app.models import Conversation, Message

    convs = [
        (1, [("user", "发票抬头写错了能改吗?"), ("assistant", "发货前可在订单页自助修改,已开的电子发票支持红冲重开。")]),
        (2, [("user", "国外的快递费怎么算?"), ("assistant", "海外件暂不支持直邮,港澳台首重 30 元。")]),
        (3, [("user", "大件商品寄过来运费谁出?"), ("assistant", "质量问题退货的运费由商家承担。")]),
    ]
    with db_session_factory() as s:
        for conv_id, pairs in convs:
            s.add(Conversation(id=conv_id, user_id="guest"))
            s.add_all([Message(conversation_id=conv_id, role=r, content=c) for r, c in pairs])
        # 干扰:纯工具轨迹 assistant(content 空)应被跳过
        s.add(Message(conversation_id=2, role="assistant", content=None,
                      tool_calls=[{"name": "query_order", "args": {}, "id": "c1"}]))
        # 干扰:没有 user 消息的会话
        s.add(Conversation(id=9, user_id="guest"))
        s.add(Message(conversation_id=9, role="assistant", content="你好,请问有什么可以帮你?"))
        s.commit()


def _library_with_invoice_chunk(db_session_factory):
    """库内预置一条「发票」知识(与对话 1 挖出的问法同词面 → 应判重复)。"""
    kb = KnowledgeBaseStore(db_session_factory)
    kb.replace_doc_chunks("x.md", [
        Chunk("交易", "发票", "支持电子发票。", "x.md > 交易 > 发票", "policy", False),
    ])
    return kb


MAPPING = {
    "发票": [("发票抬头写错了能改吗", "发货前可自助修改,支持红冲重开")],
    "快递费": [("国外的快递费怎么算", "港澳台首重 30 元,续重 15 元/kg")],
    "运费": [("大件商品寄过来运费谁出", "质量问题退货运费商家承担")],
}


def test_conversation_text_skips_tool_and_empty(db_session_factory):
    _seed_conversations(db_session_factory)
    text = _conversation_text(db_session_factory, 2)
    assert "快递费" in text and "港澳台" in text and "query_order" not in text
    assert _conversation_text(db_session_factory, 9) is None  # 无 user 提问 → 不可挖


def test_mine_extracts_dedups_and_promotes(db_session_factory):
    _seed_conversations(db_session_factory)
    kb = _library_with_invoice_chunk(db_session_factory)
    stats = mine(kb, FakeEmbedding(), StubMineModel(MAPPING), dedup_threshold=0.9)
    assert stats["conversations"] == 3
    assert stats["extracted"] == 3
    # 发票问法与库内同词面 → discarded;海外运费为新知 → kept;
    # 大件运费与批内已 kept 的海外运费同词面 → 批内去重 discarded
    assert stats["kept"] == 1 and stats["discarded"] == 2
    assert kb.extracted_items() == []  # 全部出清(extracted → kept/discarded)
    kept = [c for c in kb.all_dedup_texts() if "港澳台" in c]
    assert len(kept) == 1 and kept[0].startswith("对话挖掘\n")  # 入库且带 category


def test_mine_skips_already_mined_conversations(db_session_factory):
    _seed_conversations(db_session_factory)
    kb = KnowledgeBaseStore(db_session_factory)
    stub = StubMineModel(MAPPING)
    first = mine(kb, FakeEmbedding(), stub)
    assert first["conversations"] == 3
    second = mine(kb, FakeEmbedding(), stub)
    assert second["conversations"] == 0 and second["extracted"] == 0


def test_mine_vectorizes_kept_chunks(db_session_factory):
    _seed_conversations(db_session_factory)
    kb = KnowledgeBaseStore(db_session_factory)
    from tests.helpers import FakeVectorStore

    stats = mine(kb, FakeEmbedding(), StubMineModel(MAPPING), dedup_threshold=0.9,
                 vectors=FakeVectorStore())
    assert stats["vectorized"] == stats["kept"]
    assert kb.pending_chunks() == []
