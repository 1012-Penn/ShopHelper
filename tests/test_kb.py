from app.chunking import Chunk
from app.kb import KnowledgeBaseStore, vector_text, vectorize_pending
from tests.helpers import FakeEmbedding, FakeVectorStore


def _chunk(q, a, section="退货政策.md > 退货政策 > 节"):
    return Chunk(
        category="退货政策", questions=q, answer=a, section_path=section,
        content_type="policy", is_key_clause=False,
    )


def test_vector_text_three_fields():
    assert vector_text("售后", "怎么退货\n咋退", "联系客服") == "售后\n怎么退货\n咋退\n联系客服"


def test_replace_doc_chunks_is_idempotent_and_scoped(db_session_factory):
    kb = KnowledgeBaseStore(db_session_factory)
    old_ids, new_ids = kb.replace_doc_chunks("退货政策.md", [_chunk("q1", "a1"), _chunk("q2", "a2")])
    assert old_ids == [] and len(new_ids) == 2
    # 重跑同文档:旧块删除、新块重插,数量一致
    # (SQLite 的 rowid 复用会让新旧 id 相交,MySQL InnoDB 不会;对双写正确性无影响——upsert 同 id 覆盖)
    old_ids2, new_ids2 = kb.replace_doc_chunks("退货政策.md", [_chunk("q1", "a1"), _chunk("q2", "a2")])
    assert sorted(old_ids2) == sorted(new_ids)
    # 其他文档不受影响
    kb.replace_doc_chunks("手册.md", [_chunk("m1", "a", section="手册.md > 手册 > 节")])
    assert len(kb.doc_chunk_ids("手册.md")) == 1
    assert len(kb.doc_chunk_ids("退货政策.md")) == 2


def test_replace_doc_chunks_sets_prev_next(db_session_factory):
    kb = KnowledgeBaseStore(db_session_factory)
    _, ids = kb.replace_doc_chunks("退货政策.md", [_chunk("q1", "a1"), _chunk("q2", "a2"), _chunk("q3", "a3")])
    chunks = kb.get_chunks(ids)
    assert chunks[0].prev_chunk_id is None and chunks[0].next_chunk_id == ids[1]
    assert chunks[1].prev_chunk_id == ids[0] and chunks[1].next_chunk_id == ids[2]
    assert chunks[2].prev_chunk_id == ids[1] and chunks[2].next_chunk_id is None


def test_vectorize_pending_marks_done_and_resumes(db_session_factory):
    kb = KnowledgeBaseStore(db_session_factory)
    vectors = FakeVectorStore()
    kb.replace_doc_chunks("退货政策.md", [_chunk(f"q{i}", f"a{i}") for i in range(4)])
    done = vectorize_pending(kb, vectors, FakeEmbedding(), max_chunks=2)  # 模拟中断
    assert done == 2 and vectors.count() == 2
    assert len(kb.pending_chunks()) == 2
    done2 = vectorize_pending(kb, vectors, FakeEmbedding())  # 重跑补齐
    assert done2 == 2 and vectors.count() == 4
    assert kb.pending_chunks() == []
    rows = kb.get_chunks(kb.doc_chunk_ids("退货政策.md"))
    assert all(r.vectorize_status == "done" and r.vector_id == str(r.id) for r in rows)


def test_vectorize_upsert_overwrite_no_duplicate(db_session_factory):
    kb = KnowledgeBaseStore(db_session_factory)
    vectors = FakeVectorStore()
    kb.replace_doc_chunks("退货政策.md", [_chunk("q1", "a1")])
    kb.replace_doc_chunks("退货政策.md", [_chunk("q1", "a1")])  # 重建:旧向量遗留 Milvus
    vectors.upsert([999], [[0.1] * 64])  # 干扰项
    vectorize_pending(kb, vectors, FakeEmbedding())
    assert vectors.count() == 2  # 1 真块 + 1 干扰项,无重复


def test_staging_flow_and_mined_chunks(db_session_factory):
    kb = KnowledgeBaseStore(db_session_factory)
    assert kb.already_mined_refs() == set()
    kb.insert_staging("b1", "3", [("国外的快递费怎么算", "国际件运费如下…")])
    assert kb.already_mined_refs() == {"3"}
    items = kb.extracted_items()
    assert len(items) == 1 and items[0].status == "extracted"
    ids = kb.insert_mined_chunks(items)
    chunk = kb.get_chunks(ids)[0]
    assert chunk.category == "对话挖掘" and chunk.content_type == "qa_mined"
    assert chunk.section_path == "mined:3" and chunk.vectorize_status == "pending"
    assert chunk.questions == "国外的快递费怎么算"
    kb.set_staging_status([items[0].id], "kept")
    assert kb.extracted_items() == []


def test_conversation_ids_with_user_talks(db_session_factory):
    from app.models import Conversation, Message

    with db_session_factory() as s:
        s.add_all([
            Conversation(id=1, user_id="guest"),
            Conversation(id=2, user_id="guest"),
            Conversation(id=3, user_id="guest"),  # 无 user 消息,不应入选
        ])
        s.add_all([
            Message(conversation_id=1, role="user", content="退货"),
            Message(conversation_id=2, role="assistant", content="你好"),
        ])
        s.commit()
    kb = KnowledgeBaseStore(db_session_factory)
    assert kb.conversation_ids_with_user_talks(exclude_refs={"9"}) == [1]
    assert kb.conversation_ids_with_user_talks(exclude_refs={"1"}) == []
