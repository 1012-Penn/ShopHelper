"""知识库 MySQL 持久层 + 双写向量化循环。

双写契约(spec §5):先写 knowledge_chunks(pending),向量化成功后回填 vector_id 转 done;
Milvus 主键 = chunk 主键,upsert 覆盖天然幂等,中断重跑只补 pending。
"""
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.chunking import Chunk
from app.models import Conversation, KnowledgeChunk, Message, QaStaging


def vector_text(category: str, questions: str, answer: str) -> str:
    """进向量的三格拼文本(spec §5):category + questions + answer。"""
    return f"{category}\n{questions}\n{answer}"


class KnowledgeBaseStore:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._factory = session_factory

    @property
    def session_factory(self) -> sessionmaker:
        return self._factory

    # ---------- knowledge_chunks ----------

    def doc_chunk_ids(self, doc_name: str) -> list[int]:
        with self._factory() as session:
            rows = session.scalars(
                select(KnowledgeChunk.id).where(KnowledgeChunk.section_path.like(f"{doc_name} >%"))
            ).all()
            return list(rows)

    def doc_chunk_ids_all(self) -> list[int]:
        with self._factory() as session:
            return list(session.scalars(select(KnowledgeChunk.id)).all())

    def replace_doc_chunks(self, doc_name: str, chunks: list[Chunk]) -> tuple[list[int], list[int]]:
        """按文档重建(幂等):删旧块 → 插新块(pending)→ 回填 prev/next。返回 (旧 id, 新 id)。"""
        with self._factory() as session:
            old_ids = list(
                session.scalars(
                    select(KnowledgeChunk.id).where(KnowledgeChunk.section_path.like(f"{doc_name} >%"))
                ).all()
            )
            if old_ids:
                session.query(KnowledgeChunk).where(
                    KnowledgeChunk.id.in_(old_ids)
                ).delete(synchronize_session=False)
            rows = [
                KnowledgeChunk(
                    category=c.category,
                    questions=c.questions,
                    answer=c.answer,
                    section_path=c.section_path,
                    content_type=c.content_type,
                    is_key_clause=1 if c.is_key_clause else 0,
                )
                for c in chunks
            ]
            session.add_all(rows)
            session.flush()  # 拿自增 id
            new_ids = [r.id for r in rows]
            for i, row in enumerate(rows):
                row.prev_chunk_id = new_ids[i - 1] if i > 0 else None
                row.next_chunk_id = new_ids[i + 1] if i + 1 < len(rows) else None
            session.commit()
            return old_ids, new_ids

    def pending_chunks(self) -> list[KnowledgeChunk]:
        with self._factory() as session:
            return session.scalars(
                select(KnowledgeChunk)
                .where(KnowledgeChunk.vectorize_status == "pending")
                .order_by(KnowledgeChunk.id)
            ).all()

    def mark_vectorized(self, chunk_ids: list[int]) -> None:
        if not chunk_ids:
            return
        with self._factory() as session:
            for cid in chunk_ids:
                obj = session.get(KnowledgeChunk, cid)
                if obj is not None:
                    obj.vector_id = str(cid)
                    obj.vectorize_status = "done"
            session.commit()

    def get_chunks(self, ids: list[int]) -> list[KnowledgeChunk]:
        if not ids:
            return []
        with self._factory() as session:
            rows = session.scalars(
                select(KnowledgeChunk).where(KnowledgeChunk.id.in_(ids))
            ).all()
            by_id = {r.id: r for r in rows}
            return [by_id[i] for i in ids if i in by_id]

    def all_dedup_texts(self) -> list[str]:
        with self._factory() as session:
            rows = session.scalars(select(KnowledgeChunk)).all()
            return [vector_text(r.category, r.questions, r.answer) for r in rows]

    # ---------- qa_extraction_staging ----------

    def already_mined_refs(self) -> set[str]:
        with self._factory() as session:
            rows = session.scalars(
                select(QaStaging.source_ref).where(QaStaging.source_ref.is_not(None))
            ).all()
            return set(rows)

    def conversation_ids_with_user_talks(self, exclude_refs: set[str] | None = None) -> list[int]:
        """messages 表里有 user 消息的会话 id;已挖过(source_ref 命中)的排除。"""
        exclude_refs = exclude_refs or set()
        with self._factory() as session:
            conv_ids = session.scalars(
                select(Message.conversation_id).where(Message.role == "user").distinct()
            ).all()
            return [c for c in conv_ids if str(c) not in exclude_refs]

    def insert_staging(self, batch_no: str, source_ref: str, qas: list[tuple[str, str]]) -> None:
        with self._factory() as session:
            for q, a in qas:
                session.add(QaStaging(batch_no=batch_no, source_ref=source_ref, question=q, answer=a))
            session.commit()

    def extracted_items(self) -> list[QaStaging]:
        with self._factory() as session:
            return session.scalars(
                select(QaStaging).where(QaStaging.status == "extracted").order_by(QaStaging.id)
            ).all()

    def set_staging_status(self, ids: list[int], status: str) -> None:
        if not ids:
            return
        with self._factory() as session:
            session.query(QaStaging).where(QaStaging.id.in_(ids)).update(
                {"status": status}, synchronize_session=False
            )
            session.commit()

    def insert_mined_chunks(self, items: list[QaStaging]) -> list[int]:
        """kept 问答对 → knowledge_chunks(pending);questions=问法,category 固定「对话挖掘」。"""
        with self._factory() as session:
            rows = [
                KnowledgeChunk(
                    category="对话挖掘",
                    questions=i.question,
                    answer=i.answer,
                    section_path=f"mined:{i.source_ref}",
                    content_type="qa_mined",
                    is_key_clause=0,
                )
                for i in items
            ]
            session.add_all(rows)
            session.commit()
            return [r.id for r in rows]


def vectorize_pending(kb: KnowledgeBaseStore, vectors, embedder, *, max_chunks: int | None = None, batch_size: int = 16) -> int:
    """扫 pending → embed → Milvus upsert → 回填 done;成功一批记一批,返回处理块数。

    中断安全:upsert 与 mark_vectorized 都幂等,重跑对已 upsert 未 mark 的块只是重复覆盖。
    """
    vectors.ensure_collection()
    done = 0
    while max_chunks is None or done < max_chunks:
        limit = batch_size if max_chunks is None else min(batch_size, max_chunks - done)
        rows = kb.pending_chunks()
        if not rows:
            break
        rows = rows[:limit]
        texts = [vector_text(r.category, r.questions, r.answer) for r in rows]
        vecs = embedder.embed(texts)
        vectors.upsert([
            {"id": r.id, "vector": v, "text": t, "category": r.category}
            for r, v, t in zip(rows, vecs, texts)
        ])
        kb.mark_vectorized([r.id for r in rows])
        done += len(rows)
    return done
