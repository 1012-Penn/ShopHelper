"""低置信度问题的数据飞轮:标准化、语义去重、审核与知识库待向量化写入。"""
import logging
from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from app.models import KnowledgeChunk, LowConfidenceQuestion, ReviewQueue

logger = logging.getLogger(__name__)


class ReviewConflictError(ValueError):
    """已有核准内容冲突,或当前审核状态不允许重试索引。"""


@dataclass(frozen=True)
class NormalizedQuestion:
    normalized_question: str
    ai_suggested_answer: str


class NormalizedQuestionOutput(BaseModel):
    normalized_question: str = Field(description="标准 FAQ 式问题")
    ai_suggested_answer: str = Field(description="供人工审核参考的示例答案")


class DedupDecisionOutput(BaseModel):
    same_meaning: bool = Field(description="是否与候选缺口表达同一个问题")
    matched_review_id: int | None = Field(default=None)


class FlywheelService:
    """业务层保留模型注入口,默认实现保证离线脚本/测试也能跑通闭环。

    生产接线可把 ``normalizer`` 换成结构化模型调用,把 ``deduper`` 换成模型语义判断；
    两者都只返回稳定的小对象,数据库事务和去重落点仍由本服务统一负责。
    """

    def __init__(self, session_factory: sessionmaker, *,
                 normalizer: Callable[[str], NormalizedQuestion] | None = None,
                 deduper: Callable[[str, list[ReviewQueue]], int | None] | None = None,
                 kb=None, vectors=None, embedder=None) -> None:
        self._factory = session_factory
        self._normalizer = normalizer or self._default_normalizer
        self._deduper = deduper or self._default_deduper
        self._kb, self._vectors, self._embedder = kb, vectors, embedder

    def insert(self, source: str, conversation_id: int | None, question: str, reason: str = "",
               retrieved_chunks: list[dict] | None = None, matched_review_id: int | None = None) -> int:
        """兼容现有图节点的池接口;命中 id 参数仅用于迁移期兼容。"""
        del matched_review_id
        return self.capture(source, conversation_id, question, reason, retrieved_chunks)

    @staticmethod
    def _default_normalizer(raw: str) -> NormalizedQuestion:
        question = " ".join((raw or "").split()).strip()[:512]
        return NormalizedQuestion(question, "待人工审核后补充答案。")

    @staticmethod
    def _default_deduper(normalized: str, candidates: list[ReviewQueue]) -> int | None:
        for row in candidates:
            if row.normalized_question == normalized:
                return row.id
        return None

    def capture(self, source: str, conversation_id: int | None, raw_question: str,
                reason: str, retrieved_chunks: list[dict] | None) -> int:
        """三入口统一落池并归并到 review_queue,返回缺口行 id。"""
        try:
            normalized = self._normalizer(raw_question)
        except Exception:
            normalized = self._default_normalizer(raw_question)
        with self._factory() as session:
            candidates = list(session.scalars(
                select(ReviewQueue).where(ReviewQueue.review_status == "待审")
                .order_by(ReviewQueue.id)
            ).all())
            try:
                matched_id = self._deduper(normalized.normalized_question, candidates)
            except Exception:
                matched_id = self._default_deduper(normalized.normalized_question, candidates)
            if matched_id is None:
                review = ReviewQueue(
                    normalized_question=normalized.normalized_question[:512],
                    ai_suggested_answer=normalized.ai_suggested_answer,
                    occurrence_count=1,
                )
                session.add(review)
                session.flush()
                matched_id = review.id
            else:
                review = session.get(ReviewQueue, matched_id)
                review.occurrence_count = (review.occurrence_count or 0) + 1
            session.add(LowConfidenceQuestion(
                conversation_id=conversation_id, raw_question=raw_question,
                source=source, reason=reason, matched_review_id=matched_id,
                retrieved_chunks=retrieved_chunks or None,
            ))
            session.commit()
            return matched_id

    def list_reviews(self, status: str = "待审") -> list[dict]:
        with self._factory() as session:
            rows = session.execute(
                select(ReviewQueue, KnowledgeChunk)
                .outerjoin(KnowledgeChunk, KnowledgeChunk.review_id == ReviewQueue.id)
                .where(ReviewQueue.review_status == status)
                .order_by(ReviewQueue.occurrence_count.desc(), ReviewQueue.updated_at.desc())
            ).all()
            return [self._review_dict(row, chunk) for row, chunk in rows]

    def detail(self, review_id: int) -> dict | None:
        with self._factory() as session:
            row = session.get(ReviewQueue, review_id)
            if row is None:
                return None
            chunk = session.scalar(select(KnowledgeChunk).where(KnowledgeChunk.review_id == review_id))
            result = self._review_dict(row, chunk)
            questions = session.scalars(
                select(LowConfidenceQuestion)
                .where(LowConfidenceQuestion.matched_review_id == review_id)
                .order_by(LowConfidenceQuestion.created_at, LowConfidenceQuestion.id)
            ).all()
            result["original_questions"] = [
                {"id": item.id, "raw_question": item.raw_question, "source": item.source,
                 "reason": item.reason or "", "retrieved_chunks": item.retrieved_chunks or []}
                for item in questions
            ]
            return result

    @staticmethod
    def _review_dict(row: ReviewQueue, chunk: KnowledgeChunk | None = None) -> dict:
        return {
            "id": row.id, "normalized_question": row.normalized_question,
            "ai_suggested_answer": row.ai_suggested_answer or "",
            "occurrence_count": row.occurrence_count, "review_status": row.review_status,
            "approved_answer": row.approved_answer or "",
            "knowledge_chunk_id": chunk.id if chunk is not None else None,
            "index_status": chunk.vectorize_status if chunk is not None else None,
        }

    def approve(self, review_id: int, approved_answer: str) -> int:
        """先原子保存审核与唯一知识,再尽力索引;同答案重放返回原知识 id。"""
        answer = (approved_answer or "").strip()
        if not answer:
            raise ValueError("核准答案不能为空")
        try:
            chunk_id = self._save_approval(review_id, answer)
        except IntegrityError:
            # MySQL 对审核行加锁;唯一约束同时兜住无行锁后端及并发创建竞争。
            with self._factory() as session:
                chunk = session.scalar(select(KnowledgeChunk)
                                       .where(KnowledgeChunk.review_id == review_id))
                if chunk is None:
                    raise  # 不吞掉其他数据库约束错误
                self._check_answer(chunk, answer)
                chunk_id = chunk.id
        self._try_index(chunk_id)
        return chunk_id

    @staticmethod
    def _check_answer(chunk: KnowledgeChunk, answer: str) -> None:
        if chunk.answer != answer:
            raise ReviewConflictError("该问题已有已核准答案,重复提交不得修改")

    def _save_approval(self, review_id: int, answer: str) -> int:
        with self._factory() as session:
            review = session.scalar(select(ReviewQueue).where(ReviewQueue.id == review_id)
                                    .with_for_update())
            if review is None:
                raise KeyError(f"review_queue 不存在:{review_id}")
            chunk = session.scalar(select(KnowledgeChunk).where(KnowledgeChunk.review_id == review_id))
            if chunk is not None:
                self._check_answer(chunk, answer)
            else:
                chunk = KnowledgeChunk(
                    review_id=review_id, category="飞轮补全",
                    questions=review.normalized_question, answer=answer,
                    section_path=f"review_queue:{review_id}", content_type="faq", is_key_clause=0,
                )
                session.add(chunk)
            review.review_status = "通过"
            review.approved_answer = answer
            session.flush()
            chunk_id = chunk.id
            session.commit()
            return chunk_id

    def _try_index(self, chunk_id: int) -> None:
        if self._kb is None or self._vectors is None or self._embedder is None:
            return
        if self.index_status(chunk_id) == "done":
            return
        from app.kb import vectorize_pending

        try:
            vectorize_pending(self._kb, self._vectors, self._embedder, chunk_ids=[chunk_id])
        except Exception:
            logger.warning("审核已保存,索引待重试 chunk_id=%s", chunk_id, exc_info=True)

    def index_status(self, chunk_id: int) -> str:
        with self._factory() as session:
            chunk = session.get(KnowledgeChunk, chunk_id)
            if chunk is None:
                raise KeyError(f"knowledge_chunk 不存在:{chunk_id}")
            return chunk.vectorize_status

    def retry_index(self, review_id: int) -> int:
        with self._factory() as session:
            review = session.get(ReviewQueue, review_id)
            if review is None:
                raise KeyError(f"review_queue 不存在:{review_id}")
            if review.review_status != "通过":
                raise ReviewConflictError("仅已通过审核的问题可重试索引")
            chunk = session.scalar(select(KnowledgeChunk).where(KnowledgeChunk.review_id == review_id))
            if chunk is None:
                raise ReviewConflictError("该审核尚未关联知识条目")
            chunk_id = chunk.id
        self._try_index(chunk_id)
        return chunk_id

    def reject(self, review_id: int) -> None:
        with self._factory() as session:
            review = session.get(ReviewQueue, review_id)
            if review is None:
                raise KeyError(f"review_queue 不存在:{review_id}")
            review.review_status = "驳回"
            session.commit()
