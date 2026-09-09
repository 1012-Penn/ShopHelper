"""低置信度问题的数据飞轮:标准化、语义去重、审核与知识库待向量化写入。"""
from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from app.models import KnowledgeChunk, LowConfidenceQuestion, ReviewQueue


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
            rows = session.scalars(
                select(ReviewQueue).where(ReviewQueue.review_status == status)
                .order_by(ReviewQueue.occurrence_count.desc(), ReviewQueue.updated_at.desc())
            ).all()
            return [self._review_dict(row) for row in rows]

    def detail(self, review_id: int) -> dict | None:
        with self._factory() as session:
            row = session.get(ReviewQueue, review_id)
            if row is None:
                return None
            result = self._review_dict(row)
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
    def _review_dict(row: ReviewQueue) -> dict:
        return {
            "id": row.id, "normalized_question": row.normalized_question,
            "ai_suggested_answer": row.ai_suggested_answer or "",
            "occurrence_count": row.occurrence_count, "review_status": row.review_status,
            "approved_answer": row.approved_answer or "",
        }

    def approve(self, review_id: int, approved_answer: str) -> int:
        """审核通过:写 knowledge_chunks(pending),由后续 vectorize_pending 双写。"""
        answer = (approved_answer or "").strip()
        if not answer:
            raise ValueError("核准答案不能为空")
        with self._factory() as session:
            review = session.get(ReviewQueue, review_id)
            if review is None:
                raise KeyError(f"review_queue 不存在:{review_id}")
            review.review_status = "通过"
            review.approved_answer = answer
            chunk = KnowledgeChunk(
                category="飞轮补全", questions=review.normalized_question, answer=answer,
                section_path=f"review_queue:{review_id}", content_type="faq", is_key_clause=0,
            )
            session.add(chunk)
            session.flush()
            chunk_id = chunk.id
            session.commit()
            if self._kb is not None and self._vectors is not None and self._embedder is not None:
                from app.kb import vectorize_pending
                vectorize_pending(self._kb, self._vectors, self._embedder)
            return chunk_id

    def reject(self, review_id: int) -> None:
        with self._factory() as session:
            review = session.get(ReviewQueue, review_id)
            if review is None:
                raise KeyError(f"review_queue 不存在:{review_id}")
            review.review_status = "驳回"
            session.commit()
