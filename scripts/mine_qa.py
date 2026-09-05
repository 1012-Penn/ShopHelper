"""从历史客服对话挖知识:分批 LLM 抽 QA → 暂存表 → 整体去重 → 保留项入库并向量化。

流程(spec §6):
1. 选料:messages 表里有 user 提问的会话,已挖过的(source_ref 在 staging)不重挖;
2. 分批抽取:每次运行处理 --batch-size 通会话,batch_no 记批次,LLM 抽 [question, answer] 写暂存表;
3. 整体去重:全部 extracted 项,question 向量与「库内已有知识」「批内已保留项」比余弦,
   超过阈值判重复置 discarded,存活置 kept 并入 knowledge_chunks(pending);
4. 收尾复用 build_kb 的向量化循环。

定时:crontab 外挂调度,示例(每小时一次):
  0 * * * * cd /path/to/ShopHelper && .venv/bin/python -m scripts.mine_qa >> logs/mine_qa.log 2>&1
"""
import argparse
from datetime import datetime

from pydantic import BaseModel, Field

from app.config import Settings
from app.db import make_engine, make_session_factory
from app.embedding import cosine, make_embedder
from app.kb import KnowledgeBaseStore, vectorize_pending
from app.vector_store import KnowledgeVectorStore


class MinedQA(BaseModel):
    question: str = Field(description="用户的原始问法,尽量保留原话")
    answer: str = Field(description="客服给出的答案,可适当精简")


class MinedQAList(BaseModel):
    items: list[MinedQA] = Field(description="从对话中抽取的问答对列表,没有可抽取的则为空")


def _conversation_text(factory, conversation_id: int) -> str | None:
    """user 提问 + 紧随其后的 assistant 正文(跳过纯工具调用消息),拼成对话文本;无完整问答返回 None。"""
    from app.models import Message
    from sqlalchemy import select

    with factory() as session:
        rows = session.scalars(
            select(Message).where(Message.conversation_id == conversation_id).order_by(Message.id)
        ).all()
    lines: list[str] = []
    pending_user: str | None = None
    for row in rows:
        if row.role == "user" and row.content:
            pending_user = row.content
        elif row.role == "assistant" and row.content and pending_user is not None:
            lines.append(f"用户:{pending_user}")
            lines.append(f"客服:{row.content}")
            pending_user = None
    return "\n".join(lines) if lines else None


def mine(
    kb: KnowledgeBaseStore,
    embedder,
    extract_model,
    *,
    batch_size: int = 5,
    dedup_threshold: float = 0.88,
    vectors=None,
    batch_no: str | None = None,
) -> dict:
    stats = {"conversations": 0, "extracted": 0, "kept": 0, "discarded": 0, "vectorized": 0}
    mined = kb.already_mined_refs()
    conv_ids = kb.conversation_ids_with_user_talks(exclude_refs=mined)[:batch_size]
    batch_no = batch_no or f"mine-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    for conv_id in conv_ids:
        text = _conversation_text(kb.session_factory, conv_id)
        if not text:
            continue
        result = extract_model.invoke(text)
        raw_items = getattr(result, "items", None) or []
        qas = []
        for i in raw_items:
            q = i["question"] if isinstance(i, dict) else i.question
            a = i["answer"] if isinstance(i, dict) else i.answer
            if q and a:
                qas.append((q, a))
        kb.insert_staging(batch_no, str(conv_id), qas)
        stats["conversations"] += 1
        stats["extracted"] += len(qas)

    items = kb.extracted_items()
    if items:
        q_vecs = embedder.embed([i.question for i in items])
        lib_texts = kb.all_dedup_texts()
        lib_vecs = embedder.embed(lib_texts) if lib_texts else []
        kept_ids: list[int] = []
        kept_vecs: list[list[float]] = []
        discarded_ids: list[int] = []
        for idx, item in enumerate(items):
            qv = q_vecs[idx]
            dup = max((cosine(qv, lv) for lv in lib_vecs), default=0.0) >= dedup_threshold
            if not dup:  # 批内两两:与已保留项比
                dup = max((cosine(qv, kv) for kv in kept_vecs), default=0.0) >= dedup_threshold
            if dup:
                discarded_ids.append(item.id)
            else:
                kept_ids.append(item.id)
                kept_vecs.append(qv)
        kb.set_staging_status(kept_ids, "kept")
        kb.set_staging_status(discarded_ids, "discarded")
        kept_items = [i for i in items if i.id in set(kept_ids)]
        kb.insert_mined_chunks(kept_items)
        stats["kept"] = len(kept_ids)
        stats["discarded"] = len(discarded_ids)

    if vectors is not None:
        stats["vectorized"] = vectorize_pending(kb, vectors, embedder)
    return stats


def _make_extract_model(settings: Settings):
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        temperature=settings.extract_temperature,
    )
    return model.with_structured_output(MinedQAList, method="function_calling")


def main() -> None:
    parser = argparse.ArgumentParser(description="历史对话挖知识(分批抽取→暂存→去重→入库)")
    parser.add_argument("--batch-size", type=int, default=5, help="本次处理的会话数上限")
    args = parser.parse_args()

    settings = Settings()
    kb = KnowledgeBaseStore(make_session_factory(make_engine(settings)))
    vectors = KnowledgeVectorStore(settings.milvus_db_path, dim=settings.embedding_dim)
    embedder = make_embedder(settings)
    extract_model = _make_extract_model(settings)
    stats = mine(
        kb, embedder, extract_model,
        batch_size=args.batch_size,
        dedup_threshold=settings.dedup_threshold,
        vectors=vectors,
    )
    print(
        f"挖知识完成:会话 {stats['conversations']} 通,抽取 {stats['extracted']} 条,"
        f"保留 {stats['kept']} 条,去重丢弃 {stats['discarded']} 条,向量化 {stats['vectorized']} 条"
    )


if __name__ == "__main__":
    main()
