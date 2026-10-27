"""Build fixed evidence arms without changing retrieval or rewrite behavior."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path

from app.evidence_slate import EvidenceDocument, select_evidence_slate


def evidence_rows_from_ids(
    ids: Iterable[int],
    scores: Iterable[float],
    rows: Mapping[int, dict],
    *,
    top_k: int,
) -> list[dict]:
    result = []
    for chunk_id, score in zip(ids, scores):
        row = rows.get(int(chunk_id))
        if row is None:
            continue
        result.append(
            {
                "chunk_id": int(chunk_id),
                "section_path": str(row.get("section_path") or ""),
                "text": str(row.get("text") or ""),
                "score": round(float(score), 9),
            }
        )
        if len(result) >= top_k:
            break
    return result


def select_from_saved_rerank(
    *,
    case_id: str,
    query: str,
    artifact: dict,
    rows: Mapping[int, dict],
    top_k: int,
) -> dict:
    candidate_ids = [int(item) for item in artifact["candidate_ids"]]
    ranked_ids = [candidate_ids[int(index)] for index, _score in artifact["ranked"]]
    scores = [float(score) for _index, score in artifact["ranked"]]
    documents = {
        chunk_id: EvidenceDocument(
            chunk_id=chunk_id,
            section_path=str(row.get("section_path") or ""),
            text=str(row.get("text") or ""),
        )
        for chunk_id, row in rows.items()
        if chunk_id in set(candidate_ids)
    }
    selection = select_evidence_slate(
        query,
        ranked_ids=ranked_ids,
        documents=documents,
        top_k=top_k,
    )
    score_by_id = dict(zip(ranked_ids, scores))
    return {
        "case_id": case_id,
        "baseline": evidence_rows_from_ids(ranked_ids, scores, rows, top_k=top_k),
        "slate": evidence_rows_from_ids(
            selection.ordered_ids,
            [score_by_id[chunk_id] for chunk_id in selection.ordered_ids],
            rows,
            top_k=top_k,
        ),
        "selection_facets": list(selection.facets),
        "candidate_pool_k": len(ranked_ids),
    }


def load_saved_dev_reranks(path: Path) -> dict[str, dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(item["case_id"]): item for item in payload.get("cases") or []}


def load_chunk_rows(sqlite_path: Path) -> dict[int, dict]:
    import sqlite3

    connection = sqlite3.connect(f"file:{Path(sqlite_path)}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT id, category, questions, answer, section_path FROM knowledge_chunks"
        ).fetchall()
    finally:
        connection.close()
    return {
        int(row[0]): {
            "chunk_id": int(row[0]),
            "category": str(row[1] or ""),
            "question": str(row[2] or ""),
            "answer": str(row[3] or ""),
            "section_path": str(row[4] or ""),
            "text": f"{row[2] or ''!s}\n{row[3] or ''!s}".strip(),
        }
        for row in rows
    }


def configure_retrieval_environment(
    *, credential_env_file: Path, sqlite_db: Path, snapshot: Path
) -> None:
    import os

    from dotenv import dotenv_values

    values = dotenv_values(credential_env_file)
    for key in (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "EMBEDDING_API_KEY",
        "EMBEDDING_API_BASE",
        "RERANK_API_KEY",
        "RERANK_API_BASE",
    ):
        value = values.get(key)
        if value:
            os.environ[key] = str(value)
    os.environ.setdefault("OPENAI_API_KEY", "not-used-for-retrieval")
    os.environ["DATABASE_URL"] = f"sqlite:///{Path(sqlite_db).resolve()}"
    os.environ["MILVUS_DB_PATH"] = str(Path(snapshot))
    os.environ["OPENAI_MODEL"] = "deepseek-flash"
    os.environ["LLM_MAX_RETRIES"] = "0"
    os.environ["HYBRID_CANDIDATES"] = "50"
    os.environ["RERANK_TOP_K"] = "10"
    os.environ["RETRIEVAL_EVIDENCE_SLATE_ENABLED"] = "true"
    os.environ["RETRIEVAL_RERANK_POOL_K"] = "50"
