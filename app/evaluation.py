"""自动评估轮次结果的持久化与趋势读取。"""
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from app.models import EvalRun


class EvalRunStore:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._factory = session_factory

    def record(self, triggered_by: str, dataset_size: int, metrics: dict) -> int:
        if triggered_by not in {"定时", "手动"}:
            raise ValueError("triggered_by 必须是定时或手动")
        with self._factory() as session:
            row = EvalRun(triggered_by=triggered_by, dataset_size=dataset_size, metrics=metrics)
            session.add(row)
            session.commit()
            return row.id

    def list_runs(self, limit: int = 30) -> list[dict]:
        with self._factory() as session:
            rows = session.scalars(select(EvalRun).order_by(EvalRun.created_at, EvalRun.id)
                                   .limit(limit)).all()
            return [{"id": row.id, "triggered_by": row.triggered_by,
                     "dataset_size": row.dataset_size, "metrics": row.metrics,
                     "created_at": row.created_at.isoformat() if row.created_at else ""}
                    for row in rows]
