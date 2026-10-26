#!/usr/bin/env python3
"""CH11 可靠执行可复跑验收：真实子进程、独立 SQLite 连接与固定负载测量。

不调用模型、不连接共享 MySQL/Milvus。默认输出 JSON；可用 --json-out 保存。
崩溃子进程只写隔离临时库，随后由父进程用新 engine/factory 验证恢复。
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.execution.store import (ExecutionStore, RequestConflict, SessionBusy,
                                 UnsafeResume)
from app.models import Base, Conversation, Ticket
from app.tickets import create_ticket_idempotent


def make_engine(db_path: str):
    return create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False, "timeout": 30},
    )


def make_factory(engine):
    return sessionmaker(bind=engine, expire_on_commit=False)


def init_db(db_path: str) -> None:
    engine = make_engine(db_path)
    Base.metadata.create_all(engine)
    engine.dispose()


def _emit(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def child_main(mode: str, db_path: str) -> int:
    engine = make_engine(db_path)
    factory = make_factory(engine)
    store = ExecutionStore(factory)
    conversation_id = 900
    owner = f"child-{mode}"
    if mode == "crash-before-event":
        run = store.reserve_run(
            conversation_id=conversation_id,
            client_request_id="child-before-event",
            request_payload={"message": "崩溃前未持久事件", "session_id": conversation_id},
            owner_token=owner,
            lease_seconds=-1,
        )
        _emit({"run_id": run.run_id})
        os._exit(23)
    if mode == "crash-after-event":
        run = store.reserve_run(
            conversation_id=conversation_id,
            client_request_id="child-after-event",
            request_payload={"message": "已持久首事件", "session_id": conversation_id},
            owner_token=owner,
            lease_seconds=-1,
        )
        store.append_event(run.run_id, {"type": "session", "session_id": conversation_id})
        _emit({"run_id": run.run_id})
        os._exit(24)
    if mode == "crash-after-write-start":
        run = store.reserve_run(
            conversation_id=conversation_id,
            client_request_id="child-write-start",
            request_payload={"message": "写前崩溃", "session_id": conversation_id},
            owner_token=owner,
            lease_seconds=-1,
        )
        store.start_tool_attempt(
            run.run_id,
            tool_call_id="child-write-call",
            tool_name="create_ticket",
            access="write",
            arguments={"description": "模拟写响应丢失"},
            idempotency_key="child-confirm-key",
        )
        _emit({"run_id": run.run_id})
        os._exit(25)
    if mode == "complete":
        run = store.reserve_run(
            conversation_id=conversation_id,
            client_request_id="child-complete",
            request_payload={"message": "子进程完成", "session_id": conversation_id},
            owner_token=owner,
            lease_seconds=60,
        )
        store.append_event(run.run_id, {"type": "session", "session_id": conversation_id})
        store.append_event(run.run_id, {"type": "token", "content": "完成"})
        store.append_event(run.run_id, {"type": "done"})
        store.finish_run(
            run.run_id,
            status="completed",
            result={"answer": "完成"},
            usage={"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
            duration_ms=7,
        )
        _emit({"run_id": run.run_id})
        return 0
    if mode == "resume":
        run_id = os.environ["RELIABILITY_RUN_ID"]
        store.resume_run(run_id, owner_token=owner, lease_seconds=60)
        store.append_event(run_id, {"type": "token", "content": "恢复完成"})
        store.append_event(run_id, {"type": "done"})
        store.finish_run(
            run_id,
            status="completed",
            result={"answer": "恢复完成"},
            usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            duration_ms=3,
        )
        _emit({"run_id": run_id})
        return 0
    raise SystemExit(f"unknown child mode: {mode}")


def _spawn(mode: str, db_path: str, *, env: dict | None = None) -> dict:
    completed = subprocess.run(
        [sys.executable, __file__, "--child", mode, "--db", db_path],
        cwd=Path(__file__).resolve().parent.parent,
        env={**os.environ, **(env or {})},
        capture_output=True,
        text=True,
        timeout=30,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"{mode} child produced no result: {completed.stderr}")
    return {**json.loads(lines[-1]), "returncode": completed.returncode}


def same_session_race(db_path: str) -> dict:
    engine = make_engine(db_path)
    barrier = threading.Barrier(2)
    outcomes: list[str] = []
    lock = threading.Lock()

    def attempt(suffix: str) -> None:
        store = ExecutionStore(make_factory(engine))
        barrier.wait(timeout=10)
        try:
            store.reserve_run(
                conversation_id=500,
                client_request_id=f"race-{suffix}",
                request_payload={"message": suffix, "session_id": 500},
                owner_token=f"owner-{suffix}",
                lease_seconds=60,
            )
            with lock:
                outcomes.append("created")
        except SessionBusy:
            with lock:
                outcomes.append("busy")

    threads = [threading.Thread(target=attempt, args=(name,)) for name in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    engine.dispose()
    return {
        "created": outcomes.count("created"),
        "busy": outcomes.count("busy"),
        "threads_finished": sum(not thread.is_alive() for thread in threads),
    }


def restart_evidence(base: Path) -> dict:
    complete_db = str(base / "complete.db")
    init_db(complete_db)
    child = _spawn("complete", complete_db)
    store = ExecutionStore(make_factory(make_engine(complete_db)))
    completed = store.get_run(child["run_id"])
    replay_types = [event.event["type"] for event in store.read_events(child["run_id"])]

    before_db = str(base / "before.db")
    init_db(before_db)
    before = _spawn("crash-before-event", before_db)
    before_store = ExecutionStore(make_factory(make_engine(before_db)))
    before_store.recover_expired_lease(900)
    before_status = before_store.get_run(before["run_id"])
    resumed = _spawn("resume", before_db, env={"RELIABILITY_RUN_ID": before["run_id"]})
    before_final = before_store.get_run(before["run_id"])

    after_db = str(base / "after.db")
    init_db(after_db)
    after = _spawn("crash-after-event", after_db)
    after_store = ExecutionStore(make_factory(make_engine(after_db)))
    after_store.recover_expired_lease(900)
    after_status = after_store.get_run(after["run_id"])
    after_types = [event.event["type"] for event in after_store.read_events(after["run_id"])]

    write_db = str(base / "write.db")
    init_db(write_db)
    write = _spawn("crash-after-write-start", write_db)
    write_store = ExecutionStore(make_factory(make_engine(write_db)))
    write_store.recover_expired_lease(900)
    write_status = write_store.get_run(write["run_id"])
    try:
        write_store.resume_run(write["run_id"], owner_token="new", lease_seconds=60)
        resume_blocked = False
    except UnsafeResume:
        resume_blocked = True

    return {
        "completed_child_replay": {
            "parent_status": completed["status"],
            "answer": completed["result"]["answer"],
            "events": replay_types,
            "child_exit": child["returncode"],
        },
        "crash_before_event": {
            "recovered_status": before_status["status"],
            "resume_child_exit": resumed["returncode"],
            "final_status": before_final["status"],
            "attempt": before_final["attempt"],
        },
        "crash_after_event": {
            "recovered_status": after_status["status"],
            "events": after_types,
        },
        "crash_after_write_start": {
            "status": write_status["status"],
            "write_state": write_status["write_state"],
            "auto_resume_blocked": resume_blocked,
        },
    }


class FlakyFactory:
    def __init__(self, factory) -> None:
        self._factory = factory
        self.fail_next = False

    def __call__(self):
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("injected database outage")
        return self._factory()


def db_failure_and_write_replay(base: Path) -> dict:
    db_path = str(base / "recovery.db")
    init_db(db_path)
    real_factory = make_factory(make_engine(db_path))
    flaky = FlakyFactory(real_factory)
    store = ExecutionStore(flaky)
    run = store.reserve_run(
        conversation_id=700,
        client_request_id="db-failure",
        request_payload={"message": "数据库故障", "session_id": 700},
        owner_token="owner",
        lease_seconds=60,
    )
    flaky.fail_next = True
    failed = False
    try:
        store.append_event(run.run_id, {"type": "session", "session_id": 700})
    except RuntimeError as exc:
        failed = "injected database outage" in str(exc)
    event_id = store.append_event(run.run_id, {"type": "session", "session_id": 700})
    sequence = [event.seq for event in store.read_events(run.run_id)]
    store.finish_run(run.run_id, status="completed", duration_ms=1)

    with real_factory() as session:
        session.add(Conversation(id=700, user_id="guest"))
        session.commit()
    first = create_ticket_idempotent(
        real_factory, conversation_id=700, description="响应丢失",
        ticket_type="售后", confirm_key="lost-response-confirm-key",
    )
    retry = create_ticket_idempotent(
        real_factory, conversation_id=700, description="响应丢失",
        ticket_type="售后", confirm_key="lost-response-confirm-key",
    )
    with real_factory() as session:
        ticket_count = len(session.scalars(select(Ticket)).all())
    return {
        "db_failure_recovered": failed,
        "event_after_retry": event_id > 0,
        "event_sequences": sequence,
        "lost_response_same_ticket": first["ticket_no"] == retry["ticket_no"],
        "lost_response_deduplicated": retry.get("deduplicated") is True,
        "ticket_count": ticket_count,
    }


def duplicate_request(db_path: str) -> dict:
    store = ExecutionStore(make_factory(make_engine(db_path)))
    payload = {"message": "同键", "session_id": 600}
    first = store.reserve_run(
        conversation_id=600, client_request_id="same-client-request",
        request_payload=payload, owner_token="a", lease_seconds=60)
    second = store.reserve_run(
        conversation_id=600, client_request_id="same-client-request",
        request_payload=payload, owner_token="b", lease_seconds=60)
    conflict = False
    try:
        store.reserve_run(
            conversation_id=600, client_request_id="same-client-request",
            request_payload={"message": "被改", "session_id": 600},
            owner_token="c", lease_seconds=60)
    except RequestConflict:
        conflict = True
    return {
        "same_run": first.run_id == second.run_id,
        "second_was_created": second.created,
        "changed_payload_conflict": conflict,
    }


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    rank = max(1, int(round(p * len(ordered) + 0.4999)))
    return ordered[min(rank - 1, len(ordered) - 1)]


def benchmark(base: Path) -> dict:
    db_path = str(base / "benchmark.db")
    init_db(db_path)
    engine = make_engine(db_path)
    workers = min(8, max(2, os.cpu_count() or 2))
    runs_per_worker = 4
    durations: list[float] = []
    first_events: list[float] = []
    successes = 0

    def workload(worker_index: int) -> list[tuple[float, float, bool]]:
        store = ExecutionStore(make_factory(engine))
        local = []
        for iteration in range(runs_per_worker):
            start = time.perf_counter()
            conversation_id = 1000 + worker_index
            ok = False
            first_ms = 0.0
            try:
                run = store.reserve_run(
                    conversation_id=conversation_id,
                    client_request_id=f"bench-{worker_index}-{iteration}",
                    request_payload={
                        "message": "固定负载",
                        "session_id": conversation_id,
                    },
                    owner_token=f"bench-{worker_index}",
                    lease_seconds=60,
                )
                store.append_event(run.run_id, {"type": "session", "session_id": conversation_id})
                first_ms = (time.perf_counter() - start) * 1000
                store.append_event(run.run_id, {"type": "token", "content": "ok"})
                store.append_event(run.run_id, {"type": "done"})
                store.finish_run(
                    run.run_id, status="completed",
                    result={"answer": "ok"}, duration_ms=0)
                ok = True
            finally:
                local.append(((time.perf_counter() - start) * 1000, first_ms, ok))
        return local

    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(workload, range(workers)):
            for duration, first_ms, ok in result:
                durations.append(duration)
                first_events.append(first_ms)
                successes += int(ok)
    wall_s = time.perf_counter() - wall_start
    engine.dispose()
    total_runs = workers * runs_per_worker
    return {
        "backend": "isolated_file_sqlite",
        "configured_workers": workers,
        "hardware_logical_cpus": os.cpu_count(),
        "runs": total_runs,
        "success_rate": successes / total_runs,
        "wall_s": wall_s,
        "throughput_runs_per_s": total_runs / wall_s if wall_s else 0.0,
        "run_p50_ms": percentile(durations, 0.50),
        "run_p95_ms": percentile(durations, 0.95),
        "first_persisted_event_p50_ms": percentile(first_events, 0.50),
        "first_persisted_event_p95_ms": percentile(first_events, 0.95),
        "mean_run_ms": statistics.fmean(durations),
    }


def run_acceptance() -> dict:
    with tempfile.TemporaryDirectory(prefix="shophelper-reliability-") as td:
        base = Path(td)
        race_db = str(base / "race.db")
        init_db(race_db)
        duplicate_db = str(base / "duplicate.db")
        init_db(duplicate_db)
        result = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "same_session_race": same_session_race(race_db),
            "duplicate_request": duplicate_request(duplicate_db),
            "restart": restart_evidence(base),
            "dbfault_and_write_response_loss": db_failure_and_write_replay(base),
            "benchmark": benchmark(base),
            "model_calls": 0,
            "paid_calls": 0,
        }
        return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", choices=[
        "crash-before-event", "crash-after-event", "crash-after-write-start",
        "complete", "resume",
    ])
    parser.add_argument("--db")
    parser.add_argument("--json-out")
    args = parser.parse_args()
    if args.child:
        if not args.db:
            raise SystemExit("--db is required for child mode")
        return child_main(args.child, args.db)
    result = run_acceptance()
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.json_out:
        Path(args.json_out).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
