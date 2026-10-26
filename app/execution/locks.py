"""按 session factory 共享的写事务锁；只解决 SQLite 单连接游标互踩。"""
from threading import RLock


def write_lock_for(session_factory) -> RLock:
    lock = getattr(session_factory, "_shophelper_write_lock", None)
    if lock is None:
        lock = RLock()
        setattr(session_factory, "_shophelper_write_lock", lock)
    return lock
