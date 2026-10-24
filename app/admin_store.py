"""ch14 后台管理员与审计存储。

路由只拿 DTO，不接触 ORM/password_hash；审计写入失败不阻断业务。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from app.admin_security import PBKDF2_ITERATIONS, hash_password, verify_password
from app.models import AdminAuditLog, AdminUser

logger = logging.getLogger(__name__)

ROLE_VIEWER = "viewer"
ROLE_OPERATOR = "operator"
ROLE_ADMIN = "admin"
ROLES = (ROLE_VIEWER, ROLE_OPERATOR, ROLE_ADMIN)
_ROLE_RANK = {ROLE_VIEWER: 1, ROLE_OPERATOR: 2, ROLE_ADMIN: 3}
_AUDIT_RESULTS = ("成功", "失败")


def normalize_username(username: str) -> str:
    return (username or "").strip().lower()


def role_allows(actual: str, required: str) -> bool:
    """未知角色 fail closed。"""
    return _ROLE_RANK.get(actual, 0) >= _ROLE_RANK.get(required, 999)


@dataclass(frozen=True)
class AdminUserView:
    id: int
    username: str
    role: str
    is_active: bool
    session_version: int


@dataclass(frozen=True)
class LoginResult:
    user: AdminUserView | None
    reason: str  # ok | bad_credentials | locked | inactive | unknown


class AdminStore:
    def __init__(self, factory: sessionmaker, *,
                 password_iterations: int = PBKDF2_ITERATIONS) -> None:
        self._factory = factory
        self._iterations = max(1, int(password_iterations))
        self._dummy_hash: str | None = None

    @staticmethod
    def _view(row: AdminUser) -> AdminUserView:
        return AdminUserView(row.id, row.username, row.role,
                             bool(row.is_active), row.session_version)

    def _dummy(self) -> str:
        """未知用户的等价耗时校验；惰性生成，迭代次数与真实账号一致。"""
        if self._dummy_hash is None:
            self._dummy_hash = hash_password("dummy-password", iterations=self._iterations)
        return self._dummy_hash

    def get_by_id(self, user_id: int) -> AdminUserView | None:
        with self._factory() as session:
            row = session.get(AdminUser, user_id)
            return self._view(row) if row else None

    def get_by_username(self, username: str) -> AdminUserView | None:
        name = normalize_username(username)
        if not name:
            return None
        with self._factory() as session:
            row = session.scalar(select(AdminUser).where(AdminUser.username == name))
            return self._view(row) if row else None

    def authenticate(self, username: str, password: str, *, max_failures: int,
                     lock_seconds: int, now: datetime | None = None) -> LoginResult:
        """统一登录判定；任何失败原因都由路由映射为同一条 401。"""
        now = now or datetime.utcnow()
        name = normalize_username(username)
        with self._factory() as session:
            row = session.scalar(select(AdminUser).where(AdminUser.username == name)) if name else None
            if row is None:
                verify_password(password or "", self._dummy())
                return LoginResult(None, "unknown")
            if not row.is_active:
                # 仍做一次真实校验，降低“禁用/不存在”的时序差
                verify_password(password or "", row.password_hash)
                return LoginResult(self._view(row), "inactive")
            if row.locked_until and row.locked_until > now:
                return LoginResult(self._view(row), "locked")
            if row.locked_until and row.locked_until <= now:
                row.failed_login_count = 0
                row.locked_until = None
            if not verify_password(password or "", row.password_hash):
                row.failed_login_count = (row.failed_login_count or 0) + 1
                if row.failed_login_count >= max(1, int(max_failures)):
                    row.locked_until = now + timedelta(seconds=max(1, int(lock_seconds)))
                session.commit()
                return LoginResult(self._view(row), "bad_credentials")
            row.failed_login_count = 0
            row.locked_until = None
            row.last_login_at = now
            session.commit()
            return LoginResult(self._view(row), "ok")

    def create_user(self, username: str, password: str, role: str = ROLE_VIEWER) -> AdminUserView:
        name = normalize_username(username)
        if not 1 <= len(name) <= 64 or any(ord(ch) < 32 for ch in name):
            raise ValueError("用户名不合法")
        if role not in ROLES:
            raise ValueError("角色不合法")
        if not isinstance(password, str) or not 12 <= len(password) <= 256:
            raise ValueError("口令长度需在 12..256 位")
        with self._factory() as session:
            row = AdminUser(username=name, password_hash=hash_password(
                password, iterations=self._iterations), role=role,
                is_active=True, session_version=1)
            session.add(row)
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
                raise ValueError("用户名已存在")
            session.refresh(row)
            return self._view(row)

    def list_users(self) -> list[dict]:
        with self._factory() as session:
            rows = session.scalars(select(AdminUser).order_by(AdminUser.id)).all()
            return [{
                "id": row.id, "username": row.username, "role": row.role,
                "is_active": bool(row.is_active),
                "locked_until": row.locked_until.isoformat() if row.locked_until else None,
                "last_login_at": row.last_login_at.isoformat() if row.last_login_at else None,
                "session_version": row.session_version,
            } for row in rows]

    def set_active(self, username: str, active: bool) -> bool:
        name = normalize_username(username)
        with self._factory() as session:
            row = session.scalar(select(AdminUser).where(AdminUser.username == name))
            if row is None:
                return False
            row.is_active = bool(active)
            # 启停都 bump：禁用前签发的 token 不能在重新启用后复活
            row.session_version = (row.session_version or 1) + 1
            session.commit()
            return True

    def reset_password(self, username: str, password: str) -> bool:
        if not isinstance(password, str) or not 12 <= len(password) <= 256:
            raise ValueError("口令长度需在 12..256 位")
        name = normalize_username(username)
        with self._factory() as session:
            row = session.scalar(select(AdminUser).where(AdminUser.username == name))
            if row is None:
                return False
            row.password_hash = hash_password(password, iterations=self._iterations)
            row.failed_login_count = 0
            row.locked_until = None
            row.session_version = (row.session_version or 1) + 1
            session.commit()
            return True

    def unlock(self, username: str) -> bool:
        name = normalize_username(username)
        with self._factory() as session:
            row = session.scalar(select(AdminUser).where(AdminUser.username == name))
            if row is None:
                return False
            row.failed_login_count = 0
            row.locked_until = None
            session.commit()
            return True

    def count_admins(self, *, active_only: bool = False) -> int:
        with self._factory() as session:
            stmt = select(func.count(AdminUser.id))
            if active_only:
                stmt = stmt.where(AdminUser.is_active.is_(True))
            return int(session.scalar(stmt) or 0)


class AdminAuditStore:
    """append-only 审计。写失败只 warning，不阻断业务（对齐 tool_audit_logs）。"""

    def __init__(self, factory: sessionmaker) -> None:
        self._factory = factory

    def log(self, *, admin_user_id: int | None, username: str, role: str,
            action: str, target_type: str, target_id: str | int | None = None,
            result: str = "成功", detail: dict | None = None,
            ip: str | None = None, user_agent: str | None = None) -> None:
        if result not in _AUDIT_RESULTS:
            raise ValueError("审计结果不合法")
        with self._factory() as session:
            session.add(AdminAuditLog(
                admin_user_id=admin_user_id,
                username=(normalize_username(username) or "unknown")[:64],
                role=(role or "anonymous")[:16],
                action=(action or "")[:64],
                target_type=(target_type or "")[:32],
                target_id=(str(target_id)[:64] if target_id is not None else None),
                result=result,
                detail=detail,
                ip=(ip[:64] if ip else None),
                user_agent=(user_agent[:255] if user_agent else None),
            ))
            session.commit()

    def log_best_effort(self, **kwargs) -> None:
        try:
            self.log(**kwargs)
        except Exception:
            logger.warning("后台审计写入失败(不拦业务):%s",
                           kwargs.get("action"), exc_info=True)

    def list_logs(self, *, limit: int = 50, offset: int = 0,
                  action: str | None = None, username: str | None = None,
                  result: str | None = None) -> list[dict]:
        limit = max(1, min(int(limit), 200))
        offset = max(0, int(offset))
        with self._factory() as session:
            stmt = select(AdminAuditLog).order_by(AdminAuditLog.id.desc())
            if action:
                stmt = stmt.where(AdminAuditLog.action == action)
            if username:
                stmt = stmt.where(AdminAuditLog.username == normalize_username(username))
            if result:
                stmt = stmt.where(AdminAuditLog.result == result)
            rows = session.scalars(stmt.offset(offset).limit(limit)).all()
            return [{
                "id": row.id, "admin_user_id": row.admin_user_id,
                "username": row.username, "role": row.role,
                "action": row.action, "target_type": row.target_type,
                "target_id": row.target_id, "result": row.result,
                "detail": row.detail or {}, "ip": row.ip,
                "user_agent": row.user_agent,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            } for row in rows]
