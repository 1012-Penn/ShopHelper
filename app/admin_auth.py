"""ch14 FastAPI 后台鉴权依赖：会话解析、角色依赖、CSRF、页面守卫。"""
from __future__ import annotations

import asyncio
import hmac
import logging
from dataclasses import dataclass
from urllib.parse import quote, urlsplit

from fastapi import Depends, HTTPException, Request
from fastapi.responses import PlainTextResponse, RedirectResponse

from app.admin_security import SESSION_COOKIE, safe_next_path, verify_session
from app.admin_store import AdminStore, role_allows

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AdminPrincipal:
    user_id: int
    username: str
    role: str
    csrf_token: str
    session_version: int


def origin_matches(request: Request) -> bool:
    """存在 Origin 时必须同源；忽略 scheme 以兼容 TLS 终止代理。"""
    origin = request.headers.get("origin")
    if not origin:
        return True
    if origin == "null":
        return False
    try:
        parsed = urlsplit(origin)
        expected_host = request.url.hostname or ""
        if not parsed.hostname or parsed.hostname.lower() != expected_host.lower():
            return False
        if parsed.port is not None and request.url.port is not None \
                and parsed.port != request.url.port:
            return False
        return True
    except Exception:
        return False


def read_principal(request: Request) -> AdminPrincipal | None:
    """验 Cookie 并查库；角色只取 DB 值，token 里的 role 不作为授权依据。"""
    secret = request.app.state.settings.admin_session_secret
    if not secret:
        return None
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    payload = verify_session(token, secret)
    if payload is None:
        return None
    store: AdminStore = request.app.state.admin_store
    try:
        user = store.get_by_id(payload["uid"])
    except Exception:
        logger.warning("[auth] 后台鉴权查库失败", exc_info=True)
        raise HTTPException(status_code=503, detail="后台鉴权暂不可用")
    if user is None or not user.is_active or user.session_version != payload["ver"]:
        return None
    if not role_allows(user.role, "viewer"):
        return None
    return AdminPrincipal(user.id, user.username, user.role,
                          payload["csrf"], user.session_version)


def require_admin(request: Request) -> AdminPrincipal:
    principal = read_principal(request)
    if principal is None:
        raise HTTPException(status_code=401, detail="请先登录")
    return principal


def require_role(*roles: str):
    def dependency(principal: AdminPrincipal = Depends(require_admin)) -> AdminPrincipal:
        if principal.role not in roles:
            raise HTTPException(status_code=403, detail="权限不足")
        return principal
    return dependency


require_viewer = require_role("viewer", "operator", "admin")
require_operator = require_role("operator", "admin")
require_admin_role = require_role("admin")


def require_csrf(request: Request,
                 principal: AdminPrincipal = Depends(require_admin)) -> AdminPrincipal:
    if request.method.upper() in {"POST", "PUT", "PATCH", "DELETE"}:
        token = request.headers.get("x-csrf-token", "")
        if not token or not hmac.compare_digest(token, principal.csrf_token):
            raise HTTPException(status_code=403, detail="CSRF 校验失败,请刷新页面重试")
        if not origin_matches(request):
            raise HTTPException(status_code=403, detail="请求来源校验失败")
    return principal


async def page_guard(request: Request):
    """页面守卫：未登录 303 到登录页；角色不足 403；通过返回 principal。"""
    principal = await asyncio.to_thread(read_principal, request)
    if principal is None:
        target = request.url.path
        if request.url.query:
            target += "?" + request.url.query
        next_path = safe_next_path(target)
        return RedirectResponse(f"/admin/login?next={quote(next_path, safe='')}",
                                status_code=303)
    if not role_allows(principal.role, "viewer"):
        return PlainTextResponse("权限不足", status_code=403)
    return principal


def audit_context(request: Request) -> dict:
    ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")
    return {"ip": ip, "user_agent": user_agent[:255] if user_agent else None}
