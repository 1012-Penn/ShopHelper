"""ch14 后台管理员 API：登录、登出、会话信息与审计查询。"""
from __future__ import annotations

import asyncio
import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field

from app.admin_auth import (AdminPrincipal, audit_context, origin_matches,
                            require_admin, require_admin_role, require_csrf)
from app.admin_security import (SESSION_COOKIE, new_csrf_token, safe_next_path,
                                sign_session)
from app.admin_store import AdminAuditStore, AdminStore, ROLE_OPERATOR, role_allows

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["admin"])


class AdminLoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)
    next: str | None = None


def _cookie_secure(request: Request) -> bool:
    return bool(request.app.state.settings.admin_cookie_secure)


@router.post("/login")
async def login(body: AdminLoginRequest, request: Request, response: Response) -> dict:
    settings = request.app.state.settings
    if not settings.admin_session_secret:
        raise HTTPException(status_code=503,
                            detail="后台鉴权未配置:请设置 ADMIN_SESSION_SECRET")
    if not origin_matches(request):
        raise HTTPException(status_code=403, detail="请求来源校验失败")

    store: AdminStore = request.app.state.admin_store
    audit: AdminAuditStore = request.app.state.admin_audit_store
    try:
        result = await asyncio.to_thread(
            store.authenticate, body.username, body.password,
            max_failures=settings.admin_login_max_failures,
            lock_seconds=settings.admin_login_lock_seconds,
        )
    except Exception:
        logger.warning("[auth] 后台登录查库失败", exc_info=True)
        raise HTTPException(status_code=503, detail="后台鉴权暂不可用")

    if result.reason != "ok" or result.user is None:
        await asyncio.to_thread(
            audit.log_best_effort,
            admin_user_id=result.user.id if result.user else None,
            username=result.user.username if result.user else body.username,
            role=result.user.role if result.user else "anonymous",
            action="admin.login_failed", target_type="admin_session",
            result="失败", detail={"reason": result.reason},
            **audit_context(request),
        )
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    user = result.user
    ttl = max(60, int(settings.admin_session_ttl_seconds))
    csrf = new_csrf_token()
    now = int(time.time())
    token = sign_session({
        "uid": user.id, "username": user.username, "role": user.role,
        "ver": user.session_version, "iat": now, "exp": now + ttl, "csrf": csrf,
    }, settings.admin_session_secret)
    response.set_cookie(SESSION_COOKIE, token, max_age=ttl, httponly=True,
                        samesite="lax", secure=_cookie_secure(request), path="/")
    response.headers["Cache-Control"] = "no-store"

    await asyncio.to_thread(
        audit.log_best_effort,
        admin_user_id=user.id, username=user.username, role=user.role,
        action="admin.login", target_type="admin_session",
        detail={"expires_in": ttl}, **audit_context(request),
    )
    return {"username": user.username, "role": user.role, "csrf_token": csrf,
            "expires_in": ttl, "redirect_to": safe_next_path(body.next)}


@router.post("/logout")
async def logout(request: Request, response: Response,
                 principal: AdminPrincipal = Depends(require_csrf)) -> dict:
    response.delete_cookie(SESSION_COOKIE, path="/", httponly=True,
                           samesite="lax", secure=_cookie_secure(request))
    response.headers["Cache-Control"] = "no-store"
    audit: AdminAuditStore = request.app.state.admin_audit_store
    await asyncio.to_thread(
        audit.log_best_effort,
        admin_user_id=principal.user_id, username=principal.username,
        role=principal.role, action="admin.logout",
        target_type="admin_session", **audit_context(request),
    )
    return {"logged_out": True}


@router.get("/session")
async def session_info(request: Request, response: Response,
                       principal: AdminPrincipal = Depends(require_admin)) -> dict:
    response.headers["Cache-Control"] = "no-store"
    return {"username": principal.username, "role": principal.role,
            "csrf_token": principal.csrf_token,
            "can_write": role_allows(principal.role, ROLE_OPERATOR)}


@router.get("/audit")
async def audit_list(request: Request, response: Response,
                     limit: int = Query(50, ge=1, le=200),
                     offset: int = Query(0, ge=0),
                     action: str | None = None,
                     username: str | None = None,
                     result: str | None = None,
                     _principal: AdminPrincipal = Depends(require_admin_role)) -> dict:
    store: AdminAuditStore = request.app.state.admin_audit_store
    items = await asyncio.to_thread(store.list_logs, limit=limit, offset=offset,
                                    action=action, username=username, result=result)
    response.headers["Cache-Control"] = "no-store"
    return {"items": items, "limit": limit, "offset": offset}
