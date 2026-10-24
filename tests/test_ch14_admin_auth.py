"""ch14 后台鉴权：安全原语、Store、HTTP 角色矩阵与审计。"""
import time
from datetime import datetime, timedelta
from urllib.parse import quote

import pytest

from app.admin_security import (SESSION_COOKIE, hash_password, new_csrf_token,
                                safe_next_path, sign_session, verify_password,
                                verify_session)
from app.admin_store import (AdminAuditStore, AdminStore, ROLE_ADMIN, ROLE_OPERATOR,
                             ROLE_VIEWER, role_allows)
from tests.helpers import admin_login, csrf_headers, fake_chat, seed_admin


def _store(factory) -> AdminStore:
    return AdminStore(factory, password_iterations=2_000)


# ---------- 安全原语 ----------

def test_hash_password_roundtrip_and_random_salt():
    first = hash_password("admin-pass-123", iterations=2_000)
    second = hash_password("admin-pass-123", iterations=2_000)
    assert first.startswith("pbkdf2_sha256$2000$")
    assert first != second
    assert verify_password("admin-pass-123", first)
    assert verify_password("admin-pass-123", second)
    assert not verify_password("wrong-pass", first)


@pytest.mark.parametrize("bad", [
    "",
    "plain-text",
    "pbkdf2_sha256$0$c2FsdA$aGFzaA",
    "pbkdf2_sha256$99999999$c2FsdA$aGFzaA",
    "md5$1000$c2FsdA$aGFzaA",
    "pbkdf2_sha256$1000$$",
    "pbkdf2_sha256$not-a-number$c2FsdA$aGFzaA",
])
def test_verify_password_rejects_malformed_hash(bad):
    assert verify_password("anything", bad) is False


def test_session_sign_and_verify_roundtrip():
    now = int(time.time())
    payload = {"uid": 7, "username": "admin", "role": "admin", "ver": 3,
               "csrf": new_csrf_token(), "iat": now, "exp": now + 60}
    token = sign_session(payload, "test-secret")
    assert verify_session(token, "test-secret") == payload


def test_session_rejects_tampering_and_bad_secrets():
    now = int(time.time())
    payload = {"uid": 7, "ver": 1, "csrf": "csrf", "iat": now, "exp": now + 60}
    token = sign_session(payload, "test-secret")
    body, signature = token.split(".")
    flipped = ("A" if signature[0] != "A" else "B") + signature[1:]
    assert verify_session(f"{body}.{flipped}", "test-secret") is None
    assert verify_session(token, "another-secret") is None
    assert verify_session(token, "") is None
    assert verify_session("not-a-token", "test-secret") is None


def test_session_rejects_expired_and_future_tokens():
    now = int(time.time())
    expired = sign_session({"uid": 1, "ver": 1, "csrf": "c", "iat": now - 120,
                            "exp": now - 1}, "s")
    future = sign_session({"uid": 1, "ver": 1, "csrf": "c", "iat": now + 300,
                           "exp": now + 600}, "s")
    assert verify_session(expired, "s") is None
    assert verify_session(future, "s") is None


@pytest.mark.parametrize("value,expected", [
    ("/admin/evals", "/admin/evals"),
    ("/admin/review?status=%E5%BE%85%E5%AE%A1", "/admin/review?status=%E5%BE%85%E5%AE%A1"),
    ("/admin/review#frag", "/admin/review"),
    (None, "/admin/review"),
    ("", "/admin/review"),
    ("https://evil.example/admin/review", "/admin/review"),
    ("//evil.example/admin/review", "/admin/review"),
    ("/admin/login", "/admin/review"),
    ("/admin/assets/admin.js", "/admin/review"),
    ("/foo", "/admin/review"),
    ("/admin/%0d%0aSet-Cookie:x", "/admin/review"),
    ("/admin/review\\evil", "/admin/review"),
])
def test_safe_next_path(value, expected):
    assert safe_next_path(value) == expected


# ---------- Store ----------

def test_create_and_lookup_user(db_session_factory):
    store = _store(db_session_factory)
    user = store.create_user("Admin", "admin-pass-123", ROLE_ADMIN)
    assert user.username == "admin" and user.role == ROLE_ADMIN
    assert user.session_version == 1
    assert store.get_by_username("ADMIN").id == user.id
    assert store.get_by_id(user.id).username == "admin"
    assert store.get_by_id(999_999) is None
    assert store.list_users()[0]["username"] == "admin"

    with pytest.raises(ValueError):
        store.create_user("admin", "another-pass-123")
    with pytest.raises(ValueError):
        store.create_user("x", "short")
    with pytest.raises(ValueError):
        store.create_user("x", "long-enough-pass", "root")


def test_authenticate_success_failure_and_unknown(db_session_factory):
    store = _store(db_session_factory)
    store.create_user("operator", "operator-pass-123", ROLE_OPERATOR)
    now = datetime(2026, 10, 25, 12, 0, 0)

    bad = store.authenticate("operator", "wrong-pass", max_failures=3,
                             lock_seconds=60, now=now)
    assert bad.reason == "bad_credentials"

    ok = store.authenticate(" Operator ", "operator-pass-123", max_failures=3,
                            lock_seconds=60, now=now)
    assert ok.reason == "ok" and ok.user.role == ROLE_OPERATOR
    assert store.get_by_username("operator").session_version == 1

    unknown = store.authenticate("nobody", "whatever-pass", max_failures=3,
                                 lock_seconds=60, now=now)
    assert unknown.reason == "unknown" and unknown.user is None


def test_lockout_and_unlock_after_window(db_session_factory):
    store = _store(db_session_factory)
    store.create_user("locked", "locked-pass-123", ROLE_VIEWER)
    now = datetime(2026, 10, 25, 12, 0, 0)

    for _ in range(3):
        assert store.authenticate("locked", "wrong-pass", max_failures=3,
                                  lock_seconds=60, now=now).reason == "bad_credentials"

    assert store.authenticate("locked", "locked-pass-123", max_failures=3,
                              lock_seconds=60, now=now).reason == "locked"
    assert store.authenticate("locked", "locked-pass-123", max_failures=3,
                              lock_seconds=60,
                              now=now + timedelta(seconds=30)).reason == "locked"
    assert store.authenticate("locked", "locked-pass-123", max_failures=3,
                              lock_seconds=60,
                              now=now + timedelta(seconds=61)).reason == "ok"


def test_inactive_user_cannot_authenticate(db_session_factory):
    store = _store(db_session_factory)
    store.create_user("disabled", "disabled-pass-123", ROLE_VIEWER)
    assert store.set_active("disabled", False)
    result = store.authenticate("disabled", "disabled-pass-123",
                                max_failures=5, lock_seconds=60)
    assert result.reason == "inactive"


def test_password_reset_and_active_toggle_bump_session_version(db_session_factory):
    store = _store(db_session_factory)
    user = store.create_user("alice", "alice-pass-123", ROLE_VIEWER)

    assert store.reset_password("alice", "alice-pass-456")
    assert store.get_by_id(user.id).session_version == 2
    assert store.authenticate("alice", "alice-pass-456", max_failures=5,
                              lock_seconds=60).reason == "ok"
    assert store.set_active("alice", False)
    assert store.get_by_id(user.id).session_version == 3
    assert store.set_active("alice", True)
    assert store.get_by_id(user.id).session_version == 4
    assert store.unlock("alice")
    assert store.get_by_id(user.id).session_version == 4


def test_role_allows_is_fail_closed():
    assert role_allows(ROLE_ADMIN, ROLE_VIEWER)
    assert role_allows(ROLE_OPERATOR, ROLE_VIEWER)
    assert not role_allows(ROLE_VIEWER, ROLE_OPERATOR)
    assert not role_allows("unknown", ROLE_VIEWER)


# ---------- 审计 Store ----------

def test_audit_log_truncation_and_filters(db_session_factory):
    store = AdminAuditStore(db_session_factory)
    store.log(admin_user_id=1, username="Admin", role="admin",
              action="review.approve", target_type="review_queue", target_id=3,
              detail={"review_id": 3}, ip="127.0.0.1", user_agent="pytest" * 100)
    store.log(admin_user_id=None, username="unknown", role="anonymous",
              action="admin.login_failed", target_type="admin_session",
              result="失败")

    items = store.list_logs()
    assert len(items) == 2
    assert items[0]["action"] == "admin.login_failed"  # id desc
    assert items[0]["result"] == "失败"
    assert items[0]["detail"] == {}

    filtered = store.list_logs(action="review.approve")
    assert len(filtered) == 1
    assert filtered[0]["username"] == "admin"
    assert filtered[0]["target_id"] == "3"
    assert len(filtered[0]["user_agent"]) == 255
    assert store.list_logs(username="ADMIN")[0]["action"] == "review.approve"


def test_audit_best_effort_does_not_raise(db_session_factory, monkeypatch):
    store = AdminAuditStore(db_session_factory)

    def boom(**kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr(store, "log", boom)
    store.log_best_effort(admin_user_id=None, username="x", role="anonymous",
                          action="admin.login", target_type="admin_session")


# ---------- 登录 / 会话 / 登出 ----------

async def test_login_sets_cookie_and_session(make_client):
    client, app = await make_client(fake_chat("ok"))
    seed_admin(app, "admin", "admin-pass-123", "admin")

    response = await client.post("/api/admin/login",
                                 json={"username": "ADMIN", "password": "admin-pass-123"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["username"] == "admin" and body["role"] == "admin"
    assert body["redirect_to"] == "/admin/review"
    cookie = response.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=lax" in cookie and "path=/" in cookie

    session = await client.get("/api/admin/session")
    assert session.status_code == 200
    assert session.json()["username"] == "admin"
    assert session.json()["csrf_token"] == body["csrf_token"]
    assert session.json()["can_write"] is True


async def test_login_failure_is_generic_and_audited(make_client):
    client, app = await make_client(fake_chat("ok"))
    seed_admin(app, "admin", "admin-pass-123", "admin")

    wrong = await client.post("/api/admin/login",
                              json={"username": "admin", "password": "wrong-pass-123"})
    unknown = await client.post("/api/admin/login",
                                json={"username": "nobody", "password": "wrong-pass-123"})
    assert wrong.status_code == 401 and wrong.json()["detail"] == "用户名或密码错误"
    assert unknown.status_code == 401 and unknown.json()["detail"] == "用户名或密码错误"

    rows = app.state.admin_audit_store.list_logs(action="admin.login_failed")
    assert len(rows) == 2
    assert rows[0]["admin_user_id"] is None  # 最新一条是未知用户
    assert rows[1]["admin_user_id"] is not None
    assert all(row["result"] == "失败" for row in rows)


async def test_login_503_without_secret(make_client, monkeypatch):
    client, app = await make_client(fake_chat("ok"))
    monkeypatch.setattr(app.state.settings, "admin_session_secret", "")
    response = await client.post("/api/admin/login",
                                 json={"username": "admin", "password": "admin-pass-123"})
    assert response.status_code == 503
    assert "ADMIN_SESSION_SECRET" in response.json()["detail"]


async def test_login_rejects_cross_origin(make_client):
    client, app = await make_client(fake_chat("ok"))
    seed_admin(app, "admin", "admin-pass-123", "admin")
    response = await client.post(
        "/api/admin/login", json={"username": "admin", "password": "admin-pass-123"},
        headers={"Origin": "https://evil.example"})
    assert response.status_code == 403


async def test_logout_requires_csrf_and_clears_cookie(make_client):
    client, app = await make_client(fake_chat("ok"))
    seed_admin(app, "admin", "admin-pass-123", "admin")
    csrf = await admin_login(client)

    no_csrf = await client.post("/api/admin/logout")
    assert no_csrf.status_code == 403
    logged_out = await client.post("/api/admin/logout", headers=csrf_headers(csrf))
    assert logged_out.status_code == 200
    assert logged_out.json()["logged_out"] is True
    assert (await client.get("/api/admin/session")).status_code == 401


async def test_session_requires_login(make_client):
    client, _ = await make_client(fake_chat("ok"))
    assert (await client.get("/api/admin/session")).status_code == 401


async def test_audit_api_is_admin_only(make_client):
    client, app = await make_client(fake_chat("ok"))
    seed_admin(app, "op", "operator-pass-123", "operator")
    seed_admin(app, "adm", "admin-pass-123", "admin")

    csrf = await admin_login(client, "op", "operator-pass-123")
    assert (await client.get("/api/admin/audit")).status_code == 403
    await client.post("/api/admin/logout", headers=csrf_headers(csrf))

    await admin_login(client, "adm", "admin-pass-123")
    response = await client.get("/api/admin/audit")
    assert response.status_code == 200
    assert any(item["action"] == "admin.login" for item in response.json()["items"])


async def test_expired_and_tampered_cookie_are_rejected(make_client):
    client, app = await make_client(fake_chat("ok"))
    seed_admin(app, "admin", "admin-pass-123", "admin")
    await admin_login(client)
    user = app.state.admin_store.get_by_username("admin")
    now = int(time.time())
    expired = sign_session({
        "uid": user.id, "username": user.username, "role": user.role,
        "ver": user.session_version, "csrf": "csrf", "iat": now - 120, "exp": now - 1,
    }, app.state.settings.admin_session_secret)
    client.cookies.clear()
    client.cookies.set(SESSION_COOKIE, expired)
    assert (await client.get("/api/admin/session")).status_code == 401

    login = await client.post("/api/admin/login",
                              json={"username": "admin", "password": "admin-pass-123"})
    assert login.status_code == 200
    valid = login.cookies.get(SESSION_COOKIE)
    body, signature = valid.split(".")
    tampered_sig = ("A" if signature[0] != "A" else "B") + signature[1:]
    client.cookies.clear()
    client.cookies.set(SESSION_COOKIE, f"{body}.{tampered_sig}")
    assert (await client.get("/api/admin/session")).status_code == 401


async def test_disabled_user_and_version_bump_invalidate_cookie(make_client):
    client, app = await make_client(fake_chat("ok"))
    seed_admin(app, "admin", "admin-pass-123", "admin")
    await admin_login(client)
    app.state.admin_store.set_active("admin", False)
    assert (await client.get("/api/admin/session")).status_code == 401

    app.state.admin_store.set_active("admin", True)
    await admin_login(client)
    app.state.admin_store.reset_password("admin", "new-admin-pass-123")
    assert (await client.get("/api/admin/session")).status_code == 401


async def test_secure_cookie_flag(make_client):
    client, app = await make_client(fake_chat("ok"))
    seed_admin(app, "admin", "admin-pass-123", "admin")
    app.state.settings.admin_cookie_secure = True
    response = await client.post("/api/admin/login",
                                 json={"username": "admin", "password": "admin-pass-123"})
    assert response.status_code == 200
    assert "secure" in response.headers["set-cookie"].lower()


# ---------- 页面与后台 API 鉴权 ----------

@pytest.mark.parametrize("path", ["/admin/review", "/admin/evals", "/admin/topics"])
async def test_admin_pages_redirect_when_unauthenticated(make_client, path):
    client, _ = await make_client(fake_chat("ok"))
    response = await client.get(path)
    assert response.status_code == 303
    assert response.headers["location"].startswith("/admin/login?next=")
    assert quote(path, safe="") in response.headers["location"]


async def test_login_page_and_shared_js_are_public(make_client):
    client, _ = await make_client(fake_chat("ok"))
    page = await client.get("/admin/login")
    assert page.status_code == 200 and "后台登录" in page.text
    script = await client.get("/admin/assets/admin.js")
    assert script.status_code == 200 and "AdminAuth" in script.text


async def test_admin_pages_accessible_after_login(make_client):
    client, app = await make_client(fake_chat("ok"))
    seed_admin(app, "viewer", "viewer-pass-123", "viewer")
    await admin_login(client, "viewer", "viewer-pass-123")
    for path in ("/admin/review", "/admin/evals", "/admin/topics"):
        response = await client.get(path)
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "/admin/assets/admin.js" in response.text


@pytest.mark.parametrize("path", [
    "/api/review-queue",
    "/api/review-queue/1",
    "/api/topics/distribution",
    "/api/eval-runs",
    "/api/usage/by-intent",
])
async def test_read_apis_require_login(make_client, path):
    client, _ = await make_client(fake_chat("ok"))
    response = await client.get(path)
    assert response.status_code == 401
    assert response.json()["detail"] == "请先登录"


@pytest.mark.parametrize("path,payload", [
    ("/api/review-queue/1/approve", {"approved_answer": "答案"}),
    ("/api/review-queue/1/reject", None),
    ("/api/topics/classify-batch", {"limit": 1}),
])
async def test_write_apis_require_login(make_client, path, payload):
    client, _ = await make_client(fake_chat("ok"))
    response = await (client.post(path, json=payload) if payload is not None
                      else client.post(path))
    assert response.status_code == 401


async def test_viewer_cannot_write(make_client):
    client, app = await make_client(fake_chat("ok"))
    seed_admin(app, "viewer", "viewer-pass-123", "viewer")
    csrf = await admin_login(client, "viewer", "viewer-pass-123")
    for path, payload in [
        ("/api/review-queue/1/approve", {"approved_answer": "答案"}),
        ("/api/review-queue/1/reject", None),
        ("/api/topics/classify-batch", {"limit": 1}),
    ]:
        response = await (client.post(path, json=payload, headers=csrf_headers(csrf))
                          if payload is not None
                          else client.post(path, headers=csrf_headers(csrf)))
        assert response.status_code == 403
        assert response.json()["detail"] == "权限不足"


async def test_operator_writes_with_csrf_and_audit(make_client):
    from app.flywheel import FlywheelService
    from app.store import TopicStore
    from app.topic_classifier import StubTopicClassifier

    client, app = await make_client(fake_chat("ok"))
    app.state.flywheel = FlywheelService(app.state.session_factory)
    app.state.topic_store = TopicStore(app.state.session_factory)
    app.state.topic_classifier = StubTopicClassifier({})
    seed_admin(app, "op", "operator-pass-123", "operator")
    csrf = await admin_login(client, "op", "operator-pass-123")

    approve_id = app.state.flywheel.insert("retrieval_low_conf", 1, "会飞的手机怎么买", "证据置信度低")
    reject_id = app.state.flywheel.insert("self_check", 1, "积分怎么兑换", "模型自评")

    approved = await client.post(f"/api/review-queue/{approve_id}/approve",
                                 json={"approved_answer": "手机目前不能飞。"},
                                 headers=csrf_headers(csrf))
    assert approved.status_code == 200
    rejected = await client.post(f"/api/review-queue/{reject_id}/reject",
                                 headers=csrf_headers(csrf))
    assert rejected.status_code == 200
    classified = await client.post("/api/topics/classify-batch", json={"limit": 5},
                                   headers=csrf_headers(csrf))
    assert classified.status_code == 200

    rows = app.state.admin_audit_store.list_logs()
    actions = [row["action"] for row in rows]
    assert "review.approve" in actions
    assert "review.reject" in actions
    assert "topics.classify_batch" in actions
    approve_row = next(row for row in rows if row["action"] == "review.approve")
    assert approve_row["target_id"] == str(approve_id)
    assert approve_row["username"] == "op"
    assert "approved_answer" not in (approve_row["detail"] or {})
    assert approve_row["detail"]["answer_chars"] == len("手机目前不能飞。")


async def test_operator_write_without_csrf_is_forbidden(make_client):
    client, app = await make_client(fake_chat("ok"))
    seed_admin(app, "op", "operator-pass-123", "operator")
    await admin_login(client, "op", "operator-pass-123")
    response = await client.post("/api/topics/classify-batch", json={"limit": 1})
    assert response.status_code == 403
    assert "CSRF" in response.json()["detail"]


async def test_audit_failure_does_not_block_write(make_client, monkeypatch):
    from app.flywheel import FlywheelService

    client, app = await make_client(fake_chat("ok"))
    app.state.flywheel = FlywheelService(app.state.session_factory)
    seed_admin(app, "op", "operator-pass-123", "operator")
    csrf = await admin_login(client, "op", "operator-pass-123")
    review_id = app.state.flywheel.insert("retrieval_low_conf", 1, "测试审计失败", "reason")

    def boom(**kwargs):
        raise RuntimeError("audit down")

    monkeypatch.setattr(app.state.admin_audit_store, "log", boom)
    response = await client.post(f"/api/review-queue/{review_id}/approve",
                                 json={"approved_answer": "答案仍然写入。"},
                                 headers=csrf_headers(csrf))
    assert response.status_code == 200
    assert response.json()["approved"] is True
