"""ch14 后台鉴权安全原语：口令哈希、会话签名、CSRF 与 next 路径校验。

只依赖标准库，不碰 FastAPI/DB，便于单测与复用。
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from urllib.parse import urlsplit

PBKDF2_ALGORITHM = "pbkdf2_sha256"
PBKDF2_ITERATIONS = 600_000
PBKDF2_MAX_ITERATIONS = 5_000_000
SESSION_COOKIE = "shophelper_admin_session"
SESSION_MAX_TOKEN_LEN = 4096


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64d(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def hash_password(password: str, *, iterations: int = PBKDF2_ITERATIONS) -> str:
    """生成 pbkdf2_sha256$迭代次数$salt$hash；空口令与非法迭代次数直接拒绝。"""
    if not isinstance(password, str) or not password:
        raise ValueError("口令不能为空")
    iterations = int(iterations)
    if not 1 <= iterations <= PBKDF2_MAX_ITERATIONS:
        raise ValueError("PBKDF2 迭代次数不合法")
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"{PBKDF2_ALGORITHM}${iterations}${_b64e(salt)}${_b64e(digest)}"


def verify_password(password: str, encoded: str) -> bool:
    """校验口令；任何格式/解码异常一律返回 False，不向调用方抛异常。"""
    try:
        algorithm, iterations_s, salt_s, digest_s = encoded.split("$")
        if algorithm != PBKDF2_ALGORITHM:
            return False
        iterations = int(iterations_s)
        if not 1 <= iterations <= PBKDF2_MAX_ITERATIONS:
            return False
        salt = _b64d(salt_s)
        expected = _b64d(digest_s)
        if not salt or not expected:
            return False
        candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        return hmac.compare_digest(candidate, expected)
    except Exception:
        return False


def sign_session(payload: dict, secret: str) -> str:
    """payload 紧凑 JSON → base64url，再对 payload 文本做 HMAC-SHA256。"""
    if not secret:
        raise ValueError("会话密钥不能为空")
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True,
                      ensure_ascii=False).encode("utf-8")
    payload_b64 = _b64e(body)
    signature = hmac.new(secret.encode("utf-8"), payload_b64.encode("ascii"),
                         hashlib.sha256).digest()
    return f"{payload_b64}.{_b64e(signature)}"


def verify_session(token: str, secret: str, *, now: int | None = None) -> dict | None:
    """验签 + 验过期；任何异常返回 None，调用方按未登录处理。"""
    if not secret or not token or len(token) > SESSION_MAX_TOKEN_LEN:
        return None
    try:
        payload_b64, sig_b64 = token.split(".")
        expected = hmac.new(secret.encode("utf-8"), payload_b64.encode("ascii"),
                            hashlib.sha256).digest()
        if not hmac.compare_digest(expected, _b64d(sig_b64)):
            return None
        payload = json.loads(_b64d(payload_b64).decode("utf-8"))
        if not isinstance(payload, dict):
            return None
        current = int(time.time()) if now is None else int(now)
        exp = int(payload.get("exp", 0))
        iat = int(payload.get("iat", 0))
        if exp <= current or iat > current + 60:
            return None
        if type(payload.get("uid")) is not int or payload["uid"] <= 0:
            return None
        if type(payload.get("ver")) is not int:
            return None
        if not isinstance(payload.get("csrf"), str) or not payload["csrf"]:
            return None
        return payload
    except Exception:
        return None


def new_csrf_token() -> str:
    return secrets.token_urlsafe(32)


def safe_next_path(value: str | None, *, default: str = "/admin/review") -> str:
    """只允许登录后跳回 /admin/ 站内路径，拒绝外站、协议相对、反斜杠与控制字符。"""
    if not value:
        return default
    value = value.strip()
    if not value or len(value) > 2048:
        return default
    if any(ch in value for ch in ("\r", "\n", "\x00")):
        return default
    if "\\" in value:
        return default
    lowered = value.lower()
    if any(seq in lowered for seq in ("%0d", "%0a", "%00")):
        return default
    if not value.startswith("/admin/") or value.startswith("//"):
        return default
    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc:
        return default
    path = parsed.path
    if not path.startswith("/admin/"):
        return default
    if path == "/admin/login" or path.startswith("/admin/assets/"):
        return default
    return path + (("?" + parsed.query) if parsed.query else "")
