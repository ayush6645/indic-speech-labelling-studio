"""Authentication and strict local-vs-ngrok request detection."""
import datetime as dt
import secrets
from fastapi import HTTPException, Request

import bcrypt

from labelling_tool import config, db

SESSION_COOKIE = "lt_session"
SESSION_DAYS = 30
_PROXY_HEADERS = (
    "x-forwarded-for",
    "x-forwarded-host",
    "x-forwarded-proto",
    "x-real-ip",
    "forwarded",
    "ngrok-trace-id",
    "ngrok-skip-browser-warning",
)
_LOCAL_IPS = {"127.0.0.1", "::1", "::ffff:127.0.0.1", "localhost"}
_LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]"}


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("ascii")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("ascii"))
    except Exception:
        return False


def is_local_request(request: Request) -> bool:
    """True ONLY when the browser truly connects directly on localhost/127.0.0.1 (never over ngrok/proxies).
    ngrok runs locally and opens a TCP connection to 127.0.0.1, so we check both Host and proxy headers.
    """
    client_ip = (request.client.host if request.client else "") or ""
    if client_ip not in _LOCAL_IPS:
        return False
    hdrs = request.headers
    for h in _PROXY_HEADERS:
        if h in hdrs:
            return False
    host = hdrs.get("host", "").split(":")[0].lower().strip()
    if host not in _LOCAL_HOSTS:
        return False
    return True


def ensure_local_user(username: str = config.DEFAULT_LOCAL_USER):
    """Make sure a local admin user record exists (no password needed for localhost auto-login)."""
    with db._WRITE_LOCK:
        conn = db.connect()
        try:
            row = conn.execute("SELECT username FROM users WHERE username = ?", (username,)).fetchone()
            if not row:
                # Unusable random password until explicitly set for remote login
                pw_hash = hash_password(secrets.token_urlsafe(32))
                conn.execute(
                    "INSERT INTO users (username, display_name, password_hash, role, active, created_at) "
                    "VALUES (?, ?, ?, 'admin', 1, ?)",
                    (username, username.capitalize(), pw_hash, db.now_iso()),
                )
                conn.commit()
        finally:
            conn.close()


def create_user(username: str, password: str, display_name: str = "", role: str = "reviewer") -> dict:
    username = username.strip().lower()
    if not username or len(username) < 2 or not all(c.isalnum() or c in "._-" for c in username):
        raise ValueError("Username must be at least 2 characters (letters, digits, . _ -)")
    if len(password) < 4:
        raise ValueError("Password must be at least 4 characters")
    if role not in ("admin", "reviewer"):
        raise ValueError("Role must be 'admin' or 'reviewer'")
    with db._WRITE_LOCK:
        conn = db.connect()
        try:
            conn.execute(
                "INSERT INTO users (username, display_name, password_hash, role, active, created_at) "
                "VALUES (?, ?, ?, ?, 1, ?) "
                "ON CONFLICT(username) DO UPDATE SET display_name=excluded.display_name, "
                "password_hash=excluded.password_hash, role=excluded.role, active=1",
                (username, display_name.strip() or username, hash_password(password), role, db.now_iso()),
            )
            conn.commit()
            return {"username": username, "display_name": display_name.strip() or username, "role": role, "active": True}
        finally:
            conn.close()


def set_user_active(username: str, active: bool):
    with db._WRITE_LOCK:
        conn = db.connect()
        try:
            conn.execute("UPDATE users SET active = ? WHERE username = ?", (1 if active else 0, username))
            if not active:
                conn.execute("DELETE FROM sessions WHERE username = ?", (username,))
            conn.commit()
        finally:
            conn.close()


def list_users() -> list[dict]:
    conn = db.connect()
    try:
        rows = conn.execute(
            "SELECT username, display_name, role, active, created_at FROM users ORDER BY role, username"
        ).fetchall()
        return [dict(r) | {"active": bool(r["active"])} for r in rows]
    finally:
        conn.close()


def authenticate_user(username: str, password: str) -> dict | None:
    username = (username or "").strip().lower()
    conn = db.connect()
    try:
        row = conn.execute(
            "SELECT username, display_name, password_hash, role, active FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        if not row or not row["active"]:
            return None
        if not verify_password(password or "", row["password_hash"]):
            return None
        return {"username": row["username"], "display_name": row["display_name"], "role": row["role"]}
    finally:
        conn.close()


def create_session(username: str) -> str:
    token = secrets.token_urlsafe(32)
    now = dt.datetime.now()
    exp = now + dt.timedelta(days=SESSION_DAYS)
    with db._WRITE_LOCK:
        conn = db.connect()
        try:
            conn.execute(
                "INSERT INTO sessions (token, username, created_at, expires_at) VALUES (?, ?, ?, ?)",
                (token, username, now.isoformat(timespec="seconds"), exp.isoformat(timespec="seconds")),
            )
            conn.commit()
        finally:
            conn.close()
    return token


def delete_session(token: str):
    if not token:
        return
    with db._WRITE_LOCK:
        conn = db.connect()
        try:
            conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
            conn.commit()
        finally:
            conn.close()


def user_from_cookie(token: str | None) -> dict | None:
    if not token:
        return None
    conn = db.connect()
    try:
        row = conn.execute("""
            SELECT u.username, u.display_name, u.role, u.active, s.expires_at
            FROM sessions s
            JOIN users u ON u.username = s.username
            WHERE s.token = ?
        """, (token,)).fetchone()
        if not row or not row["active"]:
            return None
        if row["expires_at"] < dt.datetime.now().isoformat(timespec="seconds"):
            return None
        return {"username": row["username"], "display_name": row["display_name"], "role": row["role"]}
    finally:
        conn.close()


def get_current_user(request: Request, local_user: str = config.DEFAULT_LOCAL_USER) -> dict:
    """Resolve authenticated user; auto-login ONLY on genuine local requests."""
    token = request.cookies.get(SESSION_COOKIE)
    u = user_from_cookie(token)
    if u:
        u["local"] = is_local_request(request)
        return u
    if is_local_request(request):
        ensure_local_user(local_user)
        conn = db.connect()
        try:
            row = conn.execute(
                "SELECT username, display_name, role FROM users WHERE username = ?", (local_user,)
            ).fetchone()
            if row:
                return {"username": row["username"], "display_name": row["display_name"], "role": row["role"], "local": True}
        finally:
            conn.close()
        return {"username": local_user, "display_name": local_user.capitalize(), "role": "admin", "local": True}
    raise HTTPException(status_code=401, detail="Authentication required")
