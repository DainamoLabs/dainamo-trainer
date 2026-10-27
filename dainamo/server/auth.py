"""Sign-in, for when the server can be reached from other machines.

``python -m dainamo --host 0.0.0.0`` turns it on with a generated access key, kept in data/auth.json
and printed at startup; ``--password`` (or DAINAMO_PASSWORD) sets your own instead. A browser signs in
once and keeps a cookie for 30 days; scripts can send ``Authorization: Bearer <key>``. Opening any page
with ``?key=<key>`` signs that browser in, so the printed link works on a phone.

Only the API is protected: the web UI's own files load without signing in, and hold no data."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import secrets
from urllib.parse import parse_qsl, urlencode, urlsplit

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel
from starlette.datastructures import Headers

from . import db

COOKIE = "dainamo_session"
MAX_AGE = 30 * 24 * 3600
KEY_FILE = os.path.join(db.DATA_DIR, "auth.json")
OPEN_PATHS = {"/api/auth", "/api/auth/login", "/api/auth/logout"}
PROTECTED_PAGES = {"/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"}

_password: str | None = None  # None: sign-in is off
_secret = b""


def _stored() -> dict:
    try:
        with open(KEY_FILE) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _save(data: dict) -> None:
    os.makedirs(db.DATA_DIR, exist_ok=True)
    with open(KEY_FILE + ".tmp", "w") as f:
        json.dump(data, f)
    if os.name != "nt":
        os.chmod(KEY_FILE + ".tmp", 0o600)
    os.replace(KEY_FILE + ".tmp", KEY_FILE)


def generated_key(new: bool = False) -> str:
    """This install's access key, created on first use (or replaced with ``new``)."""
    data = _stored()
    if new or not data.get("key"):
        data["key"] = secrets.token_urlsafe(18)
        _save(data)
    return data["key"]


def enable(password: str | None) -> None:
    """Require ``password`` for the API from now on; None turns sign-in off."""
    global _password, _secret
    if password is not None and not password:
        raise ValueError("The password can't be empty")
    data = _stored()
    if password is not None and not data.get("secret"):
        data["secret"] = secrets.token_hex(32)
        _save(data)
    _password, _secret = password, bytes.fromhex(data.get("secret", ""))


def enabled() -> bool:
    return _password is not None


def _session_value() -> str:
    # changing the password signs every browser out
    return hmac.new(_secret, _password.encode(), hashlib.sha256).hexdigest()


def _matches(given: str | None) -> bool:
    return given is not None and hmac.compare_digest(given.encode(), _password.encode())


def signed_in(request: Request) -> bool:
    if not enabled():
        return True
    cookie = request.cookies.get(COOKIE)
    if cookie and hmac.compare_digest(cookie, _session_value()):
        return True
    header = request.headers.get("authorization", "")
    return header.lower().startswith("bearer ") and _matches(header[7:].strip())


def _set_cookie(response: Response) -> None:
    response.set_cookie(COOKIE, _session_value(), max_age=MAX_AGE, httponly=True, samesite="lax", path="/")


class AuthMiddleware:
    """Answers 401 to API requests without a session; turns ``?key=`` on a page into a session."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not enabled():
            return await self.app(scope, receive, send)
        request = Request(scope)
        path = scope["path"]
        if (path.startswith("/api/") or path in PROTECTED_PAGES) and path not in OPEN_PATHS:
            if not signed_in(request):
                return await JSONResponse({"detail": "Sign in to continue"}, status_code=401)(scope, receive, send)
        elif "key" in request.query_params:
            query = [(k, v) for k, v in parse_qsl(scope.get("query_string", b"").decode()) if k != "key"]
            # one leading slash: "//evil.example/x" would send the browser to another site
            target = "/" + path.lstrip("/\\") + (f"?{urlencode(query)}" if query else "")
            response = RedirectResponse(target, status_code=303)
            if _matches(request.query_params["key"]):
                _set_cookie(response)
            else:
                await asyncio.sleep(1)
            return await response(scope, receive, send)
        return await self.app(scope, receive, send)


SAFE_METHODS = ("GET", "HEAD", "OPTIONS")


class CrossSiteGuard:
    """Refuses requests that change something (POST, PUT, DELETE...) when a browser says another site sent them.
    Without it, any web page open on this machine could make the browser start, stop and upload here: the host check
    doesn't stop that, and with sign-in off there's no cookie to withhold. ``Sec-Fetch-Site`` is what browsers say
    (a reverse proxy doesn't change it); for one that doesn't send it, ``Origin`` must name this server. Scripts and
    curl send neither, and are let through as before."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["method"] not in SAFE_METHODS and scope["path"].startswith("/api/"):
            headers = Headers(scope=scope)
            site = headers.get("sec-fetch-site")
            if site is not None:
                refused = site not in ("same-origin", "none")
            else:
                origin = headers.get("origin")
                hosts = {headers.get("host"), headers.get("x-forwarded-host")} - {None}
                refused = origin is not None and urlsplit(origin).netloc not in hosts
            if refused:
                response = JSONResponse({"detail": "Requests from other websites aren't accepted"}, status_code=403)
                return await response(scope, receive, send)
        return await self.app(scope, receive, send)


router = APIRouter(prefix="/api/auth")


@router.get("")
def auth_status(request: Request):
    return {"required": enabled(), "signed_in": signed_in(request)}


class LoginIn(BaseModel):
    password: str


@router.post("/login")
async def login(body: LoginIn, response: Response):
    if not enabled():
        return {"ok": True}
    if not _matches(body.password.strip()):
        await asyncio.sleep(1)  # slows down guessing
        raise HTTPException(401, "Wrong password or key")
    _set_cookie(response)
    return {"ok": True}


@router.post("/logout")
def logout(response: Response):
    response.delete_cookie(COOKIE, path="/")
    return {"ok": True}
