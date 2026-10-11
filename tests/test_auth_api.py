"""JWT 登录鉴权测试：注册 / 登录 / 刷新 / 退出 / 鉴权保护（临时 SQLite，不依赖 GPU）。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from injection_guard.config import get_settings
from injection_guard.persistence.database import get_database

USER = {"username": "alice", "password": "s3cret-pass"}


def _client(tmp_path, monkeypatch, **env) -> TestClient:
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'auth.db'}")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    get_database.cache_clear()
    from injection_guard.api.app import create_app

    return TestClient(create_app())


@pytest.fixture()
def client(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as test_client:
        yield test_client
    get_settings.cache_clear()
    get_database.cache_clear()


def test_register_login_and_me(client: TestClient) -> None:
    registered = client.post("/api/auth/register", json=USER)
    assert registered.status_code == 201
    body = registered.json()
    assert body["user"]["username"] == "alice"
    assert "password" not in body["user"] and "password_hash" not in body["user"]
    assert client.cookies.get("ig_refresh_token")  # refresh 走 httpOnly Cookie

    token = body["access_token"]
    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["username"] == "alice"

    logged_in = client.post("/api/auth/login", json=USER)
    assert logged_in.status_code == 200
    assert logged_in.json()["access_token"]


def test_duplicate_username_conflicts(client: TestClient) -> None:
    assert client.post("/api/auth/register", json=USER).status_code == 201
    assert client.post("/api/auth/register", json=USER).status_code == 409


def test_weak_password_rejected(client: TestClient) -> None:
    resp = client.post("/api/auth/register", json={"username": "bob", "password": "short"})
    assert resp.status_code == 400
    assert "error" in resp.json()


def test_login_failure_is_generic(client: TestClient) -> None:
    client.post("/api/auth/register", json=USER)
    wrong_password = client.post(
        "/api/auth/login", json={"username": "alice", "password": "nope-nope-nope"}
    )
    unknown_user = client.post(
        "/api/auth/login", json={"username": "ghost", "password": "nope-nope-nope"}
    )
    assert wrong_password.status_code == 401
    assert unknown_user.status_code == 401
    assert wrong_password.json() == unknown_user.json()  # 不泄露账号是否存在


def test_requires_authentication(client: TestClient) -> None:
    assert client.get("/api/sessions").status_code == 401
    assert client.get("/api/auth/me").status_code == 401
    bad = client.get("/api/auth/me", headers={"Authorization": "Bearer not-a-jwt"})
    assert bad.status_code == 401


def test_refresh_rotates_token(client: TestClient) -> None:
    client.post("/api/auth/register", json=USER)
    refreshed = client.post("/api/auth/refresh", json={})
    assert refreshed.status_code == 200
    token = refreshed.json()["access_token"]
    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200


def test_refresh_rejected_without_cookie(client: TestClient) -> None:
    client.cookies.clear()
    assert client.post("/api/auth/refresh", json={}).status_code == 401


def test_logout_clears_refresh_cookie(client: TestClient) -> None:
    client.post("/api/auth/register", json=USER)
    assert client.post("/api/auth/logout", json={}).status_code == 200
    assert not client.cookies.get("ig_refresh_token")
    assert client.post("/api/auth/refresh", json={}).status_code == 401


def test_logout_all_devices_revokes_existing_tokens(client: TestClient) -> None:
    token = client.post("/api/auth/register", json=USER).json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    assert client.get("/api/auth/me", headers=headers).status_code == 200

    assert client.post("/api/auth/logout", json={"all_devices": True}, headers=headers).status_code == 200
    # 令牌版本已递增，旧 access token 立即失效
    assert client.get("/api/auth/me", headers=headers).status_code == 401


def test_registration_can_be_disabled(tmp_path, monkeypatch) -> None:
    with _client(tmp_path, monkeypatch, ALLOW_REGISTRATION="false") as test_client:
        assert test_client.post("/api/auth/register", json=USER).status_code == 403
    get_settings.cache_clear()
    get_database.cache_clear()
