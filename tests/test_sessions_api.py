"""会话持久化 API 测试：临时 PostgreSQL schema，不依赖 GPU / 模型。

覆盖：会话与消息的完整生命周期、检测记录写入、标题自动生成、
分页列表、删除级联、清空、**跨用户隔离**，以及未登录返回 401。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from injection_guard.config import get_settings
from injection_guard.persistence.database import get_database

PASSWORD = "s3cret-pass"


def _client(postgres_database_url, monkeypatch) -> TestClient:
    monkeypatch.setenv("DATABASE_URL", postgres_database_url)
    get_settings.cache_clear()
    get_database.cache_clear()
    from injection_guard.api.app import create_app

    return TestClient(create_app())


def _register(client: TestClient, username: str) -> dict:
    resp = client.post("/api/auth/register", json={"username": username, "password": PASSWORD})
    assert resp.status_code == 201
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


@pytest.fixture()
def client(postgres_database_url, monkeypatch):
    with _client(postgres_database_url, monkeypatch) as test_client:
        test_client.headers.update(_register(test_client, "owner"))
        yield test_client


def test_sessions_require_authentication(postgres_database_url, monkeypatch) -> None:
    with _client(postgres_database_url, monkeypatch) as test_client:
        assert test_client.get("/api/sessions").status_code == 401
        assert test_client.post("/api/sessions", json={}).status_code == 401


def test_session_and_message_lifecycle(client: TestClient) -> None:
    created = client.post("/api/sessions", json={})
    assert created.status_code == 201
    session_id = created.json()["id"]

    listing = client.get("/api/sessions").json()["sessions"]
    assert [item["id"] for item in listing] == [session_id]
    assert listing[0]["message_count"] == 0

    user_message = client.post(
        f"/api/sessions/{session_id}/messages",
        json={"role": "user", "content": "你好，帮我写一段代码"},
    )
    assert user_message.status_code == 201
    assert user_message.json()["status"] == "done"

    assistant = client.post(
        f"/api/sessions/{session_id}/messages", json={"role": "assistant", "content": ""}
    ).json()
    assert assistant["status"] == "streaming"
    assistant_id = assistant["id"]

    patched = client.patch(
        f"/api/sessions/{session_id}/messages/{assistant_id}",
        json={"content": "好的", "status": "done"},
    ).json()
    assert patched["content"] == "好的"
    assert patched["status"] == "done"

    detection = client.put(
        f"/api/sessions/{session_id}/messages/{assistant_id}/detection",
        json={"is_safe": True, "stopped_at": None, "input_detection": {"is_safe": True}},
    )
    assert detection.status_code == 200
    assert detection.json()["input_detection"]["is_safe"] is True

    detail = client.get(f"/api/sessions/{session_id}").json()
    assert detail["title"].startswith("你好")  # 首条用户消息自动生成标题
    assert [message["role"] for message in detail["messages"]] == ["user", "assistant"]
    assert detail["detections"][assistant_id]["is_safe"] is True

    renamed = client.patch(f"/api/sessions/{session_id}", json={"title": "改名后"}).json()
    assert renamed["title"] == "改名后"

    assert client.delete(f"/api/sessions/{session_id}").status_code == 204
    assert client.get(f"/api/sessions/{session_id}").status_code == 404


def test_sessions_are_isolated_per_user(client: TestClient) -> None:
    alice = _register(client, "alice")
    bob = _register(client, "bob")
    session_id = client.post("/api/sessions", json={}, headers=alice).json()["id"]
    client.post(
        f"/api/sessions/{session_id}/messages",
        json={"role": "user", "content": "hi"},
        headers=alice,
    )

    assert client.get("/api/sessions", headers=bob).json()["sessions"] == []
    assert client.get(f"/api/sessions/{session_id}", headers=bob).status_code == 404
    assert client.patch(
        f"/api/sessions/{session_id}", json={"title": "x"}, headers=bob
    ).status_code == 404
    assert client.delete(f"/api/sessions/{session_id}", headers=bob).status_code == 404
    assert client.get("/api/sessions", headers=alice).json()["sessions"] != []


def test_clear_sessions(client: TestClient) -> None:
    for _ in range(3):
        client.post("/api/sessions", json={})
    assert client.delete("/api/sessions").json()["deleted"] == 3
    assert client.get("/api/sessions").json()["sessions"] == []


def test_invalid_role_rejected(client: TestClient) -> None:
    session_id = client.post("/api/sessions", json={}).json()["id"]
    response = client.post(
        f"/api/sessions/{session_id}/messages", json={"role": "system", "content": "x"}
    )
    assert response.status_code == 400
    assert "error" in response.json()
