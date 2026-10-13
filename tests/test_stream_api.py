"""``POST /api/chat_stream`` SSE 端点测试（桩服务，不加载模型）。"""

from __future__ import annotations

import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

from injection_guard.api.app import create_app
from injection_guard.api.dependencies import ServiceBusyError, get_defense_service
from injection_guard.api.security import get_current_user


class _StubService:
    def __init__(self, events):
        self._events = events

    def stream_detect(self, **_kw):
        yield from self._events


class _BusyService:
    def stream_detect(self, **_kw):
        raise ServiceBusyError("busy")
        yield  # pragma: no cover


def _client(service) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_defense_service] = lambda: service
    # 鉴权已由 auth 测试覆盖，这里只测 SSE：桩掉当前用户（路由需要 user.id）
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id="stub-user")
    return TestClient(app)


def _parse_sse(text: str):
    """解析 SSE 文本为 ``[(event, data), ...]``（忽略心跳注释）。"""
    events = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block or block.startswith(":"):
            continue
        event = None
        data_lines = []
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].strip())
        if event is not None:
            events.append((event, json.loads("\n".join(data_lines))))
    return events


def test_stream_returns_event_sequence():
    events = [
        ("input_detection", {"is_safe": True}),
        ("rag", {"enabled": False}),
        ("delta", {"content": "你"}),
        ("delta", {"content": "好"}),
        ("done", {"is_safe": True, "stopped_at": None, "model_response": "你好"}),
    ]
    client = _client(_StubService(events))
    resp = client.post("/api/chat_stream", json={"user_input": "你好"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    parsed = _parse_sse(resp.text)
    assert [name for name, _ in parsed] == [name for name, _ in events]
    assert parsed[-1][1]["model_response"] == "你好"


def test_stream_missing_user_input_returns_400_json():
    client = _client(_StubService([]))
    resp = client.post("/api/chat_stream", json={"model_response": "x"})
    assert resp.status_code == 400
    assert resp.json()["error"]


def test_stream_emits_error_event_on_busy():
    client = _client(_BusyService())
    resp = client.post("/api/chat_stream", json={"user_input": "你好"})
    assert resp.status_code == 200
    parsed = _parse_sse(resp.text)
    assert parsed[-1] == ("error", {"error": "service_busy"})


# ── 生成结束时服务端落库助手正文（不再依赖前端 PATCH）────────────────────


def _persist_client(service, user_id: str) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_defense_service] = lambda: service
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=user_id)
    return TestClient(app)


def test_stream_persists_assistant_content_from_done_event(tmp_path, monkeypatch):
    """done 事件里的 model_response 由服务端写回会话，刷新页面也能看到回复。"""
    from injection_guard.config import get_settings
    from injection_guard.persistence.database import get_database
    from injection_guard.persistence.repository import SessionRepository

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'stream.db'}")
    get_settings.cache_clear()
    get_database.cache_clear()
    try:
        events = [
            ("delta", {"content": "你好"}),
            ("done", {"is_safe": True, "stopped_at": None, "model_response": "你好！很高兴见到你"}),
        ]
        with _persist_client(_StubService(events), "u1") as client:
            db = get_database()  # create_app 已建表
            with db.session() as session:
                repo = SessionRepository(session, user_id="u1")
                chat = repo.create_session(title="t")
                message = repo.append_message(chat.id, "assistant")

            resp = client.post(
                "/api/chat_stream",
                json={
                    "user_input": "你好",
                    "session_id": chat.id,
                    "assistant_message_id": message.id,
                },
            )

        assert resp.status_code == 200
        with db.session() as session:
            stored = SessionRepository(session, user_id="u1").get_session(chat.id)
            assert stored.messages[0].content == "你好！很高兴见到你"
            assert stored.messages[0].status == "done"
    finally:
        get_settings.cache_clear()
        get_database.cache_clear()


def test_stream_persists_output_blocked_text(tmp_path, monkeypatch):
    """输出检测拦截：落库拦截文案，而不是模型原文。"""
    from injection_guard.agent.state import OUTPUT_BLOCKED_TEXT
    from injection_guard.config import get_settings
    from injection_guard.persistence.database import get_database
    from injection_guard.persistence.repository import SessionRepository

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'stream2.db'}")
    get_settings.cache_clear()
    get_database.cache_clear()
    try:
        events = [
            ("output_detection", {"is_safe": False, "failed_rules": ["R1"], "total_rules": 5}),
            ("done", {"is_safe": False, "stopped_at": "output_detection", "model_response": "危险原文"}),
        ]
        with _persist_client(_StubService(events), "u2") as client:
            db = get_database()
            with db.session() as session:
                repo = SessionRepository(session, user_id="u2")
                chat = repo.create_session(title="t")
                message = repo.append_message(chat.id, "assistant")

            resp = client.post(
                "/api/chat_stream",
                json={
                    "user_input": "你好",
                    "session_id": chat.id,
                    "assistant_message_id": message.id,
                },
            )

        assert resp.status_code == 200
        with db.session() as session:
            stored = SessionRepository(session, user_id="u2").get_session(chat.id)
            assert stored.messages[0].content == OUTPUT_BLOCKED_TEXT
    finally:
        get_settings.cache_clear()
        get_database.cache_clear()


def test_stream_without_session_ids_skips_persistence(tmp_path, monkeypatch):
    """未带会话定位（脚本 / 旧前端）时不做落库，接口行为不变。"""
    from injection_guard.config import get_settings
    from injection_guard.persistence.database import get_database

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'stream3.db'}")
    get_settings.cache_clear()
    get_database.cache_clear()
    try:
        events = [("done", {"is_safe": True, "stopped_at": None, "model_response": "hi"})]
        with _persist_client(_StubService(events), "u3") as client:
            resp = client.post("/api/chat_stream", json={"user_input": "你好"})
        assert resp.status_code == 200
        assert _parse_sse(resp.text)[-1][0] == "done"
    finally:
        get_settings.cache_clear()
        get_database.cache_clear()


def test_final_assistant_content_mapping():
    """终态 -> 落库正文：输入拦截 / 输出拦截 / 正常。"""
    from injection_guard.agent.state import (
        INPUT_BLOCKED_TEXT,
        OUTPUT_BLOCKED_TEXT,
        final_assistant_content,
    )

    assert final_assistant_content(model_response=None, stopped_at="input_detection") == INPUT_BLOCKED_TEXT
    assert final_assistant_content(model_response="危险原文", stopped_at="output_detection") == OUTPUT_BLOCKED_TEXT
    assert final_assistant_content(model_response="正常回复", stopped_at=None) == "正常回复"
    assert final_assistant_content(model_response="跳过输出检测", stopped_at="output_skipped") == "跳过输出检测"
