"""``POST /api/chat_stream`` SSE 端点测试（桩服务，不加载模型）。"""

from __future__ import annotations

import json

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
    # 鉴权已由 auth 测试覆盖，这里只测 SSE：桩掉当前用户
    app.dependency_overrides[get_current_user] = lambda: object()
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
