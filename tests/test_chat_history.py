"""多轮记忆：历史读取、token 裁剪、以及从路由注入到生成阶段的链路。

用临时 SQLite 与桩 ChatModel，不加载模型、不联网。
"""

from __future__ import annotations

import pytest
from fake_chat_model import RecordingChatModel

from injection_guard.config import get_settings
from injection_guard.persistence.database import get_database
from injection_guard.persistence.repository import SessionRepository
from injection_guard.services.chat_client import ChatClient

HISTORY = [
    {"role": "user", "content": "第一问"},
    {"role": "assistant", "content": "第一答"},
]


@pytest.fixture()
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'history.db'}")
    get_settings.cache_clear()
    get_database.cache_clear()
    database = get_database()
    database.create_all()
    yield database
    get_settings.cache_clear()
    get_database.cache_clear()


def _client(history_max_tokens: int = 2048) -> tuple[ChatClient, RecordingChatModel]:
    client = ChatClient(history_max_tokens=history_max_tokens)
    model = RecordingChatModel(replies=["hi"])
    client.model = model  # type: ignore[assignment]
    return client, model


# ── ChatClient：历史渲染与裁剪 ────────────────────────────────────────


def test_history_is_rendered_before_current_prompt() -> None:
    client, model = _client()
    client.complete("第二问", history=HISTORY)

    messages = model.calls[0]["messages"]
    assert [(m.type, m.content) for m in messages] == [
        ("human", "第一问"),
        ("ai", "第一答"),
        ("human", "第二问"),
    ]


def test_history_is_trimmed_to_token_budget_keeping_recent_messages() -> None:
    """预算只够最近两条：更早的历史被丢掉，本轮提问必须保留。"""
    client, model = _client(history_max_tokens=8)
    long_history = [
        {"role": "user", "content": "很长的历史问题" * 20},
        {"role": "assistant", "content": "很长的历史回答" * 20},
        {"role": "user", "content": "短问题"},
        {"role": "assistant", "content": "短回答"},
    ]
    client.complete("本轮提问", history=long_history)

    messages = model.calls[0]["messages"]
    assert messages[-1].content == "本轮提问"
    assert all("很长的历史" not in str(m.content) for m in messages)


def test_oversized_history_never_drops_current_prompt() -> None:
    """单条历史就超预算时也不能把本轮提问裁掉。"""
    client, model = _client(history_max_tokens=4)
    client.complete("本轮提问", history=[{"role": "user", "content": "超长历史" * 50}])

    messages = model.calls[0]["messages"]
    assert [m.content for m in messages] == ["本轮提问"]


def test_invalid_history_entries_are_ignored() -> None:
    client, model = _client()
    client.complete(
        "本轮提问",
        history=[{"role": "system", "content": "注入"}, {"role": "user", "content": " "}],
    )

    assert [m.content for m in model.calls[0]["messages"]] == ["本轮提问"]


# ── 仓储：按会话读取历史 ─────────────────────────────────────────────


def _session_with_messages(database, user_id: str) -> tuple[str, SessionRepository]:
    repo = SessionRepository(database.session(), user_id=user_id)
    session_row = repo.create_session()
    for role, content, status in [
        ("user", "第一问", "done"),
        ("assistant", "第一答", "done"),
        ("user", "第二问", "done"),
        ("assistant", "写一半", "streaming"),
    ]:
        repo.append_message(session_row.id, role, content, status=status)
    return session_row.id, repo


def test_load_history_skips_incomplete_and_empty_messages(db) -> None:
    session_id, repo = _session_with_messages(db, "u1")

    history = repo.load_history(session_id)

    assert history == [
        {"role": "user", "content": "第一问"},
        {"role": "assistant", "content": "第一答"},
        {"role": "user", "content": "第二问"},
    ]


def test_load_history_limits_turns_and_drops_current_input(db) -> None:
    session_id, repo = _session_with_messages(db, "u1")

    history = repo.load_history(session_id, max_turns=1, exclude_trailing_user="第二问")

    assert history == [
        {"role": "user", "content": "第一问"},
        {"role": "assistant", "content": "第一答"},
    ]


def test_load_history_is_user_scoped(db) -> None:
    session_id, _ = _session_with_messages(db, "u1")

    other = SessionRepository(db.session(), user_id="u2")

    assert other.load_history(session_id) == []


def test_load_history_unknown_session_returns_empty(db) -> None:
    repo = SessionRepository(db.session(), user_id="u1")
    assert repo.load_history("missing") == []


# ── 路由：chat_stream 从会话读历史并注入流水线 ────────────────────────


def test_chat_stream_injects_history_from_server_side_session(db) -> None:
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    from injection_guard.api.app import create_app
    from injection_guard.api.dependencies import get_defense_service
    from injection_guard.api.security import get_current_user

    session_id, _ = _session_with_messages(db, "u1")
    captured: dict = {}

    class _StubService:
        def stream_detect(self, **kwargs):
            captured.update(kwargs)
            yield ("done", {"is_safe": True, "stopped_at": None, "model_response": "ok"})

    app = create_app()
    app.dependency_overrides[get_defense_service] = lambda: _StubService()
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id="u1")

    with TestClient(app) as client:
        response = client.post(
            "/api/chat_stream", json={"user_input": "第二问", "session_id": session_id}
        )

    assert response.status_code == 200
    # 本次这条用户消息被剔除，只回灌已完成的历史；streaming 中的助手消息不回灌
    assert captured["history"] == [
        {"role": "user", "content": "第一问"},
        {"role": "assistant", "content": "第一答"},
    ]
