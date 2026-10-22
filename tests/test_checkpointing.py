"""P1 LangGraph PostgreSQL 检查点工厂单测。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from injection_guard.agent import checkpointing
from injection_guard.config import Settings


def _settings(database_url: str = "postgresql://user:pw@localhost/db"):
    return SimpleNamespace(
        database_url=database_url,
    )


def test_build_checkpointer_uses_postgres(monkeypatch):
    calls = []
    saver = object()

    def _fake_postgres(database_url):
        calls.append(database_url)
        return saver

    monkeypatch.setattr(checkpointing, "_postgres_saver", _fake_postgres)

    assert checkpointing.build_checkpointer(_settings()) is saver
    assert calls == ["postgresql://user:pw@localhost/db"]


def test_checkpoint_failure_is_not_hidden(monkeypatch):
    def _boom(*_args, **_kwargs):
        raise RuntimeError("checkpoint package missing")

    monkeypatch.setattr(checkpointing, "_postgres_saver", _boom)
    with pytest.raises(RuntimeError, match="checkpoint package missing"):
        checkpointing.build_checkpointer(_settings())


def test_checkpointer_initializes_postgres_tables(postgres_database_url: str) -> None:
    settings = Settings(database_url=postgres_database_url)
    saver = checkpointing.build_checkpointer(settings)
    try:
        with saver.conn.connection() as conn:
            row = conn.execute("SELECT to_regclass('checkpoint_migrations')").fetchone()
        assert row is not None and row["to_regclass"] == "checkpoint_migrations"
    finally:
        saver.conn.close()
