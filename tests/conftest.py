"""pytest 公共配置：源码路径与隔离的 PostgreSQL 测试 schema。"""

from __future__ import annotations

import atexit
import os
import sys
import uuid
from pathlib import Path

import psycopg
import pytest
from psycopg import sql

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from injection_guard.config import DEFAULT_DATABASE_URL, get_settings  # noqa: E402
from injection_guard.persistence.database import (  # noqa: E402
    ensure_database_exists,
    get_database,
)


def _database_url(schema: str) -> str:
    """在固定 PostgreSQL DSN 上通过 search_path 指向独立 schema。"""
    return f"{DEFAULT_DATABASE_URL}?options=-csearch_path%3D{schema}"


def _create_schema(schema: str) -> None:
    with psycopg.connect(DEFAULT_DATABASE_URL, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))


def _drop_schema(schema: str) -> None:
    with psycopg.connect(DEFAULT_DATABASE_URL, autocommit=True) as conn:
        conn.execute(
            sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema))
        )


def _dispose_database() -> None:
    try:
        get_database().dispose()
    except Exception:
        # 测试可能临时把 DATABASE_URL 改成非法值，清理阶段不应因此报错。
        pass
    finally:
        get_database.cache_clear()
        get_settings.cache_clear()


# 模块级 ``app = create_app()`` 会在测试收集时导入并建表；先提供一个临时 schema，
# 避免测试触达业务 schema。每个用例再由 autouse fixture 切换到独立的 schema。
_BOOTSTRAP_SCHEMA = f"ig_bootstrap_{uuid.uuid4().hex[:12]}"
ensure_database_exists(DEFAULT_DATABASE_URL)
_create_schema(_BOOTSTRAP_SCHEMA)
os.environ["DATABASE_URL"] = _database_url(_BOOTSTRAP_SCHEMA)
atexit.register(_drop_schema, _BOOTSTRAP_SCHEMA)

# 测试里不预热真实检测器（BERT 加载依赖 GPU / 权重）；由各用例自行构造桩。
os.environ.setdefault("WARMUP_ON_STARTUP", "false")


@pytest.fixture(autouse=True)
def postgres_database_url(monkeypatch: pytest.MonkeyPatch):
    """为每个测试创建独立 PostgreSQL schema，并切换应用的 DATABASE_URL。"""
    _dispose_database()
    schema = f"ig_test_{uuid.uuid4().hex[:12]}"
    _create_schema(schema)
    url = _database_url(schema)
    monkeypatch.setenv("DATABASE_URL", url)
    get_settings.cache_clear()
    get_database.cache_clear()
    try:
        yield url
    finally:
        _dispose_database()
        _drop_schema(schema)
