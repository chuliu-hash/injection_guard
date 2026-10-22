"""PostgreSQL 引擎、会话工厂与 FastAPI 依赖（SQLAlchemy 2.0，同步）。

项目只支持 PostgreSQL；``postgresql://`` 会在内部规范到 psycopg v3 驱动。引擎按 URL
缓存，测试可通过 ``get_database.cache_clear()`` + 环境变量 ``DATABASE_URL`` 覆盖。
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Iterator
from urllib.parse import urlsplit, urlunsplit

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from ..config import get_settings
from .models import Base

logger = logging.getLogger(__name__)


def _split_postgres_url(database_url: str) -> tuple[str, str, str, str, str]:
    parsed = urlsplit(database_url)
    driver = parsed.scheme.lower()
    if driver not in {"postgresql", "postgresql+psycopg"}:
        raise ValueError(
            "DATABASE_URL 仅支持 PostgreSQL（postgresql:// 或 postgresql+psycopg://）"
        )
    return parsed.scheme, parsed.netloc, parsed.path, parsed.query, parsed.fragment


def sqlalchemy_database_url(database_url: str) -> str:
    """规范为 SQLAlchemy + psycopg v3 URL；拒绝 PostgreSQL 之外的数据库。"""
    _, netloc, path, query, fragment = _split_postgres_url(database_url)
    return urlunsplit(("postgresql+psycopg", netloc, path, query, fragment))


def psycopg_database_url(database_url: str) -> str:
    """规范为 libpq/psycopg 可直接使用的 URL。"""
    _, netloc, path, query, fragment = _split_postgres_url(database_url)
    return urlunsplit(("postgresql", netloc, path, query, fragment))


def _create_engine(database_url: str) -> Engine:
    return create_engine(sqlalchemy_database_url(database_url), pool_pre_ping=True, future=True)


class Database:
    """封装引擎与 session 工厂，便于依赖注入与测试替换。"""

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self.engine = _create_engine(database_url)
        self._session_factory = sessionmaker(
            bind=self.engine, autoflush=False, expire_on_commit=False, future=True
        )

    def create_all(self) -> None:
        """建表（幂等）。首次运行时创建 schema，已存在则跳过。"""
        Base.metadata.create_all(self.engine)

    def session(self) -> Session:
        return self._session_factory()

    def dispose(self) -> None:
        self.engine.dispose()


@lru_cache(maxsize=1)
def get_database() -> Database:
    """进程级数据库单例（测试可用 ``get_database.cache_clear()`` 覆盖）。"""
    return Database(get_settings().database_url)


def get_db_session() -> Iterator[Session]:
    """FastAPI 依赖：每个请求一个会话，异常回滚、结束关闭。"""
    session = get_database().session()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
