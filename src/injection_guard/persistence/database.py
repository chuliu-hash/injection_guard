"""数据库引擎、会话工厂与 FastAPI 依赖（SQLAlchemy 2.0，同步）。

默认使用仓库内 SQLite 文件（``DATABASE_URL`` 可切换 PostgreSQL）。引擎按 URL 缓存，
测试可通过 ``get_database.cache_clear()`` + 环境变量 ``DATABASE_URL`` 覆盖。
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from ..config import get_settings
from .models import Base

logger = logging.getLogger(__name__)


def _is_memory_sqlite(database_url: str) -> bool:
    return ":memory:" in database_url or database_url.rstrip("/") == "sqlite://"


def _prepare_sqlite_path(database_url: str) -> None:
    """确保 SQLite 文件所在目录存在（否则首次建库会失败）。"""
    path = database_url[len("sqlite:///") :]
    if path and path != ":memory:":
        Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)


def _create_engine(database_url: str) -> Engine:
    kwargs: Dict[str, Any] = {"pool_pre_ping": True, "future": True}
    if database_url.startswith("sqlite"):
        _prepare_sqlite_path(database_url)
        # FastAPI 在线程池里执行同步依赖，需关闭 SQLite 的线程校验
        kwargs["connect_args"] = {"check_same_thread": False}
        if _is_memory_sqlite(database_url):
            kwargs["poolclass"] = StaticPool  # 内存库：所有连接共享同一实例
    engine = create_engine(database_url, **kwargs)
    if database_url.startswith("sqlite"):
        _enable_sqlite_foreign_keys(engine)
    return engine


def _enable_sqlite_foreign_keys(engine: Engine) -> None:
    """SQLite 默认不校验外键；显式打开以支持级联删除。"""

    @event.listens_for(engine, "connect")
    def _set_pragma(dbapi_connection, _connection_record) -> None:  # noqa: ANN001
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


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
