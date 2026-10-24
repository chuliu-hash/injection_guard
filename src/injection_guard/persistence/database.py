"""PostgreSQL 引擎、会话工厂与 FastAPI 依赖（SQLAlchemy 2.0，同步）。

项目只支持 PostgreSQL；``postgresql://`` 会在内部规范到 psycopg v3 驱动。连接业务库
之前会先检查数据库是否存在，缺失时自动创建，再建立 SQLAlchemy 引擎。引擎按 URL 缓存，
测试可通过 ``get_database.cache_clear()`` + 环境变量 ``DATABASE_URL`` 覆盖。
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Iterator
from urllib.parse import unquote, urlsplit, urlunsplit

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from ..config import get_settings
from .models import Base

logger = logging.getLogger(__name__)

# 创建目标数据库时必须先连到一个已存在的维护库；PostgreSQL 标准安装同时提供
# postgres / template1，前者不可用时依次回退。
_MAINTENANCE_DATABASES = ("postgres", "template1")


def _split_postgres_url(database_url: str) -> tuple[str, str, str, str, str]:
    parsed = urlsplit(database_url)
    driver = parsed.scheme.lower()
    if driver not in {"postgresql", "postgresql+psycopg"}:
        raise ValueError(
            "DATABASE_URL 仅支持 PostgreSQL（postgresql:// 或 postgresql+psycopg://）"
        )
    if not database_name(database_url):
        raise ValueError("DATABASE_URL 必须显式包含数据库名，例如 ...:5432/injection_guard")
    return parsed.scheme, parsed.netloc, parsed.path, parsed.query, parsed.fragment


def database_name(database_url: str) -> str:
    """从 DSN 中取数据库名（URL 解码后）；缺失时返回空串。"""
    return unquote(urlsplit(database_url).path.lstrip("/")).strip()


def sqlalchemy_database_url(database_url: str) -> str:
    """规范为 SQLAlchemy + psycopg v3 URL；拒绝 PostgreSQL 之外的数据库。"""
    _, netloc, path, query, fragment = _split_postgres_url(database_url)
    return urlunsplit(("postgresql+psycopg", netloc, path, query, fragment))


def psycopg_database_url(database_url: str) -> str:
    """规范为 libpq/psycopg 可直接使用的 URL。"""
    _, netloc, path, query, fragment = _split_postgres_url(database_url)
    return urlunsplit(("postgresql", netloc, path, query, fragment))


def _maintenance_conninfo(database_url: str, maintenance_database: str) -> str:
    """把业务 DSN 的数据库名替换为维护库，保留 SSL 等连接参数。

    测试 DSN 可能带 ``options=-csearch_path=...``；创建数据库前删掉该参数，避免维护
    连接依赖尚未创建的 schema。
    """
    params = conninfo_to_dict(psycopg_database_url(database_url))
    params["dbname"] = maintenance_database
    params.pop("options", None)
    return make_conninfo(**params)


def _connect_maintenance(database_url: str) -> psycopg.Connection:
    candidates = [os.environ.get("POSTGRES_MAINTENANCE_DB"), *_MAINTENANCE_DATABASES]
    last_error: Exception | None = None
    for candidate in dict.fromkeys(name for name in candidates if name):
        try:
            return psycopg.connect(
                _maintenance_conninfo(database_url, candidate), autocommit=True
            )
        except psycopg.OperationalError as exc:
            last_error = exc
    raise RuntimeError(
        "无法连接 PostgreSQL 维护库以创建业务数据库；"
        "请确认数据库服务可用，或通过 POSTGRES_MAINTENANCE_DB 指定可连接的维护库"
    ) from last_error


def ensure_database_exists(database_url: str) -> bool:
    """确保 DSN 指向的数据库存在；返回是否执行了创建。

    ``CREATE DATABASE`` 不能在普通事务中执行，因此维护连接使用 autocommit。并发启动时
    可能同时尝试创建，捕获 ``DuplicateDatabase`` 后按已存在处理。
    """
    target = database_name(database_url)
    with _connect_maintenance(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (target,))
            if cur.fetchone() is not None:
                return False
            try:
                cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(target)))
            except psycopg.errors.DuplicateDatabase:
                return False
            except psycopg.errors.InsufficientPrivilege as exc:
                raise RuntimeError(
                    f"当前 PostgreSQL 用户无权创建数据库 {target!r}；"
                    "请由管理员预先创建，或为 DATABASE_URL 使用有 CREATEDB 权限的账号"
                ) from exc
    logger.info("[DB] 已创建 PostgreSQL 数据库: %s", target)
    return True


def _create_engine(database_url: str) -> Engine:
    return create_engine(sqlalchemy_database_url(database_url), pool_pre_ping=True, future=True)


class Database:
    """封装引擎与 session 工厂，便于依赖注入与测试替换。"""

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self.name = database_name(database_url)
        ensure_database_exists(database_url)
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
