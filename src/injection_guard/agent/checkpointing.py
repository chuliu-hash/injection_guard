"""LangGraph PostgreSQL 检查点工厂（P1）。

检查点与会话数据共用 ``DATABASE_URL`` 指向的 PostgreSQL。这里不提供内存或 SQLite
后端，也不做静默回退：数据库不可用时直接让启动失败，避免线上误以为状态可恢复。
"""

from __future__ import annotations

import logging
from typing import Any

from ..config import Settings
from ..persistence.database import psycopg_database_url

logger = logging.getLogger(__name__)


def _postgres_saver(database_url: str):
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    pool = ConnectionPool(
        psycopg_database_url(database_url),
        min_size=1,
        max_size=5,
        open=True,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    pool.wait()
    saver = PostgresSaver(pool)
    saver.setup()
    return saver


def build_checkpointer(settings: Settings) -> Any:
    """构建并初始化 PostgreSQL 检查点；连接或建表失败时向上抛出。"""
    logger.info("[LangGraph] 使用 PostgreSQL 检查点")
    return _postgres_saver(settings.database_url)
