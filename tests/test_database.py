"""PostgreSQL-only 数据库 URL 规范与拒绝逻辑。"""

from __future__ import annotations

import pytest

from injection_guard.persistence.database import psycopg_database_url, sqlalchemy_database_url

POSTGRES_URL = "postgresql://chuliu:123456@127.0.0.1:5432/injection_guard"


def test_plain_postgres_url_uses_psycopg_v3_for_sqlalchemy() -> None:
    assert sqlalchemy_database_url(POSTGRES_URL) == (
        "postgresql+psycopg://chuliu:123456@127.0.0.1:5432/injection_guard"
    )
    assert psycopg_database_url(POSTGRES_URL) == POSTGRES_URL


def test_sqlalchemy_url_is_kept_idempotently() -> None:
    url = "postgresql+psycopg://chuliu:123456@127.0.0.1:5432/injection_guard"
    assert sqlalchemy_database_url(url) == url
    assert psycopg_database_url(url) == POSTGRES_URL


@pytest.mark.parametrize("url", ["sqlite:///test.db", "mysql://user:pw@localhost/db"])
def test_non_postgres_urls_are_rejected(url: str) -> None:
    with pytest.raises(ValueError, match="仅支持 PostgreSQL"):
        sqlalchemy_database_url(url)
    with pytest.raises(ValueError, match="仅支持 PostgreSQL"):
        psycopg_database_url(url)


def test_postgres_url_requires_explicit_database_name() -> None:
    with pytest.raises(ValueError, match="必须显式包含数据库名"):
        sqlalchemy_database_url("postgresql://chuliu:123456@127.0.0.1:5432")
