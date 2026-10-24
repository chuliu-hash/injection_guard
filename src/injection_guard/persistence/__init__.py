"""会话/消息/检测记录的服务端持久化层（SQLAlchemy 2.0）。

浏览器的 ``localStorage`` 只适合临时状态；企业级要求会话数据**落库、按用户隔离、
可多端同步、可审计**。本包提供 ORM 模型、引擎/会话工厂与按 ``user_id`` 隔离的仓储，
由 ``api/session_routes.py`` 暴露 REST 接口给前端。
"""

from .database import Database, ensure_database_exists, get_database, get_db_session
from .models import Base, ChatMessageRow, ChatSessionRow, DetectionRow, UserRow
from .repository import DEFAULT_TITLE, SessionRepository
from .user_repository import UserRepository

__all__ = [
    "Base",
    "ChatMessageRow",
    "ChatSessionRow",
    "Database",
    "DEFAULT_TITLE",
    "DetectionRow",
    "SessionRepository",
    "UserRepository",
    "UserRow",
    "ensure_database_exists",
    "get_database",
    "get_db_session",
]
