"""会话仓储：所有读写都以构造时注入的 ``user_id`` 为边界。

上层（REST 路由）不做任何过滤，隔离在本层统一保证——只要带上别人的 session_id
也查不到、改不了，返回 ``None``。
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from ..config import get_settings
from .models import ChatMessageRow, ChatSessionRow, DetectionRow

DEFAULT_TITLE = "新会话"
_VALID_ROLES = ("user", "assistant")


def _now_ms() -> int:
    return int(time.time() * 1000)


def auto_title(text: str, max_len: int) -> str:
    """由首条用户消息生成标题（与前端 ``truncate`` 行为一致）。"""
    normalized = " ".join(text.split())
    if not normalized:
        return DEFAULT_TITLE
    return normalized[:max_len] + ("…" if len(normalized) > max_len else "")


class SessionRepository:
    def __init__(self, session: Session, *, user_id: str) -> None:
        self.db = session
        self.user_id = user_id

    # ── 会话 ──────────────────────────────────────────────────────────
    def list_sessions(self) -> List[Dict[str, Any]]:
        """会话列表（含消息数），按最近更新倒序——不加载消息正文。"""
        stmt = (
            select(ChatSessionRow, func.count(ChatMessageRow.id))
            .outerjoin(ChatMessageRow, ChatMessageRow.session_id == ChatSessionRow.id)
            .where(ChatSessionRow.user_id == self.user_id)
            .group_by(ChatSessionRow.id)
            .order_by(ChatSessionRow.updated_at.desc())
        )
        return [{"session": row, "message_count": count} for row, count in self.db.execute(stmt)]

    def get_session(self, session_id: str) -> Optional[ChatSessionRow]:
        stmt = select(ChatSessionRow).where(
            ChatSessionRow.id == session_id, ChatSessionRow.user_id == self.user_id
        )
        return self.db.execute(stmt).scalar_one_or_none()

    def create_session(self, title: str = DEFAULT_TITLE) -> ChatSessionRow:
        now = _now_ms()
        row = ChatSessionRow(
            id=str(uuid.uuid4()),
            user_id=self.user_id,
            title=title.strip() or DEFAULT_TITLE,
            created_at=now,
            updated_at=now,
        )
        self.db.add(row)
        self.db.commit()
        self.db.refresh(row)
        return row

    def rename_session(self, session_id: str, title: str) -> Optional[ChatSessionRow]:
        row = self.get_session(session_id)
        if row is None:
            return None
        row.title = title.strip()
        row.updated_at = _now_ms()
        self.db.commit()
        self.db.refresh(row)
        return row

    def delete_session(self, session_id: str) -> bool:
        row = self.get_session(session_id)
        if row is None:
            return False
        self.db.delete(row)  # ORM 级联删除 messages -> detections
        self.db.commit()
        return True

    def clear_sessions(self) -> int:
        """清空当前用户的全部会话（连带消息与检测记录）。返回删除的会话数。"""
        session_ids = select(ChatSessionRow.id).where(ChatSessionRow.user_id == self.user_id)
        message_ids = select(ChatMessageRow.id).where(ChatMessageRow.session_id.in_(session_ids))
        self.db.execute(delete(DetectionRow).where(DetectionRow.message_id.in_(message_ids)))
        self.db.execute(delete(ChatMessageRow).where(ChatMessageRow.session_id.in_(session_ids)))
        result = self.db.execute(
            delete(ChatSessionRow).where(ChatSessionRow.user_id == self.user_id)
        )
        self.db.commit()
        return int(result.rowcount or 0)

    # ── 消息 ──────────────────────────────────────────────────────────
    def append_message(
        self,
        session_id: str,
        role: str,
        content: str = "",
        status: Optional[str] = None,
    ) -> Optional[ChatMessageRow]:
        session_row = self.get_session(session_id)
        if session_row is None:
            return None
        if role not in _VALID_ROLES:
            raise ValueError(f"非法 role: {role}")

        now = _now_ms()
        row = ChatMessageRow(
            id=str(uuid.uuid4()),
            session_id=session_id,
            role=role,
            content=content or "",
            status=status or ("streaming" if role == "assistant" else "done"),
            created_at=now,
        )
        self.db.add(row)
        if role == "user" and session_row.title == DEFAULT_TITLE:
            session_row.title = auto_title(content, get_settings().session_title_max_len)
        session_row.updated_at = now
        self.db.commit()
        self.db.refresh(row)
        return row

    def patch_message(
        self, session_id: str, message_id: str, patch: Dict[str, Any]
    ) -> Optional[ChatMessageRow]:
        row = self._get_message(session_id, message_id)
        if row is None:
            return None
        for field in ("content", "status", "error"):
            if patch.get(field) is not None:
                setattr(row, field, patch[field])
        self._touch(session_id)
        self.db.commit()
        self.db.refresh(row)
        return row

    def put_detection(
        self, session_id: str, message_id: str, record: Dict[str, Any]
    ) -> Optional[DetectionRow]:
        if self._get_message(session_id, message_id) is None:
            return None
        row = self.db.get(DetectionRow, message_id)
        if row is None:
            row = DetectionRow(message_id=message_id)
            self.db.add(row)
        row.is_safe = bool(record.get("is_safe", True))
        row.stopped_at = record.get("stopped_at")
        row.input_detection = record.get("input_detection")
        row.rag = record.get("rag")
        row.output_detection = record.get("output_detection")
        self._touch(session_id)
        self.db.commit()
        self.db.refresh(row)
        return row

    # ── 内部 ──────────────────────────────────────────────────────────
    def _get_message(self, session_id: str, message_id: str) -> Optional[ChatMessageRow]:
        stmt = (
            select(ChatMessageRow)
            .join(ChatSessionRow, ChatSessionRow.id == ChatMessageRow.session_id)
            .where(
                ChatMessageRow.id == message_id,
                ChatMessageRow.session_id == session_id,
                ChatSessionRow.user_id == self.user_id,
            )
        )
        return self.db.execute(stmt).scalar_one_or_none()

    def _touch(self, session_id: str) -> None:
        session_row = self.get_session(session_id)
        if session_row is not None:
            session_row.updated_at = _now_ms()
