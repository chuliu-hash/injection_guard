"""会话 / 消息 / 检测记录的 ORM 模型。

三张表，均以 ``user_id`` 做行级隔离（多租户/多用户互不可见）：

- ``chat_sessions``：一个会话一行；
- ``chat_messages``：一条消息一行（按 ``created_at`` 排序）；
- ``message_detections``：与助手消息**一对一**，以 ``message_id`` 为主键。

``created_at`` / ``updated_at`` 沿用前端契约，统一为**毫秒整数**（非 datetime）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlalchemy import JSON, BigInteger, Boolean, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """所有 ORM 模型的基类。"""


class UserRow(Base):
    """账号：登录凭据 + 令牌版本（用于「退出所有设备」/ 改密后失效旧令牌）。"""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    username: Mapped[str] = mapped_column(String(128), nullable=False, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    # 自增版本号：递增即吊销该用户此前签发的全部 access / refresh token
    token_version: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[int] = mapped_column(BigInteger, nullable=False)


class ChatSessionRow(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    # 拥有者：所有查询强制带此条件，实现用户间隔离
    user_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[int] = mapped_column(BigInteger, nullable=False)
    updated_at: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)

    messages: Mapped[List["ChatMessageRow"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ChatMessageRow.created_at",
        lazy="selectin",
    )


class ChatMessageRow(Base):
    __tablename__ = "chat_messages"
    __table_args__ = (Index("ix_chat_messages_session_created", "session_id", "created_at"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="done")
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[int] = mapped_column(BigInteger, nullable=False)

    session: Mapped["ChatSessionRow"] = relationship(back_populates="messages")
    detection: Mapped[Optional["DetectionRow"]] = relationship(
        back_populates="message",
        cascade="all, delete-orphan",
        uselist=False,
        lazy="selectin",
    )


class DetectionRow(Base):
    __tablename__ = "message_detections"

    message_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("chat_messages.id", ondelete="CASCADE"),
        primary_key=True,
    )
    is_safe: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    # 提前终止位置：input_detection / output_detection / output_skipped / null
    stopped_at: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    # 结构化检测详情，原样存 JSON（输入检测 / RAG / 输出检测三段）
    input_detection: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    rag: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    output_detection: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)

    message: Mapped["ChatMessageRow"] = relationship(back_populates="detection")
