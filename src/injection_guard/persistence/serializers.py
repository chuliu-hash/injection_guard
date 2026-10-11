"""ORM 行 -> API 响应字典（wire 格式为 snake_case，前端在 api 层转 camelCase）。"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .models import ChatMessageRow, ChatSessionRow, DetectionRow, UserRow


def user_to_dict(row: UserRow) -> Dict[str, Any]:
    """对外暴露的用户信息（**绝不包含密码哈希**）。"""
    return {
        "id": row.id,
        "username": row.username,
        "is_active": row.is_active,
        "created_at": row.created_at,
    }


def message_to_dict(row: ChatMessageRow) -> Dict[str, Any]:
    return {
        "id": row.id,
        "role": row.role,
        "content": row.content,
        "status": row.status,
        "error": row.error,
        "created_at": row.created_at,
    }


def detection_to_dict(row: DetectionRow) -> Dict[str, Any]:
    return {
        "is_safe": row.is_safe,
        "stopped_at": row.stopped_at,
        "input_detection": row.input_detection,
        "rag": row.rag,
        "output_detection": row.output_detection,
    }


def session_summary(row: ChatSessionRow, message_count: Optional[int] = None) -> Dict[str, Any]:
    count = len(row.messages) if message_count is None else message_count
    return {
        "id": row.id,
        "title": row.title,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
        "message_count": count,
    }


def session_detail(row: ChatSessionRow) -> Dict[str, Any]:
    detail = session_summary(row)
    detail["messages"] = [message_to_dict(message) for message in row.messages]
    detail["detections"] = {
        message.id: detection_to_dict(message.detection)
        for message in row.messages
        if message.detection is not None
    }
    return detail
