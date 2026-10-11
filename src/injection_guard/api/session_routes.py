"""会话 / 消息 / 检测记录的 REST 路由（服务端持久化 + 按用户隔离）。

浏览器不再持有权威数据：前端只做内存缓存，所有增删改查都经此接口落库，
因而支持多端同步、换设备不丢、按账号隔离与后续审计。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Response
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from ..persistence.database import get_db_session
from ..persistence.repository import DEFAULT_TITLE, SessionRepository
from ..persistence.serializers import (
    detection_to_dict,
    message_to_dict,
    session_detail,
    session_summary,
)
from .schemas import (
    ChatMessageModel,
    ClearSessionsResponse,
    CreateMessageRequest,
    CreateSessionRequest,
    DetectionRecordModel,
    PatchMessageRequest,
    RenameSessionRequest,
    SessionDetail,
    SessionListResponse,
    SessionSummary,
)
from .security import get_current_user_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


def get_repository(
    db: Session = Depends(get_db_session),
    user_id: str = Depends(get_current_user_id),
) -> SessionRepository:
    return SessionRepository(db, user_id=user_id)


def _not_found() -> JSONResponse:
    # 与项目其它接口一致：错误体为 {"error": ...}（而非 FastAPI 默认的 detail）
    return JSONResponse(status_code=404, content={"error": "会话不存在"})


@router.get("", response_model=SessionListResponse, summary="列出当前用户的会话")
def list_sessions(repository: SessionRepository = Depends(get_repository)):
    return {
        "sessions": [
            session_summary(item["session"], item["message_count"])
            for item in repository.list_sessions()
        ]
    }


@router.post("", response_model=SessionDetail, status_code=201, summary="新建会话")
def create_session(
    payload: CreateSessionRequest | None = None,
    repository: SessionRepository = Depends(get_repository),
):
    row = repository.create_session(title=(payload.title if payload else None) or DEFAULT_TITLE)
    return session_detail(row)


@router.delete("", response_model=ClearSessionsResponse, summary="清空当前用户的全部会话")
def clear_sessions(repository: SessionRepository = Depends(get_repository)):
    return {"deleted": repository.clear_sessions()}


@router.get("/{session_id}", response_model=SessionDetail, summary="获取会话详情（含消息与检测记录）")
def get_session(session_id: str, repository: SessionRepository = Depends(get_repository)):
    row = repository.get_session(session_id)
    if row is None:
        return _not_found()
    return session_detail(row)


@router.patch("/{session_id}", response_model=SessionSummary, summary="重命名会话")
def rename_session(
    session_id: str,
    payload: RenameSessionRequest,
    repository: SessionRepository = Depends(get_repository),
):
    if not payload.title.strip():
        return JSONResponse(status_code=400, content={"error": "title 不能为空"})
    row = repository.rename_session(session_id, payload.title)
    if row is None:
        return _not_found()
    return session_summary(row)


@router.delete("/{session_id}", status_code=204, summary="删除会话（连带消息与检测记录）")
def delete_session(session_id: str, repository: SessionRepository = Depends(get_repository)):
    if not repository.delete_session(session_id):
        return _not_found()
    return Response(status_code=204)


@router.post(
    "/{session_id}/messages",
    response_model=ChatMessageModel,
    status_code=201,
    summary="追加一条消息",
)
def create_message(
    session_id: str,
    payload: CreateMessageRequest,
    repository: SessionRepository = Depends(get_repository),
):
    if payload.role not in ("user", "assistant"):
        return JSONResponse(status_code=400, content={"error": "role 必须是 user 或 assistant"})
    row = repository.append_message(session_id, payload.role, payload.content or "", payload.status)
    if row is None:
        return _not_found()
    return message_to_dict(row)


@router.patch(
    "/{session_id}/messages/{message_id}",
    response_model=ChatMessageModel,
    summary="更新消息（流式结束时的内容 / 状态 / 错误）",
)
def patch_message(
    session_id: str,
    message_id: str,
    payload: PatchMessageRequest,
    repository: SessionRepository = Depends(get_repository),
):
    row = repository.patch_message(session_id, message_id, payload.model_dump(exclude_unset=True))
    if row is None:
        return _not_found()
    return message_to_dict(row)


@router.put(
    "/{session_id}/messages/{message_id}/detection",
    response_model=DetectionRecordModel,
    summary="写入 / 覆盖检测记录",
)
def put_detection(
    session_id: str,
    message_id: str,
    payload: DetectionRecordModel,
    repository: SessionRepository = Depends(get_repository),
):
    row = repository.put_detection(session_id, message_id, payload.model_dump())
    if row is None:
        return _not_found()
    return detection_to_dict(row)
