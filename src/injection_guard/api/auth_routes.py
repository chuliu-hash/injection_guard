"""认证路由：注册 / 登录 / 刷新 / 退出 / 当前用户。

令牌策略：
- **access token** 在 JSON 响应体返回，前端放内存、随 ``Authorization`` 头发送；
- **refresh token** 以 **httpOnly Cookie** 下发（JS 不可读，抗 XSS 窃取），
  仅用于 ``POST /api/auth/refresh`` 换取新 access token（并轮换 refresh）。

失败一律返回 ``{"error": ...}``，与项目其它接口契约一致。
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from ..auth.passwords import BCRYPT_MAX_BYTES
from ..auth.tokens import TokenError, create_access_token, create_refresh_token, decode_token
from ..config import get_settings
from ..persistence.database import get_db_session
from ..persistence.models import UserRow
from ..persistence.serializers import user_to_dict
from ..persistence.user_repository import UserRepository
from .schemas import (
    LoginRequest,
    LogoutRequest,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
    UserModel,
)
from .security import get_current_user, get_optional_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _error(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": message})


def _validate_credentials(username: str, password: str) -> Optional[str]:
    settings = get_settings()
    if not username.strip():
        return "用户名不能为空"
    if len(username.strip()) > 64:
        return "用户名过长（≤64 字符）"
    if len(password) < settings.password_min_length:
        return f"密码至少 {settings.password_min_length} 位"
    if len(password.encode("utf-8")) > BCRYPT_MAX_BYTES:
        return f"密码过长（≤{BCRYPT_MAX_BYTES} 字节）"
    return None


def _set_refresh_cookie(response: Response, token: str) -> None:
    settings = get_settings()
    response.set_cookie(
        key=settings.refresh_cookie_name,
        value=token,
        max_age=settings.refresh_token_expire_days * 24 * 3600,
        httponly=True,
        secure=settings.cookie_secure,
        samesite=settings.cookie_samesite,  # type: ignore[arg-type]
        path=settings.refresh_cookie_path,
    )


def _clear_refresh_cookie(response: Response) -> None:
    settings = get_settings()
    response.delete_cookie(
        key=settings.refresh_cookie_name, path=settings.refresh_cookie_path
    )


def _token_response(user: UserRow, access_token: str) -> dict:
    settings = get_settings()
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": settings.access_token_expire_minutes * 60,
        "user": user_to_dict(user),
    }


@router.post("/register", response_model=TokenResponse, status_code=201, summary="注册并登录")
def register(
    payload: RegisterRequest,
    response: Response,
    db: Session = Depends(get_db_session),
):
    settings = get_settings()
    if not settings.allow_registration:
        return _error(403, "已关闭自助注册")
    message = _validate_credentials(payload.username, payload.password)
    if message:
        return _error(400, message)

    repo = UserRepository(db)
    if repo.get_by_username(payload.username) is not None:
        return _error(409, "用户名已存在")

    user = repo.create(payload.username, payload.password)
    logger.info("新用户注册: %s", user.username)
    _set_refresh_cookie(response, create_refresh_token(user))
    return _token_response(user, create_access_token(user))


@router.post("/login", response_model=TokenResponse, summary="登录")
def login(
    payload: LoginRequest,
    response: Response,
    db: Session = Depends(get_db_session),
):
    user = UserRepository(db).authenticate(payload.username, payload.password)
    if user is None:
        # 不区分「账号不存在」与「密码错误」，避免用户名枚举
        return _error(401, "用户名或密码错误")
    _set_refresh_cookie(response, create_refresh_token(user))
    return _token_response(user, create_access_token(user))


@router.post("/refresh", response_model=TokenResponse, summary="用刷新令牌换取新的 access token")
def refresh(
    request: Request,
    response: Response,
    payload: Optional[RefreshRequest] = None,
    db: Session = Depends(get_db_session),
):
    settings = get_settings()
    token = (payload.refresh_token if payload else None) or request.cookies.get(
        settings.refresh_cookie_name
    )
    if not token:
        return _error(401, "缺少刷新令牌")
    try:
        claims = decode_token(token, expected_type="refresh")
    except TokenError:
        return _error(401, "刷新令牌无效或已过期")

    user = UserRepository(db).get_by_id(claims["sub"])
    if user is None or not user.is_active or user.token_version != claims.get("ver"):
        return _error(401, "刷新令牌已失效，请重新登录")

    # 轮换：每次刷新都换发新的 refresh，降低泄露窗口
    _set_refresh_cookie(response, create_refresh_token(user))
    return _token_response(user, create_access_token(user))


@router.post("/logout", summary="退出登录（清 Cookie；all_devices 则吊销全部令牌）")
def logout(
    response: Response,
    payload: Optional[LogoutRequest] = None,
    user: Optional[UserRow] = Depends(get_optional_user),
    db: Session = Depends(get_db_session),
):
    if payload and payload.all_devices and user is not None:
        UserRepository(db).bump_token_version(user)
    _clear_refresh_cookie(response)
    return {"ok": True}


@router.get("/me", response_model=UserModel, summary="当前登录用户")
def me(user: UserRow = Depends(get_current_user)):
    return user_to_dict(user)
