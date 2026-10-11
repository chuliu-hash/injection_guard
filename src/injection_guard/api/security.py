"""请求身份解析：基于 JWT 的鉴权依赖。

- ``get_current_user``：校验 ``Authorization: Bearer <access token>``，返回 ``UserRow``；
  校验通过还需满足「用户存在 / 未停用 / 令牌版本一致」。
- ``get_current_user_id``：只需要用户 id 的依赖（会话仓储用）。
- ``require_user``：给路由做 router/route 级 ``dependencies=[...]`` 用。
- ``get_optional_user``：令牌缺失/无效时不报错（用于 logout 等）。

令牌版本（``ver``）与 ``users.token_version`` 比对，使「退出所有设备 / 改密」可立即吊销旧令牌。
"""

from __future__ import annotations

from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from ..auth.tokens import TokenError, decode_token
from ..persistence.database import get_db_session
from ..persistence.models import UserRow
from ..persistence.user_repository import UserRepository

# auto_error=False：缺失时交由我们统一返回 401（而非 FastAPI 默认 403）
_bearer = HTTPBearer(auto_error=False, description="JWT access token")


def _unauthorized(detail: str = "未认证或登录已过期") -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def _user_from_token(token: str, db: Session, expected_type: str) -> Optional[UserRow]:
    try:
        claims = decode_token(token, expected_type=expected_type)
    except TokenError:
        return None
    user = UserRepository(db).get_by_id(claims["sub"])
    if user is None or not user.is_active:
        return None
    if user.token_version != claims.get("ver"):
        return None
    return user


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    db: Session = Depends(get_db_session),
) -> UserRow:
    """返回当前登录用户；未携带 / 无效 / 被吊销则 401。"""
    if credentials is None or not credentials.credentials:
        raise _unauthorized()
    user = _user_from_token(credentials.credentials, db, expected_type="access")
    if user is None:
        raise _unauthorized()
    return user


def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    db: Session = Depends(get_db_session),
) -> Optional[UserRow]:
    """令牌有效则返回用户，否则 ``None``（不报错）。"""
    if credentials is None or not credentials.credentials:
        return None
    return _user_from_token(credentials.credentials, db, expected_type="access")


def get_current_user_id(user: UserRow = Depends(get_current_user)) -> str:
    return user.id


# 路由级依赖别名（`dependencies=[Depends(require_user)]`）
require_user = get_current_user
