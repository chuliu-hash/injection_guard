"""JWT 签发与校验（HS256）。

令牌分两类，用 ``type`` 声明，互不通用：

- ``access``：短时效，前端放内存、随 ``Authorization: Bearer`` 头发送；
- ``refresh``：长时效，仅经 httpOnly Cookie 下发，用于换取新的 access token。

两者都带 ``ver``（用户令牌版本）：与 ``users.token_version`` 不符即视为已吊销。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import jwt

from ..config import get_settings
from ..persistence.models import UserRow

ACCESS_TYPE = "access"
REFRESH_TYPE = "refresh"


class TokenError(Exception):
    """令牌缺失、过期、签名错误或类型不符。"""


def _encode(claims: Dict[str, Any], expires_delta: timedelta) -> str:
    settings = get_settings()
    now = datetime.now(timezone.utc)
    payload = {**claims, "iat": now, "exp": now + expires_delta}
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def create_access_token(user: UserRow) -> str:
    settings = get_settings()
    return _encode(
        {
            "sub": user.id,
            "username": user.username,
            "type": ACCESS_TYPE,
            "ver": user.token_version,
        },
        timedelta(minutes=settings.access_token_expire_minutes),
    )


def create_refresh_token(user: UserRow) -> str:
    settings = get_settings()
    return _encode(
        {"sub": user.id, "type": REFRESH_TYPE, "ver": user.token_version},
        timedelta(days=settings.refresh_token_expire_days),
    )


def decode_token(token: str, expected_type: str) -> Dict[str, Any]:
    """校验签名 / 过期 / 类型，返回 claims；失败抛 :class:`TokenError`。"""
    settings = get_settings()
    try:
        claims = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
    except jwt.PyJWTError as exc:  # 含 ExpiredSignatureError / InvalidTokenError 等
        raise TokenError(str(exc)) from exc
    if claims.get("type") != expected_type:
        raise TokenError("令牌类型不匹配")
    if not claims.get("sub"):
        raise TokenError("令牌缺少 sub")
    return claims
