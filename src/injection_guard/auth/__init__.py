"""认证原语：密码哈希与 JWT 签发 / 校验。

与 FastAPI / 存储解耦，便于单测；HTTP 层见 ``api/security.py`` 与 ``api/auth_routes.py``。
"""

from .passwords import hash_password, verify_password
from .tokens import TokenError, create_access_token, create_refresh_token, decode_token

__all__ = [
    "TokenError",
    "create_access_token",
    "create_refresh_token",
    "decode_token",
    "hash_password",
    "verify_password",
]
