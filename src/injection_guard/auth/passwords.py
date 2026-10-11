"""密码哈希：bcrypt（自带盐、抗暴力破解）。明文密码永不落库、永不进日志。"""

from __future__ import annotations

import bcrypt

# bcrypt 仅处理前 72 字节；注册时按字节上限校验（见 api/auth_routes.py）
BCRYPT_MAX_BYTES = 72


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        # 非法哈希格式一律视为校验失败，不抛异常（避免泄露内部状态）
        return False
