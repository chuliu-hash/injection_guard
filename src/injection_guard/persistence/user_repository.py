"""用户仓储：账号创建、查询与口令校验。

用户名大小写不敏感（存储原样、比较时 lower）；密码只存 bcrypt 哈希。
"""

from __future__ import annotations

import time
import uuid
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..auth.passwords import hash_password, verify_password
from .models import UserRow


def _now_ms() -> int:
    return int(time.time() * 1000)


class UserRepository:
    def __init__(self, session: Session) -> None:
        self.db = session

    def get_by_id(self, user_id: str) -> Optional[UserRow]:
        return self.db.get(UserRow, user_id)

    def get_by_username(self, username: str) -> Optional[UserRow]:
        stmt = select(UserRow).where(func.lower(UserRow.username) == username.strip().lower())
        return self.db.execute(stmt).scalar_one_or_none()

    def create(self, username: str, password: str) -> UserRow:
        row = UserRow(
            id=str(uuid.uuid4()),
            username=username.strip(),
            password_hash=hash_password(password),
            is_active=True,
            token_version=0,
            created_at=_now_ms(),
        )
        self.db.add(row)
        self.db.commit()
        self.db.refresh(row)
        return row

    def authenticate(self, username: str, password: str) -> Optional[UserRow]:
        """校验用户名 + 口令；成功返回用户，失败返回 ``None``（不区分账号/密码错误）。"""
        user = self.get_by_username(username)
        if user is None or not user.is_active:
            return None
        if not verify_password(password, user.password_hash):
            return None
        return user

    def bump_token_version(self, user: UserRow) -> UserRow:
        """递增令牌版本，吊销该用户已签发的全部令牌（退出所有设备 / 改密）。"""
        user.token_version += 1
        self.db.commit()
        self.db.refresh(user)
        return user
