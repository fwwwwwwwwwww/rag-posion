"""用户与会话服务。

职责：
  - 用户注册 / 认证 / 改密（密码一律经 bcrypt 哈希后入库）
  - 登录失败限流（锁定策略），抵御在线暴力破解
  - 首次启动时按环境变量创建管理员账号
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Optional, Tuple

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import User
from app.security import (
    dummy_verify,
    hash_password,
    needs_rehash,
    validate_password_strength,
    validate_username,
    verify_password,
)

logger = logging.getLogger("rag.auth.service")

# 连续失败达到该次数后锁定账号
MAX_FAILED_ATTEMPTS = 5
# 锁定时长（分钟）。真实系统常用递增锁定或验证码，这里用固定值。
LOCK_MINUTES = 15


class AuthError(Exception):
    """认证相关错误（对外只暴露安全措辞，不泄露内部细节）。"""


def get_user(db: Session, username: str) -> Optional[User]:
    if not username:
        return None
    return db.scalar(select(User).where(User.username == username))


def count_users(db: Session) -> int:
    from sqlalchemy import func
    return int(db.scalar(select(func.count()).select_from(User)) or 0)


def create_user(db: Session, username: str, password: str,
                display_name: Optional[str] = None,
                role: str = "user") -> User:
    """创建用户。用户名/密码都会先做校验，不合格直接抛 ValueError。"""
    ok, why = validate_username(username)
    if not ok:
        raise ValueError(why)
    username = username.strip()

    ok, why = validate_password_strength(password)
    if not ok:
        raise ValueError(why)

    if get_user(db, username) is not None:
        raise ValueError("该用户名已被占用")

    user = User(
        username=username,
        password_hash=hash_password(password),   # 只存哈希
        display_name=display_name or username,
        role=role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    logger.info("已创建用户 %s（角色 %s）", username, role)
    return user


def authenticate(db: Session, username: str, password: str) -> Tuple[Optional[User], str]:
    """校验用户名密码。

    返回 ``(user, error)``：成功时 user 非空、error 为空。

    **安全措辞**：无论是"用户不存在"还是"密码错误"，对外都返回同一句话，
    避免账号枚举。用户不存在时也跑一次等价的假校验，避免用响应时间区分。
    """
    user = get_user(db, username or "")

    if user is None:
        dummy_verify()
        return None, "用户名或密码不正确"

    # 账号锁定检查（放在密码校验之前，避免绕过锁定继续试）
    now = datetime.now()
    if user.locked_until and user.locked_until > now:
        left = int((user.locked_until - now).total_seconds() // 60) + 1
        return None, f"账号已临时锁定，请 {left} 分钟后再试"

    if not user.is_active:
        return None, "账号已被禁用"

    if not verify_password(password, user.password_hash):
        user.failed_attempts = (user.failed_attempts or 0) + 1
        if user.failed_attempts >= MAX_FAILED_ATTEMPTS:
            user.locked_until = now + timedelta(minutes=LOCK_MINUTES)
            user.failed_attempts = 0
            logger.warning("用户 %s 连续失败过多，锁定至 %s",
                           user.username, user.locked_until)
        db.commit()
        return None, "用户名或密码不正确"

    # 登录成功：清零失败计数；必要时按新参数重算哈希
    user.failed_attempts = 0
    user.locked_until = None
    user.last_login_at = now
    if needs_rehash(user.password_hash):
        try:
            user.password_hash = hash_password(password)
            logger.info("已为 %s 升级密码哈希参数", user.username)
        except Exception:
            pass
    db.commit()
    db.refresh(user)
    return user, ""


def change_password(db: Session, user: User, old_password: str,
                    new_password: str) -> None:
    """修改密码。必须先验证旧密码。"""
    if not verify_password(old_password, user.password_hash):
        raise AuthError("原密码不正确")
    ok, why = validate_password_strength(new_password)
    if not ok:
        raise ValueError(why)
    if verify_password(new_password, user.password_hash):
        raise ValueError("新密码不能与原密码相同")
    user.password_hash = hash_password(new_password)
    user.updated_at = datetime.now()
    db.commit()
    logger.info("用户 %s 已修改密码", user.username)


def ensure_default_admin(db: Session) -> Optional[User]:
    """首次启动时创建管理员账号。

    密码来源优先级：环境变量 ``ADMIN_PASSWORD`` > 随机生成并打印到日志。
    **绝不使用硬编码默认密码**（那等于给所有人留了后门）。
    """
    if db.scalar(select(User.id).limit(1)) is not None:
        return None

    username = os.getenv("ADMIN_USERNAME", "admin")
    password = os.getenv("ADMIN_PASSWORD", "").strip()

    generated = False
    if not password:
        import secrets
        # 随机密码：满足强度校验（字母+数字+符号）
        password = secrets.token_urlsafe(12) + "a1!"
        generated = True

    try:
        user = create_user(db, username, password,
                           display_name="管理员", role="admin")
    except ValueError as e:
        logger.error("创建默认管理员失败：%s", e)
        return None

    if generated:
        # 只打印一次，提示尽快改密；生产环境应改用环境变量注入
        logger.warning("=" * 62)
        logger.warning("已创建默认管理员账号（随机密码，仅本次显示）")
        logger.warning("    用户名: %s", username)
        logger.warning("    密码  : %s", password)
        logger.warning("请登录后立即修改密码，或用 ADMIN_PASSWORD 环境变量指定。")
        logger.warning("=" * 62)
    return user
