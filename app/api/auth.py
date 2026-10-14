"""鉴权接口：登录 / 登出 / 当前用户 / 改密。

令牌传递方式（两种都支持）
--------------------------
1. **HttpOnly Cookie**（浏览器默认走这条）
   - ``HttpOnly``  → JS 读不到，**防 XSS 窃取**
   - ``SameSite=Lax`` → 缓解 CSRF
   - ``Secure``   → 仅在 HTTPS 下发送（本地 http 调试时自动关闭）
   - ``Path=/``
   - 前端无需也无法读取令牌内容，因此不存在"令牌存 localStorage 被 XSS 拿走"的问题。
2. **Authorization: Bearer <token>**（脚本/接口调用）

安全措施清单
------------
- 密码：bcrypt 加盐哈希入库，**永不返回**
- 登录：失败限流 + 账号锁定；错误措辞不区分"用户不存在/密码错误"
- 会话：HMAC 签名 + 过期时间 + 恒定时间比较
- 响应：``User.to_safe_dict()`` 保证不含 password_hash
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.config import SESSION_COOKIE_NAME, SESSION_TTL, SESSION_SECRET
from app.db.engine import get_db
from app.db.models import User
from app.security import get_signer
from app.services import auth_service

logger = logging.getLogger("rag.api.auth")

router = APIRouter(prefix="/auth", tags=["auth"])

# 全局单例：必须与中间件（app/auth.py）用同一个密钥，否则签发的令牌验不过
signer = get_signer(SESSION_SECRET, SESSION_TTL)

# Cookie 是否加 Secure：本地 http 调试时不能加，否则浏览器不回传
_COOKIE_SECURE = os.getenv("SESSION_COOKIE_SECURE", "false").lower() in ("true", "1", "yes")


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1, max_length=128)


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=32)
    password: str = Field(..., min_length=8, max_length=128)
    display_name: Optional[str] = Field(None, max_length=64)


class ChangePasswordRequest(BaseModel):
    old_password: str = Field(..., min_length=1, max_length=128)
    new_password: str = Field(..., min_length=8, max_length=128)


class UserOut(BaseModel):
    id: int
    username: str
    display_name: Optional[str] = None
    role: str
    is_active: bool
    last_login_at: Optional[str] = None
    created_at: Optional[str] = None


def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        max_age=SESSION_TTL,
        httponly=True,          # JS 不可读 → 防 XSS 窃取
        samesite="lax",         # 缓解 CSRF
        secure=_COOKIE_SECURE,  # HTTPS 下自动为 True
        path="/",
    )


def _clear_session_cookie(response: Response) -> None:
    response.delete_cookie(SESSION_COOKIE_NAME, path="/")


def current_user(request: Request, db: Session = Depends(get_db)) -> User:
    """从 Cookie 或 Bearer 头解析当前用户。失败抛 401。"""
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth[7:]

    payload = signer.verify(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="登录状态已失效，请重新登录",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = db.get(User, payload.get("uid"))
    # 用户被删除/禁用后，旧令牌立即失效（不只是靠过期时间）
    if user is None or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="账号不存在或已被禁用")
    return user


def require_admin(user: User = Depends(current_user)) -> User:
    if user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="需要管理员权限")
    return user


# ---------------- 路由 ----------------
@router.post("/login", response_model=UserOut)
async def login(payload: LoginRequest, response: Response,
                db: Session = Depends(get_db)):
    """登录。成功后在 HttpOnly Cookie 写入会话令牌，响应体不含任何令牌。"""
    user, err = auth_service.authenticate(db, payload.username, payload.password)
    if user is None:
        # 统一 401 + 统一措辞，避免账号枚举
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail=err or "用户名或密码不正确")

    token = signer.issue(user.username, user.id)
    _set_session_cookie(response, token)
    logger.info("用户 %s 登录成功", user.username)
    return UserOut(**user.to_safe_dict())


@router.post("/logout")
async def logout(response: Response):
    """登出：清除 Cookie。"""
    _clear_session_cookie(response)
    return {"status": "ok"}


@router.get("/me", response_model=UserOut)
async def me(user: User = Depends(current_user)):
    """返回当前登录用户（不含任何密码字段）。"""
    return UserOut(**user.to_safe_dict())


@router.post("/change-password")
async def change_password(payload: ChangePasswordRequest,
                          user: User = Depends(current_user),
                          db: Session = Depends(get_db)):
    """修改自己的密码。必须提供原密码。"""
    try:
        auth_service.change_password(db, user, payload.old_password,
                                     payload.new_password)
    except auth_service.AuthError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return {"status": "ok", "message": "密码已更新，请重新登录"}


@router.get("/users", response_model=list[UserOut])
async def list_users(user: User = Depends(require_admin),
                     db: Session = Depends(get_db)):
    """列出用户（仅管理员）。"""
    from sqlalchemy import select
    rows = db.scalars(select(User).order_by(User.id)).all()
    return [UserOut(**u.to_safe_dict()) for u in rows]


@router.post("/users", response_model=UserOut, status_code=201)
async def register_user(payload: RegisterRequest,
                        user: User = Depends(require_admin),
                        db: Session = Depends(get_db)):
    """新建用户（仅管理员）。"""
    try:
        u = auth_service.create_user(db, payload.username, payload.password,
                                     payload.display_name)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return UserOut(**u.to_safe_dict())


@router.get("/needs-setup")
async def needs_setup(db: Session = Depends(get_db)):
    """是否还没有任何用户（前端据此决定是否提示初始账号）。"""
    return {"needs_setup": auth_service.count_users(db) == 0}
