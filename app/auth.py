"""鉴权中间件。

同时支持两种凭据：

1. **会话 Cookie**（浏览器默认）：登录后由服务端写入 HttpOnly Cookie。
   HttpOnly 让 JS 读不到，因此即使页面被 XSS 注入也偷不走会话；
   SameSite=Lax 缓解 CSRF；Secure 在 HTTPS 下自动启用。
2. **Authorization: Bearer <token>**：脚本与接口调用，
   支持旧的静态 AUTH_TOKEN（便于自动化）以及登录签发的会话令牌。

校验顺序：先看 Bearer 头（脚本优先），再看 Cookie。
"""
import hmac
import logging

from fastapi import HTTPException, status

from app.config import AUTH_TOKEN, PUBLIC_PATHS, SESSION_COOKIE_NAME, SESSION_SECRET, SESSION_TTL
from app.security import get_signer

logger = logging.getLogger("rag.auth")

# 用全局单例：与 app/api/auth.py 的签发方共享同一密钥，
# 否则"登录成功却立刻 401"（各自随机生成密钥时必然发生）
_signer = get_signer(SESSION_SECRET, SESSION_TTL)


def _valid_session(token: str) -> bool:
    return _signer.verify(token) is not None


def _valid_static_token(token: str) -> bool:
    return bool(AUTH_TOKEN) and hmac.compare_digest(token, AUTH_TOKEN)


def is_authenticated(auth_header: str = "", cookie_token: str = "") -> bool:
    """判断凭据是否有效（供中间件与依赖共用）。"""
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        # 会话令牌或静态令牌，任一有效即可
        if _valid_session(token) or _valid_static_token(token):
            return True
    if cookie_token and _valid_session(cookie_token):
        return True
    return False


async def verify_token(
    credentials=None,
) -> str:
    """兼容旧的依赖用法（HTTPBearer）。"""
    if credentials is None or not getattr(credentials, "credentials", None):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少凭据，请先登录或提供 Bearer Token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = credentials.credentials
    if _valid_session(token) or _valid_static_token(token):
        return token
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="凭据无效或已过期",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def auth_middleware(request, call_next):
    """全局鉴权中间件。公开路径与非 /api 路径放行，其余要求有效凭据。"""
    from fastapi.responses import JSONResponse

    path = request.url.path
    if not path.startswith("/api"):
        return await call_next(request)
    if path in PUBLIC_PATHS:
        return await call_next(request)

    auth_header = request.headers.get("Authorization", "")
    cookie_token = request.cookies.get(SESSION_COOKIE_NAME, "")

    if is_authenticated(auth_header, cookie_token):
        return await call_next(request)

    # 区分"没带凭据"与"凭据失效"，方便前端决定是跳登录页还是提示重新登录
    has_any = auth_header.startswith("Bearer ") or bool(cookie_token)
    return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content={
            "detail": "登录状态已失效，请重新登录" if has_any
                      else "未登录，请先登录",
            "code": "session_expired" if has_any else "unauthenticated",
        },
        headers={"WWW-Authenticate": "Bearer"},
    )
