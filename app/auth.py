"""简易 Token 鉴权（演示用）。

通过 Bearer Token 校验，令牌在 .env 或环境变量 AUTH_TOKEN 中配置，
默认值为 `rag-demo-token`。
"""
import hmac

from fastapi import HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.config import AUTH_TOKEN, PUBLIC_PATHS

_bearer = HTTPBearer(auto_error=False)


async def verify_token(
    credentials: HTTPAuthorizationCredentials = None,
) -> str:
    """Bearer Token 校验依赖。通过则返回 token，失败则 401。"""
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少 Bearer Token，请在 Authorization 头中提供",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = credentials.credentials
    if not hmac.compare_digest(token, AUTH_TOKEN):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token 无效",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token


async def auth_middleware(request, call_next):
    """全局鉴权中间件。公开路径和非 /api 路径放行，其余要求 Bearer Token。"""
    from fastapi.responses import JSONResponse

    path = request.url.path
    if not path.startswith("/api"):
        return await call_next(request)
    if path in PUBLIC_PATHS:
        return await call_next(request)

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": "缺少 Bearer Token"},
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = auth_header[7:]
    if not hmac.compare_digest(token, AUTH_TOKEN):
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": "Token 无效"},
            headers={"WWW-Authenticate": "Bearer"},
        )
    return await call_next(request)
