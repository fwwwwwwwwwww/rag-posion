"""会话令牌：HMAC 签名的自包含 token。

设计取舍
--------
- **不用 JWT 库**：项目没装 PyJWT，而一个"带签名+过期"的令牌只需 HMAC-SHA256，
  自己实现反而更可控（JWT 的 `alg=none` 之类的历史坑也就不用管了）。
- **格式**：``base64url(payload).base64url(signature)``，
  payload 是 ``{"sub": 用户名, "uid": id, "iat": 签发时间, "exp": 过期时间}``。
- **签名恒定时间比较**，防时序侧信道。
- **密钥来自配置**：`SESSION_SECRET`。若未配置则启动时随机生成
  （副作用：进程重启后所有会话失效——比用一个硬编码默认密钥安全得多）。
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import secrets
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("rag.auth.token")

# 会话有效期（秒），默认 12 小时
DEFAULT_TTL = 12 * 3600


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64d(text: str) -> bytes:
    pad = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + pad)


class SessionSigner:
    """签发并校验会话令牌。"""

    def __init__(self, secret: str, ttl: int = DEFAULT_TTL,
                 secret_file: Optional[str] = None):
        if not secret:
            # 未配置密钥：生成随机密钥并**持久化到文件**。
            #
            # 为什么必须持久化：signer 会被实例化多次（中间件一份、路由一份），
            # 而且生产环境可能起多个 worker。若每次都各自随机生成，
            # "A 签发的令牌 B 验不过"，表现为**登录成功后立刻 401**。
            # 实测正是这个 bug（排查脚本已清理，这条验证链路由
            # tests/test_login_flow.py 保留）。
            secret = self._load_or_create_secret(secret_file)
        self._secret = secret.encode("utf-8")
        self.ttl = ttl

    @staticmethod
    def _load_or_create_secret(secret_file: Optional[str]) -> str:
        import os
        import secrets as _secrets

        if not secret_file:
            base = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))          # 项目根目录
            secret_file = os.path.join(base, "data", ".session_secret")

        try:
            if os.path.exists(secret_file):
                with open(secret_file, "r", encoding="utf-8") as f:
                    s = f.read().strip()
                if s:
                    return s

            s = _secrets.token_urlsafe(48)
            os.makedirs(os.path.dirname(secret_file), exist_ok=True)
            with open(secret_file, "w", encoding="utf-8") as f:
                f.write(s)
            # 尽量收紧权限（Windows 上 chmod 行为有限，失败也不影响功能）
            try:
                os.chmod(secret_file, 0o600)
            except Exception:
                pass
            logger.warning(
                "未配置 SESSION_SECRET，已生成随机密钥并保存到 %s。"
                "生产环境建议用环境变量显式指定（多机部署时尤其重要）。",
                secret_file)
            return s
        except Exception as e:
            # 落盘失败也不能让服务起不来：退化为进程内随机密钥
            logger.warning("会话密钥落盘失败（%s），改用进程内临时密钥："
                           "重启后登录会失效", e)
            return _secrets.token_urlsafe(48)

    def _sign(self, payload_b64: str) -> str:
        mac = hmac.new(self._secret, payload_b64.encode("ascii"), hashlib.sha256)
        return _b64e(mac.digest())

    def issue(self, username: str, uid: int, ttl: Optional[int] = None) -> str:
        now = int(time.time())
        payload = {
            "sub": username,
            "uid": uid,
            "iat": now,
            "exp": now + int(ttl or self.ttl),
            # 随机串：即使同一用户同一秒登录，令牌也不同
            "jti": secrets.token_urlsafe(8),
        }
        payload_b64 = _b64e(json.dumps(payload, separators=(",", ":"),
                                       ensure_ascii=False).encode("utf-8"))
        return f"{payload_b64}.{self._sign(payload_b64)}"

    def verify(self, token: Optional[str]) -> Optional[Dict[str, Any]]:
        """校验令牌。任何异常/过期/签名不符都返回 None。"""
        if not token or "." not in token:
            return None
        try:
            payload_b64, sig = token.rsplit(".", 1)
        except ValueError:
            return None

        expected = self._sign(payload_b64)
        # 恒定时间比较，避免逐字节猜测签名
        if not hmac.compare_digest(sig, expected):
            return None

        try:
            payload = json.loads(_b64d(payload_b64).decode("utf-8"))
        except Exception:
            return None

        if not isinstance(payload, dict):
            return None
        exp = payload.get("exp")
        if not isinstance(exp, int) or exp < int(time.time()):
            return None
        if not payload.get("sub"):
            return None
        return payload


# 全局共享的签名器（单例）。
#
# 为什么必须是单例：签名依赖密钥，而"签发"与"校验"可能发生在不同模块
# （app/api/auth.py 只签发、app/auth.py 中间件只校验）。
# 如果各自 new 一个，未配置 SESSION_SECRET 时两者会拿到不同的随机密钥，
# 结果就是**登录成功却立刻 401**。这也是多 worker 部署下必须共享密钥的原因。
_SHARED_SIGNER: Optional[SessionSigner] = None


def get_signer(secret: str = "", ttl: int = DEFAULT_TTL) -> SessionSigner:
    """获取全局唯一的签名器（首次调用时按参数初始化）。"""
    global _SHARED_SIGNER
    if _SHARED_SIGNER is None:
        _SHARED_SIGNER = SessionSigner(secret, ttl=ttl)
    return _SHARED_SIGNER
