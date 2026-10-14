"""密码哈希与校验。

**安全要点（为什么这么做）**

1. **绝不存明文**，也不存"可逆加密"。数据库里只保存**单向哈希**。
2. **用专门的自适应哈希函数**（bcrypt），不要用 MD5/SHA256 之流：
   后者算得太快，离线爆破成本极低。bcrypt 自带 cost 因子，可随硬件升级调高。
3. **每个用户独立随机盐**（bcrypt 会自动生成并嵌在哈希串里），
   使相同密码得到不同哈希，防止彩虹表与"哪些用户同密码"的推断。
4. **校验必须恒定时间比较**（bcrypt 内部已做），避免通过响应时间侧信道猜密码。
5. **72 字节上限**：bcrypt 只取前 72 字节，超出部分被静默忽略。
   这会让"超长密码的不同后缀"被判为同一个密码，因此先做 SHA-256 预哈希再交给
   bcrypt（业内常见做法），既规避截断，又不改变熵。
6. **登录失败信息不区分"用户不存在"与"密码错误"**，避免账号枚举。
   用户不存在时也跑一次假校验，避免用响应时间区分。
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import re
from typing import Optional, Tuple

logger = logging.getLogger("rag.auth.password")

try:
    import bcrypt
    _BCRYPT_OK = True
except ImportError:                      # 理论上不会发生，留个兜底
    _BCRYPT_OK = False
    logger.warning("bcrypt 未安装，密码哈希降级为 scrypt（仍然安全但请尽快安装 bcrypt）")

# bcrypt cost：12 在现代 CPU 上约 0.2-0.3s，兼顾安全与体验。
# 注意：调高会让登录变慢，但攻击者离线爆破同样变慢，这是期望的性质。
BCRYPT_ROUNDS = 12

# scrypt 兜底参数（仅在 bcrypt 不可用时使用）
_SCRYPT_N, _SCRYPT_R, _SCRYPT_P = 2 ** 14, 8, 1

# 密码最小长度。上限用于防止超长输入拖慢哈希（DoS）
MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 128

# 常见弱密码（做最小拦截；真实系统应接入更大字典或撞库检测）
_COMMON_WEAK = {
    "12345678", "123456789", "1234567890", "password", "password1", "passw0rd",
    "qwertyui", "abc12345", "11111111", "88888888", "admin123", "root1234",
    "iloveyou", "letmein1", "welcome1", "changeme", "test1234", "rag12345",
}


def _prehash(password: str) -> bytes:
    """先用 SHA-256 压成固定长度，规避 bcrypt 的 72 字节截断。"""
    return hashlib.sha256(password.encode("utf-8")).digest()


def validate_password_strength(password: str) -> Tuple[bool, str]:
    """检查密码强度，返回 (是否合格, 原因)。"""
    if not isinstance(password, str):
        return False, "密码格式不正确"
    if len(password) < MIN_PASSWORD_LENGTH:
        return False, f"密码至少 {MIN_PASSWORD_LENGTH} 位"
    if len(password) > MAX_PASSWORD_LENGTH:
        return False, f"密码不能超过 {MAX_PASSWORD_LENGTH} 位"
    if password.lower() in _COMMON_WEAK:
        return False, "该密码过于常见，请更换"
    # 至少两类字符，避免"aaaaaaaa"这类
    kinds = sum(bool(re.search(p, password)) for p in
                (r"[a-z]", r"[A-Z]", r"\d", r"[^A-Za-z0-9]"))
    if kinds < 2:
        return False, "密码需包含字母、数字、符号中的至少两类"
    return True, ""


def validate_username(username: str) -> Tuple[bool, str]:
    """用户名规则：3-32 位，字母/数字/下划线/连字符/中文。"""
    if not isinstance(username, str) or not username.strip():
        return False, "用户名不能为空"
    username = username.strip()
    if not (3 <= len(username) <= 32):
        return False, "用户名长度需在 3-32 位之间"
    if not re.fullmatch(r"[\w\-\u4e00-\u9fff]+", username):
        return False, "用户名只能包含字母、数字、下划线、连字符或中文"
    return True, ""


def hash_password(password: str) -> str:
    """生成密码哈希。返回的字符串**可以安全地存进数据库**。"""
    ok, why = validate_password_strength(password)
    if not ok:
        raise ValueError(why)

    if _BCRYPT_OK:
        return bcrypt.hashpw(_prehash(password),
                             bcrypt.gensalt(rounds=BCRYPT_ROUNDS)).decode("ascii")

    # 兜底：scrypt（内存硬，同样适合存密码）
    import os
    salt = os.urandom(16)
    dk = hashlib.scrypt(_prehash(password), salt=salt,
                        n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=32)
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: Optional[str]) -> bool:
    """校验密码。任何异常都返回 False（**绝不因异常而放行**）。"""
    if not password or not stored:
        return False
    try:
        if stored.startswith("scrypt$"):
            _, n, r, p, salt_hex, dk_hex = stored.split("$")
            dk = hashlib.scrypt(_prehash(password), salt=bytes.fromhex(salt_hex),
                                n=int(n), r=int(r), p=int(p), dklen=32)
            return hmac.compare_digest(dk.hex(), dk_hex)
        if _BCRYPT_OK:
            return bcrypt.checkpw(_prehash(password), stored.encode("ascii"))
    except Exception as e:
        logger.warning("密码校验异常（按失败处理）：%s", e)
    return False


def dummy_verify() -> None:
    """用户不存在时也做一次等价耗时的校验，避免用响应时间枚举账号。

    坑：不要在这里现场 hashpw（那会让耗时翻倍，反而更容易区分）。
    模块加载时预生成一个假哈希，这里只做一次 checkpw 即可。
    """
    if _DUMMY_HASH:
        try:
            bcrypt.checkpw(_prehash("timing-equalizer"), _DUMMY_HASH)
        except Exception:
            pass


def needs_rehash(stored: Optional[str]) -> bool:
    """判断已有哈希是否需要按当前参数重算（cost 提升后渐进升级）。"""
    if not stored:
        return False
    if stored.startswith("scrypt$"):
        return _BCRYPT_OK          # 能升级到 bcrypt 就升级
    if not _BCRYPT_OK:
        return False
    try:
        # bcrypt 哈希串形如 $2b$12$...，第三段就是 cost
        parts = stored.split("$")
        if len(parts) >= 3 and parts[1].startswith("2"):
            return int(parts[2]) < BCRYPT_ROUNDS
    except Exception:
        return True                # 解析不了就当需要重算
    return False


# 模块加载时预生成假哈希（见 dummy_verify 的说明）
if _BCRYPT_OK:
    try:
        _DUMMY_HASH = bcrypt.hashpw(_prehash("timing-equalizer"),
                                    bcrypt.gensalt(rounds=BCRYPT_ROUNDS))
    except Exception:
        _DUMMY_HASH = None
else:
    _DUMMY_HASH = None
