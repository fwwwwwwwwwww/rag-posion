"""鉴权接口端到端测试。

用**临时数据库**，不碰项目里的 rag_experiments.db。
覆盖：
  1. 未登录访问受保护接口 → 401
  2. 登录成功 → 下发 HttpOnly Cookie，且响应体**不含**任何令牌
  3. 带 Cookie 访问受保护接口 → 200
  4. 错误密码 → 401，且措辞不泄露"用户是否存在"
  5. 改密：旧密码校验、新密码强度
  6. 登录失败限流 / 账号锁定
  7. 响应中绝不出现 password_hash
"""
import os
import sys
import tempfile

TMPDIR = tempfile.mkdtemp(prefix="rag_auth_test_")
DB_PATH = os.path.join(TMPDIR, "auth_test.db")
os.environ["DATABASE_URL"] = f"sqlite:///{DB_PATH}"
os.environ["SESSION_SECRET"] = "test-secret-for-unit-tests-only"
os.environ["SESSION_COOKIE_SECURE"] = "false"

sys.path.insert(0, r"E:\RAG2")

from fastapi.testclient import TestClient

from app.db.engine import SessionLocal
from app.db.init_db import init_db
from app.services import auth_service
from server import app

init_db()
client = TestClient(app)

PASS = []
FAIL = []


def check(name, cond, extra=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'OK  ' if cond else 'FAIL'} {name}" + (f"  {extra}" if extra else ""))


# ---------------- 1. 未登录 ----------------
print("\n[1] 未登录访问受保护接口")
r = client.get("/api/agent/status")
check("未登录访问 /api/agent/status 返回 401", r.status_code == 401, f"实际 {r.status_code}")
body = r.json()
check("返回 code=unauthenticated", body.get("code") == "unauthenticated", str(body))

r = client.get("/api/auth/me")
check("未登录访问 /api/auth/me 返回 401", r.status_code == 401, f"实际 {r.status_code}")

# ---------------- 2. 创建用户 ----------------
print("\n[2] 创建用户（密码入库前必须哈希）")
db = SessionLocal()
try:
    user = auth_service.create_user(db, "alice", "AlicePass1!", display_name="Alice")
    uid = user.id
    check("创建用户成功", uid is not None)
    # 直接查库确认没存明文
    from app.db.models import User
    row = db.get(User, uid)
    check("库中不含明文密码", "AlicePass1!" not in (row.password_hash or ""))
    check("库中密码为 bcrypt 哈希", (row.password_hash or "").startswith("$2"))
    check("哈希长度合理", len(row.password_hash) == 60, str(len(row.password_hash)))
finally:
    db.close()

# ---------------- 3. 弱密码/弱用户名应被拒 ----------------
print("\n[3] 密码与用户名规则")
db = SessionLocal()
try:
    for bad in ["short1!", "12345678", "alllowercase"]:
        try:
            auth_service.create_user(db, f"u_{abs(hash(bad)) % 9999}", bad)
            check(f"弱密码 {bad!r} 应被拒绝", False)
        except ValueError as e:
            check(f"弱密码 {bad!r} 被拒绝", True, str(e))
    try:
        auth_service.create_user(db, "ab", "ValidPass1!")
        check("过短用户名应被拒绝", False)
    except ValueError:
        check("过短用户名被拒绝", True)
    try:
        auth_service.create_user(db, "alice", "AnotherPass1!")
        check("重复用户名应被拒绝", False)
    except ValueError:
        check("重复用户名被拒绝", True)
finally:
    db.close()

# ---------------- 4. 登录 ----------------
print("\n[4] 登录")
r = client.post("/api/auth/login",
                json={"username": "alice", "password": "WrongPass1!"})
check("错误密码返回 401", r.status_code == 401, f"实际 {r.status_code}")
wrong_msg = r.json().get("detail", "")

r2 = client.post("/api/auth/login",
                 json={"username": "no_such_user", "password": "WrongPass1!"})
nonexistent_msg = r2.json().get("detail", "")
check("不存在的用户与密码错误措辞一致（防账号枚举）",
      wrong_msg == nonexistent_msg, f"{wrong_msg!r} vs {nonexistent_msg!r}")

r = client.post("/api/auth/login", json={"username": "alice", "password": "AlicePass1!"})
check("正确密码登录成功", r.status_code == 200, f"实际 {r.status_code}")
data = r.json()
check("响应含用户信息", data.get("username") == "alice", str(data))
check("响应**不含** password_hash", "password_hash" not in data, str(list(data.keys())))
check("响应**不含**任何 token 字段",
      not any("token" in k.lower() for k in data), str(list(data.keys())))
cookie = r.headers.get("set-cookie", "")
check("下发了会话 Cookie", "rag_session=" in cookie, cookie[:80])
check("Cookie 标记 HttpOnly（防 XSS 窃取）", "HttpOnly" in cookie, cookie[:120])
check("Cookie 标记 SameSite", "samesite" in cookie.lower(), cookie[:120])

# ---------------- 5. 带 Cookie 访问 ----------------
print("\n[5] 登录后可访问受保护接口")
r = client.get("/api/auth/me")
check("/api/auth/me 返回 200", r.status_code == 200, f"实际 {r.status_code}")
me = r.json()
check("me 返回正确用户", me.get("username") == "alice")
check("me 不含 password_hash", "password_hash" not in me, str(list(me.keys())))

# ---------------- 6. 改密 ----------------
print("\n[6] 修改密码")
r = client.post("/api/auth/change-password",
                json={"old_password": "WrongOld1!", "new_password": "NewPass123!"})
check("旧密码错误被拒绝", r.status_code == 400, f"实际 {r.status_code}")

r = client.post("/api/auth/change-password",
                json={"old_password": "AlicePass1!", "new_password": "weak"})
# 422 = 被 Pydantic 的 min_length=8 拦下（schema 层）；400 = 被服务层强度校验拦下。
# 两者都算"拒绝"，schema 层更早，属于纵深防御。
check("新密码过弱被拒绝", r.status_code in (400, 422), f"实际 {r.status_code}")

r = client.post("/api/auth/change-password",
                json={"old_password": "AlicePass1!", "new_password": "AlicePass1!"})
check("新旧密码相同被拒绝", r.status_code == 400, f"实际 {r.status_code}")

r = client.post("/api/auth/change-password",
                json={"old_password": "AlicePass1!", "new_password": "NewPass123!"})
check("改密成功", r.status_code == 200, f"实际 {r.status_code}")

db = SessionLocal()
try:
    from app.db.models import User
    row = db.get(User, uid)
    from app.security import verify_password
    check("新密码可校验", verify_password("NewPass123!", row.password_hash))
    check("旧密码已失效", not verify_password("AlicePass1!", row.password_hash))
    check("库中仍无明文", "NewPass123!" not in (row.password_hash or ""))
finally:
    db.close()

# ---------------- 7. 登出 ----------------
print("\n[7] 登出")
r = client.post("/api/auth/logout")
check("登出返回 200", r.status_code == 200)
r = client.get("/api/auth/me")
check("登出后访问受保护接口返回 401", r.status_code == 401, f"实际 {r.status_code}")

# ---------------- 8. 登录限流 ----------------
print("\n[8] 登录失败限流/锁定")
for i in range(auth_service.MAX_FAILED_ATTEMPTS):
    client.post("/api/auth/login", json={"username": "alice", "password": "Bad1Pass!"})
r = client.post("/api/auth/login", json={"username": "alice", "password": "NewPass123!"})
detail = r.json().get("detail", "")
check("连续失败后账号被锁定", "锁定" in detail, detail)
check("锁定期间即使密码正确也拒绝", r.status_code == 401, f"实际 {r.status_code}")

# ---------------- 9. 非法令牌 ----------------
print("\n[9] 伪造/篡改令牌")
r = client.get("/api/auth/me", headers={"Authorization": "Bearer forged.token.value"})
check("伪造令牌被拒绝", r.status_code == 401, f"实际 {r.status_code}")

sys.path.insert(0, r"E:\RAG2")
from app.api.auth import signer
good = signer.issue("alice", uid)
tampered = good[:-4] + ("aaaa" if not good.endswith("aaaa") else "bbbb")
r = client.get("/api/auth/me", headers={"Authorization": f"Bearer {tampered}"})
check("篡改签名的令牌被拒绝", r.status_code == 401, f"实际 {r.status_code}")

r = client.get("/api/auth/me", headers={"Authorization": f"Bearer {good}"})
check("合法令牌可访问", r.status_code == 200, f"实际 {r.status_code}")

print(f"\n{'=' * 58}")
print(f"通过 {len(PASS)} 项，失败 {len(FAIL)} 项")
if FAIL:
    print("失败项：")
    for f in FAIL:
        print("  -", f)
print("=" * 58)
sys.exit(1 if FAIL else 0)
