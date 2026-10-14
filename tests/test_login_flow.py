"""对**真实运行中的服务**做登录流程集成验证。

覆盖前端依赖的完整链路：
  未登录被拦 → 登录 → 带 Cookie 访问 → 会话有效性 → 登出 → 再被拦
"""
import json
import sys
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8000"
USER = "admin"
PWD = "AdminTest123!"

PASS, FAIL = [], []


def check(name, cond, extra=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'OK  ' if ok(cond) else 'FAIL'} {name}" + (f"  {extra}" if extra else ""))


def ok(c):
    return bool(c)


class Session:
    """带 Cookie 的简易会话。"""

    def __init__(self):
        self.cookie = ""

    def req(self, method, path, body=None, use_cookie=True):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(BASE + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        if use_cookie and self.cookie:
            req.add_header("Cookie", self.cookie)
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read().decode()
                # 记录 Set-Cookie
                sc = r.headers.get("Set-Cookie")
                if sc and "rag_session=" in sc:
                    self.cookie = sc.split(";")[0]
                return r.status, (json.loads(raw) if raw.strip().startswith(("{", "[")) else raw), dict(r.headers)
        except urllib.error.HTTPError as e:
            raw = e.read().decode()
            try:
                body = json.loads(raw)
            except Exception:
                body = raw
            return e.code, body, dict(e.headers)


s = Session()

print("[1] 未登录：受保护接口应被拦")
st, body, _ = s.req("GET", "/api/agent/status", use_cookie=False)
check("/api/agent/status 未登录返回 401", st == 401, f"实际 {st}")
check("返回 code=unauthenticated", isinstance(body, dict) and body.get("code") == "unauthenticated",
      str(body)[:100])

st, body, _ = s.req("GET", "/api/experiments", use_cookie=False)
check("/api/experiments 未登录返回 401", st == 401, f"实际 {st}")

print("\n[2] 公开路径应放行")
st, body, _ = s.req("GET", "/api/health", use_cookie=False)
check("/api/health 免登录", st == 200, f"实际 {st}")
st, body, _ = s.req("GET", "/api/attacks", use_cookie=False)
check("/api/attacks 免登录", st == 200, f"实际 {st}")

print("\n[3] 错误密码")
st, body, _ = s.req("POST", "/api/auth/login",
                    {"username": USER, "password": "WrongPassword1!"}, use_cookie=False)
check("错误密码返回 401", st == 401, f"实际 {st}")

print("\n[4] 正确登录")
st, body, headers = s.req("POST", "/api/auth/login",
                          {"username": USER, "password": PWD}, use_cookie=False)
check("登录成功 200", st == 200, f"实际 {st} {str(body)[:120]}")
if st == 200:
    check("响应含用户名", body.get("username") == USER, str(body))
    check("响应**不含** password_hash", "password_hash" not in body, str(list(body.keys())))
    check("已获得会话 Cookie", bool(s.cookie), s.cookie[:40])
print()

print("[5] 带 Cookie 访问受保护接口")
st, body, _ = s.req("GET", "/api/auth/me")
check("/api/auth/me 返回 200", st == 200, f"实际 {st}")
if st == 200:
    check("me 是当前用户", body.get("username") == USER)
    check("me 不含 password_hash", "password_hash" not in body, str(list(body.keys())))

st, body, _ = s.req("GET", "/api/agent/status")
check("/api/agent/status 带会话可访问", st == 200, f"实际 {st}")

st, body, _ = s.req("GET", "/api/attacks/strategies")
check("/api/attacks/strategies 带会话可访问", st == 200, f"实际 {st}")

print("\n[6] 登出后会话立即失效")
st, body, _ = s.req("POST", "/api/auth/logout")
check("登出返回 200", st == 200, f"实际 {st}")
# 手动清掉本地 Cookie 模拟浏览器已被清除
s.cookie = ""
st, body, _ = s.req("GET", "/api/auth/me")
check("登出后 /api/auth/me 返回 401", st == 401, f"实际 {st}")

print("\n[7] 静态资源")
for path in ["/", "/index.html"]:
    st, body, _ = s.req("GET", path, use_cookie=False)
    check(f"{path} 返回 200", st == 200, f"实际 {st}")

print(f"\n{'=' * 58}")
print(f"通过 {len(PASS)} 项，失败 {len(FAIL)} 项")
for f in FAIL:
    print("  - 失败:", f)
print("=" * 58)
sys.exit(1 if FAIL else 0)
