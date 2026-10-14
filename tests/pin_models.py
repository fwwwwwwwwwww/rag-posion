"""把 embedding 模型与对话模型都固定在显存里常驻。

背景：Ollama 默认 5 分钟卸载模型。本项目要在两个模型间来回切换
（检索用 qwen3-embedding 4.36GB，对话/判定用 qwen3.5:2b 2.35GB），
每次切换都重新加载，窗口期还会返回**空响应**——实测这会让判定把
一次成功的攻击误判为"未采纳"。

两个模型共 6.7GB，显卡 8GB，本来是可以同时常驻的。
"""
import json
import sys
import urllib.request

sys.path.insert(0, r"E:\RAG2")

BASE = "http://localhost:11434"
EMBED = "qwen3-embedding:latest"
LLM = "qwen3.5:2b"


def post(path, payload, timeout=600):
    req = urllib.request.Request(
        f"{BASE}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


print("1) 固定 embedding 模型常驻（keep_alive=-1）...")
try:
    r = post("/api/embed", {"model": EMBED,
                            "input": ["预热"], "keep_alive": -1})
    dim = len((r.get("embeddings") or [[]])[0])
    print(f"   OK，向量维度 {dim}")
except Exception as e:
    print("   失败:", e)

print("2) 固定对话模型常驻（keep_alive=-1）...")
try:
    r = post("/api/chat", {"model": LLM,
                           "messages": [{"role": "user", "content": "ok"}],
                           "stream": False,
                           "think": False,
                           "keep_alive": -1,
                           "options": {"num_predict": 8}})
    print("   OK，回复:", (r.get("message", {}).get("content") or "")[:20])
except Exception as e:
    print("   失败:", e)

print("\n3) 当前常驻模型:")
try:
    with urllib.request.urlopen(f"{BASE}/api/ps", timeout=10) as r:
        ps = json.loads(r.read().decode())
    for m in ps.get("models", []):
        print(f"   {m.get('name'):28s} vram={m.get('size_vram', 0)/1e9:.2f}GB "
              f"expires={m.get('expires_at')}")
except Exception as e:
    print("   查询失败:", e)
