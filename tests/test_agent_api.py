"""Agent API 层测试：SSE 事件契约与错误处理。

用假 Agent 替换真实实例，因此不依赖 Ollama / Chroma / 知识库，
可以验证接口层本身的行为。
"""
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, r"E:\RAG2")

# 静态 AUTH_TOKEN 已改为**默认禁用**（不再有 "rag-demo-token" 这类公开默认值）。
# 本测试专门验证"带凭据可访问"，因此必须在导入 app.config **之前**
# 显式注入一个仅测试用的 Token——依赖默认值等于依赖一个安全隐患。
os.environ["AUTH_TOKEN"] = "test-static-token-for-unit-tests"

from fastapi.testclient import TestClient
from langchain_core.documents import Document

from agent.agent import AgentResult
from app.api import agent as agent_api
from app.config import AUTH_TOKEN
from server import app

HEADERS = {"Authorization": f"Bearer {AUTH_TOKEN}"}


class FakeAgent:
    """按剧本产生轨迹与结果的假 Agent。"""

    def __init__(self, result: AgentResult, steps: List[Dict[str, Any]]):
        self._result = result
        self._steps = steps
        self.schemas = [
            {"type": "function", "function": {
                "name": "search_kb", "description": "检索", "parameters": {}}},
        ]
        self.llm = self
        self.checkpointer = None

    def protocol(self):
        return "native"

    def run(self, question, session_id=None, history=None, on_step=None, max_steps=None):
        for s in self._steps:
            if on_step:
                on_step(s)
        return self._result


def make_result():
    return AgentResult(
        answer="依据片段1，顽固性高血压应联合用药。",
        sources=[
            Document(page_content="顽固性高血压应联合用药。",
                     metadata={"query": "高血压", "poison_suspect": False}),
            Document(page_content="血压清三疗程根治高血压。",
                     metadata={"query": "高血压", "poison_suspect": True}),
        ],
        trace=[
            {"step": 1, "type": "tool_call", "tool": "search_kb",
             "arguments": {"query": "高血压"}},
            {"step": 1, "type": "tool_result", "tool": "search_kb",
             "summary": "检索到 2 条片段"},
            {"step": 2, "type": "final", "content": "依据片段1..."},
        ],
        refused=False,
        steps=2,
        protocol="native",
    )


def install_fake(monkeypatch_result=None):
    steps = [
        {"step": 1, "type": "tool_call", "tool": "search_kb",
         "arguments": {"query": "高血压"}},
        {"step": 1, "type": "tool_result", "tool": "search_kb",
         "summary": "检索到 2 条片段"},
    ]
    fake = FakeAgent(monkeypatch_result or make_result(), steps)
    agent_api._agent_state.update(ready=True, agent=fake, error=None, kb_count=123)
    return fake


def parse_sse(text: str):
    """把 SSE 文本解析成 (event, data) 列表。"""
    events = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        name, data = None, None
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[7:].strip()
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        if name:
            events.append((name, data))
    return events


def test_agent_stream_contract():
    """SSE 必须按 session → step* → sources → done 的顺序推送。"""
    install_fake()
    client = TestClient(app)
    with client.stream("POST", "/api/agent/chat",
                       json={"question": "顽固性高血压用什么药？"},
                       headers=HEADERS) as resp:
        assert resp.status_code == 200, resp.status_code
        body = "".join(resp.iter_text())

    events = parse_sse(body)
    names = [n for n, _ in events]
    assert names[0] == "session", f"首个事件应为 session: {names}"
    assert names.count("step") == 2, f"应有 2 个 step 事件: {names}"
    assert names[-2:] == ["sources", "done"], f"结尾应为 sources, done: {names}"

    done = dict(events)["done"]
    assert "联合用药" in done["answer"]
    assert done["protocol"] == "native"
    assert done["session_id"], "应回传 session_id"

    sources = dict(events)["sources"]["sources"]
    assert len(sources) == 2
    assert any(s["poison_suspect"] for s in sources), "投毒标记未透传"
    print(f"[OK] SSE 契约: 事件序列 = {names}")
    print(f"     答案: {done['answer'][:40]}")
    print(f"     sources: {len(sources)} 条，投毒标记 "
          f"{[s['poison_suspect'] for s in sources]}")


def test_agent_ask_endpoint():
    """非流式接口一次性返回答案 + 轨迹 + 来源。"""
    install_fake()
    client = TestClient(app)
    r = client.post("/api/agent/ask",
                    json={"question": "顽固性高血压用什么药？", "session_id": "s1"},
                    headers=HEADERS)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["session_id"] == "s1"
    assert len(data["trace"]) == 3
    assert len(data["sources"]) == 2
    assert data["steps"] == 2
    print(f"[OK] /ask: steps={data['steps']}, trace={len(data['trace'])} 步, "
          f"sources={len(data['sources'])} 条")


def test_refused_result_surfaces():
    """拒答结果必须透传到接口层。"""
    res = make_result()
    res.answer = "【无法回答】知识库中没有足够依据。"
    res.refused = True
    res.refusal_reason = "没有找到相关资料"
    res.sources = []
    install_fake(res)
    client = TestClient(app)
    r = client.post("/api/agent/ask", json={"question": "外星人医保？"},
                    headers=HEADERS)
    data = r.json()
    assert data["refused"] is True
    assert "无法回答" in data["answer"]
    print(f"[OK] 拒答透传: refused={data['refused']}, "
          f"reason={data['refusal_reason']}")


def test_tools_and_status():
    """工具清单与状态接口。"""
    install_fake()
    client = TestClient(app)
    t = client.get("/api/agent/tools", headers=HEADERS).json()
    assert t["tools"] and t["tools"][0]["name"] == "search_kb"
    s = client.get("/api/agent/status", headers=HEADERS).json()
    assert s["ready"] is True
    assert s["protocol"] == "native"
    assert s["kb_chunks"] == 123
    print(f"[OK] /tools 返回 {len(t['tools'])} 个工具；/status protocol={s['protocol']}, "
          f"kb_chunks={s['kb_chunks']}")


def test_auth_required():
    """未带 token 必须 401。"""
    install_fake()
    client = TestClient(app)
    r = client.post("/api/agent/ask", json={"question": "x"})
    assert r.status_code == 401, r.status_code
    print("[OK] 鉴权生效: 无 token -> 401")


if __name__ == "__main__":
    for fn in (test_agent_stream_contract, test_agent_ask_endpoint,
               test_refused_result_surfaces, test_tools_and_status,
               test_auth_required):
        fn()
        print("-" * 70)
    print("API 层测试全部通过")
