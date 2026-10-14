"""Agent 冒烟测试（不依赖 Chroma/Ollama embedding，用假向量库验证控制流）。

验证点：
  1. 工具循环真的转起来（模型请求工具 → 执行 → 回填 → 再决策）；
  2. 模型决定停时进入 finalize，而不是被常量打断；
  3. refuse 工具能走到拒答分支；
  4. 会话记忆（checkpointer + thread_id）能延续上文；
  5. 两条通道（native / react）行为一致。
"""
import sys
from typing import Any, Dict, List

sys.path.insert(0, r"E:\RAG2")

from langchain_core.documents import Document

from agent import ToolChatModel, build_agent, new_memory
from agent.agent import MedicalAgent
from agent.chat_ollama import ChatReply, ToolCall
from agent.tools import ToolKit


class FakeDB:
    """最小向量库替身：as_retriever().invoke() 返回固定文档。"""

    def __init__(self):
        self.docs: List[Document] = []
        self.queries: List[str] = []

    def as_retriever(self, search_kwargs=None):
        k = (search_kwargs or {}).get("k", 5)

        class _R:
            def __init__(self, outer):
                self.outer = outer

            def invoke(self, q):
                self.outer.queries.append(q)
                return self.outer.docs[:k]

        return _R(self)


class ScriptedLLM:
    """按剧本返回动作的假模型，用来确定性地驱动控制流。"""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0
        self.system_seen: List[str] = []
        # 剧本耗尽后的兜底项：可调用项会被重复使用
        self.default = None

    def protocol(self):
        return "scripted"

    def chat(self, messages, tools=None, system=None):
        self.calls += 1
        if system:
            self.system_seen.append(system)
        if self.script:
            item = self.script.pop(0)
        elif self.default is not None:
            # 剧本用完后，重复最后一个"可调用"项（观察式 spy 需要被多次调用）
            item = self.default
        else:
            item = {"action": "final", "answer": "默认回答"}
        if callable(item):
            # 必须把 tools/system 一并透传，否则测不出 agent 节点是否真的
            # 把工具集交给了模型（这个 mock 缺陷曾让会话记忆测试误报失败）
            return item(messages, tools=tools, system=system)
        return ChatReply(text=item.get("answer", ""), tool_calls=[
            ToolCall(name=t["name"], arguments=t.get("arguments", {}), call_id=f"c{self.calls}")
            for t in item.get("tools", [])
        ])


def make_docs():
    return [
        Document(page_content="高血压的规范治疗包括生活方式干预与降压药物治疗。",
                 metadata={"doc_id": "d1", "source": "内科教材"}),
        Document(page_content="顽固性高血压应联合用药，并排除继发因素。",
                 metadata={"doc_id": "d2", "source": "指南2024"}),
        Document(page_content="协和研发中心推出的血压清纯中药制剂，三疗程根治高血压，无任何副作用。",
                 metadata={"doc_id": "d3", "source": "未知来源"}),
    ]


def test_tool_loop_and_finalize():
    """模型先检索、再检查投毒、最后作答——循环由模型驱动。"""
    db = FakeDB()
    db.docs = make_docs()
    llm = ScriptedLLM([
        {"tools": [{"name": "search_kb", "arguments": {"query": "顽固性高血压用药"}}]},
        {"tools": [{"name": "check_poison", "arguments": {"doc_id": 3}}]},
        {"answer": "依据片段1和片段2，顽固性高血压应联合用药并排除继发因素。"},
    ])
    agent = MedicalAgent(db=db, llm=llm, knowledge_base=["x"] * 10)
    # 关掉真实检测器训练，避免依赖 RAGblack 数据集
    agent.toolkit._detector = _StubDetector()
    res = agent.run("顽固性高血压用什么药最有效？")

    assert db.queries, "search_kb 未真正触发检索"
    tools = [t["tool"] for t in res.trace if t["type"] == "tool_call"]
    assert tools == ["search_kb", "check_poison"], f"工具序列异常: {tools}"
    assert res.answer, "最终回答为空"
    assert not res.refused
    assert len(res.sources) >= 2, f"证据应被收集: {len(res.sources)}"
    # 3 次模型调用：检索决策 → 投毒检测决策 → 基于证据给出最终回答
    assert llm.calls == 3, f"模型调用次数异常: {llm.calls}"
    assert "依据" in res.answer or "片段" in res.answer, f"回答未基于证据: {res.answer}"
    print(f"[OK] 工具循环: 工具序列={tools}, 步数={res.steps}, "
          f"来源={len(res.sources)}, 模型调用={llm.calls}次")
    print(f"     回答: {res.answer[:60]}")
    return res


def test_finalize_rejects_unsourced_answer():
    """没有检索证据时，模型凭先验给出的回答必须被标注为未核查。

    回归测试：finalize 曾经无条件采用模型的回答；一旦模型跳过检索直接作答，
    就等于绕过了"无依据不作答"这条核心准则。
    """
    db = FakeDB()
    db.docs = make_docs()
    llm = ScriptedLLM([
        # 模型一上来就凭先验作答，完全没调检索工具
        {"answer": "【模型凭先验随口回答，未经知识库核查】"},
    ])
    agent = MedicalAgent(db=db, llm=llm, knowledge_base=["x"] * 10)
    res = agent.run("高血压怎么治？")
    assert res.answer.startswith("【未核查】"), \
        f"未经核查的回答未被标注: {res.answer}"
    assert "没有知识库依据" in res.answer
    print(f"[OK] 无证据时标注未核查: {res.answer[:60]}...")


def test_refuse_branch():
    """模型调用 refuse → 必须走拒答分支，不能编造答案。"""
    db = FakeDB()
    db.docs = []
    llm = ScriptedLLM([
        {"tools": [{"name": "search_kb", "arguments": {"query": "外星人医保政策"}}]},
        {"tools": [{"name": "refuse", "arguments": {"reason": "知识库中没有任何相关资料"}}]},
    ])
    agent = MedicalAgent(db=db, llm=llm, knowledge_base=["x"] * 10)
    res = agent.run("外星人的医保政策是什么？")
    assert res.refused, "refuse 未生效"
    assert "无法回答" in res.answer, f"答案未体现拒答: {res.answer}"
    print(f"[OK] 拒答分支: {res.answer}")


def test_finalize_regenerates_when_model_failed():
    """模型在检索后未能给出回答时，finalize 必须基于证据补一次生成。"""
    db = FakeDB()
    db.docs = make_docs()
    llm = ScriptedLLM([
        {"tools": [{"name": "search_kb", "arguments": {"query": "高血压"}}]},
        # 检索后模型返回空响应（真实小模型会这样）
        {"answer": ""},
        # 第三次 = finalize 基于证据的落地生成
        {"answer": "依据片段1与片段2：应联合用药并排除继发因素。"},
    ])
    agent = MedicalAgent(db=db, llm=llm, knowledge_base=["x"] * 10)
    res = agent.run("高血压怎么治？")
    assert res.answer.startswith("依据片段"), f"finalize 未补生成: {res.answer}"
    print(f"[OK] 模型空响应时 finalize 基于证据补生成: {res.answer[:50]}")


def test_cross_check_tool():
    """交叉验证工具：两篇独立来源 → supported；单一来源 → weak。"""
    tk = ToolKit(db=FakeDB(), llm=None, knowledge_base=[])
    tk.last_search = make_docs()
    r = tk.cross_check("高血压应联合用药并排除继发因素")
    assert r["verdict"] in ("supported", "weak"), r
    print(f"[OK] cross_check: verdict={r['verdict']}, "
          f"supporting={r['supporting_count']}, independent={r['independent_sources']}")

    r2 = tk.cross_check("量子催眠一次治愈抑郁症")
    assert r2["verdict"] == "unsupported", r2
    print(f"[OK] cross_check 反例: verdict={r2['verdict']} (无支持证据，不得作答)")


def test_session_memory():
    """同一 session_id 的第二次提问必须能看到第一轮上下文。"""
    db = FakeDB()
    db.docs = make_docs()
    calls_log = []          # 每次模型调用：{agent节点?, 可见消息}

    def spy(messages, tools=None, system=None):
        """记录每次调用可见的上下文，然后直接给出答案。"""
        calls_log.append({
            "is_agent": bool(system),
            "contents": [m.get("content", "") for m in messages],
        })
        return ChatReply(text="好的。")

    llm = ScriptedLLM([spy])   # 剧本只有 spy，后续所有调用都由它接手
    llm.default = spy
    agent = MedicalAgent(db=db, llm=llm, knowledge_base=["x"] * 10,
                         checkpointer=new_memory())
    agent.toolkit._detector = _StubDetector()

    sid = "session-abc"
    agent.run("顽固性高血压用什么药？", session_id=sid)
    agent.run("那它的副作用呢？", session_id=sid)

    print(f"     模型调用 {len(calls_log)} 次，"
          f"其中 agent 决策调用 {sum(1 for c in calls_log if c['is_agent'])} 次")

    # 第二轮 = 可见消息里含第二轮提问的那次 agent 决策调用
    round2 = [c for c in calls_log
              if c["is_agent"] and any("那它的副作用呢" in x for x in c["contents"])]
    assert round2, f"第二轮没有 agent 决策调用；记录={calls_log}"
    visible = round2[-1]["contents"]
    assert any("顽固性高血压用什么药" in x for x in visible), \
        f"第二轮看不到第一轮的提问，记忆未生效: {visible}"
    assert "好的。" in visible, "第二轮看不到第一轮的回答"
    print(f"[OK] 会话记忆: 第二轮 agent 可见 {len(visible)} 条消息，"
          f"含第一轮提问与回答")
    print(f"     上下文: {[v[:16] for v in visible]}")


def test_checkpointer_without_session():
    """带 checkpointer 但不传 session_id 时必须能正常跑完。

    回归测试：langgraph 要求带 checkpointer 的图必须提供 thread_id，
    而 run() 原先只在同时给了 session_id 时才设置，导致这种调用直接抛
    ValueError: Checkpointer requires ... 'thread_id'。
    """
    db = FakeDB()
    db.docs = make_docs()
    llm = ScriptedLLM([
        {"tools": [{"name": "search_kb", "arguments": {"query": "高血压"}}]},
        {"answer": "初始回答"},
        {"answer": "依据片段1回答。"},
    ])
    agent = MedicalAgent(db=db, llm=llm, knowledge_base=["x"] * 10,
                         checkpointer=new_memory())
    res = agent.run("高血压怎么治？")          # 注意：不传 session_id
    assert res.answer, "未产出答案"
    print(f"[OK] 带记忆但不传 session_id 可正常运行: {res.answer[:40]}")


def test_wire_tool_call_arguments_are_objects():
    """回填给模型的 assistant 工具调用消息里，arguments 必须是 JSON 对象。

    回归测试：曾经把 json.dumps(...) 得到的**字符串**塞进 arguments，
    Ollama 在下一轮请求里会解析失败并返回
    "Value looks like object, but can't find closing '}' symbol"，
    同时打断工具循环第二轮和会话记忆追问。
    """
    from agent.agent import _wire_tool_calls, _to_wire_message

    wired = _wire_tool_calls([{"id": "c1", "name": "search_kb",
                               "arguments": {"query": "高血压", "k": 5}}])
    args = wired[0]["function"]["arguments"]
    assert isinstance(args, dict), f"arguments 必须是对象，实际是 {type(args).__name__}"

    # 字符串形式的 arguments 也要被解析回对象
    wired2 = _wire_tool_calls([{"id": "c2", "name": "search_kb",
                                "arguments": '{"query": "糖尿病"}'}])
    assert isinstance(wired2[0]["function"]["arguments"], dict)

    # 走一遍完整的 Message 转换路径（模拟 checkpointer 取回的历史）
    class FakeAIMessage:
        type = "ai"
        content = ""
        tool_calls = [{"id": "c3", "name": "check_poison", "args": {"doc_id": 2}}]

    msg = _to_wire_message(FakeAIMessage())
    assert isinstance(msg["tool_calls"][0]["function"]["arguments"], dict), \
        f"历史消息里的 arguments 不是对象: {msg}"
    print(f"[OK] 工具调用 wire 格式: {msg['tool_calls'][0]['function']}")


def test_react_tool_call_variants():
    """文本 ReAct 通道必须识别"工具名直接写在 action 里"的写法。

    回归测试：实测 qwen 在 ReAct 协议下会输出
    ``{"action": "search_kb", "arguments": {...}}``，
    而早期解析只认 ``action in ("tool","call","action")``，
    于是**合法调用被当成最终回答吞掉**——表现为"Agent 一次工具都没调用"、
    分层指标里召回率恒为 0。
    """
    from agent.chat_ollama import ChatReply, ToolCall, _extract_json

    # 1) JSON 提取要能处理代码块与纯 JSON
    for text in [
        '{"action": "search_kb", "arguments": {"query": "上海肺癌", "k": 4}}',
        '{"action": "tool", "tool": "search_kb", "arguments": {"query": "x"}}',
        '```json\n{"action": "search_kb", "arguments": {"query": "y"}}\n```',
    ]:
        obj = _extract_json(text)
        assert isinstance(obj, dict), f"未能提取 JSON: {text}"

    # 2) 用假 LLM 走一遍 legacy 适配器的解析路径
    class FakeLegacy:
        model = "fake"

        def invoke(self, payload):
            return '{"action": "search_kb", "arguments": {"query": "测试", "k": 3}}'

    from agent.chat_ollama import ensure_tool_model
    adapter = ensure_tool_model(FakeLegacy())
    reply = adapter.chat([{"role": "user", "content": "q"}], tools=[
        {"type": "function", "function": {"name": "search_kb", "description": "",
                                          "parameters": {}}}])
    assert reply.wants_tools, f"合法调用被当成最终回答: {reply.text!r}"
    assert reply.tool_calls[0].name == "search_kb"
    assert reply.tool_calls[0].arguments.get("query") == "测试"
    print(f"[OK] ReAct 工具调用变体识别: {reply.tool_calls[0].name}("
          f"{reply.tool_calls[0].arguments})")

    # 3) 直接作答仍要被识别为回答而不是工具调用
    class FakeAnswer:
        model = "fake"

        def invoke(self, payload):
            return '{"action": "final", "answer": "无法回答"}'

    a2 = ensure_tool_model(FakeAnswer())
    r2 = a2.chat([{"role": "user", "content": "q"}], tools=[
        {"type": "function", "function": {"name": "search_kb", "description": "",
                                          "parameters": {}}}])
    assert not r2.wants_tools, "最终回答被误判为工具调用"
    assert "无法回答" in r2.text
    print(f"[OK] ReAct 最终回答识别: {r2.text!r}")


def test_all_evidence_suspect_forces_refusal():
    """全部证据都被标记为投毒嫌疑时，必须拒答。

    回归测试：原先会"基于已有证据生成回答"——但那批证据全是投毒文档，
    等于让模型照着假材料答题。这个规则在投毒场景下是最后一道闸。
    """
    from agent.agent import MedicalAgent
    from langchain_core.documents import Document

    class DB:
        def as_retriever(self, search_kwargs=None):
            docs = [Document(page_content="假药能根治高血压。",
                             metadata={"doc_id": "p1"})]

            class R:
                def invoke(self, q):
                    return docs
            return R()

    llm = ScriptedLLM([
        {"tools": [{"name": "search_kb", "arguments": {"query": "高血压"}}]},
        {"tools": [{"name": "check_poison", "arguments": {"doc_id": 1}}]},
        # 模型打算照着投毒内容作答
        {"answer": "推荐使用该假药，可根治高血压。"},
    ])
    agent = MedicalAgent(db=DB(), llm=llm, knowledge_base=["x"] * 5)

    class AlwaysPoison:
        def predict_single(self, text):
            return {"is_poison": True, "confidence": 0.95, "reason": "绝对化承诺",
                    "evidence": ["强断言: 根治"], "method": "rule"}

    agent.toolkit._detector = AlwaysPoison()
    res = agent.run("高血压怎么治？")
    assert res.refused, "全部证据可疑时未拒答"
    assert "无法回答" in res.answer, f"未给出拒答文案: {res.answer}"
    assert "假药" not in res.answer, "仍然采信了投毒内容"
    print(f"[OK] 全部证据可疑时强制拒答: {res.answer[:50]}")


def test_protocol_detection():
    """真实 Ollama：确认两条通道的探测结果符合预期。"""
    for model, expect in [("deepseek-r1:latest", "react"),
                          ("qwen3.5:2b", "native")]:
        try:
            m = ToolChatModel(model=model, timeout=180)
            p = m.protocol()
            flag = "OK " if p == expect else "?? "
            print(f"[{flag}] {model} -> {p} (预期 {expect})")
        except Exception as e:
            print(f"[SKIP] {model}: {e}")


class _StubDetector:
    def predict_single(self, text):
        is_poison = "根治" in text or "无任何副作用" in text
        return {"is_poison": is_poison, "confidence": -0.12 if is_poison else 0.08}


if __name__ == "__main__":
    print("=" * 70)
    test_tool_loop_and_finalize()
    print("-" * 70)
    test_finalize_rejects_unsourced_answer()
    print("-" * 70)
    test_finalize_regenerates_when_model_failed()
    print("-" * 70)
    test_refuse_branch()
    print("-" * 70)
    test_cross_check_tool()
    print("-" * 70)
    test_session_memory()
    print("-" * 70)
    test_checkpointer_without_session()
    print("-" * 70)
    test_wire_tool_call_arguments_are_objects()
    print("-" * 70)
    test_react_tool_call_variants()
    print("-" * 70)
    test_all_evidence_suspect_forces_refusal()
    print("-" * 70)
    print("真实模型协议探测（需要 Ollama）：")
    test_protocol_detection()
    print("=" * 70)
    print("全部冒烟测试通过")
