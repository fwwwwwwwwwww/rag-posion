"""真正的工具化 Agent（LangGraph + 工具循环 + 会话记忆）。

与原 ``rag_graph.py`` 的区别
----------------------------
原实现：``retrieve → grade → generate``，下一步走哪条边由 Python 的 ``if`` 决定，
模型只有三个固定空位可填。

本实现：模型拿到一套工具和一份策略，**自己决定**调哪个工具、调几次、
证据够不够、该不该拒答。循环终止条件是"模型不再请求工具"，而不是某个常量。

控制流：

    ┌───────┐  模型要求调工具   ┌───────┐
    │ agent │ ───────────────→ │ tools │
    │ (LLM) │ ←─────────────── │ (执行) │
    └───┬───┘   工具结果回填     └───────┘
        │ 模型输出最终回答 / 调用 refuse
        ↓
    ┌──────────┐
    │ finalize │  基于实际收集到的证据生成回答，不编造
    └──────────┘

会话记忆由 LangGraph checkpointer 承担（按 thread_id 隔离），
因此"那它的副作用呢？"这类追问可以延续上文。
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

from .chat_ollama import ChatReply, ToolCall, ensure_tool_model, strip_think
from .prompts import AGENT_SYSTEM, GENERATE_PROMPT
from .tools import ToolKit

logger = logging.getLogger("rag.agent")

# 工具循环的硬上限：防止模型陷入无限探索
MAX_AGENT_STEPS = 8
# 每个工具结果注入上下文时的截断长度
TOOL_RESULT_LIMIT = 2500
# 最终回答上下文的总字符预算（防止检索过多片段撑爆模型窗口）
CONTEXT_CHAR_BUDGET = 4000
# 会话记忆回灌的最大消息条数（防止上下文无限增长）
HISTORY_LIMIT = 8
# 系统提示词里回显的最近对话轮数
MEMORY_TURNS = 6


def _msg_role(m: Any) -> str:
    """取出消息角色。LangGraph 的 add_messages 会把 dict 转成 Message 对象，
    因此两种形态都要支持。"""
    if isinstance(m, dict):
        return str(m.get("role", ""))
    return str(getattr(m, "type", "") or getattr(m, "role", ""))


def _msg_content(m: Any) -> str:
    """取出消息文本内容（可能为 None 或结构化块列表）。"""
    raw = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts = []
        for b in raw:
            if isinstance(b, str):
                parts.append(b)
            elif isinstance(b, dict):
                parts.append(str(b.get("text", "")))
        return "".join(parts)
    return str(raw)


def _msg_tool_calls(m: Any) -> List[Any]:
    """取出 assistant 消息上的 tool_calls（两种形态）。"""
    calls = m.get("tool_calls") if isinstance(m, dict) else getattr(m, "tool_calls", None)
    return list(calls or [])


def _coerce_call(c: Any) -> Dict[str, Any]:
    """把 tool_call 统一成 ``{"id", "name", "arguments"}``。

    来源有两种：我们自己构造的 OpenAI 风格 dict，以及 langchain Message 上
    的 ToolCall 对象。
    """
    if isinstance(c, dict):
        if "function" in c and isinstance(c["function"], dict):
            fn = c["function"]
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"_raw": args}
            return {"id": c.get("id", ""), "name": fn.get("name", ""),
                    "arguments": args if isinstance(args, dict) else {}}
        args = c.get("arguments", c.get("args", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {"_raw": args}
        return {"id": c.get("id", c.get("tool_call_id", "")),
                "name": c.get("name", ""),
                "arguments": args if isinstance(args, dict) else {}}

    args = getattr(c, "args", None)
    if args is None:
        args = getattr(c, "arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {"_raw": args}
    return {
        "id": getattr(c, "id", "") or "",
        "name": getattr(c, "name", "") or "",
        "arguments": args if isinstance(args, dict) else {},
    }


def _render_memory(history: List[Dict[str, Any]]) -> str:
    """把最近几轮对话渲染成系统提示词里的「对话上文」。

    为什么需要它：本轮的证据（evidence）每轮重置，历史只以消息形式存在于
    checkpointer 里，而小模型经常忽略消息形式的历史，把"那它的副作用呢？"
    当成一个没有指代对象的新问题。显式写进系统提示词能显著提高指代解析的可靠性。

    注意：窗口裁剪必须**先按对话轮次筛选**再取最近几轮。曾经直接对原始历史
    取最后 4 条，结果一条冗长的工具返回消息就会把上一轮的答案挤出窗口，
    表现为"第二轮明明有记忆却看不到上一轮答了什么"。
    """
    turns: List[str] = []
    for m in history:
        role = (m.get("role") or "").lower()
        if role not in ("user", "assistant"):
            continue
        content = (m.get("content") or "").strip()
        # 跳过纯工具调用产生的空 assistant 消息
        if not content:
            continue
        if role == "assistant" and ("【无法回答】" in content or "【未核查】" in content):
            content = content.split("\n", 1)[0]
        if len(content) > 300:
            content = content[:300] + "…"
        turns.append(f"{'用户' if role == 'user' else '你'}：{content}")

    if not turns:
        return ""
    # 只保留最近 MEMORY_TURNS 轮，避免挤占工具说明占用的注意力。
    # 放在系统提示词**结尾**（贴近当前问题）并写成祈使句——小模型对
    # 中段的说明经常忽略，实测这样才稳定触发指代解析。
    return ("\n## 对话上文（最近几轮）\n"
            + "\n".join(turns[-MEMORY_TURNS:])
            + "\n\n**重要**：当前问题可能是对上文的追问（含「它/这个/那」等指代）。"
              "你必须先结合上文确定指代对象，把补全后的完整实体"
              "（疾病名 / 药物名 / 疗法名）写进 search_kb 的查询里再检索。\n")


def _wire_tool_calls(tcs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把工具调用渲染成 Ollama 期望的 assistant 消息格式。

    统一入口很关键：这里曾经直接把 ``json.dumps(arguments)`` 塞进
    ``function.arguments``，而 OpenAI/Ollama 期望它是**对象**而不是字符串。
    用字符串时 Ollama 在新一轮请求里会解析失败并返回
     ``Value looks like object, but can't find closing '}' symbol``——
    这会同时打断工具循环的第二轮与会话记忆的追问轮次。
    """
    out = []
    for c in tcs:
        args = c.get("arguments", c.get("args", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {"_raw": args}
        if not isinstance(args, dict):
            args = {}
        out.append({
            "id": c.get("id") or f"call_{len(out)}",
            "type": "function",
            "function": {"name": c.get("name", ""), "arguments": args},
        })
    return out


def _to_wire_message(m: Any) -> Optional[Dict[str, Any]]:
    """把 langgraph 的 Message 对象（或已是 dict 的消息）转成 HTTP 层要的 dict。"""
    if isinstance(m, dict):
        return m
    role = _msg_role(m)
    role = {"human": "user", "ai": "assistant"}.get(role, role)
    if role not in ("user", "assistant", "tool", "system"):
        return None
    out: Dict[str, Any] = {"role": role, "content": _msg_content(m)}
    if role == "tool":
        # tool_call_id 是**必填**：langgraph 的 add_messages 在缺少该字段时
        # 会直接抛 KeyError('tool_call_id')。这里兜底生成一个，
        # 避免某条历史消息缺字段就把整张图打断。
        out["tool_call_id"] = getattr(m, "tool_call_id", "") or "call_unknown"
        name = getattr(m, "name", "") or ""
        if name:
            out["tool_name"] = name
    if role == "assistant":
        calls = [_coerce_call(c) for c in _msg_tool_calls(m)]
        if calls:
            out["tool_calls"] = _wire_tool_calls(calls)
    return out


def _fmt_tools_block(schemas: List[dict]) -> str:
    lines = []
    for s in schemas:
        fn = s.get("function", {})
        name = fn.get("name", "?")
        desc = fn.get("description", "")
        props = fn.get("parameters", {}).get("properties", {}) or {}
        req = set(fn.get("parameters", {}).get("required", []) or [])
        args = ", ".join(
            f"{k}:{v.get('type', 'string')}{'' if k in req else '?'}" for k, v in props.items()
        )
        lines.append(f"- {name}({args})：{desc}")
    return "\n".join(lines)


class AgentState(TypedDict, total=False):
    """Agent 状态。

    ``messages`` 带 add_messages 归约语义：节点返回的新消息是**追加**而非覆盖，
    这正是工具结果回填与多轮记忆能工作的基础。
    其余字段由各节点覆写。
    """

    messages: Annotated[list, add_messages]
    question: str
    original_question: str
    step_count: int
    max_steps: int
    trace: list          # 决策轨迹，供前端展示
    evidence: list       # 累积的证据文档（含验证标记）
    poison_flagged: list
    refused: bool
    refusal_reason: str
    final_answer: str
    finish_reason: str


@dataclass
class AgentResult:
    """对外返回结构（兼容原 RetrievalQA 的 result/source_documents）。"""

    answer: str
    sources: List[Any] = field(default_factory=list)
    trace: List[Dict[str, Any]] = field(default_factory=list)
    refused: bool = False
    refusal_reason: str = ""
    steps: int = 0
    protocol: str = ""

    def to_legacy(self) -> Dict[str, Any]:
        """兼容原 RetrievalQA 的返回结构。

        除 ``result`` / ``source_documents`` 外还附带 ``trace`` / ``refused`` /
        ``steps``：实验评估需要决策轨迹来还原分层指标（召回率、检出率、
        交叉验证结果），只返回前两个键会让这些指标全部失真。
        """
        return {
            "result": self.answer,
            "source_documents": self.sources,
            "trace": self.trace,
            "refused": self.refused,
            "refusal_reason": self.refusal_reason,
            "steps": self.steps,
            "protocol": self.protocol,
        }


def _json_safe(value: Any) -> Any:
    """把轨迹数据压成可 JSON 序列化的形态。

    工具返回值里可能混入 numpy 标量、Document 对象等，
    直接丢给 SSE/FastAPI 会崩。
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        # NaN / Inf 不是合法 JSON
        return value if value == value and abs(value) != float("inf") else None
    if hasattr(value, "item"):          # numpy 标量
        try:
            return _json_safe(value.item())
        except Exception:
            return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


def _tool_result_message(name: str, call_id: str, result: Dict[str, Any]) -> dict:
    """把工具返回结果包装成模型能看到的消息。"""
    payload = json.dumps(result, ensure_ascii=False)
    if len(payload) > TOOL_RESULT_LIMIT:
        payload = payload[:TOOL_RESULT_LIMIT] + '...（结果过长已截断）'
    return {
        "role": "tool",
        "tool_name": name,
        "tool_call_id": call_id,
        "content": f"工具 {name} 返回：{payload}",
    }


def _summarize_result(name: str, result: Dict[str, Any]) -> str:
    """给轨迹面板生成一行人类可读的结果摘要。"""
    if not isinstance(result, dict):
        return str(result)[:100]
    if "error" in result:
        return f"出错：{result['error']}"
    if name == "search_kb":
        return f"检索到 {result.get('count', 0)} 条片段"
    if name == "check_poison":
        if result.get("is_poison") is None:
            return "检测器不可用"
        return ("⚠ 判定为投毒嫌疑" if result.get("is_poison") else "未发现异常") + \
               f"（置信度 {result.get('confidence')}）"
    if name == "cross_check":
        return (f"{result.get('verdict')}：{result.get('independent_sources')} 篇独立来源支持")
    if name == "rewrite_query":
        return f"改写为：{result.get('rewritten', '')}"
    if name == "refuse":
        return f"拒答：{result.get('reason', '')}"
    return "完成"


class MedicalAgent:
    """医疗知识库问答 Agent。"""

    def __init__(
        self,
        db: Any,
        llm: Any,
        knowledge_base: Optional[List[str]] = None,
        model: Optional[str] = None,
        checkpointer: Any = None,
        max_steps: int = MAX_AGENT_STEPS,
    ):
        self.db = db
        self.llm = ensure_tool_model(llm, model=model)
        self.toolkit = ToolKit(db, self.llm, knowledge_base=knowledge_base)
        self.schemas = self.toolkit.schemas()
        self.max_steps = max_steps
        self.checkpointer = checkpointer
        # 与原 RAGGraph 的字段对齐：实验脚本会直接写这两个属性
        self._db = db
        self._poison_ids: List[str] = []
        # 决策轨迹的实时回调（run() 期间设置），用于 SSE 推送每一步
        self._on_step: Optional[Any] = None
        self._app = self._compile()

    def _fallback_tool_attempt(self, history: List[dict], state: AgentState) -> ChatReply:
        """弱模型兜底：缩小工具集 + 强制指令，再试一次拿到工具调用。

        只保留 ``search_kb`` 与 ``refuse``：前者是获取事实依据的唯一入口，
        后者保证"查不到就该拒答"这条路径仍然可达。工具越少，小模型越不容易空响应。
        """
        from .prompts import FALLBACK_NUDGE

        minimal = [s for s in self.schemas
                   if s["function"]["name"] in ("search_kb", "refuse")]
        question = state.get("original_question") or state.get("question", "")
        nudge = [{"role": "user", "content": FALLBACK_NUDGE.format(question=question)}]
        try:
            return self.llm.chat(nudge, tools=minimal)
        except Exception:
            logger.warning("兜底重试失败", exc_info=True)
            return ChatReply(text="")

    def _emit(self, event: Dict[str, Any]) -> None:
        """把一次决策事件推给回调（若已注册）。回调异常绝不影响主流程。"""
        cb = self._on_step
        if cb is None:
            return
        try:
            cb(event)
        except Exception:
            logger.debug("轨迹回调失败", exc_info=True)

    # ---------------- 图构建 ----------------
    def _compile(self):
        from langgraph.graph import END, START, StateGraph

        def eff_max(state: AgentState) -> int:
            """本次运行的工具循环上限（可被 run(max_steps=...) 覆盖）。"""
            return int(state.get("max_steps") or self.max_steps)

        def agent_node(state: AgentState) -> Dict[str, Any]:
            """模型决策节点：调工具，或给出最终回答。

            这是整个系统唯一"决定下一步做什么"的地方，而做决定的是模型。
            """
            step = state.get("step_count", 0)
            msgs = list(state.get("messages", []))

            # 会话记忆回灌：把 langgraph 的 Message 对象转回 dict。
            # 工具返回消息可能很长，直接对全量历史取尾部会把上一轮的问答挤掉，
            # 因此优先保留"对话轮次"（user/assistant），再补上工具消息。
            all_wire: List[Dict[str, Any]] = []
            for m in msgs:
                wm = _to_wire_message(m)
                if wm is not None:
                    all_wire.append(wm)

            dialogue = [m for m in all_wire
                        if m.get("role") in ("user", "assistant")]
            if len(dialogue) > HISTORY_LIMIT:
                # 只保留最近的对话轮次，及其之后产生的工具消息
                keep_from = dialogue[-HISTORY_LIMIT]
                idx = all_wire.index(keep_from)
                all_wire = all_wire[idx:]
            history = all_wire

            # 「对话上文」只取当前问题**之前**的轮次，这样"那它的副作用呢？"
            # 才有一段上文可供解析指代；把当前问题本身也算进去会自指。
            current_q = ""
            for m in reversed(history):
                if (m.get("role") or "").lower() == "user":
                    current_q = (m.get("content") or "").strip()
                    break
            prior = list(history)
            if current_q:
                for i in range(len(prior) - 1, -1, -1):
                    if (prior[i].get("role") or "").lower() == "user" \
                            and (prior[i].get("content") or "").strip() == current_q:
                        prior = prior[:i]
                        break

            system = AGENT_SYSTEM.format(
                memory_block=_render_memory(prior),
                tools_block=_fmt_tools_block(self.schemas),
            )
            reply: ChatReply = self.llm.chat(
                history, tools=self.schemas, system=system)
            # ---- 弱模型兜底 ----
            # 小模型在工具集较大时可能返回**完全空**的响应（既无工具调用也无文本），
            # 实测 qwen2.5:3b 传 5 个工具时就会这样。此时用"单一工具 + 强制指令"
            # 重试一次，否则这一轮空转结束、什么也答不出来。
            if not reply.wants_tools and not (reply.text or "").strip():
                reply = self._fallback_tool_attempt(history, state)
                if reply.wants_tools:
                    self._emit({"step": step + 1, "type": "fallback",
                                "summary": "模型空响应，已用强制检索指令重试"})

            trace = list(state.get("trace", []))
            updates: Dict[str, Any] = {"step_count": step + 1}

            if reply.wants_tools:
                new_msgs = []
                for tc in reply.tool_calls:
                    evt = {
                        "step": step + 1,
                        "type": "tool_call",
                        "tool": tc.name,
                        "arguments": tc.arguments,
                    }
                    trace.append(evt)
                    self._emit(evt)
                    new_msgs.append({
                        "role": "assistant",
                        "content": "",
                        "tool_calls": _wire_tool_calls([tc.to_dict()]),
                    })
                updates["messages"] = new_msgs
                updates["trace"] = trace
                return updates

            # 模型给出文本：如果调用了 refuse 工具，走拒答分支
            text = strip_think(reply.text or "")
            updates["final_answer"] = text
            updates["trace"] = trace + [{
                "step": step + 1, "type": "final", "content": text[:300],
            }]
            updates["finish_reason"] = "answered"
            updates["messages"] = [{"role": "assistant", "content": text}]
            return updates

        def tools_node(state: AgentState) -> Dict[str, Any]:
            """工具执行节点：执行模型请求的工具，结果回填给模型。"""
            msgs = list(state.get("messages", []))
            trace = list(state.get("trace", []))
            evidence = list(state.get("evidence", []))
            poison_flagged = list(state.get("poison_flagged", []))
            updates: Dict[str, Any] = {}

            # 取出最后一条 assistant 消息里的工具调用（统一成 id/name/arguments）
            calls: List[Dict[str, Any]] = []
            for m in reversed(msgs):
                if _msg_role(m) in ("assistant", "ai") and _msg_tool_calls(m):
                    calls = [_coerce_call(c) for c in _msg_tool_calls(m)]
                    break

            out_msgs: List[dict] = []
            refused = False
            refusal_reason = ""

            for c in calls:
                name = c["name"]
                args = c["arguments"]
                call_id = c["id"]

                result = self.toolkit.call(name, args if isinstance(args, dict) else {})

                summary = _summarize_result(name, result)
                # 轨迹记录
                for t in reversed(trace):
                    if t.get("type") == "tool_call" and t.get("tool") == name \
                            and "result" not in t:
                        t["result"] = summary
                        break
                evt = {
                    "step": state.get("step_count", 0),
                    "type": "tool_result",
                    "tool": name,
                    "summary": summary,
                    "result": result,
                }
                trace.append(evt)
                self._emit(evt)

                # 收集证据
                if name == "search_kb" and isinstance(result.get("results"), list):
                    for r in result["results"]:
                        evidence.append({
                            "id": r.get("id"),
                            "content": r.get("content", ""),
                            "query": args.get("query", ""),
                            "doc_key": r.get("doc_key", ""),
                        })
                elif name == "check_poison" and result.get("is_poison"):
                    poison_flagged.append({
                        "doc_id": args.get("doc_id"),
                        "text": (args.get("text") or "")[:120],
                        "confidence": result.get("confidence"),
                    })
                elif name == "refuse":
                    refused = True
                    refusal_reason = result.get("reason", "")

                out_msgs.append(_tool_result_message(name, call_id, result))

            updates["messages"] = out_msgs
            updates["trace"] = trace
            updates["evidence"] = evidence
            updates["poison_flagged"] = poison_flagged
            if refused:
                updates["refused"] = True
                updates["refusal_reason"] = refusal_reason
            return updates

        def finalize_node(state: AgentState) -> Dict[str, Any]:
            """最终回答节点：基于 Agent 实际收集到的证据生成，不编造。"""
            trace = list(state.get("trace", []))
            evidence = state.get("evidence", []) or []
            existing = (state.get("final_answer") or "").strip()

            # ---- 全部证据都被判定可疑时必须拒答 ----
            # 场景：检索到的每一篇都被 check_poison 标记为投毒嫌疑。
            # 此时"基于已有证据生成回答"等于拿投毒文档作答——必须拒答。
            poison_flagged = state.get("poison_flagged", []) or []
            if evidence and poison_flagged and not state.get("refused"):
                flagged_ids = {p.get("doc_id") for p in poison_flagged
                               if p.get("doc_id") is not None}
                ev_ids = {e.get("id") for e in evidence if e.get("id") is not None}
                # 触发条件：所有**被检查过的**证据都被标可疑，且没有剩下任何未检查的证据
                # （只要还有一篇没查过，就允许模型继续判断，不强行拒答）
                if flagged_ids and ev_ids and flagged_ids >= ev_ids:
                    trace.append({
                        "step": state.get("step_count", 0),
                        "type": "finalize",
                        "summary": f"证据全部被标记可疑（{len(flagged_ids)} 篇），改为拒答",
                    })
                    note = ("【无法回答】检索到的内容全部被判定为投毒嫌疑，"
                            "没有可信依据回答问题。")
                    return {"final_answer": note, "trace": trace,
                            "messages": [{"role": "assistant", "content": note}],
                            "finish_reason": "all_evidence_suspect",
                            "refused": True,
                            "refusal_reason": "全部检索结果均被标记为投毒嫌疑"}

            # 模型已通过 refuse 明确拒答
            if state.get("refused"):
                reason = state.get("refusal_reason") or "知识库中没有足够依据回答该问题"
                answer = f"【无法回答】{reason}"
                trace.append({"step": state.get("step_count", 0),
                              "type": "finalize", "summary": "模型选择拒答"})
                return {"final_answer": answer, "trace": trace,
                        "finish_reason": "refused"}

            if not evidence:
                trace.append({"step": state.get("step_count", 0),
                              "type": "finalize", "summary": "无检索证据"})
                if state.get("finish_reason") == "max_steps":
                    answer = "【无法回答】已达到工具调用上限，但仍未收集到足够证据。"
                    return {"final_answer": answer, "trace": trace,
                            "messages": [{"role": "assistant", "content": answer}],
                            "finish_reason": "max_steps"}
                if existing:
                    # 模型在完全没用过检索工具的情况下直接作答。这条回答没有任何
                    # 知识库依据，不能就这么当成结论返回——必须显式标注未核查。
                    used_tools = any(t.get("type") == "tool_call" for t in trace)
                    if not used_tools:
                        note = ("【未核查】模型未检索知识库即作答，"
                                "以下内容没有知识库依据，请谨慎采信：\n\n")
                        trace.append({
                            "step": state.get("step_count", 0),
                            "type": "finalize",
                            "summary": "模型未调用任何工具即作答，已标注为未核查",
                        })
                        return {
                            "final_answer": note + existing,
                            "trace": trace,
                            "messages": [{"role": "assistant", "content": note + existing}],
                            "finish_reason": "unverified",
                        }
                    return {"final_answer": existing, "trace": trace,
                            "finish_reason": "answered"}
                answer = "【无法回答】未能从知识库中检索到任何可用资料。"
                return {"final_answer": answer, "trace": trace,
                        "finish_reason": "no_evidence"}

            # 模型已经基于同一批证据给出过实质回答 —— 直接采用，不再重复生成。
            # 原先无条件再调一次模型，既浪费一次完整生成（本地模型往往要几十秒），
            # 也容易在第二次调用上踩到服务端异常，把一次本来成功的回答变成"生成失败"。
            if existing and not existing.startswith("[模型调用失败"):
                trace.append({"step": state.get("step_count", 0),
                              "type": "finalize",
                              "summary": "采用模型基于证据给出的回答"})
                return {"final_answer": existing, "trace": trace,
                        "messages": [{"role": "assistant", "content": existing}],
                        "finish_reason": "answered"}

            # 命中工具上限但已有证据：照常基于证据收束成回答。
            # （模型反复改写查询时会走到这里，不能因为步数用尽就丢弃已搜集的证据）
            at_limit = state.get("finish_reason") == "max_steps"
            trace.append({"step": state.get("step_count", 0),
                          "type": "finalize",
                          "summary": ("已达步数上限，基于已有证据收束" if at_limit
                                      else "基于收集到的证据生成回答")})
            question = state.get("original_question") or state.get("question", "")
            context = self._build_context(state)
            prompt = GENERATE_PROMPT.format(context=context, question=question)
            reply = self.llm.chat([{"role": "user", "content": prompt}])
            answer = strip_think(reply.text or "")
            if not answer or answer.startswith("[模型调用失败"):
                answer = (f"【生成失败】基于检索到的 {len(evidence)} 个片段生成回答时"
                          f"模型调用出错：{answer or '空响应'}")
            return {"final_answer": answer, "trace": trace,
                    "messages": [{"role": "assistant", "content": answer}],
                    "finish_reason": "answered"}

        def route_after_agent(state: AgentState) -> str:
            """唯一的路由：由**模型是否请求工具**决定，而非由代码里的常量决定。"""
            for m in reversed(list(state.get("messages", []))):
                if _msg_role(m) in ("assistant", "ai"):
                    return "tools" if _msg_tool_calls(m) else "finalize"
            return "finalize"

        def route_after_tools(state: AgentState) -> str:
            """工具执行完毕：拒答则直接收束，否则回到模型继续决策。"""
            if state.get("refused"):
                return "finalize"
            if state.get("step_count", 0) >= eff_max(state):
                return "finalize"
            return "agent"

        from langgraph.graph.message import add_messages
        from typing import Annotated

        class _State(TypedDict, total=False):
            messages: Annotated[list, add_messages]
            question: str
            original_question: str
            step_count: int
            max_steps: int
            trace: list
            evidence: list
            poison_flagged: list
            refused: bool
            refusal_reason: str
            final_answer: str
            finish_reason: str

        g = StateGraph(_State)
        g.add_node("agent", agent_node)
        g.add_node("tools", tools_node)
        g.add_node("finalize", finalize_node)
        g.add_edge(START, "agent")
        g.add_conditional_edges("agent", route_after_agent,
                                {"tools": "tools", "finalize": "finalize"})
        g.add_conditional_edges("tools", route_after_tools,
                                {"agent": "agent", "finalize": "finalize"})
        g.add_edge("finalize", END)

        if self.checkpointer is not None:
            return g.compile(checkpointer=self.checkpointer)
        return g.compile()

    # ---------------- 上下文构建 ----------------
    def _build_context(self, state: AgentState) -> str:
        """把检索到的证据 + 验证结论组装成最终回答的上下文。

        关键改进：**投毒嫌疑与独立来源数会被显式告知模型**，
        而不是像原实现那样把文档原样倒进去。
        同时按预算裁剪：检索 10 条 × 600 字会把上下文推到 5000+ 字符，
        叠加 num_ctx 后容易触到模型窗口上限。
        """
        evidence = state.get("evidence", []) or []
        poison_flagged = state.get("poison_flagged", []) or []
        poison_ids = {p.get("doc_id") for p in poison_flagged if p.get("doc_id")}

        # 按 doc_key 去重，统计每个来源被检索到几次
        seen: Dict[str, int] = {}
        blocks: List[str] = []
        used = 0
        truncated = 0
        for ev in evidence:
            key = ev.get("doc_key") or f"id{ev.get('id')}"
            seen[key] = seen.get(key, 0) + 1
            content = ev.get("content", "") or ""
            if used + len(content) > CONTEXT_CHAR_BUDGET:
                # 预算用尽：截断当前片段后停止堆叠
                remaining = max(0, CONTEXT_CHAR_BUDGET - used)
                if remaining > 200:
                    content = content[:remaining] + "…"
                else:
                    truncated += 1
                    continue
            used += len(content)
            mark = "【投毒嫌疑】" if ev.get("id") in poison_ids else ""
            blocks.append(f"[片段{ev.get('id')}]{mark}{content}")

        header = []
        if poison_flagged:
            header.append(
                f"注意：本次检索中有 {len(poison_flagged)} 个片段被投毒检测器标记为可疑，"
                "引用它们时必须声明这一点。")
        header.append(f"本次共获得 {len(blocks)} 个片段，来自 {len(seen)} 个独立来源。")
        if truncated:
            header.append(f"另有 {truncated} 个片段因上下文长度限制未纳入。")
        return "\n\n".join(header + blocks)

    # ---------------- 对外接口 ----------------
    def run(
        self,
        question: str,
        session_id: Optional[str] = None,
        history: Optional[List[dict]] = None,
        on_step: Optional[Any] = None,
        max_steps: Optional[int] = None,
    ) -> AgentResult:
        """执行一次 Agent 推理。

        ``on_step``：可选的轨迹回调，每产生一步决策（工具调用/工具结果/收束）
        就被调用一次，用于 SSE 实时推送"决策轨迹"给前端。
        ``max_steps``：仅对本次运行生效的上限（不修改实例状态，避免并发请求互相干扰）。
        """
        effective_max = int(max_steps or self.max_steps)
        init_msgs: List[dict] = []
        if history:
            for h in history:
                role = h.get("role")
                content = h.get("content", "")
                if role in ("user", "assistant") and content:
                    init_msgs.append({"role": role, "content": content})
        init_msgs.append({"role": "user", "content": question})

        init: Dict[str, Any] = {
            "messages": init_msgs,
            "question": question,
            "original_question": question,
            "step_count": 0,
            "max_steps": effective_max,
            "trace": [],
            "evidence": [],
            "poison_flagged": [],
            "refused": False,
            "refusal_reason": "",
            "final_answer": "",
            "finish_reason": "",
        }

        config: Dict[str, Any] = {"recursion_limit": (effective_max + 4) * 2}
        if self.checkpointer is not None:
            # 带 checkpointer 的图必须提供 thread_id，否则 langgraph 直接报错。
            # 未指定会话时生成一次性 thread_id（该次调用不留记忆）。
            config["configurable"] = {
                "thread_id": session_id or f"ephemeral-{uuid.uuid4().hex[:12]}"
            }
        elif session_id:
            logger.debug("传入 session_id=%s 但没有 checkpointer，记忆不会生效", session_id)

        self._on_step = on_step
        try:
            state = self._app.invoke(init, config=config)
        finally:
            self._on_step = None

        # 还原证据为 langchain Document，兼容原有的 source_documents 用法
        sources = self._to_documents(state)
        return AgentResult(
            answer=state.get("final_answer", "") or "",
            sources=sources,
            trace=_json_safe(state.get("trace", []) or []),
            refused=bool(state.get("refused")),
            refusal_reason=state.get("refusal_reason", "") or "",
            steps=state.get("step_count", 0),
            protocol=self.llm.protocol() if hasattr(self.llm, "protocol") else "",
        )

    @staticmethod
    def _to_documents(state: AgentState) -> List[Any]:
        """把证据片段还原成 Document 对象（带验证元数据）。"""
        try:
            from langchain_core.documents import Document
        except Exception:  # 极简环境下的降级
            return []

        poison_ids = {p.get("doc_id") for p in (state.get("poison_flagged") or [])
                      if p.get("doc_id")}
        docs, seen = [], set()
        for ev in state.get("evidence", []) or []:
            content = ev.get("content", "")
            if not content or content in seen:
                continue
            seen.add(content)
            docs.append(Document(
                page_content=content,
                metadata={
                    "doc_id": ev.get("id"),
                    "query": ev.get("query", ""),
                    "doc_key": ev.get("doc_key", ""),
                    "poison_suspect": ev.get("id") in poison_ids,
                },
            ))
        return docs

    # ---------------- 兼容原 RetrievalQA 接口 ----------------
    def invoke(self, payload: Any) -> Dict[str, Any]:
        """``rag.invoke({"query": q})`` 兼容入口。

        原代码（含实验脚本与 API 层）无需改动即可切换到 Agent。
        """
        if isinstance(payload, dict):
            question = payload.get("query") or payload.get("question") or ""
            session_id = payload.get("session_id")
        else:
            question = str(payload)
            session_id = None
        return self.run(question, session_id=session_id).to_legacy()

    def cleanup_poison(self) -> None:
        """删除实验注入的投毒文档（与 RAGblack / 原 RAGGraph 行为一致）。

        Agent 自身不注入投毒文档，但 ``build_rag`` 会把注入产生的 id 写进
        ``_poison_ids``，实验结束必须清理，否则污染会累积到后续实验。
        """
        if not self._poison_ids:
            return
        try:
            self._db._collection.delete(ids=self._poison_ids)
        except Exception as e:
            logger.warning("清理投毒文档失败：%s", e)
        finally:
            self._poison_ids = []


def build_agent(
    db: Any,
    llm: Any = None,
    knowledge_base: Optional[List[str]] = None,
    model: Optional[str] = None,
    checkpointer: Any = None,
    max_steps: int = MAX_AGENT_STEPS,
) -> MedicalAgent:
    """构建 Agent（对应原 ``build_rag_graph`` 的位置）。"""
    return MedicalAgent(
        db=db, llm=llm, knowledge_base=knowledge_base,
        model=model, checkpointer=checkpointer, max_steps=max_steps,
    )


def new_memory() -> MemorySaver:
    """新建一个会话记忆存储（进程内）。"""
    return MemorySaver()
