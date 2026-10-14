"""支持工具调用的 Ollama 对话模型（原生 tool_calls + 文本 ReAct 双通道）。

为什么需要这一层
----------------
原项目用的是 ``langchain_community.llms.Ollama``，它有两个致命限制：

1. **没有 ``bind_tools``**，无法把工具交给模型；
2. 它是纯补全接口（``LLM``），不返回 ``tool_calls`` 结构。

而且实测表明：``deepseek-r1:latest`` 这个项目默认模型**根本不支持**原生工具调用
（发送 tools 后返回的是散文，``message.tool_calls`` 为空），
``qwen2.5:3b`` / ``qwen3.5:2b`` 才支持。

因此本模块直接对接 Ollama 的 ``/api/chat``，提供两条通道：

- ``native``：走标准 tools 字段，模型返回结构化 ``tool_calls``（qwen 系列）；
- ``react``：把工具说明写进提示词，要求模型输出 JSON 决策，由本模块解析
  （deepseek-r1 等推理模型走这条）。

用哪种通道**不需要人工配置**：构造时自动探测一次并缓存结果。
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"

# 探测结果缓存：模型是否支持原生工具调用
_TOOL_SUPPORT_CACHE: Dict[str, bool] = {}

# 推理模型的思考段落（deepseek-r1 等）
_THINK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    """去掉推理模型的思考段落，并清理常见残留标记。

    原项目的正则只匹配 ``<think>...</think>``，遇到 ``thinking`` 变体、
    大写形式或未闭合标签就会把思考内容泄漏进解析逻辑，这里一并处理。
    """
    if not text:
        return ""
    text = _THINK_RE.sub("", text)
    # 未闭合的思考段落：直接截断到最后一个闭合标记之后
    for marker in ("</think>", "</thinking>"):
        if marker in text.lower():
            idx = text.lower().rindex(marker)
            text = text[idx + len(marker):]
    # 偶然出现的孤立标记
    text = re.sub(r"</?think(?:ing)?>", "", text, flags=re.IGNORECASE)
    return text.strip()


def _extract_json(text: str) -> Optional[dict]:
    """从模型输出里稳健地抠出第一个 JSON 对象。

    模型常见的三种输出：纯 JSON、```json 代码块、JSON 前后带解释文字。
    这里按优先级逐个尝试。
    """
    if not text:
        return None
    candidates: List[str] = []

    # 1) 代码块
    for block in re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL):
        candidates.append(block.strip())
    # 2) 全文
    candidates.append(text.strip())
    # 3) 第一个平衡花括号片段
    depth, start = 0, -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                candidates.append(text[start:i + 1])
                start = -1

    for cand in candidates:
        if not cand:
            continue
        try:
            obj = json.loads(cand)
        except Exception:
            # 容忍尾随逗号
            try:
                obj = json.loads(re.sub(r",\s*([}\]])", r"\1", cand))
            except Exception:
                continue
        if isinstance(obj, dict):
            return obj
    return None


@dataclass
class ToolCall:
    """一次工具调用请求。"""

    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    call_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "arguments": self.arguments, "call_id": self.call_id}


@dataclass
class ChatReply:
    """一次模型回复：要么要求调工具，要么给出最终文本。"""

    text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    # 模型原始输出，便于排查解析失败原因
    raw: str = ""

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
        }


def format_tools_for_prompt(tools: List[dict]) -> str:
    """把工具的 JSON Schema 渲染成人类可读的提示词片段（react 通道用）。"""
    lines = []
    for t in tools:
        fn = t.get("function", t)
        name = fn.get("name", "?")
        desc = fn.get("description", "")
        params = fn.get("parameters", {}).get("properties", {})
        required = set(fn.get("parameters", {}).get("required", []) or [])
        arg_bits = []
        for pname, pspec in params.items():
            ptype = pspec.get("type", "string")
            note = "" if pname in required else "，可选"
            arg_bits.append(f"{pname}: {ptype}{note}")
        lines.append(f"- {name}({', '.join(arg_bits)}) — {desc}")
    return "\n".join(lines)


class ToolChatModel:
    """带工具调用能力的 Ollama 对话模型。"""

    def __init__(
        self,
        model: str = "qwen2.5:3b",
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        temperature: float = 0.0,
        num_ctx: int = 8192,
        num_predict: int = 1024,
        timeout: int = 600,
        force_protocol: Optional[str] = None,
        think: Optional[bool] = False,
        keep_alive: Optional[Any] = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.timeout = timeout
        # 思考模型的推理段落会吃掉整个 num_predict 预算，导致 content 为空。
        # 实测 qwen3.5:2b 默认输出 9000+ 字思考、正文为空；think=False 后才正常。
        # 工具调用场景不需要暴露思考过程，因此默认关闭。
        # 传 None 表示不设置该参数（兼容不支持 think 字段的旧版 Ollama）。
        self.think = think
        # keep_alive：控制模型常驻时长。默认情况下 Ollama 5 分钟就卸载模型，
        # 而本项目会在"embedding 模型"与"对话模型"之间来回切换，
        # 每次切换都要重新加载（十几秒），窗口期还会返回空响应。
        # 传 -1 表示常驻不卸载。
        self.keep_alive = keep_alive
        # force_protocol: None=自动探测, "native"/"react"=强制走某条通道
        self.force_protocol = force_protocol
        self._protocol: Optional[str] = None

    # ---------------- 底层 HTTP ----------------
    def _post(self, path: str, payload: dict) -> dict:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            # Ollama 的详细错误在响应体里，直接抛 e 只能看到 "HTTP Error 400"，
            # 无法排查（例如上下文超长、模型不支持的参数）。这里把响应体带出来。
            detail = ""
            try:
                detail = e.read().decode("utf-8", errors="replace")[:1200]
            except Exception:
                pass
            raise RuntimeError(
                f"Ollama HTTP {e.code}: {detail or e.reason}") from e

    def _chat_payload(self, messages: List[dict], tools: Optional[List[dict]] = None) -> dict:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                # 限制生成长度：不设的话本地小模型可能一直生成到超时
                "num_predict": self.num_predict,
            },
        }
        if self.think is not None:
            payload["think"] = self.think
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        if tools:
            payload["tools"] = tools
        return payload

    # ---------------- 协议探测 ----------------
    def protocol(self) -> str:
        """返回该模型应使用的通道：``native`` 或 ``react``。"""
        if self._protocol is not None:
            return self._protocol
        if self.force_protocol in ("native", "react"):
            self._protocol = self.force_protocol
            return self._protocol

        key = f"{self.base_url}|{self.model}"
        if key in _TOOL_SUPPORT_CACHE:
            self._protocol = "native" if _TOOL_SUPPORT_CACHE[key] else "react"
            return self._protocol

        # 用一次极小的工具调用探测
        probe_tools = [{
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add two numbers",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                    "required": ["a", "b"],
                },
            },
        }]
        native = False
        try:
            resp = self._post("/api/chat", {
                "model": self.model,
                "messages": [{"role": "user", "content": "Use the add tool for 2+2."}],
                "tools": probe_tools,
                "stream": False,
                "think": False,  # 探测不需要思考，能显著提速
                "options": {"temperature": 0.0, "num_predict": 64},
            })
            native = bool(resp.get("message", {}).get("tool_calls"))
        except Exception:
            native = False

        _TOOL_SUPPORT_CACHE[key] = native
        self._protocol = "native" if native else "react"
        return self._protocol

    @property
    def supports_native_tools(self) -> bool:
        return self.protocol() == "native"

    def warmup(self) -> bool:
        """预热模型，把对话模型常驻显存；返回是否拿到非空输出。

        为什么需要：本地同时存在 embedding 模型（本项目用 4.36GB 的
        qwen3-embedding）与对话模型（2.35GB）。做一次检索就会把 embedding
        模型载入显存，紧接着的第一次对话调用需要**重新加载对话模型**，
        这个窗口期 Ollama 会返回**空 content**（不是报错、不是超时）。

        实测后果：采纳性判定拿到空文本，于是一次**成功的攻击被误判为"未采纳"**
        （回答明明推荐了假药，却被记成 rejected）。因此批量评测前必须先预热。
        """
        try:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "ok"}],
                "stream": False,
                "options": {"num_predict": 8, "temperature": 0.0},
            }
            if self.think is not None:
                payload["think"] = self.think
            if self.keep_alive is not None:
                payload["keep_alive"] = self.keep_alive
            resp = self._post("/api/chat", payload)
            content = (resp.get("message", {}) or {}).get("content") or ""
            if not content.strip():
                logger.warning("预热返回空内容（模型可能仍在加载）")
                return False
            return True
        except Exception as e:
            logger.warning("模型预热失败：%s", e)
            return False

    # ---------------- 对外调用 ----------------
    def chat(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        system: Optional[str] = None,
    ) -> ChatReply:
        """发一轮对话。有 tools 时按探测结果选择通道，可能返回工具调用请求。"""
        msgs: List[dict] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(messages)

        if not tools:
            return self._plain(msgs)

        if self.protocol() == "native":
            return self._native(msgs, tools)
        return self._react(msgs, tools)

    def _plain(self, msgs: List[dict]) -> ChatReply:
        """无工具的普通生成。"""
        try:
            resp = self._post("/api/chat", self._chat_payload(msgs))
        except Exception as e:
            return ChatReply(text=f"[模型调用失败: {e}]", raw=str(e))
        message = resp.get("message", {}) or {}
        raw = message.get("content", "") or ""
        # 区分"真的没话说"与"思考吃光了预算"：后者要给出可操作的提示，
        # 否则调用方只会看到一个莫名其妙的空答案。
        if not raw.strip():
            thinking = message.get("thinking", "") or ""
            if thinking:
                return ChatReply(
                    text=(f"[模型调用失败: 模型把全部生成预算用在了思考上"
                          f"（thinking {len(thinking)} 字），正文为空。"
                          f"请设置 think=False 或增大 num_predict。]"),
                    raw=thinking,
                )
        return ChatReply(text=strip_think(raw), raw=raw)

    def _native(self, msgs: List[dict], tools: List[dict]) -> ChatReply:
        """原生 tools 通道：模型返回结构化 tool_calls。"""
        try:
            resp = self._post("/api/chat", self._chat_payload(msgs, tools))
        except Exception as e:
            return ChatReply(text=f"[模型调用失败: {e}]", raw=str(e))

        message = resp.get("message", {}) or {}
        raw = message.get("content", "") or ""
        calls: List[ToolCall] = []
        for i, tc in enumerate(message.get("tool_calls") or []):
            fn = tc.get("function", {}) or {}
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"_raw": args}
            if not isinstance(args, dict):
                args = {}
            calls.append(ToolCall(
                name=fn.get("name", ""),
                arguments=args,
                call_id=tc.get("id") or f"call_{i}",
            ))
        # 原生通道下 content 可能就是最终答案（模型没调工具时）
        return ChatReply(text=strip_think(raw), tool_calls=calls, raw=raw)

    def _react(
        self,
        msgs: List[dict],
        tools: List[dict],
    ) -> ChatReply:
        """文本 ReAct 通道：把工具写进提示词，要求模型输出 JSON 决策。

        适配 deepseek-r1 这类不支持原生工具调用的推理模型。
        """
        names = [t.get("function", {}).get("name") for t in tools]
        valid_names = {n for n in names if n}
        protocol = (
            "\n\n【工具调用协议】\n"
            "你可以调用以下工具：\n"
            f"{format_tools_for_prompt(tools)}\n\n"
            "每次只输出一个 JSON 对象，不要输出任何其它文字、解释或 Markdown 代码块标记。\n"
            "需要调用工具时：\n"
            '{\"action\": \"tool\", \"tool\": \"工具名\", \"arguments\": {参数对象}}\n'
            "已经可以直接回答时：\n"
            '{\"action\": \"final\", \"answer\": \"你的最终回答\"}\n'
            f"tool 字段必须是这些之一：{', '.join([n for n in names if n])}"
        )

        # 把协议附加到最后一条 system 消息上（无 system 则新增）
        msgs = list(msgs)
        if msgs and msgs[0].get("role") == "system":
            msgs[0] = {**msgs[0], "content": msgs[0]["content"] + protocol}
        else:
            msgs.insert(0, {"role": "system", "content": protocol.lstrip("\n")})

        try:
            resp = self._post("/api/chat", self._chat_payload(msgs))
        except Exception as e:
            return ChatReply(text=f"[模型调用失败: {e}]", raw=str(e))

        raw = resp.get("message", {}).get("content", "") or ""
        clean = strip_think(raw)
        obj = _extract_json(clean)

        if not isinstance(obj, dict):
            # 没给出协议要求的 JSON：把这段文本当作最终回答
            # （模型在协议提示下仍直接作答，是最常见的降级情形）
            return ChatReply(text=clean, raw=raw)

        action = str(obj.get("action", "") or obj.get("type", "")).strip()
        action_l = action.lower()
        tool_name = (obj.get("tool") or obj.get("name") or obj.get("tool_name")
                     or obj.get("function"))
        if isinstance(tool_name, dict):          # 形如 {"function": {"name": ...}}
            tool_name = tool_name.get("name")

        # 兼容多种写法：模型经常把工具名直接写在 action 字段里
        # （实测 qwen 系会输出 {"action": "search_kb", "arguments": {...}}），
        # 也有写 {"tool": "search_kb"}、{"name": "search_kb"} 的。
        # 早期只认 action in ("tool","call","action")，会把这类**合法调用**
        # 当成最终回答吞掉——表现为"Agent 一次工具都没调用"。
        if not tool_name and action in valid_names:
            tool_name = action
        is_tool_call = bool(tool_name) and (
            action_l in ("tool", "call", "action", "function") or action in valid_names
        )

        if is_tool_call:
            args = obj.get("arguments", obj.get("args", obj.get("parameters", {})))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"_raw": args}
            if not isinstance(args, dict):
                args = {}
            return ChatReply(
                text="",
                tool_calls=[ToolCall(name=str(tool_name), arguments=args,
                                     call_id=f"react_{tool_name}")],
                raw=raw,
            )

        # action=final 或未声明 action 但给了 answer
        answer = obj.get("answer") or obj.get("final_answer") or obj.get("text")
        if isinstance(answer, str) and answer.strip():
            return ChatReply(text=answer.strip(), raw=raw)
        return ChatReply(text=clean, raw=raw)


class _LegacyOllamaAdapter:
    """把 ``langchain_community.llms.Ollama`` 适配成 ToolChatModel 的接口。

    旧版 LLM 没有工具能力，因此永远走 react 通道；保留它是为了让
    ``build_rag_graph(db, llm)`` 这类既有调用点不必改动。
    """

    def __init__(self, llm: Any, model: str = "", base_url: str = DEFAULT_OLLAMA_BASE_URL):
        self._llm = llm
        self.model = model or getattr(llm, "model", "legacy-ollama")
        self.base_url = base_url
        self.force_protocol = "react"

    @property
    def supports_native_tools(self) -> bool:
        return False

    def protocol(self) -> str:
        return "react"

    def _flatten(self, msgs: List[dict]) -> Any:
        """把消息列表压成旧版 LLM 能接受的形式。

        关键取舍：把 assistant 的工具调用与 tool 返回**渲染成纯文本**，
        而不是保留 ``role="tool"`` + ``tool_call_id`` 的消息。

        原因：``langchain_community.llms.Ollama`` 是非对话式的旧接口，
        对 role-based 的工具消息支持不完整——实测会在第二轮（历史里含
        tool 消息时）抛 ``KeyError: 'tool_call_id'``。而旧接口对
        (role, content) 纯文本对是稳定的。文本化之后模型依然能看到
        "我调了什么工具、拿到了什么结果"，只是少了结构化字段，
        对 ReAct 通道来说本来就是文本协议，没有损失。
        """
        if not msgs:
            return ""
        # 单条 user 消息直接传字符串，语义最清晰
        if len(msgs) == 1 and msgs[0].get("role") == "user":
            return msgs[0].get("content", "")

        pairs: List[tuple] = []
        for m in msgs:
            role = m.get("role", "user")
            content = m.get("content", "") or ""

            if role == "tool":
                # 工具返回：写成 plain text，避免 role=tool 的兼容问题
                name = m.get("tool_name") or "工具"
                pairs.append(("human", f"[{name} 的返回结果]\n{content}"))
                continue

            if role == "assistant" and m.get("tool_calls"):
                # 助手发起工具调用：把调用意图写成文本
                calls = []
                for tc in m.get("tool_calls") or []:
                    fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
                    args = fn.get("arguments", {})
                    if isinstance(args, dict):
                        args = json.dumps(args, ensure_ascii=False)
                    calls.append(f'{{"action": "tool", "tool": "{fn.get("name", "")}", '
                                 f'"arguments": {args}}}')
                text = "\n".join(calls)
                if content:
                    text = f"{content}\n{text}"
                pairs.append(("ai", text))
                continue

            mapped = {"assistant": "ai", "system": "system"}.get(role, role)
            pairs.append((mapped, content))
        return pairs

    def chat(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        system: Optional[str] = None,
    ) -> ChatReply:
        msgs: List[dict] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(messages)

        if tools:
            names = [t.get("function", {}).get("name") for t in tools]
            valid_names = {n for n in names if n}
            protocol = (
                "\n\n【工具调用协议】\n"
                "你可以调用以下工具：\n"
                f"{format_tools_for_prompt(tools)}\n\n"
                "每次只输出一个 JSON 对象，不要输出任何其它文字或代码块标记。\n"
                '调用工具：{\"action\": \"tool\", \"tool\": \"工具名\", \"arguments\": {参数对象}}\n'
                '直接回答：{\"action\": \"final\", \"answer\": \"你的最终回答\"}\n'
                f"tool 字段必须是这些之一：{', '.join([n for n in names if n])}"
            )
            msgs.append({"role": "system", "content": protocol})
        else:
            valid_names = set()

        try:
            raw = self._llm.invoke(self._flatten(msgs))
        except Exception as e:
            return ChatReply(text=f"[模型调用失败: {e}]", raw=str(e))
        if not isinstance(raw, str):
            raw = str(raw)

        clean = strip_think(raw)
        if not tools:
            return ChatReply(text=clean, raw=raw)

        obj = _extract_json(clean)
        if isinstance(obj, dict):
            action = str(obj.get("action", "") or obj.get("type", "")).strip()
            action_l = action.lower()
            tool_name = (obj.get("tool") or obj.get("name") or obj.get("tool_name")
                         or obj.get("function"))
            if isinstance(tool_name, dict):
                tool_name = tool_name.get("name")

            # 同样兼容"工具名直接写在 action 里"的写法，
            # 否则合法调用会被当成最终回答吞掉（见 _react 的说明）。
            if not tool_name and action in valid_names:
                tool_name = action
            is_tool_call = bool(tool_name) and (
                action_l in ("tool", "call", "action", "function")
                or action in valid_names
            )

            if is_tool_call:
                args = obj.get("arguments", obj.get("args", obj.get("parameters", {})))
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {"_raw": args}
                if not isinstance(args, dict):
                    args = {}
                return ChatReply(
                    text="",
                    tool_calls=[ToolCall(name=str(tool_name), arguments=args,
                                         call_id=f"legacy_{tool_name}")],
                    raw=raw,
                )
            answer = obj.get("answer") or obj.get("final_answer") or obj.get("text")
            if isinstance(answer, str) and answer.strip():
                return ChatReply(text=answer.strip(), raw=raw)
        return ChatReply(text=clean, raw=raw)


def ensure_tool_model(
    llm: Any,
    model: Optional[str] = None,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    temperature: float = 0.0,
) -> Any:
    """把任意 llm 入参归一化成具备 ``chat(messages, tools)`` 的对象。

    - 已经是 :class:`ToolChatModel` / 适配器：原样返回；
    - 是旧版 ``langchain_community.llms.Ollama``：包一层 react 适配器；
    - ``None`` 或字符串：按模型名新建 :class:`ToolChatModel`。
    """
    if isinstance(llm, (ToolChatModel, _LegacyOllamaAdapter)):
        return llm
    if llm is None:
        return ToolChatModel(model=model or "qwen2.5:3b",
                             base_url=base_url, temperature=temperature)
    if isinstance(llm, str):
        return ToolChatModel(model=llm, base_url=base_url, temperature=temperature)

    # 看起来像旧版 langchain LLM（有 invoke、没有 chat/tools）
    if hasattr(llm, "invoke") and not hasattr(llm, "bind_tools"):
        return _LegacyOllamaAdapter(llm, model=model or "", base_url=base_url)

    return llm
