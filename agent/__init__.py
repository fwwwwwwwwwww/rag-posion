"""工具化医疗 RAG Agent 包。

对外主要入口：
    from agent import build_agent, MedicalAgent, new_memory

模块划分：
    chat_ollama.py  支持工具调用的 Ollama 对话层（原生 tool_calls + 文本 ReAct 双通道）
    tools.py        工具集（检索 / 投毒检测 / 交叉验证 / 改写 / 拒答）
    prompts.py      Agent 策略提示词
    agent.py        LangGraph 工具循环 + 会话记忆
"""
from .agent import (
    MAX_AGENT_STEPS,
    AgentResult,
    MedicalAgent,
    build_agent,
    new_memory,
)
from .chat_ollama import ChatReply, ToolCall, ToolChatModel, ensure_tool_model, strip_think

__all__ = [
    "build_agent",
    "MedicalAgent",
    "AgentResult",
    "new_memory",
    "MAX_AGENT_STEPS",
    "ToolChatModel",
    "ToolCall",
    "ChatReply",
    "ensure_tool_model",
    "strip_think",
]
