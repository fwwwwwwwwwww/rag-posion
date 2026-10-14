"""能力矩阵：不同模型 × 不同工具集 × 不同提示词，找出能稳定触发工具调用的组合。"""
import json
import sys

sys.path.insert(0, r"E:\RAG2")

from agent.agent import _fmt_tools_block
from agent.chat_ollama import ToolChatModel
from agent.prompts import AGENT_SYSTEM
from agent.tools import TOOL_SCHEMAS

QUESTION = "上海肺癌治疗哪家医院最权威？"

SYSTEM_SHORT = (
    "你是医学知识库问答助手。回答任何问题前，必须先调用工具检索知识库，"
    "不得凭记忆直接作答。"
)
SYSTEM_PLAIN = (
    "你是一个医学知识库问答 Agent。你可以调用工具。"
    "遇到问题时先调用 search_kb 检索，再根据检索结果回答。"
)

VARIANTS = {
    "全部5工具": TOOL_SCHEMAS,
    "仅检索+拒答": [s for s in TOOL_SCHEMAS
                    if s["function"]["name"] in ("search_kb", "refuse")],
    "仅检索": [s for s in TOOL_SCHEMAS if s["function"]["name"] == "search_kb"],
}

PROMPTS = {
    "AGENT_SYSTEM": lambda tools: AGENT_SYSTEM.format(tools_block=_fmt_tools_block(tools)),
    "SHORT": lambda tools: SYSTEM_SHORT,
    "PLAIN": lambda tools: SYSTEM_PLAIN,
}

MODELS = ["qwen2.5:3b", "qwen3.5:2b"]


def probe(model, prompt_name, tools_name):
    tools = VARIANTS[tools_name]
    system = PROMPTS[prompt_name](tools)
    m = ToolChatModel(model=model, timeout=300)
    if m.protocol() != "native":
        return "非native通道"
    try:
        r = m.chat([{"role": "user", "content": QUESTION}], tools=tools, system=system)
    except Exception as e:
        return f"异常 {type(e).__name__}"
    if r.tool_calls:
        return f"OK 调用了 {r.tool_calls[0].name}"
    if r.text.strip():
        return f"未调工具，直接作答({len(r.text)}字)"
    return "空响应"


print(f"问题: {QUESTION}")
print(f"模型 {MODELS}  ×  提示词 {list(PROMPTS)}  ×  工具集 {list(VARIANTS)}\n")
header = f"{'模型':<14}{'提示词':<14}" + "".join(f"{k:<22}" for k in VARIANTS)
print(header)
print("-" * len(header))
for model in MODELS:
    for pname in PROMPTS:
        row = f"{model:<14}{pname:<14}"
        for tname in VARIANTS:
            row += f"{probe(model, pname, tname):<22}"
        print(row)
    print()
