"""RAG 链路入口（现已重构为工具化 Agent）。

历史
----
本文件原先实现的是 ``retrieve → grade → generate`` 固定链路：
下一步走哪条边由 Python 的 ``if state["filtered"]`` 和模块常量
``MAX_QUERY_REWRITES`` 决定，模型只负责在三个固定提示词空位里填空。
那**不是 Agent，只是一条被打扮成状态图的流水线**。

现在
----
实际实现在 :mod:`agent` 包里，本文件只保留兼容外观，
让 ``RAGblack.py`` / ``app/api/legacy.py`` 等既有调用点无需改动：

    build_rag_graph(db, llm, k=3) -> MedicalAgent（接口兼容 RetrievalQA）

Agent 的真实行为：
  - 模型通过 ``bind_tools`` 式的工具协议**自主决定**调用哪个工具、调几次；
  - 工具集：``search_kb`` / ``check_poison`` / ``cross_check`` /
    ``rewrite_query`` / ``refuse``；
  - 循环终止条件是"模型不再请求工具"，而不是某个写死的常量；
  - 支持会话记忆（checkpointer），可处理多轮追问；
  - 证据不足时调用 ``refuse`` **明确拒答**，而不是关掉防御继续编造。

兼容说明
--------
``GRADE_PROMPT`` / ``REWRITE_PROMPT`` / ``_parse_grade_ids`` 是旧实现
（基于"让 LLM 输出编号再用正则解析"）的残留符号。保留它们仅为避免
外部导入报错，新代码不应再使用——相关性判断已改由 Agent 用工具完成。
"""
from __future__ import annotations

import re
from typing import Any, List, Optional

from agent import MedicalAgent, new_memory  # noqa: F401  (对外的构建入口)
from agent.agent import MAX_AGENT_STEPS
from agent.chat_ollama import strip_think
from agent.prompts import GENERATE_PROMPT  # noqa: F401  (兼容旧导入)

# ---------------- 旧符号（已废弃，仅为兼容保留） ----------------
# 相关性评分不再靠"模型输出编号 + 正则解析"，而是由 Agent 调用
# cross_check / check_poison 工具完成。正则在"共5篇"这类输出上会误判。
GRADE_PROMPT = None
REWRITE_PROMPT = None


def _parse_grade_ids(answer: str, n_docs: int) -> List[int]:
    """【已废弃】从评分回复中解析文档编号。

    保留函数签名以免破坏外部导入；Agent 已不再使用编号式评分。
    """
    text = strip_think(answer or "")
    if not text or "无" in text[:20]:
        return []
    valid = set(range(1, n_docs + 1))
    ids: List[int] = []
    for token in re.findall(r"\d+", text):
        num = int(token)
        if num in valid and num not in ids:
            ids.append(num)
    return ids


# ---------------- 兼容包装 ----------------
class RAGGraph(MedicalAgent):
    """原 ``RAGGraph`` 的兼容名。

    它现在是一个**真正的工具化 Agent**：``invoke({"query": q})`` 仍返回
    ``{"result", "source_documents"}``，但内部走的是模型自主的工具循环。
    """

    def __init__(self, app: Any = None, db: Any = None, k: int = 5,
                 llm: Any = None, model: Optional[str] = None,
                 knowledge_base: Optional[List[str]] = None,
                 session_id: Optional[str] = None):
        # 原构造签名为 (app, db, k)，app 是编译好的旧图；现在不再需要它
        super().__init__(
            db=db or getattr(app, "_db", None),
            llm=llm,
            knowledge_base=knowledge_base,
            model=model,
        )
        self._k = k
        self._session_id = session_id

    def invoke(self, payload: Any) -> dict:
        """兼容原 RetrievalQA 的 ``invoke({"query": q})`` 入口。

        返回值除了 ``result`` / ``source_documents`` 之外，还带上
        ``trace`` / ``refused`` / ``steps``——实验层（``eval.metrics``）
        要靠 trace 还原分层指标。

        踩过的坑：这里曾经只返回 ``{"result", "source_documents"}`` 两个键，
        把 Agent 的决策轨迹整个丢掉了，导致所有分层指标（召回率/检出率）
        恒为 0，看起来像"投毒根本没进检索"，实际是度量被截断了。
        """
        # 注入默认会话 id，使同一 RAGGraph 实例上的多次提问共享记忆
        if isinstance(payload, dict):
            payload = {**payload}
            payload.setdefault("session_id", self._session_id)
        return super().invoke(payload)


def build_rag_graph(db, llm, k: int = 5, knowledge_base=None, checkpointer=None,
                    model: Optional[str] = None) -> RAGGraph:
    """构建 Agent（保持原函数名与调用方式）。

    参数:
        db:             Chroma 向量库
        llm:            LLM 对象（旧版 ``langchain`` LLM 或模型名均可）
        k:              建议检索条数，作为 Agent 的默认提示（模型可自行调整）
        knowledge_base: 干净文档列表，用于训练投毒检测器（可选）
        checkpointer:   会话记忆存储，传入后支持多轮对话
        model:          模型名，llm 为 None 时使用

    返回:
        RAGGraph（MedicalAgent 子类），接口兼容原 RetrievalQA。
    """
    agent = RAGGraph(
        app=None, db=db, k=k, llm=llm, model=model,
        knowledge_base=knowledge_base,
    )
    if checkpointer is not None:
        agent.checkpointer = checkpointer
        agent._app = agent._compile()
    return agent
