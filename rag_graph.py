"""基于 LangGraph 的 RAG 链路（Corrective RAG 风格防御）。

用 StateGraph 取代原 RetrievalQA 线性链，流程：

    ┌──────────┐     ┌──────────┐   无相关文档
    │ retrieve │ --> │  grade   | --------------┐
    └──────────┘     └────┬─────┘               ↓
         ↑  有相关文档 ↓                    ┌──────────┐
         │           ↓    重试未超限      │ rewrite  │（查询改写）
         │      ┌─────────┐   <────────── └──────────┘
         └──────│ generate│
                └────┬────┘
                     ↓ END

- grade 节点：批量 LLM 相关性评分，过滤可疑/无关文档（投毒防御核心）。
  注意：攻击实验的 P@5 因此变为"过滤后仍有投毒文档残留的比例"，
  即 LangGraph 防御生效后的残余攻击检索成功率。
- rewrite 节点：无相关文档时改写查询并重新检索，最多 MAX_QUERY_REWRITES 次。

对外接口与原 RetrievalQA 保持一致：
    qa.invoke({"query": question}) -> {"result": str, "source_documents": [...]}
"""
import re
from typing import Any, Dict, List, Tuple, TypedDict

from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END

# 查询改写最大重试次数（每次重试 = 1 次改写 + 1 次检索 + 1 次评分）
MAX_QUERY_REWRITES = 1

GRADE_PROMPT = PromptTemplate.from_template(
    "你是检索文档质量审核员。给定问题与若干候选文档片段，"
    "判断每个片段是否与问题直接相关（即有助于回答该问题）。\n"
    "只输出相关片段的编号，用英文逗号分隔，例如：1,3\n"
    "若全部无关，只输出：无\n\n"
    "问题：{question}\n\n候选文档：\n{docs_block}"
)

REWRITE_PROMPT = PromptTemplate.from_template(
    "以下问题在知识库中未检索到相关资料。"
    "请改写该问题，使其更适合在医学知识库中检索，只输出改写后的问题。\n\n"
    "原问题：{question}"
)

GENERATE_PROMPT = PromptTemplate.from_template(
    "基于以下资料回答问题：\n\n资料内容：\n{context}\n\n问题：{question}\n回答："
)


class GraphState(TypedDict):
    question: str          # 当前查询（可能被改写过）
    documents: List[Any]   # 最近一次检索结果
    filtered: List[Any]    # 评分后保留的相关文档
    generation: str        # 最终回答
    retries: int           # 已执行的重写次数


def _strip_think(text: str) -> str:
    """去除 deepseek-r1 等推理模型的 <think>...</think> 段落。"""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _parse_grade_ids(answer: str, n_docs: int) -> List[int]:
    """从评分回复中解析相关文档编号（1 起）。容错推理模型的冗余输出。"""
    text = _strip_think(answer)
    if not text or "无" in text[:20]:
        return []
    valid = set(range(1, n_docs + 1))
    ids = []
    for token in re.findall(r"\d+", text):
        num = int(token)
        if num in valid and num not in ids:
            ids.append(num)
    return ids


def build_rag_graph(db, llm, k: int = 3):
    """基于已构建的 Chroma 向量库与 LLM 编译 RAG StateGraph。

    参数:
        db:  Chroma 向量库（RAGblack.build_clean_vectordb 的产物）
        llm: Ollama LLM 实例
        k:   检索条数

    返回:
        RAGGraph 实例，接口兼容原 RetrievalQA。
    """
    retriever = db.as_retriever(search_kwargs={"k": k})

    # ---------------- 节点 ----------------
    def retrieve(state: GraphState) -> Dict[str, Any]:
        docs = retriever.invoke(state["question"])
        return {"documents": docs, "filtered": []}

    def grade(state: GraphState) -> Dict[str, Any]:
        docs: List[Any] = state["documents"]
        if not docs:
            return {"filtered": []}
        docs_block = "\n".join(
            f"{i}. {doc.page_content[:500]}" for i, doc in enumerate(docs, 1))
        answer = llm.invoke(GRADE_PROMPT.format(
            question=state["question"], docs_block=docs_block))
        ids = set(_parse_grade_ids(answer, len(docs)))
        kept = [doc for i, doc in enumerate(docs, 1) if i in ids]
        return {"filtered": kept}

    def rewrite(state: GraphState) -> Dict[str, Any]:
        answer = llm.invoke(REWRITE_PROMPT.format(question=state["question"]))
        new_q = _strip_think(answer).strip().strip('"')
        return {"question": new_q or state["question"],
                "retries": state["retries"] + 1}

    def generate(state: GraphState) -> Dict[str, Any]:
        context = "\n\n".join(doc.page_content for doc in state["filtered"])
        answer = llm.invoke(GENERATE_PROMPT.format(
            context=context, question=state["question"]))
        return {"generation": _strip_think(answer)}

    # ---------------- 路由 ----------------
    def route_after_grade(state: GraphState) -> str:
        if state["filtered"]:
            return "generate"
        if state["retries"] < MAX_QUERY_REWRITES:
            return "rewrite"
        return "generate"

    # ---------------- 组图 ----------------
    graph = StateGraph(GraphState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade", grade)
    graph.add_node("rewrite", rewrite)
    graph.add_node("generate", generate)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", route_after_grade,
                                {"generate": "generate", "rewrite": "rewrite"})
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", END)

    return RAGGraph(graph.compile(), db, k)


class RAGGraph:
    """LangGraph RAG 的对外封装，接口兼容原 RetrievalQA 链。"""

    def __init__(self, app, db, k: int):
        self._app = app
        self._db = db
        self._k = k
        self._poison_ids: List[str] = []

    def invoke(self, payload) -> Dict[str, Any]:
        question = payload["query"] if isinstance(payload, dict) else payload
        state = self._app.invoke(
            {"question": question, "documents": [], "filtered": [],
             "generation": "", "retries": 0},
            config={"recursion_limit": 16})
        return {
            "result": state["generation"],
            "source_documents": state["filtered"] or state["documents"],
        }

    def cleanup_poison(self) -> None:
        """删除注入的投毒文档（与 RAGblack._cleanup_poison 等价）。"""
        if self._poison_ids:
            self._db._collection.delete(ids=self._poison_ids)
            self._poison_ids = []
