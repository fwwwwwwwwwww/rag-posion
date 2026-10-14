"""Agent 对话接口（SSE 流式推送决策轨迹）。

与 ``legacy.py`` 的 ``/chat/stream`` 的关键区别：
旧接口是"检索 → 过滤 → 生成"的固定流水线，前端只能看到最终答案；
本接口把 Agent 的**每一步决策**（调了什么工具、参数、结果摘要）实时推给前端，
让"模型在自主行动"这件事变成可见的。

事件序列：
    event: step   — 一步决策 {step, type: tool_call|tool_result, tool, ...}
    event: sources— 最终引用的证据
    event: done   — 结束 {answer, steps, refused, protocol}
    event: error  — 出错 {message}
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import (
    AGENT_MAX_STEPS,
    DEFAULT_LLM_MODEL,
    EMBEDDING_MODEL,
    RETRIEVAL_K_DEFAULT,
)
from app.db.engine import SessionLocal
import vectordb

logger = logging.getLogger("rag.api.agent")

router = APIRouter(prefix="/agent", tags=["agent"])

# ---------------- Agent 单例（懒加载，带会话记忆） ----------------
_agent_state: Dict[str, Any] = {"ready": False, "agent": None, "error": None, "kb_count": 0}
_init_lock = asyncio.Lock()


def _build_agent():
    """构造带 MemorySaver 会话记忆的 Agent。"""
    import RAGblack as exp
    from agent import build_agent, new_memory
    from langchain_chroma import Chroma
    from app.config import CHROMA_PERSIST_DIR

    embedding, _legacy_llm = exp.init_ollama(DEFAULT_LLM_MODEL)
    db = Chroma(embedding_function=embedding, persist_directory=CHROMA_PERSIST_DIR)
    chunk_count = vectordb.count_chunks(db)

    if chunk_count == 0:
        # 只有库是空的才需要加载语料重建；库已有内容时不加载，
        # 避免白白读入 33MB / 20000 篇知识库
        kb = exp.load_medical_knowledge_base()
        if not kb:
            raise RuntimeError("知识库为空且加载失败，请确认 medical_knowledge_base.json 存在")
        db = exp.build_clean_vectordb(kb, embedding)
        chunk_count = vectordb.count_chunks(db)

    agent = build_agent(
        db=db,
        llm=None,                       # 由 model 参数决定，走支持工具调用的对话层
        model=DEFAULT_LLM_MODEL,
        checkpointer=new_memory(),
        max_steps=AGENT_MAX_STEPS,
    )
    return agent, chunk_count


async def _ensure_agent():
    """异步初始化 Agent（同步构建丢线程池，避免阻塞事件循环）。"""
    if _agent_state["ready"]:
        return _agent_state["agent"]
    async with _init_lock:
        if _agent_state["ready"]:
            return _agent_state["agent"]
        try:
            agent, count = await asyncio.to_thread(_build_agent)
            _agent_state.update(ready=True, agent=agent, kb_count=count, error=None)
            logger.info("Agent 就绪：知识库 %d 个 chunk，模型 %s，通道 %s",
                        count, DEFAULT_LLM_MODEL, agent.llm.protocol())
        except Exception as e:
            _agent_state["error"] = str(e)
            logger.exception("Agent 初始化失败")
            raise
    return _agent_state["agent"]


class AgentChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    session_id: Optional[str] = Field(None, description="会话 id，留空则自动生成（用于多轮记忆）")
    max_steps: Optional[int] = Field(None, ge=1, le=20, description="覆盖本次工具循环上限")


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _build_sources(result) -> List[Dict[str, Any]]:
    out = []
    for rank, doc in enumerate(result.sources, start=1):
        meta = getattr(doc, "metadata", {}) or {}
        out.append({
            "rank": rank,
            "content": (getattr(doc, "page_content", "") or "")[:300],
            "poison_suspect": bool(meta.get("poison_suspect")),
            "query": meta.get("query", ""),
        })
    return out


async def _agent_stream(req: AgentChatRequest):
    """SSE 生成器：把 Agent 的决策轨迹与最终答案逐步推给前端。"""
    session_id = req.session_id or f"auto-{uuid.uuid4().hex[:12]}"
    yield _sse("session", {"session_id": session_id})

    try:
        agent = await _ensure_agent()
    except Exception as e:
        yield _sse("error", {"message": f"Agent 初始化失败: {e}"})
        return

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    sentinel = object()

    def on_step(event: Dict[str, Any]) -> None:
        """工作线程里的回调：把事件安全地投递到事件循环的队列。"""
        loop.call_soon_threadsafe(queue.put_nowait, event)

    async def run_agent():
        """在线程池里跑 Agent（同步阻塞），结束后投递结束标记。"""
        try:
            # max_steps 按请求传入，不修改共享的 agent 实例（避免并发互相干扰）
            result = await asyncio.to_thread(
                agent.run, req.question, session_id, None, on_step, req.max_steps)
            return result
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, sentinel)

    task = asyncio.create_task(run_agent())

    # 边跑边把轨迹推给前端
    while True:
        item = await queue.get()
        if item is sentinel:
            break
        yield _sse("step", item)

    try:
        result = await task
    except Exception as e:
        logger.exception("Agent 执行失败")
        yield _sse("error", {"message": f"Agent 执行失败: {e}"})
        return

    sources = _build_sources(result)
    yield _sse("sources", {"sources": sources})
    yield _sse("done", {
        "answer": result.answer,
        "session_id": session_id,
        "steps": result.steps,
        "refused": result.refused,
        "refusal_reason": result.refusal_reason,
        "protocol": result.protocol,
        "trace": result.trace,
    })

    # 落库审计（与旧接口共用 AttackLog 表）
    try:
        await asyncio.to_thread(_save_log, req.question, session_id, result, sources)
    except Exception as e:
        logger.warning("写入问答日志失败：%s", e)


def _save_log(question: str, session_id: str, result, sources: List[Dict[str, Any]]):
    from app.db.models import AttackLog
    db = SessionLocal()
    try:
        db.add(AttackLog(
            question=question,
            is_poison=any(s.get("poison_suspect") for s in sources),
            answer=result.answer,
            sources=sources,
        ))
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ---------------- 路由 ----------------
@router.post("/chat")
async def agent_chat(req: AgentChatRequest):
    """流式对话：返回 SSE，含决策轨迹。"""
    return StreamingResponse(
        _agent_stream(req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/ask")
async def agent_ask(req: AgentChatRequest):
    """非流式对话：一次性返回答案 + 完整决策轨迹。"""
    try:
        agent = await _ensure_agent()
    except Exception as e:
        return {"error": f"Agent 初始化失败: {e}"}

    session_id = req.session_id or f"auto-{uuid.uuid4().hex[:12]}"

    try:
        result = await asyncio.to_thread(
            agent.run, req.question, session_id, None, None, req.max_steps)
    except Exception as e:
        logger.exception("Agent 执行失败")
        return {"error": f"Agent 执行失败: {e}"}

    return {
        "answer": result.answer,
        "session_id": session_id,
        "steps": result.steps,
        "refused": result.refused,
        "refusal_reason": result.refusal_reason,
        "protocol": result.protocol,
        "trace": result.trace,
        "sources": _build_sources(result),
    }


@router.get("/tools")
async def agent_tools():
    """列出 Agent 可用的工具（前端用于展示"模型能做什么"）。"""
    try:
        agent = await _ensure_agent()
    except Exception as e:
        return {"error": f"Agent 初始化失败: {e}", "tools": []}
    return {
        "tools": [
            {
                "name": s["function"]["name"],
                "description": s["function"]["description"],
                "parameters": s["function"].get("parameters", {}),
            }
            for s in agent.schemas
        ]
    }


@router.get("/status")
async def agent_status():
    """Agent 状态：模型、工具调用通道、知识库规模。"""
    agent = _agent_state.get("agent")
    protocol = None
    tools_count = 0
    if agent is not None:
        try:
            protocol = agent.llm.protocol()
            tools_count = len(agent.schemas)
        except Exception:
            pass
    return {
        "ready": bool(_agent_state["ready"]),
        "error": _agent_state.get("error"),
        "model": DEFAULT_LLM_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "protocol": protocol,
        "supports_native_tools": protocol == "native",
        "tools": tools_count,
        "kb_chunks": _agent_state.get("kb_count", 0),
        "max_steps": AGENT_MAX_STEPS,
        "default_k": RETRIEVAL_K_DEFAULT,
    }


@router.delete("/session/{session_id}")
async def reset_session(session_id: str):
    """清空某个会话的记忆。"""
    agent = _agent_state.get("agent")
    if agent is None or agent.checkpointer is None:
        return {"status": "no_agent"}
    try:
        # MemorySaver 的存储是内存 dict，按其内部结构清理
        cp = agent.checkpointer
        removed = 0
        for attr in ("storage", "_storage"):
            store = getattr(cp, attr, None)
            if isinstance(store, dict):
                for key in list(store.keys()):
                    if session_id in str(key):
                        store.pop(key, None)
                        removed += 1
        return {"status": "cleared", "session_id": session_id, "removed": removed}
    except Exception as e:
        return {"status": "error", "message": str(e)}
