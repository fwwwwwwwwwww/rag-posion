"""向后兼容的原有接口（前端 index.html 依赖），并追加问答日志记录。

保留原 server.py 的：
  - GET  /api/health
  - GET  /api/attacks
  - POST /api/chat/stream  (SSE 流式，异步)
新增：
  - GET  /api/logs          (问答流水日志查询)
  - POST /api/vectorize     (批量文档向量化，丢后台处理)
"""
import asyncio
import json
from typing import Dict, List

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import (
    DEFAULT_LLM_MODEL,
    EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR,
    RETRIEVAL_K_DEFAULT,
)
from app.db.engine import get_db, SessionLocal
from app.db.models import AttackLog
import vectordb

router = APIRouter(tags=["legacy"])

_state_lock = asyncio.Lock()


def _get_exp():
    """懒加载 RAGblack 实验模块。"""
    import RAGblack as exp
    return exp


# ---------------- 全局状态（懒加载，首次请求初始化） ----------------
_state = {
    "ready": False,
    "db": None,
    "embedding": None,
    "llm": None,
    "kb_count": 0,
}
_poison_cache: Dict[int, List[str]] = {}


def _load_or_build_db(embedding):
    """优先复用磁盘上已构建的向量库；为空时才全量 embedding 构建。"""
    exp = _get_exp()
    from langchain_chroma import Chroma
    db = Chroma(embedding_function=embedding,
                persist_directory=CHROMA_PERSIST_DIR)
    chunk_count = vectordb.count_chunks(db)
    if chunk_count > 0:
        return db, chunk_count
    kb = exp.load_medical_knowledge_base()
    if not kb:
        raise RuntimeError("知识库加载失败，请确认 medical_knowledge_base.json 存在")
    return exp.build_clean_vectordb(kb, embedding), len(kb)


def _ensure_ready():
    """同步初始化 RAG 环境（在线程池中执行避免阻塞事件循环）。"""
    if _state["ready"]:
        return
    exp = _get_exp()
    embedding, llm = exp.init_ollama()
    db, count = _load_or_build_db(embedding)
    _state.update(db=db, embedding=embedding, llm=llm,
                  kb_count=count, ready=True)


async def _ensure_ready_async():
    """异步包装：将同步初始化丢到线程池。"""
    if _state["ready"]:
        return
    async with _state_lock:
        if _state["ready"]:
            return
        await asyncio.to_thread(_ensure_ready)


_agent_cache = {"agent": None, "db_id": None}


def _ensure_agent(db):
    """为旧接口构造（并缓存）一个工具化 Agent。

    旧接口原先自己做"检索 → 评分过滤 → 生成"，那条链路已随重构移除；
    现在统一走 Agent，行为与新接口保持一致。
    """
    if _agent_cache["agent"] is not None and _agent_cache["db_id"] == id(db):
        return _agent_cache["agent"]

    from agent import build_agent

    agent = build_agent(db=db, llm=None, model=DEFAULT_LLM_MODEL)
    _agent_cache.update(agent=agent, db_id=id(db))
    return agent


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000,
                          description="用户问题")
    k: int = Field(RETRIEVAL_K_DEFAULT, ge=1, le=50,
                   description="检索 Top-K")
    poisoned: bool = Field(False, description="是否注入投毒文档")
    attack_index: int = Field(0, ge=0, description="攻击场景索引")


class VectorizeRequest(BaseModel):
    documents: List[str] = Field(..., min_length=1, description="待向量化的文档列表")
    chunk_size: int = Field(500, ge=100, le=2000)
    chunk_overlap: int = Field(50, ge=0, le=500)


def _get_poison_docs(attack_index: int) -> List[str]:
    exp = _get_exp()
    if attack_index not in _poison_cache:
        task = exp.ATTACK_PAIRS[attack_index]
        _poison_cache[attack_index] = exp._generate_poison_variants(
            task, exp.POISON_DOC_COUNT)
    return _poison_cache[attack_index]


def _build_sources(docs, poison_content_set: set) -> List[Dict]:
    sources = []
    for rank, doc in enumerate(docs, start=1):
        content = doc.page_content
        meta = getattr(doc, "metadata", {}) or {}
        # 投毒标记有两个来源：本次注入的文档内容，或 Agent 的 check_poison 判定
        is_poison = (content in poison_content_set) or bool(meta.get("poison_suspect"))
        sources.append({
            "rank": rank,
            "content": content[:300],
            "is_poison": is_poison,
        })
    return sources


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _save_attack_log(question: str, is_poison: bool,
                     answer: str, sources: List[Dict]):
    """流式结束后把本次问答写入日志表。"""
    db = SessionLocal()
    try:
        db.add(AttackLog(
            question=question,
            is_poison=is_poison,
            answer=answer,
            sources=sources,
        ))
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


async def _chat_stream(req: ChatRequest):
    """异步 SSE 生成器：调用工具化 Agent，再把答案按 token 切块推出去。

    与旧实现的区别：不再由代码做"检索 → 评分过滤 → 生成"的固定流程，
    而是把问题交给 Agent（模型自主决定检索、投毒检测、交叉验证或拒答）。
    前端事件格式保持不变（sources / token / done / error），因此旧页面无需改动。
    """
    await _ensure_ready_async()
    db = _state["db"]

    poison_docs: List[str] = []
    poison_ids: List[str] = []
    poison_content_set: set = set()
    answer_text = ""
    sources: List[Dict] = []

    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, length_function=len)

    try:
        if req.poisoned:
            poison_docs = await asyncio.to_thread(_get_poison_docs, req.attack_index)
            poison_chunks = splitter.create_documents(poison_docs)
            poison_content_set = {c.page_content for c in poison_chunks}
            poison_ids = await asyncio.to_thread(db.add_documents, poison_chunks)

        # 交给 Agent 推理（同步阻塞，丢线程池）
        agent = await asyncio.to_thread(_ensure_agent, db)
        result = await asyncio.to_thread(agent.run, req.question)

        sources = _build_sources(result.sources, poison_content_set)
        # 前端只认 is_poison；Agent 的检测结论也映射到这个字段
        for s, doc in zip(sources, result.sources):
            s["is_poison"] = s["is_poison"] or bool(
                (getattr(doc, "metadata", {}) or {}).get("poison_suspect"))
        yield _sse("sources", {"sources": sources})

        # 把完整答案分块推送，模拟流式，保持前端契约不变
        answer_text = result.answer or ""
        chunk = 24
        for i in range(0, len(answer_text), chunk):
            yield _sse("token", {"text": answer_text[i:i + chunk]})

        yield _sse("done", {"status": "ok", "refused": result.refused,
                            "steps": result.steps})
        await asyncio.to_thread(
            _save_attack_log, req.question, req.poisoned, answer_text, sources)

    except Exception as e:
        yield _sse("error", {"message": str(e)})

    finally:
        if poison_ids:
            try:
                await asyncio.to_thread(vectordb.delete_ids, db, poison_ids)
            except Exception:
                pass


# ---------------- 路由 ----------------
@router.get("/health")
async def health():
    return {
        "status": "ready" if _state["ready"] else "lazy",
        "model": DEFAULT_LLM_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "kb_documents": _state["kb_count"],
    }


@router.get("/attacks")
async def list_attacks():
    exp = _get_exp()
    return [
        {"index": i, "domain": t["domain"],
         "question": t["question"], "target": t["target_answer"]}
        for i, t in enumerate(exp.ATTACK_PAIRS)
    ]


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    return StreamingResponse(
        _chat_stream(req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/vectorize")
async def vectorize_documents(req: VectorizeRequest):
    """批量文档向量化，丢后台处理，避免接口超时。"""
    from app.tasks.celery_app import submit_vectorize_task
    task_id = submit_vectorize_task(
        req.documents, req.chunk_size, req.chunk_overlap)
    return {"status": "submitted", "task_id": task_id,
            "doc_count": len(req.documents)}


@router.get("/logs")
async def list_logs(db: Session = Depends(get_db),
                   limit: int = 100, offset: int = 0):
    stmt = select(AttackLog).order_by(
        AttackLog.id.desc()).limit(limit).offset(offset)
    logs = list(db.scalars(stmt).all())
    return [
        {
            "id": l.id,
            "question": l.question,
            "is_poison": l.is_poison,
            "answer": l.answer,
            "sources": l.sources,
            "created_at": l.created_at.isoformat() if l.created_at else None,
        }
        for l in logs
    ]
