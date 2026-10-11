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
    chunk_count = db._collection.count()
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


def _grade_filter(llm, question: str, docs) -> List:
    """相关性评分过滤（与 rag_graph.grade 节点共用同一 Prompt 与解析逻辑）。"""
    if not docs:
        return []
    from rag_graph import GRADE_PROMPT, _parse_grade_ids
    docs_block = "\n".join(
        f"{i}. {doc.page_content[:500]}" for i, doc in enumerate(docs, 1))
    answer = llm.invoke(GRADE_PROMPT.format(
        question=question, docs_block=docs_block))
    ids = set(_parse_grade_ids(answer, len(docs)))
    return [doc for i, doc in enumerate(docs, 1) if i in ids]


def _build_sources(docs, poison_content_set: set) -> List[Dict]:
    sources = []
    for rank, doc in enumerate(docs, start=1):
        content = doc.page_content
        is_poison = content in poison_content_set
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
    """异步 SSE 生成器：检索 + LLM 流式输出。"""
    await _ensure_ready_async()
    db = _state["db"]
    llm = _state["llm"]

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

        retriever = db.as_retriever(search_kwargs={"k": req.k})

        # 检索（丢线程池避免阻塞）
        docs = await asyncio.to_thread(retriever.invoke, req.question)

        # 相关性评分过滤（与实验 rag_graph 的 grade 节点同一套防御逻辑）
        filtered = await asyncio.to_thread(_grade_filter, llm, req.question, docs)
        if not filtered:
            filtered = docs  # 全部被过滤时回退原始检索，避免空上下文

        sources = _build_sources(filtered, poison_content_set)
        yield _sse("sources", {"sources": sources})

        # LLM 流式输出
        from rag_graph import GENERATE_PROMPT
        context = "\n\n".join(d.page_content for d in filtered)

        def _gen():
            return list(llm.stream(GENERATE_PROMPT.format(
                context=context, question=req.question)))

        tokens = await asyncio.to_thread(_gen)
        for token in tokens:
            if token:
                answer_text += token
                yield _sse("token", {"text": token})

        yield _sse("done", {"status": "ok"})
        await asyncio.to_thread(
            _save_attack_log, req.question, req.poisoned, answer_text, sources)

    except Exception as e:
        yield _sse("error", {"message": str(e)})

    finally:
        if poison_ids:
            try:
                await asyncio.to_thread(
                    lambda: db._collection.delete(ids=poison_ids))
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
