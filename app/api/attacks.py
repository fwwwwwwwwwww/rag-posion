"""自适应攻击接口。

与旧实验接口的区别：旧接口跑的是"固定攻击 × 固定防御"，
本接口跑的是**自适应攻击战役**——攻击 Agent 每轮根据防御反馈换策略，
产出攻击轨迹（`attack_iterations` 表），而不只是一个 ASR 数字。
"""
import logging
import threading
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import DEFAULT_LLM_MODEL
from app.db.engine import get_db
from app.db.models import AttackIteration

logger = logging.getLogger("rag.api.attacks")

router = APIRouter(prefix="/attacks", tags=["attacks"])

# 防止同一场景被重复提交
_running: set = set()
_lock = threading.Lock()


class CampaignRequest(BaseModel):
    attack_index: int = Field(0, ge=0, description="攻击场景索引（对应 ATTACK_PAIRS）")
    max_rounds: int = Field(4, ge=1, le=10, description="最大迭代轮数")
    runs_per_round: int = Field(1, ge=1, le=5, description="每轮重复次数")
    doc_count: int = Field(15, ge=5, le=50, description="每轮投放的文档篇数")
    model_name: str = Field(DEFAULT_LLM_MODEL, description="攻击与判定使用的模型")
    defense_variant: str = Field("agent", description="目标防御变体")


class CampaignStarted(BaseModel):
    status: str
    attack_index: int
    message: str


def _run_campaign_job(attack_index: int, req: CampaignRequest) -> None:
    """后台线程执行：跑自适应攻击战役。"""
    import RAGblack as exp
    from langchain_chroma import Chroma
    from langchain_community.embeddings import OllamaEmbeddings
    from attack.campaign import run_campaign
    from app.config import CHROMA_PERSIST_DIR

    try:
        embedding = OllamaEmbeddings(model=exp.EMBEDDING_MODEL,
                                     base_url=exp.OLLAMA_BASE_URL)
        db = Chroma(embedding_function=embedding,
                    persist_directory=CHROMA_PERSIST_DIR)
        exp._vectordb_cache = db
        kb = exp.load_medical_knowledge_base()
        task = exp.ATTACK_PAIRS[attack_index]

        # 每个轮次都重建防御，避免上一轮的 Agent 状态串场
        def build(variant: str, poison_docs: List[str]):
            return exp.build_rag(kb, poison_docs, embedding, None,
                                 variant=variant, model_name=req.model_name)

        run_campaign(
            db=db, task=task, build_rag_fn=lambda variant, poison_docs:
                build(variant, poison_docs),
            llm=req.model_name, max_rounds=req.max_rounds,
            runs_per_round=req.runs_per_round, doc_count=req.doc_count,
            verbose=True, persist=True,
        )
        logger.info("攻击战役完成：场景 %d", attack_index)
    except Exception:
        logger.exception("攻击战役失败：场景 %d", attack_index)
    finally:
        with _lock:
            _running.discard(attack_index)


@router.post("/campaign", response_model=CampaignStarted, status_code=202)
async def start_campaign(req: CampaignRequest):
    """启动一次自适应攻击战役（后台执行，结果写入 attack_iterations）。"""
    import RAGblack as exp

    if req.attack_index >= len(exp.ATTACK_PAIRS):
        raise HTTPException(status_code=400,
                            detail=f"攻击场景索引越界（共 {len(exp.ATTACK_PAIRS)} 个）")
    with _lock:
        if req.attack_index in _running:
            raise HTTPException(status_code=409, detail="该场景的攻击战役已在运行")
        _running.add(req.attack_index)

    t = threading.Thread(target=_run_campaign_job, args=(req.attack_index, req),
                         daemon=True)
    t.start()
    return CampaignStarted(
        status="started", attack_index=req.attack_index,
        message=f"已提交后台执行，轮数上限 {req.max_rounds}，"
                f"结果写入 attack_iterations（可用 GET /api/attacks/iterations 查看）")


@router.get("/iterations")
async def list_iterations(
    db: Session = Depends(get_db),
    attack_index: Optional[int] = Query(None, description="按场景过滤"),
    limit: int = Query(200, le=1000),
):
    """查询攻击迭代轨迹。"""
    stmt = select(AttackIteration).order_by(
        AttackIteration.id.desc()).limit(limit)
    rows = list(db.scalars(stmt).all())
    if attack_index is not None:
        import RAGblack as exp
        q = exp.ATTACK_PAIRS[attack_index]["question"] if \
            attack_index < len(exp.ATTACK_PAIRS) else None
        if q:
            rows = [r for r in rows if r.question == q]
    return [
        {
            "id": r.id,
            "round_index": r.round_index,
            "domain": r.domain,
            "question": r.question,
            "strategy": r.strategy,
            "strategy_reason": r.strategy_reason,
            "injected_docs": r.injected_docs,
            "injected_chunks": r.injected_chunks,
            "retrieved_docs": r.retrieved_docs,
            "retrieved_chunks": r.retrieved_chunks,
            "detected": r.detected,
            "recall_at_k": r.recall_at_k,
            "detection_rate": r.detection_rate,
            "survive": r.survive,
            "asr": r.asr,
            "verdict": r.verdict,
            "failure_layer": r.failure_layer,
            "sample_answer": r.sample_answer,
            "seconds": r.seconds,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


@router.get("/strategies")
async def list_strategies():
    """列出可用的攻击策略（前端展示"攻击者能做什么"）。"""
    from attack.strategies import STRATEGIES, STRATEGY_NOTES
    return [
        {"name": name, "note": STRATEGY_NOTES.get(name, "")}
        for name in STRATEGIES
    ]


@router.get("/variants")
async def list_variants():
    """列出可对比的防御变体。"""
    import RAGblack as exp
    notes = {
        exp.VARIANT_NAIVE: "裸 RAG：检索后直接生成，无任何核查（对照组）",
        exp.VARIANT_FIXED: "原固定链路：retrieve → grade → generate",
        exp.VARIANT_AGENT_STAT: "Agent + 旧统计检测器（消融对照，召回率 0%）",
        exp.VARIANT_AGENT: "Agent + 新混合检测器（规则 + LLM 语义判定）",
    }
    return [{"name": v, "note": notes.get(v, "")} for v in exp.DEFENSE_VARIANTS]
