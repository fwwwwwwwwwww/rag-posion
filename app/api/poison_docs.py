"""投毒文档管理接口（异步）。"""
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.engine import get_db
from app.schemas.poison_doc import PoisonDocGenerateRequest, PoisonDocOut
from app.services import poison_doc_service

router = APIRouter(prefix="/poison-docs", tags=["poison-docs"])


@router.post("/generate", response_model=List[PoisonDocOut], status_code=201)
async def generate(payload: PoisonDocGenerateRequest, db: Session = Depends(get_db)):
    try:
        return poison_doc_service.generate_and_save(
            db, payload.attack_index, payload.count)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("", response_model=List[PoisonDocOut])
async def list_docs(
    attack_index: Optional[int] = None,
    domain: Optional[str] = None,
    limit: int = Query(100, le=500),
    offset: int = 0,
    db: Session = Depends(get_db),
):
    return poison_doc_service.list_documents(
        db, attack_index=attack_index, domain=domain, limit=limit, offset=offset)


@router.get("/{doc_id}", response_model=PoisonDocOut)
async def get_doc(doc_id: int, db: Session = Depends(get_db)):
    doc = poison_doc_service.get_document(db, doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    return doc


@router.delete("/{doc_id}")
async def delete_doc(doc_id: int, db: Session = Depends(get_db)):
    if not poison_doc_service.delete_document(db, doc_id):
        raise HTTPException(status_code=404, detail="文档不存在")
    return {"status": "deleted", "id": doc_id}


@router.delete("/by-attack/{attack_index}")
async def delete_by_attack(attack_index: int, db: Session = Depends(get_db)):
    count = poison_doc_service.delete_by_attack_index(db, attack_index)
    return {"status": "deleted", "attack_index": attack_index, "count": count}
