"""实验管理接口（异步）。"""
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.engine import get_db
from app.schemas.experiment import (
    ExperimentCreate,
    ExperimentOut,
    ExperimentDetailOut,
    ExperimentResultOut,
    CompareRequest,
)
from app.services import experiment_service

router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.post("", response_model=ExperimentOut, status_code=201)
async def create_experiment(
    payload: ExperimentCreate,
    auto_start: bool = Query(False, description="是否立即后台执行"),
    db: Session = Depends(get_db),
):
    exp = experiment_service.create_experiment(
        db,
        name=payload.name,
        exp_type=payload.exp_type,
        model_name=payload.model_name,
        params=payload.params,
    )
    if auto_start:
        experiment_service.start_experiment_async(exp.id)
    return exp


@router.get("", response_model=List[ExperimentOut])
async def list_experiments(
    limit: int = Query(50, le=200),
    offset: int = 0,
    db: Session = Depends(get_db),
):
    return experiment_service.list_experiments(db, limit=limit, offset=offset)


@router.post("/compare")
async def compare_experiments(payload: CompareRequest, db: Session = Depends(get_db)):
    return experiment_service.compare_experiments(db, payload.experiment_ids)


@router.get("/{experiment_id}", response_model=ExperimentDetailOut)
async def get_experiment(experiment_id: int, db: Session = Depends(get_db)):
    exp = experiment_service.get_experiment(db, experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail="实验不存在")
    return exp


@router.get("/{experiment_id}/results", response_model=List[ExperimentResultOut])
async def get_experiment_results(experiment_id: int, db: Session = Depends(get_db)):
    exp = experiment_service.get_experiment(db, experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail="实验不存在")
    return experiment_service.list_results(db, experiment_id)


@router.post("/{experiment_id}/start")
async def start_experiment(experiment_id: int, db: Session = Depends(get_db)):
    exp = experiment_service.get_experiment(db, experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail="实验不存在")
    if exp.status == "running":
        raise HTTPException(status_code=409, detail="实验已在运行中")
    ok = experiment_service.start_experiment_async(experiment_id)
    if not ok:
        raise HTTPException(status_code=409, detail="实验已在运行中")
    return {"status": "started", "experiment_id": experiment_id}
