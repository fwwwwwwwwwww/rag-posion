"""汇总所有 API 路由。"""
from fastapi import APIRouter

from app.api import experiments, poison_docs, legacy

api_router = APIRouter(prefix="/api")
api_router.include_router(experiments.router)
api_router.include_router(poison_docs.router)
api_router.include_router(legacy.router)
