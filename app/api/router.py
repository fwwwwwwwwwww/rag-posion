"""汇总所有 API 路由。"""
from fastapi import APIRouter

from app.api import agent, attacks, auth, experiments, poison_docs, legacy

api_router = APIRouter(prefix="/api")
api_router.include_router(auth.router)   # 登录相关放最前（多数为公开路径）
api_router.include_router(experiments.router)
api_router.include_router(poison_docs.router)
api_router.include_router(agent.router)
api_router.include_router(attacks.router)
api_router.include_router(legacy.router)
