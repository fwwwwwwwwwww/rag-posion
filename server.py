"""医疗 RAG 问答服务（FastAPI 后端入口）。

分层结构：
  app/config.py          配置与常量
  app/auth.py            鉴权中间件
  app/db/                数据库层（引擎、模型、初始化）
  app/schemas/           请求/响应模型（Pydantic）
  app/services/          业务逻辑层（实验管理、投毒文档管理）
  app/api/               API 路由层（含向后兼容的 legacy 接口）
  app/tasks/             后台任务（Celery / 线程池降级）
  frontend/              Vue3 前端工程（构建产物 dist/ 由本服务托管）

启动：
  python -m uvicorn server:app --host 127.0.0.1 --port 8000
前端开发：
  cd frontend && npm run dev   （Vite 热更新，/api 代理到 8000 端口）
"""
import logging

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.router import api_router
from app.auth import auth_middleware
from app.config import BASE_DIR
from app.db.init_db import init_db

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="医疗RAG问答服务", version="0.4")

# 全局鉴权中间件（公开路径放行，其余 /api 请求要求 Bearer Token）
app.middleware("http")(auth_middleware)


@app.on_event("startup")
def _startup():
    init_db()
    # 首次启动创建管理员账号。密码来自 ADMIN_PASSWORD 环境变量，
    # 未配置则随机生成并打印到日志——**绝不使用硬编码默认密码**。
    try:
        from app.db.engine import SessionLocal
        from app.services.auth_service import ensure_default_admin

        db = SessionLocal()
        try:
            if ensure_default_admin(db) is None:
                logging.getLogger("rag").info("已有用户，跳过默认管理员创建")
        finally:
            db.close()
    except Exception:
        logging.getLogger("rag").exception("初始化默认管理员失败，请手动创建用户")


app.include_router(api_router)

# Vue3 前端（构建产物优先，未构建时回退旧版静态页）
FRONTEND_DIST = BASE_DIR / "frontend" / "dist"
if FRONTEND_DIST.exists():
    assets = FRONTEND_DIST / "assets"
    if assets.exists():
        app.mount("/assets", StaticFiles(directory=str(assets)), name="assets")

    @app.get("/{full_path:path}")
    def spa(full_path: str):
        """SPA 回退：非 /api 路径一律返回 index.html。"""
        candidate = FRONTEND_DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(str(candidate))
        return FileResponse(str(FRONTEND_DIST / "index.html"))
else:
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

    @app.get("/")
    def index():
        return FileResponse(str(BASE_DIR / "static" / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
