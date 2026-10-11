"""SQLAlchemy 引擎与会话管理。

支持 MySQL / SQLite，MySQL 驱动不可用时自动降级 SQLite，
保证服务在无外部数据库的本地开发环境中也能启动。
"""
import logging

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from app.config import DATABASE_URL, DATA_DIR

logger = logging.getLogger("rag.db")


def _build_engine():
    url = DATABASE_URL
    try:
        engine = create_engine(
            url,
            pool_pre_ping=True,
            # SQLite 多线程下需要关闭同线程检查
            connect_args={"check_same_thread": False} if url.startswith("sqlite") else {},
        )
        # 立即探测连接，失败则降级 SQLite
        with engine.connect():
            pass
        logger.info("数据库已连接: %s", url.split("@")[-1])
        return engine
    except Exception as e:
        if url.startswith("sqlite"):
            raise
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        fallback = f"sqlite:///{(DATA_DIR / 'rag_experiments.db').as_posix()}"
        logger.warning(
            "MySQL 不可用（%s），降级为 SQLite: %s", e, fallback)
        engine = create_engine(
            fallback,
            pool_pre_ping=True,
            connect_args={"check_same_thread": False},
        )
        return engine


engine = _build_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db():
    """FastAPI 依赖：提供数据库会话。"""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
