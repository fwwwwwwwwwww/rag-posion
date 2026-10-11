"""数据库初始化：建表。

用法：
    python -m app.db.init_db
或代码中调用 init_db()。
"""
import logging

from app.db.engine import engine
from app.db.models import Base

logger = logging.getLogger("rag.db")


def init_db() -> None:
    """创建所有缺失的表（幂等）。"""
    Base.metadata.create_all(bind=engine)
    logger.info("数据库表结构已就绪")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_db()
    print("数据库初始化完成")
