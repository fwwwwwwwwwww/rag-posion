"""Celery 应用与任务定义。

当 CELERY_ENABLED=true 且 Redis 可用时，文档向量化等耗时任务走 Celery；
否则降级为线程池后台执行，保证无 Redis 环境也能运行。
"""
import logging
import threading

from app.config import CELERY_ENABLED, REDIS_URL

logger = logging.getLogger("rag.tasks")

# Celery 实例（惰性创建）
_celery = None
_celery_lock = threading.Lock()
# 模块级 Celery 任务对象（Celery 启用后由 _get_celery 动态注册）
_vectorize_task = None


def _get_celery():
    """尝试创建 Celery 实例，失败则返回 None。"""
    global _celery
    if not CELERY_ENABLED:
        return None
    if _celery is not None:
        return _celery
    with _celery_lock:
        if _celery is not None:
            return _celery
        try:
            from celery import Celery
            app = Celery("rag_tasks", broker=REDIS_URL, backend=REDIS_URL)
            app.conf.update(
                task_serializer="json",
                result_serializer="json",
                accept_content=["json"],
                timezone="Asia/Shanghai",
                enable_utc=False,
                task_track_started=True,
            )
            global _vectorize_task
            _vectorize_task = app.task(_run_vectorize_sync, name="rag.vectorize_docs")
            _celery = app
            logger.info("Celery 已启用，broker=%s", REDIS_URL)
            return app
        except ImportError:
            logger.warning("celery 未安装，降级为线程池")
            return None
        except Exception as e:
            logger.warning("Celery 连接失败（%s），降级为线程池", e)
            return None


def submit_vectorize_task(docs: list, chunk_size: int = 500,
                          chunk_overlap: int = 50) -> str:
    """提交文档向量化任务。

    优先 Celery，降级线程池。返回任务 ID。
    """
    celery = _get_celery()
    if celery is not None:
        result = _vectorize_task.delay(docs, chunk_size, chunk_overlap)
        return result.id

    # 降级：线程池
    task_id = f"thread-{threading.get_ident()}"
    t = threading.Thread(
        target=_run_vectorize_sync,
        args=(docs, chunk_size, chunk_overlap),
        daemon=True,
    )
    t.start()
    return task_id


def _run_vectorize_sync(docs: list, chunk_size: int, chunk_overlap: int):
    """线程池降级执行：将文档切片后写入 Chroma 向量库。"""
    try:
        import RAGblack as exp
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        embedding, _ = exp.init_ollama()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            length_function=len)
        chunks = splitter.create_documents(docs)

        from langchain_chroma import Chroma
        from app.config import CHROMA_PERSIST_DIR
        db = Chroma(embedding_function=embedding,
                    persist_directory=CHROMA_PERSIST_DIR)
        ids = db.add_documents(chunks)
        logger.info("向量化完成：%d 个 chunk 已写入", len(ids))
    except Exception as e:
        logger.exception("向量化失败：%s", e)

