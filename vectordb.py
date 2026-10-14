"""向量库（Chroma）操作封装。

为什么单独抽这一层
------------------
项目里到处都在直接调用 Chroma 的**私有成员**：``db._collection.count()``、
``db._collection.delete(ids=...)``、``db._collection.get(...)``。
这些不是 langchain-chroma / chromadb 的公开契约，升级依赖时可能改名或改签名，
而调用点原先散落在 ``RAGblack``、``agent``、``attack``、``app``、``tests`` 各处，
一旦失效就是一次性全崩。

把这类访问收敛到本模块后，升级依赖只需要改这一处，
并且每个函数都带了 ``get()`` 兜底路径，行为比裸调私有方法更稳。

不封装的部分：``add_documents`` / ``similarity_search*`` / ``as_retriever``
都是 langchain-chroma 的正式方法，调用方直接用即可。
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Iterator, List


def count_chunks(db: Any) -> int:
    """返回向量库中的 chunk 总数。"""
    try:
        return int(db._collection.count())
    except Exception:
        # 兜底：某些版本的 Collection 没有 count()，用 get() 的长度近似。
        return len(list_ids(db))


def list_ids(db: Any) -> List[str]:
    """取出全部 id（用于清空、体检与往返验证）。"""
    try:
        return [str(i) for i in (db.get().get("ids") or [])]
    except Exception:
        return []


def delete_ids(db: Any, ids: Iterable[str]) -> int:
    """按 id 批量删除，返回提交的 id 个数。

    空列表直接返回，避免触发 Chroma 的空删除分支。
    """
    cleaned = [str(i) for i in (ids or []) if i]
    if not cleaned:
        return 0
    db._collection.delete(ids=cleaned)
    return len(cleaned)


def clear(db: Any) -> int:
    """清空整个集合，返回删除的 id 数量。"""
    return delete_ids(db, list_ids(db))


def iter_documents(db: Any, batch: int = 1000) -> Iterator[Dict[str, Any]]:
    """分批遍历全部文档，逐条产出 ``{"id": ..., "content": ...}``。

    体检脚本需要逐条读内容（例如排查 2700 条残留投毒那类事故），
    一次性读整库在 5000+ chunk 时也能撑住，但分批更稳且内存友好。
    """
    offset = 0
    while True:
        try:
            got = db._collection.get(limit=batch, offset=offset,
                                     include=["documents"])
        except TypeError:
            # include 参数在个别版本上不支持
            got = db._collection.get(limit=batch, offset=offset)

        ids = list(got.get("ids") or [])
        docs = list(got.get("documents") or [])
        if not ids:
            return
        for i, doc_id in enumerate(ids):
            yield {"id": doc_id,
                   "content": docs[i] if i < len(docs) else ""}
        offset += len(ids)
