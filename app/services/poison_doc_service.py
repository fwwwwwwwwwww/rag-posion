"""投毒文档管理服务。

复用 RAGblack 中的投毒文档生成器（纯 Python，不依赖 Ollama），
将生成结果持久化到数据库，支持复用、检索与删除。
"""
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import PoisonDocument


def _get_exp():
    """懒加载 RAGblack 模块（避免 import 顺序问题）。"""
    import RAGblack as exp
    return exp


def generate_and_save(
    db: Session, attack_index: int, count: int = 15
) -> List[PoisonDocument]:
    """为指定攻击场景生成投毒文档并入库。"""
    exp = _get_exp()
    if attack_index < 0 or attack_index >= len(exp.ATTACK_PAIRS):
        raise ValueError(f"攻击场景索引越界: {attack_index}")
    task = exp.ATTACK_PAIRS[attack_index]
    variants = exp._generate_poison_variants(task, count)

    docs = []
    for content in variants:
        doc = PoisonDocument(
            attack_index=attack_index,
            domain=task.get("domain", "-"),
            question=task["question"],
            target_answer=task["target_answer"],
            content=content,
            keywords=task.get("keywords", []),
        )
        db.add(doc)
        docs.append(doc)
    db.commit()
    for d in docs:
        db.refresh(d)
    return docs


def list_documents(
    db: Session,
    attack_index: Optional[int] = None,
    domain: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> List[PoisonDocument]:
    stmt = select(PoisonDocument).order_by(PoisonDocument.id.desc())
    if attack_index is not None:
        stmt = stmt.where(PoisonDocument.attack_index == attack_index)
    if domain:
        stmt = stmt.where(PoisonDocument.domain == domain)
    stmt = stmt.limit(limit).offset(offset)
    return list(db.scalars(stmt).all())


def get_document(db: Session, doc_id: int) -> Optional[PoisonDocument]:
    return db.get(PoisonDocument, doc_id)


def delete_document(db: Session, doc_id: int) -> bool:
    doc = db.get(PoisonDocument, doc_id)
    if doc is None:
        return False
    db.delete(doc)
    db.commit()
    return True


def delete_by_attack_index(db: Session, attack_index: int) -> int:
    """删除指定攻击场景下的全部投毒文档，返回删除条数。"""
    stmt = select(PoisonDocument).where(PoisonDocument.attack_index == attack_index)
    docs = list(db.scalars(stmt).all())
    for d in docs:
        db.delete(d)
    db.commit()
    return len(docs)
