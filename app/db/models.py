"""ORM 模型定义。

四张核心表：
  - Experiment          实验（记录每次实验运行）
  - ExperimentResult    实验结果（每个攻击任务/领域的指标）
  - PoisonDocument      投毒文档（可复用、可检索）
  - AttackLog           攻击日志（记录问答流水，用于审计与复现）
"""
from datetime import datetime
from typing import Optional

from sqlalchemy import String, Integer, Float, Text, DateTime, ForeignKey, Boolean, JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _now() -> datetime:
    return datetime.now()


class Experiment(Base):
    """一次实验运行记录。"""

    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), index=True)
    # 实验类型: baseline / attack / defense / detection
    exp_type: Mapped[str] = mapped_column(String(32), index=True)
    model_name: Mapped[str] = mapped_column(String(64), default="deepseek-r1:latest")
    # 参数快照: 如 {"k": 3, "chunk_size": 500, "poison_count": 15}
    params: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    # 状态: pending / running / completed / failed
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    results: Mapped[list["ExperimentResult"]] = relationship(
        back_populates="experiment", cascade="all, delete-orphan")


class ExperimentResult(Base):
    """实验中单个任务（领域/问题）的指标结果。"""

    __tablename__ = "experiment_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[int] = mapped_column(
        ForeignKey("experiments.id", ondelete="CASCADE"), index=True)
    domain: Mapped[str] = mapped_column(String(64), default="-")
    question: Mapped[str] = mapped_column(Text)
    target_answer: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # 指标快照: 如 {"ASR": 0.7, "Avg_EM": 0.8, "Avg_F1": 0.6, "Avg_P@5": 0.9}
    metrics: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    experiment: Mapped["Experiment"] = relationship(back_populates="results")


class PoisonDocument(Base):
    """投毒文档，按攻击场景生成后可复用、检索、删除。"""

    __tablename__ = "poison_documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    attack_index: Mapped[int] = mapped_column(Integer, index=True)
    domain: Mapped[str] = mapped_column(String(64), default="-")
    question: Mapped[str] = mapped_column(Text)
    target_answer: Mapped[str] = mapped_column(Text)
    content: Mapped[str] = mapped_column(Text)
    keywords: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    # 是否为检测模型判定出的异常样本
    detected_poison: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


class AttackLog(Base):
    """问答流水日志，记录每次问答的检索来源与回答，用于审计与复现。"""

    __tablename__ = "attack_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("experiments.id", ondelete="SET NULL"), nullable=True, index=True)
    question: Mapped[str] = mapped_column(Text)
    is_poison: Mapped[bool] = mapped_column(Boolean, default=False)
    answer: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # 检索来源快照: [{rank, content, is_poison}, ...]
    sources: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
