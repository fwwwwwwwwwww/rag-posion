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


class User(Base):
    """平台用户。

    安全约定：
      - **只保存密码哈希**（bcrypt），永不保存明文或可逆密文；
      - ``password_hash`` 不参与任何对外序列化（见 ``to_safe_dict``）；
      - 连续失败写入 ``failed_attempts`` / ``locked_until`` 用于登录限流。
    """

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    # bcrypt 哈希串（含算法标识、cost、盐）；长度留足以便将来换算法
    password_hash: Mapped[str] = mapped_column(String(255))
    display_name: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    role: Mapped[str] = mapped_column(String(16), default="user")   # user / admin
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # ---- 登录限流 ----
    failed_attempts: Mapped[int] = mapped_column(Integer, default=0)
    locked_until: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    last_login_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    def to_safe_dict(self) -> dict:
        """对外输出。**故意不包含 password_hash**。"""
        return {
            "id": self.id,
            "username": self.username,
            "display_name": self.display_name,
            "role": self.role,
            "is_active": self.is_active,
            "last_login_at": self.last_login_at.isoformat() if self.last_login_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class AttackIteration(Base):
    """自适应攻击的一轮迭代记录。

    与原实验"生成一次、跑 N 次取平均"不同，这张表记录的是**攻击轨迹**：
    第几轮用了什么策略、为什么换这个策略、四层指标如何变化、攻击死在哪一层。
    这样"攻击如何一步步逼近成功"是可复现的，而不是只留一个最终数字。
    """

    __tablename__ = "attack_iterations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("experiments.id", ondelete="SET NULL"), nullable=True, index=True)
    round_index: Mapped[int] = mapped_column(Integer, index=True)
    domain: Mapped[str] = mapped_column(String(64), default="-")
    question: Mapped[str] = mapped_column(Text)
    claim: Mapped[str] = mapped_column(Text, default="")

    # 本轮使用的攻击策略与选择理由（由攻击 Agent 决定）
    strategy: Mapped[str] = mapped_column(String(32), index=True)
    strategy_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # 投放量
    injected_docs: Mapped[int] = mapped_column(Integer, default=0)
    injected_chunks: Mapped[int] = mapped_column(Integer, default=0)

    # 四层指标
    retrieved_docs: Mapped[int] = mapped_column(Integer, default=0)
    retrieved_chunks: Mapped[int] = mapped_column(Integer, default=0)
    detected: Mapped[int] = mapped_column(Integer, default=0)
    recall_at_k: Mapped[float] = mapped_column(Float, default=0.0)
    detection_rate: Mapped[float] = mapped_column(Float, default=0.0)
    survive: Mapped[float] = mapped_column(Float, default=0.0)
    asr: Mapped[float] = mapped_column(Float, default=0.0)

    # 采纳性判定结论 与 失败层归因
    verdict: Mapped[Optional[str]] = mapped_column(String(24), nullable=True)
    failure_layer: Mapped[Optional[str]] = mapped_column(String(24), nullable=True)

    sample_doc: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    sample_answer: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    seconds: Mapped[float] = mapped_column(Float, default=0.0)
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
