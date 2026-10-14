"""实验管理服务。

提供实验的创建、查询、对比，以及后台异步执行。
实验执行依赖 RAGblack 中的实验函数（需要 Ollama 服务与医学知识库），
因此采用后台线程执行，前端通过状态轮询获知进度。
"""
import logging
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.engine import SessionLocal
from app.db.models import Experiment, ExperimentResult

logger = logging.getLogger("rag.experiment")

# 防止同一实验被重复提交
_running_ids: set = set()
_lock = threading.Lock()


def _get_exp():
    """懒加载 RAGblack 模块。"""
    import RAGblack as exp
    return exp


def _json_safe(value: Any) -> Any:
    """将 numpy 等非 JSON 标量转换为原生类型。"""
    if value is None:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return round(float(value), 6)
    if hasattr(value, "item"):  # numpy 标量
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _df_to_metrics(row: Dict[str, Any]) -> Dict[str, Any]:
    """将 DataFrame 一行转换为 JSON-safe 指标字典。"""
    return {str(k): _json_safe(v) for k, v in row.items()}


# ---------------- CRUD ----------------
def create_experiment(
    db: Session,
    name: str,
    exp_type: str,
    model_name: str = "deepseek-r1:latest",
    params: Optional[Dict[str, Any]] = None,
) -> Experiment:
    exp = Experiment(
        name=name,
        exp_type=exp_type,
        model_name=model_name,
        params=params,
        status="pending",
    )
    db.add(exp)
    db.commit()
    db.refresh(exp)
    return exp


def get_experiment(db: Session, experiment_id: int) -> Optional[Experiment]:
    return db.get(Experiment, experiment_id)


def list_experiments(
    db: Session, limit: int = 50, offset: int = 0
) -> List[Experiment]:
    stmt = select(Experiment).order_by(Experiment.id.desc()).limit(limit).offset(offset)
    return list(db.scalars(stmt).all())


def list_results(db: Session, experiment_id: int) -> List[ExperimentResult]:
    stmt = (
        select(ExperimentResult)
        .where(ExperimentResult.experiment_id == experiment_id)
        .order_by(ExperimentResult.id.asc())
    )
    return list(db.scalars(stmt).all())


def compare_experiments(
    db: Session, experiment_ids: List[int]
) -> List[Dict[str, Any]]:
    """对比多个实验：返回每个实验的基本信息与平均指标。"""
    result = []
    for eid in experiment_ids:
        exp = db.get(Experiment, eid)
        if exp is None:
            continue
        rows = list_results(db, eid)
        # 汇总各指标的均值
        agg: Dict[str, List[float]] = {}
        for r in rows:
            if not r.metrics:
                continue
            for k, v in r.metrics.items():
                if isinstance(v, (int, float)):
                    agg.setdefault(k, []).append(float(v))
        averages = {
            k: round(sum(v) / len(v), 4) for k, v in agg.items() if v
        }
        result.append({
            "id": exp.id,
            "name": exp.name,
            "exp_type": exp.exp_type,
            "model_name": exp.model_name,
            "status": exp.status,
            "result_count": len(rows),
            "avg_metrics": averages,
            "created_at": exp.created_at.isoformat() if exp.created_at else None,
        })
    return result


# ---------------- 后台执行 ----------------
def start_experiment_async(experiment_id: int) -> bool:
    """将实验提交到后台线程执行。

    返回 False 表示实验不存在或已在运行。
    """
    db = SessionLocal()
    try:
        exp = db.get(Experiment, experiment_id)
        if exp is None or exp.status == "running":
            return False
    finally:
        db.close()

    with _lock:
        if experiment_id in _running_ids:
            return False
        _running_ids.add(experiment_id)

    t = threading.Thread(target=_run_experiment, args=(experiment_id,), daemon=True)
    t.start()
    return True


def _run_experiment(experiment_id: int) -> None:
    """后台执行实验，并把 DataFrame 结果写入 experiment_results。"""
    db = SessionLocal()
    try:
        exp = db.get(Experiment, experiment_id)
        if exp is None:
            return
        exp.status = "running"
        exp.started_at = datetime.now()
        exp.error_message = None
        db.commit()

        try:
            rag = _get_exp()
            kb = rag.load_medical_knowledge_base()
            if not kb:
                raise RuntimeError("医学知识库加载失败")

            exp_model = exp.model_name or "deepseek-r1:latest"
            df = None

            if exp.exp_type == "baseline":
                df = rag.run_baseline(kb, model_name=exp_model)
            elif exp.exp_type == "attack":
                df = rag.run_blackbox_attack(kb, model_name=exp_model)
            elif exp.exp_type == "defense":
                df = rag.defense_test(kb, model_name=exp_model)
            elif exp.exp_type == "detection":
                df = rag.test_detection_model(kb)
            else:
                raise ValueError(f"未知实验类型: {exp.exp_type}")

            if df is not None:
                _save_df_results(db, exp, df)

            exp.status = "completed"
            exp.finished_at = datetime.now()
            db.commit()
            logger.info("实验 %s 完成", experiment_id)
        except Exception as e:
            exp.status = "failed"
            exp.error_message = str(e)
            exp.finished_at = datetime.now()
            db.commit()
            logger.exception("实验 %s 失败", experiment_id)
    finally:
        db.close()
        with _lock:
            _running_ids.discard(experiment_id)


def _save_df_results(db: Session, exp: Experiment, df) -> None:
    """将 DataFrame 结果逐行写入 ExperimentResult。"""
    for row in df.to_dict(orient="records"):
        metrics = _df_to_metrics(row)
        question = str(row.get("question", row.get("问题", "")))
        domain = str(row.get("domain", row.get("领域", "-")))
        db.add(ExperimentResult(
            experiment_id=exp.id,
            domain=domain,
            question=question,
            metrics=metrics,
        ))
