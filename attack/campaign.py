"""攻击战役编排：把自适应攻击者跑成多轮迭代，并落库。

一次 campaign = 若干轮：
    生成投毒文档 → 注入 → 让防御回答 → 采集四层指标 → 分析失败层 → 生成下一轮

与原实验的区别：原实验是"生成一次、跑 N 次取平均"，
本模块是"**每轮根据反馈改变攻击策略**"，所以产出的是攻击轨迹而不是单个数字。
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

import vectordb
from .attacker import AdaptiveAttacker

logger = logging.getLogger("rag.attack.campaign")


def _inject(db, docs: List[str], chunk_size: int = 500, chunk_overlap: int = 50):
    """把投毒文档切片后写入向量库，返回 (ids, chunk_count)。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)
    chunks = splitter.create_documents(docs)
    ids = db.add_documents(chunks)
    return ids, len(chunks)


def run_campaign(
    db,
    task: Dict,
    build_rag_fn,
    llm,
    max_rounds: int = 4,
    runs_per_round: int = 1,
    stop_on_success: bool = True,
    doc_count: int = 15,
    verbose: bool = True,
    persist: bool = True,
    experiment_id: Optional[int] = None,
    defense_variant: str = "agent",
) -> Dict[str, Any]:
    """跑一次自适应攻击战役。

    参数:
        db:            向量库（投毒文档每轮注入、轮末清理）
        task:          攻击场景（含 question / target_answer / keywords / domain）
        build_rag_fn:  构造防御的函数，签名 ``variant=str -> rag``；
                       每轮都要重建，避免上一轮的 Agent 状态（记忆/证据）串场
        llm:           用于生成投毒文档与做采纳性判定
        max_rounds:    最大迭代轮数
        runs_per_round: 每轮重复数次取均值（本地模型慢，建议 1）
        persist:       是否把每轮结果写入 attack_iterations 表
    """
    from eval.metrics import AnswerJudge, collect_metrics

    question = task["question"]
    claim = task["target_answer"]
    judge = AnswerJudge(llm)
    attacker = AdaptiveAttacker(task, llm=llm, doc_count=doc_count)

    if verbose:
        print("=" * 78)
        print(f"自适应攻击战役 | {task.get('domain', '-')} | {question}")
        print(f"  攻击目标: {claim}")
        print(f"  最大轮数: {max_rounds} | 每轮重复: {runs_per_round}")
        print("=" * 78)

    failure_layer, failure_reason = "retrieval", "首轮，尚无反馈"
    campaign_start = time.time()
    rounds_out: List[Dict[str, Any]] = []

    # 预热：检索会把 embedding 模型挤进显存，紧接着的判定/生成容易拿到空响应
    if hasattr(llm, "warmup"):
        try:
            ok = llm.warmup()
            if verbose:
                print(f"  [预热] 模型 {'就绪' if ok else '可能仍在加载'}")
        except Exception:
            pass

    for rnd in range(1, max_rounds + 1):
        strategy, reason = attacker.choose_strategy(failure_layer, failure_reason)
        docs = attacker.generate_docs(strategy, rnd)

        if verbose:
            print(f"\n--- 第 {rnd} 轮 ---")
            print(f"  策略: {strategy}（{_strategy_note(strategy)}）")
            print(f"  理由: {reason}")
            print(f"  投放: {len(docs)} 篇")

        ids, chunk_count = _inject(db, docs)
        metrics_list: List[Dict[str, Any]] = []
        try:
            rag = build_rag_fn(variant=defense_variant, poison_docs=docs)
            for i in range(runs_per_round):
                t0 = time.time()
                out = rag.invoke({"query": question})
                answer = out.get("result", "") or ""
                jr = judge.judge(claim=claim, answer=answer, question=question)
                m = collect_metrics(
                    trace=out.get("trace") or [],
                    poison_docs=docs,
                    answer=answer, question=question, claim=claim,
                    refused=bool(out.get("refused")),
                    seconds=time.time() - t0,
                    steps=out.get("steps", 0),
                    injected_chunks=chunk_count,
                    judge_result=jr,
                    source_docs=out.get("source_documents") or [],
                )
                d = m.to_dict()
                d["answer"] = answer[:300]
                metrics_list.append(d)
        finally:
            # 必须清理，否则投毒累积到下一轮，指标就失去意义
            try:
                vectordb.delete_ids(db, ids)
            except Exception as e:
                logger.warning("清理投毒失败：%s", e)

        def avg(key):
            vals = [x[key] for x in metrics_list]
            return round(sum(vals) / len(vals), 4) if vals else 0.0

        agg = {
            "round": rnd,
            "strategy": strategy,
            "reason": reason,
            "docs_count": len(docs),
            "chunks_count": chunk_count,
            "asr": avg("asr"),
            "recall_at_k": avg("recall_at_k"),
            "detection_rate": avg("detection_rate"),
            "evasion": avg("evasion"),
            "survive": avg("survive"),
            "verdict": metrics_list[0].get("verdict") if metrics_list else "",
            "judge_ok": bool(metrics_list[0].get("judge_ok")) if metrics_list else False,
            "judge_reason": metrics_list[0].get("judge_reason") if metrics_list else "",
            "failure_layer": metrics_list[0].get("failure_layer") if metrics_list else "",
            "steps": avg("steps"),
            "seconds": avg("seconds"),
            "sample_answer": (metrics_list[0].get("answer") if metrics_list else ""),
            "sample_doc": docs[0][:200] if docs else "",
            "injected_docs": len(docs),
            "injected_chunks": chunk_count,
            "retrieved_docs": metrics_list[0].get("retrieved_docs") if metrics_list else 0,
            "retrieved_chunks": metrics_list[0].get("retrieved_chunks") if metrics_list else 0,
            "detected": metrics_list[0].get("detected") if metrics_list else 0,
        }
        rounds_out.append(agg)

        if verbose:
            print(f"  → verdict={agg['verdict']:14s} ASR={agg['asr']} "
                  f"召回={agg['recall_at_k']:.2f} 检出率={agg['detection_rate']:.2f} "
                  f"死在={agg['failure_layer']}"
                  + ("" if agg["judge_ok"] else "  ⚠判定失败，该轮数据不可信"))
            print(f"     回答: {agg['sample_answer'][:100]}")

        # 只把聚合指标交给攻击者做决策（不喂原始文档，避免上下文膨胀）
        attacker.record(rnd, strategy, docs, agg, reason)
        if persist:
            _persist(agg, question, claim, task, experiment_id)

        if agg["asr"] and stop_on_success:
            if verbose:
                print(f"\n  ★ 第 {rnd} 轮攻击成功（策略 {strategy}），提前结束")
            break

        failure_layer = agg["failure_layer"] or "retrieval"
        failure_reason = f"第{rnd}轮策略 {strategy} 失败于 {failure_layer}"

    summary = attacker.summary()
    summary.update({
        "question": question,
        "claim": claim,
        "domain": task.get("domain", "-"),
        "total_seconds": round(time.time() - campaign_start, 1),
        "rounds_detail": rounds_out,
    })

    if verbose:
        print("\n" + "=" * 78)
        print(f"战役结束：{'成功' if summary['succeeded'] else '失败'} | "
              f"轮数={summary['rounds']} | 尝试策略={summary['strategies_tried']} | "
              f"耗时={summary['total_seconds']}s")
        for t in summary["trajectory"]:
            print(f"  第{t['round']}轮 [{t['strategy']:12s}] ASR={t['asr']} "
                  f"召回={t['recall_at_k']} 检出={t['detection_rate']} "
                  f"死在={t['failure_layer']}")
        print("=" * 78)

    return summary


def _strategy_note(name: str) -> str:
    from .strategies import STRATEGY_NOTES
    return STRATEGY_NOTES.get(name, "")


def _persist(agg: Dict[str, Any], question: str, claim: str, task: Dict,
             experiment_id: Optional[int]) -> None:
    """把一轮攻击写入 attack_iterations 表。失败不影响主流程。"""
    try:
        from app.db.engine import SessionLocal
        from app.db.models import AttackIteration

        db = SessionLocal()
        try:
            db.add(AttackIteration(
                experiment_id=experiment_id,
                round_index=agg["round"],
                domain=task.get("domain", "-"),
                question=question,
                claim=claim,
                strategy=agg["strategy"],
                strategy_reason=agg["reason"],
                injected_docs=agg["injected_docs"],
                injected_chunks=agg["injected_chunks"],
                retrieved_docs=agg["retrieved_docs"],
                retrieved_chunks=agg["retrieved_chunks"],
                detected=agg["detected"],
                recall_at_k=agg["recall_at_k"],
                detection_rate=agg["detection_rate"],
                survive=agg["survive"],
                verdict=agg["verdict"],
                asr=agg["asr"],
                failure_layer=agg["failure_layer"],
                sample_doc=agg["sample_doc"],
                sample_answer=agg["sample_answer"],
                seconds=agg["seconds"],
            ))
            db.commit()
        finally:
            db.close()
    except Exception as e:
        logger.warning("攻击迭代落库失败（不影响流程）：%s", e)
