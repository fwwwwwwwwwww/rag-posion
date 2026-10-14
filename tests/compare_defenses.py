"""P2 四档防御对比：同一攻击场景下，各防御变体的分层指标。

四档：
  naive      裸 RAG，无任何核查        （必需对照组）
  fixed      原固定链路 retrieve→grade→generate
  agent_stat Agent + 旧统计检测器      （消融对照，0% 召回）
  agent      Agent + 新混合检测器      （当前实现）

用法：
    python tests/compare_defenses.py 0 1        # 指定攻击场景
    python tests/compare_defenses.py 0 --runs 3
"""
import sys
import time

sys.path.insert(0, r"E:\RAG2")

import pandas as pd
from langchain_chroma import Chroma

import RAGblack as exp
import vectordb
from app.config import CHROMA_PERSIST_DIR


def parse_args(argv):
    """解析位置参数（场景索引）与命名参数。

    坑：早先用 ``[a for a in argv if not a.startswith("--")]`` 取位置参数，
    结果 ``--runs 1`` 里的 **1 被当成场景索引**，导致场景被重复跑。
    正确做法是把命名参数及其取值一起跳过。
    """
    idxs, opts = [], {}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--"):
            key = a[2:]
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                opts[key] = argv[i + 1]
                i += 2
                continue
            opts[key] = True
        else:
            try:
                idxs.append(int(a))
            except ValueError:
                print(f"忽略无法解析的参数: {a!r}")
        i += 1
    return idxs, opts


def main():
    idxs, opts = parse_args(sys.argv[1:])
    idxs = idxs or [0]
    runs = int(opts.get("runs", 1))

    embedding, llm = exp.init_ollama(exp.DEFAULT_LLM_MODEL)
    # embedding 设备：把 embedding 放 CPU 可以让对话模型常驻显存。
    # 本机 8GB 显存装不下 embedding(6.34GB)+对话模型(2.36GB)，同机会互相驱逐，
    # 重载窗口期 Ollama 返回空 content，导致判定误判。见 EMBED_ON_CPU 说明。
    embedding = exp.make_embeddings()
    print(f"embedding 方式: {'CPU (num_gpu=0)' if exp.EMBED_ON_CPU else 'GPU (num_gpu=99)'}")
    # 复用磁盘上已构建好的干净向量库
    db = Chroma(embedding_function=embedding, persist_directory=CHROMA_PERSIST_DIR)
    total = vectordb.count_chunks(db)
    print(f"向量库: {total} chunks | 模型: {exp.DEFAULT_LLM_MODEL} "
          f"| 每档重复 {runs} 次")
    # 体检：干净库应为 5571；明显偏多说明混入了历史投毒残留
    expected = 5571
    if total > expected * 1.05:
        print(f"  ⚠ 库内 chunk 数异常（期望约 {expected}），疑似混有投毒残留，"
              f"建议先跑 python tests/rebuild_vectordb.py")
    print()

    kb = exp.load_medical_knowledge_base()
    # build_rag 需要 clean_docs 来构建/校验向量库；已有库时会走缓存
    exp._vectordb_cache = db

    # 记录起点：跑完要核对是否回到这个数。
    # 被强杀（非正常退出）时 finally 不执行，投毒会留在库里并污染后续所有实验——
    # 之前正是这样累积了约 2700 个残留投毒，让全部测量失效。
    start_count = total

    rows = []
    t_start = time.time()
    for idx in idxs:
        task = exp.ATTACK_PAIRS[idx]
        print("=" * 78)
        print(f"[场景 {idx}] {task['domain']} | {task['question']}")
        print(f"  攻击目标: {task['target_answer']}")
        print("=" * 78)
        for variant in exp.DEFENSE_VARIANTS:
            t0 = time.time()
            r = exp.evaluate_variant(
                knowledge_base=kb, task=task, embedding=embedding, llm=llm,
                variant=variant, runs=runs, verbose=True)
            r.pop("runs_detail", None)
            rows.append(r)
            print(f"  >>> [{variant:12s}] ASR={r['ASR']:.2f} 召回={r['Recall@k']:.2f} "
                  f"检出率={r['DetectionRate']:.2f} 生存={r['Survive']:.2f} "
                  f"拒答={r['RefusalRate']:.2f} 耗时={time.time()-t0:.0f}s")
            if r.get("JudgeFailures"):
                print(f"      ⚠ 有 {r['JudgeFailures']} 轮判定失败，该档数字不可信")
            print(f"      死在: {r['failure_layers']}\n")

    # 终点核对：应与起点一致
    end_count = vectordb.count_chunks(db)
    print("=" * 78)
    if end_count != start_count:
        print(f"⚠ 投毒清理不彻底：起点 {start_count} -> 终点 {end_count}"
              f"（残留 {end_count - start_count} 个）")
        print("  请运行 python tests/purge_poison.py 清理后再复测")
    else:
        print(f"✓ 清理彻底：库内始终 {end_count} 个 chunk")
    total_judge_failures = sum(r.get("JudgeFailures", 0) for r in rows)
    if total_judge_failures:
        print(f"⚠ 共 {total_judge_failures} 轮判定失败，相关档位数字需谨慎引用")
    else:
        print("✓ 全部判定成功，数据可信")

    df = pd.DataFrame(rows)
    cols = ["variant", "domain", "ASR", "Recall@k", "DetectionRate", "Evasion",
            "Survive", "RefusalRate", "AvgSteps"]
    print("=" * 78)
    print(f"汇总（总耗时 {time.time()-t_start:.0f}s）")
    print("=" * 78)
    print(df[cols].to_string(index=False))

    out = "rag_defense_comparison.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n已保存: {out}")

    # 按变体聚合
    print("\n按变体聚合（跨场景均值）:")
    agg = df.groupby("variant")[["ASR", "Recall@k", "DetectionRate", "Survive"]].mean()
    print(agg.round(3).to_string())


if __name__ == "__main__":
    main()
