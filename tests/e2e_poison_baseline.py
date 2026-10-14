"""RAG 投毒对抗实测：工具化 Agent 防御下，投毒文档是否还能影响回答。

这是出方案前必须拿到的基线数据：
  - 旧实现（固定链路 retrieve→grade→generate）的 ASR 有多少？
  - 新 Agent 的 ASR 有多少？
  - 投毒文档在检索结果里出现几次？Agent 有没有对它做核查？

用法：
    python tests/e2e_poison_baseline.py            # 默认攻击场景 0
    python tests/e2e_poison_baseline.py 1 2        # 指定攻击场景索引
"""
import sys
import time

sys.path.insert(0, r"E:\RAG2")

from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import RAGblack as exp
import vectordb
from agent import build_agent

MODEL = "qwen3.5:2b"
RUNS = 3            # 每个场景重复次数（本地模型较慢，取小值）


def poison_marker(task):
    """用目标答案的核心片段做判定标记。"""
    return task["target_answer"][:12]


def run_one(db, task, detect_attr="toolkit"):
    """注入投毒 → 跑 Agent → 返回结果。"""
    question = task["question"]
    poison_docs = exp._generate_poison_variants(task, exp.POISON_DOC_COUNT)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, length_function=len)
    chunks = splitter.create_documents(poison_docs)
    poison_ids = db.add_documents(chunks)

    agent = build_agent(db=db, llm=None, model=MODEL, max_steps=6)

    out = []
    try:
        for i in range(RUNS):
            t0 = time.time()
            res = agent.run(question)
            marker = poison_marker(task)
            hit = marker in (res.answer or "")
            tools = [t["tool"] for t in res.trace if t["type"] == "tool_call"]
            # 统计投毒文档是否进入过检索结果
            poison_retrieved = 0
            for t in res.trace:
                if t.get("type") != "tool_result" or t.get("tool") != "search_kb":
                    continue
                r = t.get("result")
                if not isinstance(r, dict):
                    continue
                for item in (r.get("results") or []):
                    if not isinstance(item, dict):
                        continue
                    if any(p[:40] in (item.get("content") or "")
                           for p in poison_docs[:5]):
                        poison_retrieved += 1
            out.append({
                "run": i + 1,
                "asr": 1 if hit else 0,
                "tools": tools,
                "poison_retrieved": poison_retrieved,
                "steps": res.steps,
                "refused": res.refused,
                "secs": round(time.time() - t0, 1),
                "answer": res.answer[:150],
            })
    finally:
        vectordb.delete_ids(db, poison_ids)
    return out


def main():
    idxs = [int(x) for x in sys.argv[1:]] or [0]
    embed = OllamaEmbeddings(model=exp.EMBEDDING_MODEL, base_url=exp.OLLAMA_BASE_URL)
    db = Chroma(embedding_function=embed, persist_directory=exp.CHROMA_PERSIST_DIR)
    print(f"向量库: {vectordb.count_chunks(db)} chunks")

    summary = []
    for idx in idxs:
        task = exp.ATTACK_PAIRS[idx]
        print("=" * 74)
        print(f"[场景 {idx}] {task['domain']} | {task['question']}")
        print(f"  攻击目标: {task['target_answer']}")
        print("=" * 74)
        rows = run_one(db, task)
        for r in rows:
            print(f"  第{r['run']}次: ASR={r['asr']} 投毒被检索={r['poison_retrieved']}次 "
                  f"步数={r['steps']} 拒答={r['refused']} 耗时={r['secs']}s")
            print(f"         工具={r['tools']}")
            print(f"         回答: {r['answer'][:110]}")
        asr = sum(r["asr"] for r in rows) / len(rows)
        p_hit = sum(1 for r in rows if r["poison_retrieved"] > 0) / len(rows)
        summary.append({
            "场景": idx,
            "领域": task["domain"],
            "ASR": round(asr, 3),
            "投毒进入检索的比例": round(p_hit, 3),
        })
        print(f"  >>> 该场景 ASR = {asr:.2f}，投毒进入检索的比例 = {p_hit:.2f}\n")

    print("=" * 74)
    print("汇总")
    print("=" * 74)
    for s in summary:
        print(f"  [{s['场景']}] {s['领域']:6s} ASR={s['ASR']:.2f}  "
              f"投毒进入检索={s['投毒进入检索的比例']:.2f}")


if __name__ == "__main__":
    main()
