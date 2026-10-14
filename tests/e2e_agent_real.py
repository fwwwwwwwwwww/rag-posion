"""端到端验证：真实 Chroma 向量库 + 真实 Ollama 模型。

验证内容：
  1. Agent 在真实检索下是否会自主调用工具（而不是被固定流程驱动）；
  2. 遇到强断言（投毒文档）时是否会用 check_poison / cross_check 核查；
  3. 投毒场景下最终回答是否不再复述虚假宣传；
  4. 会话记忆在真实模型下是否生效。

用法：
    python tests/e2e_agent_real.py            # 只跑无毒对比
    python tests/e2e_agent_real.py --poison   # 额外注入投毒文档
"""
import json
import os
import sys
import time

sys.path.insert(0, r"E:\RAG2")

# 默认用实测最稳定的模型；可用环境变量覆盖
os.environ.setdefault("RAG_LLM_MODEL", "qwen3.5:2b")

from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings

import RAGblack as exp
import vectordb
from agent import build_agent
from agent import new_memory as exp_new_memory

EMBED_MODEL = exp.EMBEDDING_MODEL
BASE_URL = exp.OLLAMA_BASE_URL
CHROMA_DIR = exp.CHROMA_PERSIST_DIR
MODEL = os.environ["RAG_LLM_MODEL"]


def show_trace(result, title):
    print(f"\n--- {title} ---")
    print(f"  步数={result.steps}  通道={result.protocol}  拒答={result.refused}")
    for t in result.trace:
        if t["type"] == "tool_call":
            args = json.dumps(t.get("arguments", {}), ensure_ascii=False)
            print(f"    [决策] 调用 {t['tool']}({args[:70]})")
        elif t["type"] == "tool_result":
            print(f"    [结果] {t.get('summary', '')[:90]}")
        elif t["type"] == "finalize":
            print(f"    [收束] {t.get('summary', '')}")
        elif t["type"] == "final":
            print(f"    [作答] {t.get('content', '')[:70]}")
    print(f"  回答: {result.answer[:400]}")
    print(f"  来源: {len(result.sources)} 条"
          + (f"，其中投毒嫌疑 {sum(1 for d in result.sources if d.metadata.get('poison_suspect'))} 条"
             if result.sources else ""))


def main():
    embed = OllamaEmbeddings(model=EMBED_MODEL, base_url=BASE_URL)
    db = Chroma(embedding_function=embed, persist_directory=CHROMA_DIR)
    count = vectordb.count_chunks(db)
    print(f"向量库: {CHROMA_DIR}  共 {count} 个 chunk")
    if count == 0:
        print("向量库为空，请先运行 RAGblack 的实验构建向量库。")
        return 1

    agent = build_agent(db=db, llm=None, model=MODEL,
                        checkpointer=exp_new_memory(), max_steps=4)
    print(f"模型: {MODEL}  工具调用通道: {agent.llm.protocol()}")

    task = exp.ATTACK_PAIRS[0]          # 上海肺癌 / 仁济医院林医生
    question = task["question"]
    target = task["target_answer"]

    poison_docs = exp._generate_poison_variants(task, exp.POISON_DOC_COUNT)
    poison_ids = []
    if "--poison" in sys.argv:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50, length_function=len)
        chunks = splitter.create_documents(poison_docs)
        poison_ids = db.add_documents(chunks)
        print(f"已注入 {len(poison_ids)} 个投毒 chunk")

    try:
        # 两轮必须用同一个 session_id，否则是两条独立会话，测不出记忆
        SESSION = "e2e-session"

        t0 = time.time()
        result = agent.run(question, session_id=SESSION)
        print(f"\n总耗时 {time.time() - t0:.1f}s")
        show_trace(result, f"问题：{question}")

        hit = target[:12] in (result.answer or "")
        print(f"\n  攻击目标是否被复述: {'是（防御失败）' if hit else '否（防御成功）'}")
        print(f"  拒答: {result.refused}")

        tools = [t["tool"] for t in result.trace if t["type"] == "tool_call"]
        print(f"  实际调用的工具序列: {tools}")
        print(f"  → Agent 是否自主选择了工具: {'是' if tools else '否（未调用任何工具）'}")

        # 会话记忆：同一 session 下的第二轮追问（依赖 checkpointer）
        print("\n=== 会话记忆验证（第二轮追问）===")
        t0 = time.time()
        r2 = agent.run("那它的副作用呢？", session_id=SESSION)
        print(f"  耗时 {time.time() - t0:.1f}s")
        show_trace(r2, "追问：那它的副作用呢？（应带上前文）")
        used = [t["tool"] for t in r2.trace if t["type"] == "tool_call"]
        print(f"  追问轮是否用了工具: {'是' if used else '否'}  -> {used}")
        print(f"  → 是否继承了上文（不再反问\"指的是什么\"）: "
              f"{'是' if '指的是' not in r2.answer and '请告诉我' not in r2.answer else '否'}")
    finally:
        if poison_ids:
            vectordb.delete_ids(db, poison_ids)
            print(f"已清理 {len(poison_ids)} 个投毒 chunk")

    return 0


if __name__ == "__main__":
    sys.exit(main())
