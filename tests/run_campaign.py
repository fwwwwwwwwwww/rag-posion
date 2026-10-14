"""P1 验证：自适应攻击战役是否能跑起来并产生攻击轨迹。

用法：
    python tests/run_campaign.py 0            # 场景 0，默认 3 轮
    python tests/run_campaign.py 0 --rounds 5
"""
import sys
import time

sys.path.insert(0, r"E:\RAG2")

from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings

import RAGblack as exp
import vectordb
from app.config import CHROMA_PERSIST_DIR


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    idx = int(argv[0]) if argv else 0
    rounds = 3
    if "--rounds" in sys.argv:
        rounds = int(sys.argv[sys.argv.index("--rounds") + 1])

    embedding = OllamaEmbeddings(model=exp.EMBEDDING_MODEL,
                                 base_url=exp.OLLAMA_BASE_URL)
    db = Chroma(embedding_function=embedding, persist_directory=CHROMA_PERSIST_DIR)
    exp._vectordb_cache = db
    print(f"向量库: {vectordb.count_chunks(db)} chunks | 模型: {exp.DEFAULT_LLM_MODEL}")
    kb = exp.load_medical_knowledge_base()
    task = exp.ATTACK_PAIRS[idx]

    from attack.campaign import run_campaign

    # 攻击者用原生工具调用模型（用于生成与策略选择）
    from agent import ToolChatModel
    attacker_llm = ToolChatModel(model=exp.DEFAULT_LLM_MODEL,
                                 base_url=exp.OLLAMA_BASE_URL)

    t0 = time.time()
    summary = run_campaign(
        db=db,
        task=task,
        build_rag_fn=lambda variant, poison_docs: exp.build_rag(
            kb, poison_docs, embedding, None, variant=variant,
            model_name=exp.DEFAULT_LLM_MODEL),
        llm=attacker_llm,
        max_rounds=rounds,
        runs_per_round=1,
        doc_count=15,
        verbose=True,
        persist=True,
    )
    print(f"\n总耗时 {time.time()-t0:.0f}s")

    # 打印轨迹表
    print("\n攻击轨迹:")
    print(f"  {'轮':<3}{'策略':<14}{'召回':<8}{'检出':<8}{'ASR':<5}{'失败层':<20}理由")
    for t in summary["trajectory"]:
        print(f"  {t['round']:<3}{t['strategy']:<14}"
              f"{str(t['recall_at_k']):<8}{str(t['detection_rate']):<8}"
              f"{str(t['asr']):<5}{str(t['failure_layer']):<20}"
              f"{str(t['reason'])[:40]}")

    # 验证落库
    try:
        from app.db.engine import SessionLocal
        from app.db.models import AttackIteration
        from sqlalchemy import select
        s = SessionLocal()
        rows = list(s.scalars(select(AttackIteration).order_by(
            AttackIteration.id.desc()).limit(rounds)).all())
        s.close()
        print(f"\nattack_iterations 落库 {len(rows)} 条:")
        for r in rows:
            print(f"  第{r.round_index}轮 [{r.strategy}] ASR={r.asr} "
                  f"召回={r.recall_at_k} 检出率={r.detection_rate} "
                  f"死在={r.failure_layer}")
    except Exception as e:
        print(f"落库检查失败: {e}")


if __name__ == "__main__":
    main()
