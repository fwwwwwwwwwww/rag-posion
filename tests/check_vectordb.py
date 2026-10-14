"""向量库体检：检查库里是否混有投毒残留。

为什么需要：诊断中发现库里混入了约 2700 个历史残留的投毒 chunk
（干净知识库只应产生 5571 个），导致之前所有检索指标被污染——
"投毒召回率"虚高、干净文档被挤掉，四档对比的结论因此不可信。

任何实验开跑前都应该先跑这个脚本。
"""
import sys

sys.path.insert(0, r"E:\RAG2")

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

import RAGblack as exp
from attack.detector import HybridPoisonDetector

EXPECTED_CLEAN_CHUNKS = 5571   # 3000 篇干净文档按 chunk_size=500/overlap=50 切分的结果


def main():
    d = Chroma(embedding_function=exp.make_embeddings(),
               persist_directory=exp.CHROMA_PERSIST_DIR)
    n = d._collection.count()
    print(f"库内 chunk 数: {n}")

    if n > EXPECTED_CLEAN_CHUNKS * 1.05:
        print(f"  ⚠ 明显多于干净切分数（{EXPECTED_CLEAN_CHUNKS}）"
              f"，疑似混有残留投毒")
        print(f"    建议先运行: python tests/rebuild_vectordb.py")
    elif n < EXPECTED_CLEAN_CHUNKS * 0.95:
        print(f"  ⚠ 少于干净切分数（{EXPECTED_CLEAN_CHUNKS}），可能有文档缺失")
    else:
        print(f"  ✓ 数量正常（期望 {EXPECTED_CLEAN_CHUNKS}）")

    # 抽样检测
    det = HybridPoisonDetector(llm=None, db=None, ambiguous_as_poison=False)
    total, flagged = 0, 0
    hits = []
    batch, offset = 500, 0
    while offset < n:
        got = d._collection.get(limit=batch, offset=offset, include=["documents"])
        docs = got.get("documents") or []
        if not docs:
            break
        for doc in docs:
            total += 1
            if det.detect(doc or "").is_poison:
                flagged += 1
                if len(hits) < 5:
                    hits.append((doc or "")[:70])
        offset += len(docs)

    rate = flagged / total if total else 0
    print(f"\n全量检测: {flagged}/{total} 条命中投毒特征（{rate:.2%}）")
    for h in hits:
        print(f"  {h!r}")

    if rate > 0.02:
        print("\n结论: ⚠ 库内投毒残留比例过高，实验前必须先重建")
        print("      python tests/rebuild_vectordb.py")
        return 1
    print("\n结论: ✓ 库基本干净，可以跑实验")
    return 0


if __name__ == "__main__":
    sys.exit(main())
