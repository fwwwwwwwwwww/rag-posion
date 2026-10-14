"""向量库体检：检查库里是否混有投毒残留。

为什么需要：诊断中发现库里混入了约 2700 个历史残留的投毒 chunk
（干净知识库只应产生 5568 个），导致之前所有检索指标被污染——
"投毒召回率"虚高、干净文档被挤掉，四档对比的结论因此不可信。

任何实验开跑前都应该先跑这个脚本。
"""
import sys

sys.path.insert(0, r"E:\RAG2")

from langchain_chroma import Chroma

import RAGblack as exp
import vectordb
from attack.detector import HybridPoisonDetector

# 干净重建的实测产出：3000 篇文档按 chunk_size=500/overlap=50 切分。
#
# 这里用精确值而不是量级，是因为**投毒残留按条累积**：
# 原先的 5% 数量宽容等于放行约 280 条，2% 的比例阈值等于放行约 110 条。
# 2026-10-14 那次就有 63 条投毒混进来（数量只多 1.1%、比例只占 1.12%），
# 两道阈值全部漏过，说明宽容度本身就是漏洞。
EXPECTED_CLEAN_CHUNKS = 5568


def main():
    d = Chroma(embedding_function=exp.make_embeddings(),
               persist_directory=exp.CHROMA_PERSIST_DIR)
    n = vectordb.count_chunks(d)
    print(f"库内 chunk 数: {n}")

    count_ok = n == EXPECTED_CLEAN_CHUNKS
    if n > EXPECTED_CLEAN_CHUNKS:
        print(f"  ⚠ 比干净切分数多 {n - EXPECTED_CLEAN_CHUNKS} 条"
              f"（基准 {EXPECTED_CLEAN_CHUNKS}），疑似混有残留投毒")
        print("    建议: python tests/purge_poison.py"
              "  或  python tests/rebuild_vectordb.py")
    elif n < EXPECTED_CLEAN_CHUNKS:
        print(f"  ⚠ 比干净切分数少 {EXPECTED_CLEAN_CHUNKS - n} 条"
              f"（基准 {EXPECTED_CLEAN_CHUNKS}），可能有文档缺失")
    else:
        print(f"  ✓ 数量与干净切分数一致（{EXPECTED_CLEAN_CHUNKS}）")

    # 全量检测。阈值 3 的规则层在 400 篇真实干净文档上误报为 0，
    # 因此这里不再设比例宽容——命中即异常。
    det = HybridPoisonDetector(llm=None, db=None, ambiguous_as_poison=False)
    total, flagged = 0, 0
    hits = []
    for item in vectordb.iter_documents(d, batch=500):
        doc = item["content"]
        total += 1
        if det.detect(doc or "").is_poison:
            flagged += 1
            if len(hits) < 5:
                hits.append((doc or "")[:70])

    rate = flagged / total if total else 0
    print(f"\n全量检测: {flagged}/{total} 条命中投毒特征（{rate:.2%}）")
    for h in hits:
        print(f"  {h!r}")

    if flagged:
        print(f"\n结论: ⚠ 检出 {flagged} 条投毒残留（规则层阈值 3 实测零误报），"
              f"实验前必须清理")
        print("      python tests/purge_poison.py")
        return 1
    if not count_ok:
        print("\n结论: ⚠ 未检出投毒特征，但 chunk 数与基准不一致，"
              "需人工确认是文档缺失还是基准已变")
        return 1
    print("\n结论: ✓ 库干净，数量与基准一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
