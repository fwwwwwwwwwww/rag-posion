"""彻底清除库内所有投毒 chunk，并验证结果。

用法:
    python tests/purge_poison.py          # 只清理并报告
    python tests/purge_poison.py --verify # 清理后做一次注入/清理往返验证
"""
import sys

sys.path.insert(0, r"E:\RAG2")

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

import RAGblack as exp
import vectordb
from attack.detector import HybridPoisonDetector

EXPECTED_CLEAN = 5571

# 清库必须用 GPU 嵌入（CPU 下取文档极慢），且此时不加载对话模型
d = Chroma(embedding_function=exp.make_embeddings(num_gpu=99),
           persist_directory=exp.CHROMA_PERSIST_DIR)
print(f"清理前: {vectordb.count_chunks(d)}", flush=True)

det = HybridPoisonDetector(llm=None, db=None, ambiguous_as_poison=False)


def find_poison(batch: int = 500):
    """全库扫描，返回命中投毒特征的 chunk id 列表。"""
    out = []
    for item in vectordb.iter_documents(d, batch=batch):
        if det.detect(item["content"] or "").is_poison:
            out.append(item["id"])
    return out


poison_ids = find_poison()
print(f"检出投毒 {len(poison_ids)} 条", flush=True)
if poison_ids:
    vectordb.delete_ids(d, poison_ids)
n = vectordb.count_chunks(d)
print(f"清理后: {n}（期望 {EXPECTED_CLEAN}，还差 {n - EXPECTED_CLEAN}）", flush=True)

# 反复清理，直到不再检出（删掉一批后可能又露出别的批次）
for round_i in range(5):
    poison_ids = find_poison()
    if not poison_ids:
        print(f"第 {round_i+1} 轮: 已无投毒残留", flush=True)
        break
    print(f"第 {round_i+1} 轮: 再删 {len(poison_ids)} 条", flush=True)
    vectordb.delete_ids(d, poison_ids)

final = vectordb.count_chunks(d)
print(f"\n最终: {final} 个 chunk（干净期望 {EXPECTED_CLEAN}）", flush=True)
diff = final - EXPECTED_CLEAN
if diff > 0:
    print(f"仍多出 {diff} 个——这些是检测器漏掉的投毒，或重复注入的干净副本", flush=True)
    print("建议直接重建: python tests/rebuild_vectordb.py", flush=True)
elif diff == 0:
    print("✓ 库已完全干净", flush=True)
else:
    print(f"少 {-diff} 个（误删的干净文档）", flush=True)

if "--verify" in sys.argv:
    print("\n=== 注入/清理往返验证 ===", flush=True)
    from attack import strategies as st
    task = exp.ATTACK_PAIRS[0]
    docs = st.generate("template", task, n=15)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, length_function=len)
    chunks = splitter.create_documents(docs)
    before = vectordb.count_chunks(d)
    ids = d.add_documents(chunks)
    after_add = vectordb.count_chunks(d)
    vectordb.delete_ids(d, ids)
    after_del = vectordb.count_chunks(d)
    print(f"  注入前={before} 注入后={after_add} 清理后={after_del}", flush=True)
    print(f"  {'✓ 往返干净' if after_del == before else '✗ 往返有残留'}", flush=True)
