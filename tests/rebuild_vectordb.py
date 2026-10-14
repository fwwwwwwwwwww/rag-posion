"""彻底重建干净的向量库。

背景（重要）：诊断发现原库 8269 个 chunk 里，**干净知识库只应产生 5571 个**，
也就是说混入了约 2700 个历史残留的投毒 chunk——这些是前两天多次实验
（含被中断的任务）注入后没清理干净的。

后果：之前所有检索指标都被污染了——检索结果里大量是投毒内容，
导致"投毒召回率"看起来很高、干净文档被挤掉，
四档对比的结论因此不可信。

本脚本把 collection 清空后，只用干净知识库重建。
"""
import random
import sys
import time
from itertools import islice

sys.path.insert(0, r"E:\RAG2")

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

import RAGblack as exp
import vectordb

# 批量重建时用 GPU 嵌入更快（向量数值与 CPU 一致，只是算得更快）。
# 注意：这里**不加载对话模型**，所以不会出现显存争抢；
# 实验阶段才需要 EMBED_ON_CPU=true 来给对话模型腾显存。
exp.EMBED_ON_CPU = False

t0 = time.time()
# 批量重建必须显式指定 GPU（num_gpu=99）：
# 一旦 embedding 模型以 num_gpu=0 加载过，Ollama 会一直把它留在 CPU 上，
# 实测 82 分钟一个 chunk 都写不进去。显式传 99 才会回到 GPU。
d = Chroma(embedding_function=exp.make_embeddings(num_gpu=99),
           persist_directory=exp.CHROMA_PERSIST_DIR)
print(f"重建前: {vectordb.count_chunks(d)} 个 chunk", flush=True)
print("embedding 方式: GPU (num_gpu=99)", flush=True)

# 清空
vectordb.clear(d)
print(f"已清空，剩余 {vectordb.count_chunks(d)}", flush=True)

kb = exp.load_medical_knowledge_base()
random.seed(42)
docs = kb
if len(docs) > exp.EXPERIMENT_DOC_COUNT:
    docs = random.sample(docs, exp.EXPERIMENT_DOC_COUNT)
print(f"用 {len(docs)} 篇干净文档重建", flush=True)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50, length_function=len)
chunks = splitter.create_documents(docs)
print(f"切分为 {len(chunks)} 个 chunk，开始写入（CPU embedding，较慢）", flush=True)

batch = 500
for i in range(0, len(chunks), batch):
    d.add_documents(chunks[i:i + batch])
    done = min(i + batch, len(chunks))
    print(f"  {done}/{len(chunks)}  已用 {time.time()-t0:.0f}s", flush=True)

final = vectordb.count_chunks(d)
print(f"\n重建完成: {final} 个 chunk（期望 {len(chunks)}）", flush=True)
print(f"总耗时 {time.time()-t0:.0f}s", flush=True)

# 抽查：不应再出现投毒特征
from attack.detector import HybridPoisonDetector
det = HybridPoisonDetector(llm=None, db=None, ambiguous_as_poison=False)
sample = list(islice(vectordb.iter_documents(d), 50))
flagged = sum(1 for it in sample
              if det.detect(it["content"] or "").is_poison)
print(f"抽查 50 条干净文档，命中投毒特征 {flagged} 条（应为 0 或极少）", flush=True)
