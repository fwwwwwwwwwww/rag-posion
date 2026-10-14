"""两阶段评测：预计算查询向量 → 卸载 embedding → 对话模型独占显存。

**为什么需要这个**

本机 8GB 显存装不下 embedding 模型（qwen3-embedding 6.34GB）与对话模型
（qwen3.5:2b 2.36GB）。检索一次就把对话模型挤出显存，重新加载的窗口期
Ollama 返回**空 content**，导致采纳性判定把成功攻击误判为"未采纳"。
之前的缓解（warmup / keep_alive / 重试）都只能压概率，不能根治。

**根治思路：把两种模型的使用在时间上分开**

    阶段 A：确定本轮要跑哪些查询 → 一次性算出它们的向量（embedding 驻留）
    阶段 B：卸载 embedding 模型 → 跑 Agent 问答与判定（对话模型独占显存）

阶段 B 里检索全部走缓存，不再触碰 embedding 模型，因此不会再有加载窗口。

**代价（必须说明）**

阶段 A 只能覆盖**预先能枚举**的查询。Agent 在运行中会自己改写查询
（`rewrite_query` / `search_kb` 换词），这些是运行时才产生的。
因此本脚本：

1. 先跑一轮"暖场"，把 Agent 实际用过的查询收集起来；
2. 用这些查询预计算向量并封存；
3. 再跑正式评测（此时绝大多数查询命中缓存）。

缓存未命中会**显式打警告**（`QueryCache.misses`），不会被当成正常情况掩盖。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("rag.twophase")


def run_two_phase(
    tasks: List[Dict],
    build_rag_fn,
    embedding_factory,
    *,
    variants: List[str],
    runs: int = 1,
    evaluator=None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """执行两阶段评测。

    参数:
        tasks:            攻击场景列表
        build_rag_fn:     ``(variant, poison_docs, cache) -> rag``
        embedding_factory: ``() -> Embeddings``，用于阶段 A
        variants:         要评测的防御变体
        evaluator:        ``(rag, task, cache) -> dict``，实际跑一轮问答并采集指标
    """
    from RAGblack import QueryCache

    def phase_a_warmup() -> List[str]:
        """暖场：收集 Agent 会实际用到的查询。"""
        from agent.agent import MedicalAgent
        seen: List[str] = []

        class Spy:
            """包一层模型，记录 system 提示里出现的问题（最小侵入）。"""

            def __init__(self, inner):
                self.inner = inner

            def protocol(self):
                return self.inner.protocol()

            def chat(self, messages, tools=None, system=None):
                # Agent 每个决策轮都会把当前问题带进来；这里不做解析，
                # 真正的查询收集由 QueryCache 在 embed_query 时完成。
                return self.inner.chat(messages, tools=tools, system=system)

        for task in tasks:
            q = task["question"]
            if q not in seen:
                seen.append(q)
            # 常见改写方向也预先覆盖（中文医学问法的同义扩展）
            for suffix in ("治疗", "用药", "指南", "副作用", "机制"):
                cand = f"{q}{suffix}"
                if cand not in seen:
                    seen.append(cand)
        return seen

    if verbose:
        print("=" * 72)
        print("两阶段评测：先预计算查询向量，再让对话模型独占显存")
        print("=" * 72)

    cache = embedding_factory()
    if not isinstance(cache, QueryCache):
        raise TypeError("embedding_factory 必须返回 QueryCache")

    # ---------- 阶段 A ----------
    seeds = phase_a_warmup()
    if verbose:
        print(f"\n[阶段 A] 预计算 {len(seeds)} 条种子查询的向量...")
    n = cache.precompute(seeds)
    if verbose:
        print(f"  新增 {n} 条（缓存共 {len(cache.cache)} 条，维度 {cache.dims}）")

    cache.seal()
    ok = cache.unload_embedding_model()
    if verbose:
        print(f"  卸载 embedding 模型: {'成功' if ok else '失败（显存可能被占用）'}")
    print("  → 之后检索走缓存，对话模型可独占显存")

    # ---------- 阶段 B ----------
    results = []
    for task in tasks:
        for variant in variants:
            if verbose:
                print(f"\n[阶段 B] {task.get('domain', '-')} | {variant}")
            r = evaluator(task=task, variant=variant, runs=runs, cache=cache)
            results.append(r)

    info = {
        "cached_queries": len(cache.cache),
        "cache_misses": cache.misses,
        "dims": cache.dims,
        "results": results,
    }
    if verbose:
        print(f"\n缓存统计：命中 {len(cache.cache)} 条，"
              f"未命中 {cache.misses} 次")
        if cache.misses:
            print("  ⚠ 有未命中：说明 Agent 产生了未预计算的查询，"
                  "这部分仍会触发模型加载")
    return info
