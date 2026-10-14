"""验证新检测器：对模板投毒的召回率 + 对真实医学文本的误报率。

这是 P0-1 的核心验收：旧检测器召回率 0/105 = 0%。
"""
import random
import sys

sys.path.insert(0, r"E:\RAG2")

from attack.detector import HybridPoisonDetector, rule_score
import RAGblack as exp

print("=" * 74)
print("1. 规则层单独评估（零成本）")
print("=" * 74)

kb = exp.load_medical_knowledge_base()
print(f"知识库: {len(kb)} 篇")

# 正样本：7 个场景 × 15 条模板投毒
poison = []
for t in exp.ATTACK_PAIRS:
    poison.extend(exp._generate_poison_variants(t, exp.POISON_DOC_COUNT))
print(f"投毒样本: {len(poison)} 条")

# 负样本：随机干净文档（注意用不同 seed，避免与检测器训练集重叠的巧合）
random.seed(7)
clean = random.sample(kb, 400)
print(f"干净样本: {len(clean)} 条")

det = HybridPoisonDetector(llm=None, db=None)   # 只用规则层

p_scores = [rule_score(d)[0] for d in poison]
c_scores = [rule_score(d)[0] for d in clean]

# 阈值判定
for th in (2, 3):
    tp = sum(1 for s in p_scores if s >= th)
    fp = sum(1 for s in c_scores if s >= th)
    recall = tp / len(poison)
    fpr = fp / len(clean)
    prec = tp / (tp + fp) if (tp + fp) else 0
    f1 = 2 * prec * recall / (prec + recall) if (prec + recall) else 0
    print(f"  阈值>={th}: 召回={recall:6.1%} 误报={fpr:6.1%} "
          f"精确={prec:6.1%} F1={f1:.3f}")

print(f"\n  投毒分数分布: min={min(p_scores)} max={max(p_scores)} "
      f"均值={sum(p_scores)/len(p_scores):.2f}")
print(f"  干净分数分布: min={min(c_scores)} max={max(c_scores)} "
      f"均值={sum(c_scores)/len(c_scores):.2f}")

# 对比旧检测器
print()
print("=" * 74)
print("2. 对照：旧统计检测器（IsolationForest）")
print("=" * 74)
random.seed(42)
train = random.sample(kb, min(exp.DETECTION_TRAIN_COUNT, len(kb)))
old = exp.PoisonDetectionModel()
old.fit(train)
old_tp = sum(1 for d in poison if old.predict_single(d)["is_poison"])
old_fp = sum(1 for d in clean if old.predict_single(d)["is_poison"])
print(f"  旧检测器: 召回={old_tp/len(poison):.1%} ({old_tp}/{len(poison)})  "
      f"误报={old_fp/len(clean):.1%} ({old_fp}/{len(clean)})")

print()
print("=" * 74)
print("3. 抽样看分错的干净文档（判断误报是否可接受）")
print("=" * 74)
fp_docs = [(d, rule_score(d)) for d in clean if rule_score(d)[0] >= 3][:5]
for d, (s, hits) in fp_docs:
    print(f"  [分数{s}] {d[:70]!r}")
    print(f"         命中: {hits}")
if not fp_docs:
    print("  阈值为 3 时无误报")
