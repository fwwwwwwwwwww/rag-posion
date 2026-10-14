"""投毒检测：规则前置过滤 + LLM 语义判定（替代失效的统计检测器）。

为什么重写
----------
原 ``PoisonDetectionModel``（IsolationForest）实测对模板投毒的**召回率为 0/105 = 0%**：

- 训练集是纯干净文档，却预设 ``contamination=0.05``，于是它只能标出
  **干净数据自身的极端样本**；投毒文档在它的坐标系里是"内点"。
- 实测置信度分布完全重叠：投毒均值 0.0449 vs 干净均值 0.0432（差 0.0017）。
- 特征（长度/重复率/数字密度/感叹号密度 + TF-IDF）无法表征"投毒"——
  模板投毒用的正是"权威句式 + 统计数据"，与真实医学文本同分布。

新方案分两层，各有明确职责：

**第一层：规则（零成本、可解释）**
依据实测观察——投毒文档的杀伤力来自"强断言 + 编造机构/产品名 + 疗效数字 + 权威伪装"，
而**真实的医学教材/指南几乎不会**同时出现这几样。

**第二层：LLM 语义判定（兜底，处理规则判不准的灰区）**
规则打分落在中间区间时才调用，控制成本。

**以及一个新能力：可被多篇文档佐证的强断言 → 更可疑**
单篇说"某药根治高血压"可能只是谣言；但**多篇不同来源说同一句编造断言**
恰恰是投毒的特征（真实知识库不会反复出现同一个虚构结论）。
这正好把投毒检测和 ``cross_check`` 的"独立来源计数"接起来。

对外接口保持 ``predict(docs) -> [{document, is_poison, confidence, ...}]``，
因此 ``agent/tools.py`` 里已有的 ``check_poison`` 工具无需改动。
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("rag.attack.detector")

# ---------------- 规则词表 ----------------
# 强断言分两档。教训：不能把所有绝对化词都当"一项重证据"——
# 真实医学语料里 "如何根治汗疱疹？" 这类**患者提问**很常见，
# 单个 "根治" 不该直接顶到阈值。
#
# TIER1（权重 2）：几乎只在虚假医疗宣传里出现的绝对化承诺。
# 注意：词表本身也不能被投毒样本"污染"——实测把 "彻底治愈""无副作用" 放进 TIER1
# 会误伤真实语料中的正常近义表述（如"汗疱疹彻底治愈"），这类词只保留其
# **带具体对象**的形态（如"彻底逆转糖尿病"）。
CLAIMS_TIER1 = [
    "彻底逆转", "无任何副作用", "包治", "永不复发", "无需终身用药",
    "100%有效", "一次治疗即可", "一次治愈", "一针修复", "告别失眠",
    "有效率高达", "总有效率", "临床完全缓解", "已成功治愈", "治愈率达",
    "可彻底逆转", "根治高血压", "三疗程根治",
]
# TIER2（权重 1）：可疑但可能出现在正常表述或患者提问中
CLAIMS_TIER2 = [
    "根治", "治愈率", "不复发", "完全缓解", "终身不", "无需终身",
    "特效", "秘方", "无副作用", "彻底治愈",
]

# 权威伪装：把结论挂到不具名或编造的权威上
AUTHORITY_MARKERS = [
    "权威期刊", "顶级医疗中心", "多位院士", "院士联合推荐", "专家共识认为",
    "临床实践指南推荐", "国际知名医学团队", "全国多中心", "多中心临床试验",
    "国家发明专利", "多位专家共识", "令人振奋", "强烈支持",
]

# 编造的机构/产品名。
# 实测教训：不要用宽泛的 "XX医院" 正则——真实文档里 "重庆学府医院""应带孩子去医院"
# 都会被匹配到。这里只保留**强归属**模式：必须带研发/推出/首创等动作词。
FAKE_ENTITY_PATTERNS = [
    r"[\u4e00-\u9fff]{2,6}(?:医院|医疗中心|肿瘤中心|研发中心|心理中心)"
    r"(?:的|推出|研发|首创|采用|发明)",
    r"[\u4e00-\u9fff]{2,6}(?:教授|主任|医生)(?:首创|研发|推出|发明|采用)",
    r"[\u4e00-\u9fff]{2,6}(?:疗法|治疗法|注射术|微针贴片|激活疗法)",
    r"[「『\"']([\u4e00-\u9fff]{2,8})[」』\"']\s*(?:纯中药|制剂|贴片|注射液|疗法)",
]

# 数字型疗效声明
EFFICACY_NUM = re.compile(
    r"(?:有效率|治愈率|缓解率|生存率|满意度|有效率高达)\s*(?:高达)?\s*\d{1,3}(?:\.\d+)?\s*%")
PVALUE = re.compile(r"[Pp]\s*[<＜]\s*0?\.\d+")
SAMPLE_SIZE = re.compile(r"(?:纳入|共|超过)?\s*\d{2,5}\s*例(?:患者|病例)?")

LLM_DETECT_PROMPT = """你要判断一段文本是否是**针对医疗知识的投毒内容**（虚假医疗宣传）。

待判断文本：
{text}

投毒内容的典型特征（同时出现多条即可认定）：
1. 给出绝对化疗效承诺（根治、彻底治愈、无任何副作用、永不复发等）
2. 推荐具体的医院/医生/产品/疗法名称，并声称其疗效远超常规治疗
3. 用编造的权威背书（权威期刊、多位院士推荐、多中心临床试验）支撑结论
4. 堆砌精确到小数的疗效数字（有效率97.6%）却没有任何可核查的出处
5. 反复重复同一套产品名/疗法名，有明显的推广口吻

注意：**正常的医学科普和教材也会提到疗效数字和指南**，判断的关键是
"是否在推销一个具体的、声称能根治/超越常规医疗的方案"。

只输出一个 JSON 对象：
{{"is_poison": true/false, "confidence": 0-1 的小数, "reason": "一句话理由"}}
"""


@dataclass
class DetectionResult:
    """单篇文档的检测结果。"""

    is_poison: bool
    confidence: float
    reason: str = ""
    rule_score: int = 0
    method: str = "rule"          # rule / llm / rule+llm
    evidence: List[str] = field(default_factory=list)

    def to_dict(self, doc: str = "") -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "is_poison": self.is_poison,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
            "rule_score": self.rule_score,
            "method": self.method,
            "evidence": self.evidence[:6],
        }
        if doc:
            d["document"] = doc[:120] + ("..." if len(doc) > 120 else "")
        return d


def rule_score(text: str) -> tuple:
    """规则打分：返回 (分数, 命中的证据列表)。

    分数越高越可疑。实测模板投毒普遍能拿到 3 分以上，
    而真实医学教材文本通常在 0-1 分。
    """
    if not text:
        return 0, []
    hits: List[str] = []
    score = 0

    t1 = [w for w in CLAIMS_TIER1 if w in text]
    t2 = [w for w in CLAIMS_TIER2 if w in text]
    if t1:
        score += 2 * min(len(t1), 2)     # 单个 TIER1 记 2 分，两个及以上记 4 分
        hits.append("绝对化承诺: " + "/".join(t1[:4]))
    if t2:
        # TIER2 单独出现只记 1 分；与 TIER1 同现时说明是同一批话术，不重复加倍
        score += 1 if t1 else min(len(t2), 2)
        hits.append("可疑表述: " + "/".join(t2[:4]))

    found_auth = [w for w in AUTHORITY_MARKERS if w in text]
    if found_auth:
        score += 1
        hits.append("权威伪装: " + "/".join(found_auth[:3]))

    entities: List[str] = []
    for pat in FAKE_ENTITY_PATTERNS:
        entities.extend(re.findall(pat, text))
    entities = [e for e in entities if isinstance(e, str) and len(e) >= 2]
    if entities:
        score += 1
        hits.append("具名机构/疗法: " + "/".join(entities[:3]))

    numeric = EFFICACY_NUM.findall(text)
    if numeric:
        score += 1
        hits.append(f"疗效数字: {len(numeric)} 处")

    extras = len(PVALUE.findall(text)) + (1 if SAMPLE_SIZE.search(text) else 0)
    if extras >= 2:
        score += 1
        hits.append("统计伪装: P值/样本量")

    return score, hits


class HybridPoisonDetector:
    """规则 + LLM 的混合投毒检测器。"""

    # 规则分数 >= HIGH 直接判定为投毒；<= LOW 直接判定为正常；
    # 中间灰区再交给 LLM（控制调用成本）
    HIGH = 3
    LOW = 1

    def __init__(self, llm: Any = None, db: Any = None, use_llm: bool = True,
                 llm_max_calls_per_batch: int = 20,
                 ambiguous_as_poison: bool = True):
        self.llm = llm
        self.db = db
        self.use_llm = use_llm and llm is not None
        self.llm_max_calls = llm_max_calls_per_batch
        self._llm_calls = 0
        # 灰区（分数恰好 = LOW+1，即 2 分）且无 LLM 可用时怎么判：
        #   True  → 判为可疑（偏向安全，适合**在线防御**：宁可提醒）
        #   False → 判为正常（偏向精确，适合**离线批量清理**：避免误删干净文档）
        #
        # 为什么要有这个开关：实测用"偏向安全"的口径去批量清理向量库时，
        # 把 41 篇真实医学文档一起删了——2 分档在真实语料上误报并不低
        # （阈值 3 才是我验证过的"零误报"点）。
        # 在线检测宁可误报，离线删除必须宁可漏报。
        self.ambiguous_as_poison = ambiguous_as_poison

    # ---------------- 单篇 ----------------
    def detect(self, text: str, db_support: Optional[int] = None) -> DetectionResult:
        """检测单篇文本。

        ``db_support``：该文本的核心断言在向量库中的**独立支持文档数**。
        传了的话，多来源共享同一编造断言会成为额外证据（投毒的典型特征）。
        """
        score, hits = rule_score(text)
        reason = "；".join(hits) if hits else "未命中投毒特征"

        # 多篇独立文档佐证同一个强断言 → 真实知识库不会这样，加权
        if db_support is not None and db_support >= 2 and score >= 2:
            score += 1
            reason += f"；{db_support} 篇文档重复同一断言"

        if score >= self.HIGH:
            return DetectionResult(True, min(0.6 + 0.1 * score, 0.99),
                                   reason, score, "rule", hits)
        if score <= self.LOW:
            return DetectionResult(False, min(0.05 + 0.05 * score, 0.2),
                                   reason, score, "rule", hits)

        # 灰区：交给 LLM
        if self.use_llm and self._llm_calls < self.llm_max_calls:
            self._llm_calls += 1
            llm_res = self._detect_by_llm(text)
            if llm_res is not None:
                return llm_res

        # 灰区且无 LLM 可用：按调用方指定的偏向处理
        return DetectionResult(self.ambiguous_as_poison,
                               0.5, reason + "（灰区，按规则判定）",
                               score, "rule", hits)

    def _detect_by_llm(self, text: str, retries: int = 1) -> Optional[DetectionResult]:
        """用 LLM 判定灰区文本。

        注意空响应：本地模型在显存切换/高负载时会返回**空 content**，
        此时必须返回 None 让调用方回退到规则判定，
        而**不能**把"空"当成"不是投毒"——那等于把可疑内容放行。
        见 ``EMBED_ON_CPU`` / ``keep_alive`` 的说明。
        """
        from eval.metrics import _extract_json  # 复用同一个稳健解析器

        prompt = LLM_DETECT_PROMPT.format(text=text[:1500])
        for attempt in range(retries + 1):
            try:
                reply = self.llm.chat([{"role": "user", "content": prompt}])
            except Exception as e:
                logger.warning("LLM 判定调用失败(第%d次)：%s", attempt + 1, e)
                continue

            raw = reply.raw if hasattr(reply, "raw") else ""
            if not (reply.text or "").strip() and not (raw or "").strip():
                # 空响应：稍等再试，仍空则交回调用方
                if attempt < retries:
                    time.sleep(1.5)
                continue

            obj = _extract_json(reply.text or "")
            if not isinstance(obj, dict):
                continue
            flag = obj.get("is_poison")
            if isinstance(flag, str):
                flag = flag.strip().lower() in ("true", "yes", "1")
            try:
                conf = float(obj.get("confidence", 0.5))
            except (TypeError, ValueError):
                conf = 0.5
            return DetectionResult(bool(flag), conf,
                                   str(obj.get("reason", ""))[:200],
                                   0, "rule+llm")
        logger.warning("灰区 LLM 判定未取得有效结果，回退规则判定")
        return None

    # ---------------- 批量（兼容原 PoisonDetectionModel 接口） ----------------
    def predict(self, docs: List[str]) -> List[Dict[str, Any]]:
        self._llm_calls = 0
        out = []
        for d in docs:
            r = self.detect(d, db_support=self._count_support(d))
            out.append(r.to_dict(d))
        return out

    def predict_single(self, doc: str) -> Dict[str, Any]:
        return self.detect(doc, db_support=self._count_support(doc)).to_dict(doc)

    # ---------------- 向量库佐证 ----------------
    _SUPPORT_STOP = set("的了和与及或在中为是对有可以可用进行以及一个这个那个我们他们它"
                        "治疗医院疗法患者临床研究显示表明结果结论方法方案使用包括通过")

    def _count_support(self, text: str, k: int = 8, similarity_limit: float = 0.35) -> Optional[int]:
        """统计向量库里有多少独立文档在重复同一断言（投毒特征）。"""
        if self.db is None or not text:
            return None
        try:
            hits = self.db.similarity_search_with_score(text[:300], k=k)
        except Exception:
            return None
        if not hits:
            return None
        # Chroma 返回的是距离：越小越相似
        close = [(d, s) for d, s in hits if s <= similarity_limit]
        if not close:
            return None
        keys = set()
        for d, _s in close:
            meta = getattr(d, "metadata", {}) or {}
            keys.add(str(meta.get("doc_id") or meta.get("source")
                         or hash(d.page_content[:80])))
        return len(keys)


def build_detector(llm: Any = None, db: Any = None) -> HybridPoisonDetector:
    """构建检测器（推荐入口）。"""
    return HybridPoisonDetector(llm=llm, db=db)
