"""Agent 工具集。

每个工具都是"模型可以主动选择调用"的原子能力，而不是写死在流程里的固定步骤。
与原来那条 ``retrieve → grade → generate`` 固定链路的本质区别就在这里：
**调用哪个、调几次、什么顺序，由模型决定。**

工具清单：
  - ``search_kb``      向量检索知识库
  - ``check_poison``   投毒嫌疑检测（接入 IsolationForest 检测器）
  - ``cross_check``    交叉验证某个说法是否有独立证据支持
  - ``rewrite_query``  带反馈的查询改写
  - ``refuse``         明确拒答出口（证据不足时不编造）
"""
from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("rag.agent.tools")

# 单次 search_kb 允许的检索条数上限，防止模型请求过大的 k 打爆上下文
MAX_K = 20
DEFAULT_K = 5


# ====================== 工具实现 ======================
class ToolKit:
    """持有向量库 / LLM / 检测器等依赖，向模型暴露一组工具。"""

    def __init__(self, db: Any, llm: Any, knowledge_base: Optional[List[str]] = None):
        self.db = db
        self.llm = llm
        self._knowledge_base = knowledge_base or []
        self._detector = None
        self._detector_error: Optional[str] = None
        # 检索留痕：供 cross_check 使用，也便于前端展示
        self.last_search: List[Any] = []

    # ---------------- search_kb ----------------
    def search_kb(self, query: str, k: int = DEFAULT_K) -> Dict[str, Any]:
        """在医学知识库中做向量检索。"""
        if not query or not str(query).strip():
            return {"error": "query 不能为空", "results": []}
        try:
            k = max(1, min(int(k or DEFAULT_K), MAX_K))
        except (TypeError, ValueError):
            k = DEFAULT_K

        try:
            retriever = self.db.as_retriever(search_kwargs={"k": k})
            docs = retriever.invoke(str(query))
        except Exception as e:
            logger.exception("检索失败")
            return {"error": f"检索失败: {e}", "results": []}

        self.last_search = list(docs)
        results = []
        for i, doc in enumerate(docs, 1):
            content = getattr(doc, "page_content", "") or ""
            meta = getattr(doc, "metadata", {}) or {}
            results.append({
                "id": i,
                "content": content[:600],
                "source": meta.get("source", "") or meta.get("doc_id", ""),
                # 原文档 ID 用于判断"是否来自不同独立文档"
                "doc_key": str(meta.get("doc_id") or meta.get("source") or f"chunk{i}"),
            })
        return {"query": query, "k": k, "count": len(results), "results": results}

    # ---------------- check_poison ----------------
    def _get_detector(self):
        """惰性构建投毒检测器（规则 + LLM 语义判定）。

        历史：这里原先用 ``RAGblack.PoisonDetectionModel``（IsolationForest），
        但实测它对模板投毒的**召回率是 0/105 = 0%**——训练集是纯干净文档却预设了
        ``contamination=0.05``，于是它只能标出干净数据自身的极端样本，
        投毒文档在它的坐标系里属于内点。它返回的"未发现异常"是**虚假保证**，
        反而会让 Agent 更放心地引用投毒内容。

        现改用 ``attack.detector.HybridPoisonDetector``：规则层零成本、可解释，
        灰区才调 LLM。实测召回 100%、误报 0%（对比旧模型 0% / 4%）。
        """
        if self._detector is not None or self._detector_error is not None:
            return self._detector
        try:
            from attack.detector import build_detector

            # 传入 db 后，检测器还能把"多篇文档重复同一断言"作为额外证据
            self._detector = build_detector(llm=self.llm, db=self.db)
            logger.info("投毒检测器就绪（规则 + LLM 混合）")
        except Exception as e:
            logger.warning("检测器构建失败：%s", e)
            self._detector_error = str(e)
        return self._detector

    def check_poison(self, text: str = "", doc_id: Optional[int] = None) -> Dict[str, Any]:
        """检查某段文本（或最近一次检索结果中的第 doc_id 条）的投毒嫌疑。"""
        target = text or ""
        if not target and doc_id is not None:
            try:
                idx = int(doc_id) - 1
                target = self.last_search[idx].page_content
            except (IndexError, ValueError, AttributeError):
                return {"error": f"doc_id={doc_id} 无效（请先调用 search_kb）"}
        if not target.strip():
            return {"error": "需要提供 text 或 doc_id"}

        detector = self._get_detector()
        if detector is None:
            return {
                "error": "检测器不可用",
                "detail": self._detector_error,
                "is_poison": None,
            }
        try:
            r = detector.predict_single(target)
        except Exception as e:
            return {"error": f"检测失败: {e}"}

        return {
            "is_poison": r["is_poison"],
            "confidence": r["confidence"],
            "reason": r.get("reason", ""),
            "evidence": r.get("evidence", []),
            "method": r.get("method", ""),
            "note": ("该文本被判定为虚假医疗宣传，**不得据此作答**，需在回答中说明它不可信"
                     if r["is_poison"]
                     else "未发现投毒特征，可作为证据引用"),
        }

    # ---------------- cross_check ----------------
    @staticmethod
    def _tokens(text: str) -> List[str]:
        """中文按字 + 英文/数字按词切分，用于重叠度计算。"""
        text = (text or "").lower()
        words = re.findall(r"[a-z0-9]+", text)
        chars = [c for c in re.findall(r"[\u4e00-\u9fff]", text)]
        return words + chars

    def cross_check(self, claim: str, doc_ids: Optional[List[int]] = None) -> Dict[str, Any]:
        """检验某说法是否有多篇**独立**文档支持。

        实现为确定性算法而非再问一次 LLM：这里要的是可复核的证据计数，
        再让模型自评会引入它自己的幻觉。
        """
        claim = (claim or "").strip()
        if not claim:
            return {"error": "claim 不能为空"}
        pool = self.last_search
        if doc_ids:
            picked = []
            for i in doc_ids:
                try:
                    picked.append(pool[int(i) - 1])
                except (IndexError, ValueError, TypeError):
                    continue
            if picked:
                pool = picked
        if not pool:
            return {"error": "没有可核查的文档，请先调用 search_kb"}

        claim_tokens = set(self._tokens(claim))
        if not claim_tokens:
            return {"error": "claim 无效"}

        supporting: List[Dict[str, Any]] = []
        supporting_keys = set()
        for i, doc in enumerate(pool, 1):
            content = getattr(doc, "page_content", "") or ""
            doc_tokens = set(self._tokens(content))
            if not doc_tokens:
                continue
            overlap = len(claim_tokens & doc_tokens)
            # 覆盖率 + Jaccard，避免长文档仅因篇幅长就"看起来支持"
            coverage = overlap / len(claim_tokens)
            jaccard = overlap / len(claim_tokens | doc_tokens)
            if coverage >= 0.45 and jaccard >= 0.08:
                meta = getattr(doc, "metadata", {}) or {}
                key = str(meta.get("doc_id") or meta.get("source") or f"chunk{i}")
                supporting_keys.add(key)
                supporting.append({
                    "id": i,
                    "coverage": round(coverage, 3),
                    "excerpt": content[:200],
                })

        independent = len(supporting_keys)
        return {
            "claim": claim,
            "supporting_docs": supporting,
            "supporting_count": len(supporting),
            "independent_sources": independent,
            "verdict": ("supported" if independent >= 2
                        else "weak" if independent == 1
                        else "unsupported"),
            "note": ("至少两篇独立文档支持，可作为回答依据"
                     if independent >= 2 else
                     "仅一篇文档支持，属于弱证据，回答时须说明这一点"
                     if independent == 1 else
                     "没有任何检索到的文档支持该说法，不得据此作答"),
        }

    # ---------------- rewrite_query ----------------
    def rewrite_query(self, query: str, feedback: str = "") -> Dict[str, Any]:
        """在检索效果不佳时改写查询；feedback 说明上一次为什么没查到。"""
        if not query or not str(query).strip():
            return {"error": "query 不能为空"}

        from .prompts import REWRITE_PROMPT

        prompt = REWRITE_PROMPT.format(query=query, feedback=feedback or "（无）")

        try:
            # llm 已由 ensure_tool_model 归一化为带 chat() 的对象
            if hasattr(self.llm, "chat"):
                text = self.llm.chat([{"role": "user", "content": prompt}]).text
            else:  # 兜底：极简的 invoke 接口
                from .chat_ollama import strip_think
                raw = self.llm.invoke(prompt)
                text = strip_think(raw if isinstance(raw, str) else str(raw))
        except Exception as e:
            return {"error": f"改写失败: {e}", "rewritten": query}

        text = (text or "").strip().strip('"').strip("'").splitlines()[0].strip()
        # 防止模型把整段解释当查询
        if len(text) > 120:
            text = text[:120]
        return {"original": query, "rewritten": text or query}

    # ---------------- refuse ----------------
    def refuse(self, reason: str = "") -> Dict[str, Any]:
        """拒答出口：知识库证据不足时使用，而不是编造答案。"""
        return {"refused": True, "reason": reason or "知识库中没有足够依据回答该问题"}

    # ---------------- 注册表 ----------------
    def registry(self) -> Dict[str, Callable[..., Dict[str, Any]]]:
        return {
            "search_kb": self.search_kb,
            "check_poison": self.check_poison,
            "cross_check": self.cross_check,
            "rewrite_query": self.rewrite_query,
            "refuse": self.refuse,
        }

    def schemas(self) -> List[dict]:
        """Ollama/OpenAI 风格的 tools 定义。"""
        return TOOL_SCHEMAS

    def call(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """按名字执行工具，统一异常处理。"""
        fn = self.registry().get(name)
        if fn is None:
            return {"error": f"未知工具: {name}",
                    "available": sorted(self.registry().keys())}
        kwargs = dict(arguments or {})
        kwargs.pop("_raw", None)
        try:
            return fn(**kwargs)
        except TypeError as e:
            return {"error": f"参数错误: {e}"}
        except Exception as e:
            logger.exception("工具 %s 执行失败", name)
            return {"error": f"工具执行失败: {e}"}


# ====================== 工具 Schema ======================
TOOL_SCHEMAS: List[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_kb",
            "description": "在医学知识库中进行语义检索，返回最相关的若干文档片段。"
                           "这是获取事实依据的唯一来源，任何回答都必须基于它返回的内容。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索用的查询语句，尽量具体"},
                    "k": {"type": "integer", "description": "返回条数，默认5，最大20"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_poison",
            "description": "检查一段文本是否有投毒嫌疑（虚假医疗宣传的统计特征）。"
                           "在准备引用某个片段前调用它，可传入 text 或最近一次检索结果中的 doc_id。",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "要检查的文本内容"},
                    "doc_id": {"type": "integer",
                               "description": "最近一次 search_kb 结果中的编号，与 text 二选一"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cross_check",
            "description": "交叉验证一个具体说法是否有多篇独立文档支持。"
                           "当你要下明确结论（尤其是疗效、治愈率这类断言）时先调用它。",
            "parameters": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string", "description": "待验证的具体说法"},
                    "doc_ids": {"type": "array", "items": {"type": "integer"},
                                "description": "要核查的文档编号，省略则核查全部检索结果"},
                },
                "required": ["claim"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rewrite_query",
            "description": "当检索结果与问题无关时，改写查询后重新检索。"
                           "请在 feedback 中说明上次检索为什么失败。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "原始查询"},
                    "feedback": {"type": "string", "description": "上次检索失败的原因或缺口"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "refuse",
            "description": "知识库中确实找不到依据时，明确拒答而不是编造。"
                           "这是允许且正确的选择。",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "拒答的具体原因"},
                },
                "required": ["reason"],
            },
        },
    },
]
