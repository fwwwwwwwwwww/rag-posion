# 工具化 Agent 重构说明

本目录是"把固定 RAG 流水线改造成真正工具化 Agent"的实现。

## 为什么改

改造前的链路（原 `rag_graph.py`）是：

```
retrieve → grade → generate
```

下一步走哪条边，由 Python 里的 `if state["filtered"]` 和模块常量
`MAX_QUERY_REWRITES` 决定。模型只负责往三个固定提示词空位里填文本
（评分编号 / 改写句 / 答案），**从未获得过控制权**。那是流水线，不是 Agent。

判断标准只有一个问题：**下一步做什么，是谁决定的？**

## 改成了什么

模型拿到一套工具和一份策略，自己决定调哪个工具、调几次、证据够不够、
该不该拒答。循环终止条件是"模型不再请求工具"，而不是某个写死的常量。

```
        ┌───────┐  模型要求调工具   ┌───────┐
        │ agent │ ───────────────→ │ tools │
        │ (LLM) │ ←─────────────── │ (执行) │
        └───┬───┘   工具结果回填     └───────┘
            │ 模型输出最终回答 / 调用 refuse
            ↓
        ┌──────────┐
        │ finalize │  基于实际收集到的证据生成，不编造
        └──────────┘
```

## 工具集

| 工具 | 作用 |
|---|---|
| `search_kb` | 向量检索知识库（获取事实依据的唯一入口） |
| `check_poison` | 投毒嫌疑检测（接入 `attack.detector.HybridPoisonDetector`，规则 + LLM） |
| `cross_check` | 交叉验证某说法是否有**多篇独立文档**支持 |
| `rewrite_query` | 带反馈的查询改写（检索不到时用） |
| `refuse` | 明确拒答出口（证据不足时不编造） |

## 模块

| 文件 | 职责 |
|---|---|
| `chat_ollama.py` | 工具调用 LLM 层。直连 Ollama `/api/chat`，支持原生 `tool_calls` 与文本 ReAct 双通道，**自动探测**模型属于哪种 |
| `tools.py` | 工具实现与 JSON Schema |
| `prompts.py` | Agent 策略提示词（定义的是 policy，不是填空模板） |
| `agent.py` | LangGraph 工具循环、会话记忆、决策轨迹、证据落地生成 |

## 用法

```python
from agent import build_agent, new_memory

agent = build_agent(
    db=chroma_db,
    llm=None,                    # 传 None 时按 model 名新建支持工具调用的对话层
    model="qwen3.5:2b",
    checkpointer=new_memory(),   # 传了才有会话记忆
    max_steps=6,
)

result = agent.run("顽固性高血压用什么药？", session_id="s1")
print(result.answer)             # 最终回答
print(result.trace)              # 决策轨迹（每步工具调用与结果）
print(result.sources)            # 引用的文档（带 poison_suspect 标记）
print(result.refused)            # 是否主动拒答

# 兼容旧接口
agent.invoke({"query": "..."})   # -> {"result", "source_documents"}
```

## 模型要求（实测结论，选型时务必看）

| 模型 | 原生工具调用 | 说明 |
|---|---|---|
| `qwen3.5:2b` | ✅ 稳定 | **默认**，5 个工具同时给出也能正确触发 |
| `qwen2.5:3b` | ⚠️ 不稳定 | 工具较多时会返回**完全空**的响应（已有兜底重试） |
| `deepseek-r1:latest` | ❌ 不支持 | 只能走文本 ReAct 通道，能力较弱 |

```bash
python tests/probe_tool_matrix.py    # 复现上面这张矩阵
```

## 行为准则（写进系统提示词，且由代码兜底）

1. **无依据不作答** —— 事实性陈述必须来自工具返回内容。
   若模型一次工具都没调用就直接作答，`finalize` 会强制加上「未核查」前缀。
2. **引用有门槛** —— 强断言（疗效/治愈率/根治/无副作用）须先 `check_poison` 再 `cross_check`。
3. **证据不足就拒答** —— `refuse` 是被鼓励的正确行为。
4. **衔接上文** —— 追问里的指代会结合最近几轮对话解析，并把补全后的实体写进检索查询。

## 测试

```bash
python tests/test_agent_smoke.py     # 控制流单元测试（不依赖 Ollama/Chroma）
python tests/test_agent_api.py       # API 层：SSE 事件契约、拒答透传、鉴权
python tests/e2e_agent_real.py       # 真实 Chroma + 真实 Ollama 端到端
python tests/e2e_agent_real.py --poison   # 额外注入投毒文档
```

## 注意

- **跑实验与端到端脚本需要放开文件限制**：DSH 沙箱在 `workspace-write` 下会拦截
  SQLite 的文件锁，Chroma 的元数据库会报 `attempt to write a readonly database`，
  导致向量库打不开。把会话切到**完全权限**即可。
- 检测器是 ``attack.detector.HybridPoisonDetector``（规则前置 + 灰区 LLM 判定），
  **不需要干净语料训练**，首次调用 `check_poison` 时惰性构建并缓存在进程内。
  早期版本用的 IsolationForest 对模板投毒召回率为 0%，已弃用（见该模块顶部说明）。
