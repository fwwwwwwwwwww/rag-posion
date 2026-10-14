"""应用配置。

集中管理环境变量与路径，数据库地址支持 PostgreSQL / SQLite 自动降级。
"""
import os
from pathlib import Path

# 项目根目录（app 的上一级，即 E:\RAG2）
BASE_DIR = Path(__file__).resolve().parent.parent

# 数据目录（SQLite 降级时的数据库文件存放处）
DATA_DIR = BASE_DIR / "data"


def _load_dotenv(path: Path) -> None:
    """极简 .env 加载器，避免强依赖 python-dotenv。

    仅处理 `KEY=VALUE` 形式，忽略注释与空行；已存在的环境变量不覆盖。
    """
    if not path.exists():
        return
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception:
        pass


# 加载项目根目录的 .env（若存在）
_load_dotenv(BASE_DIR / ".env")


def get_database_url() -> str:
    """解析数据库连接地址。

    优先级：
      1. 环境变量 DATABASE_URL
      2. 尝试默认 MySQL（本机 3306，配合 docker-compose）
      3. 降级为 SQLite（本地开发，零依赖）

    若指定了 MySQL 但驱动不可用，同样降级 SQLite 并告警。
    """
    url = os.getenv("DATABASE_URL", "").strip()
    if url:
        return _resolve_url(url)

    # 未配置时默认 MySQL，若连接不上则交给 engine 层再降级
    return "mysql+pymysql://rag:rag_password@localhost:3306/rag_experiments?charset=utf8mb4"


def _resolve_url(url: str) -> str:
    """将相对路径的 SQLite URL 归一化为绝对路径。"""
    if url.startswith("sqlite:///") and ":memory:" not in url:
        db_path = url.replace("sqlite:///", "", 1)
        if not Path(db_path).is_absolute():
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            return f"sqlite:///{(DATA_DIR / db_path).as_posix()}"
    return url


# Ollama 配置（供 RAG 实验复用）
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# 数据库连接地址（模块级常量，供 engine 使用）
DATABASE_URL = get_database_url()

# ---- RAG 实验相关常量（与 RAGblack.py 保持一致，便于无需加载 RAG 模块时使用）----
# 注意：Agent 需要模型具备工具调用能力。实测 deepseek-r1:latest 不支持原生
# tool_calls（会退回文本 ReAct 通道）；qwen3.5:2b 在工具较多时仍稳定，故选它。
DEFAULT_LLM_MODEL = os.getenv("RAG_LLM_MODEL", "qwen3.5:2b")
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "qwen3-embedding:latest")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "rag_poison_exp"))
RETRIEVAL_K_DEFAULT = int(os.getenv("RETRIEVAL_K_DEFAULT", "3"))
# Agent 单次提问允许的最大工具调用步数
AGENT_MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "8"))

# ---- 鉴权配置 ----
# 旧版静态 Token（保留用于脚本/接口调用，浏览器走登录会话）
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "rag-demo-token")

# 会话签名密钥。**生产环境必须显式配置**：
# 未配置时启动会随机生成，副作用是进程重启后所有登录失效
# （这比用一个硬编码默认密钥安全得多——后者等于人人都能伪造会话）。
SESSION_SECRET = os.getenv("SESSION_SECRET", "")
# 会话有效期（秒），默认 12 小时
SESSION_TTL = int(os.getenv("SESSION_TTL", str(12 * 3600)))
SESSION_COOKIE_NAME = os.getenv("SESSION_COOKIE_NAME", "rag_session")

# 不需要鉴权的路径（登录本身、健康检查、文档等）
PUBLIC_PATHS = {
    "/api/health",
    "/api/attacks",
    "/api/auth/login",
    "/api/auth/logout",
    "/api/auth/needs-setup",
    "/docs", "/openapi.json", "/redoc",
}

# ---- Celery / Redis 配置 ----
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
# 是否启用 Celery（False 时降级为线程池后台任务）
CELERY_ENABLED = os.getenv("CELERY_ENABLED", "false").lower() in ("true", "1", "yes")
