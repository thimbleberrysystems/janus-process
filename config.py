"""
config.py — centralised environment configuration.
All modules should import settings from here; never read os.environ directly.
"""
import os

from dotenv import load_dotenv

load_dotenv(override=False)  # .env values do NOT override vars already in environment

# ── LLM ───────────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")

# ── LangSmith ─────────────────────────────────────────────────────────────────
LANGSMITH_API_KEY: str = os.environ.get("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT: str = os.environ.get("LANGSMITH_PROJECT", "janus-process")

# ── Redis ─────────────────────────────────────────────────────────────────────
REDIS_URL: str = os.environ.get("REDIS_URL", "redis://localhost:6379")
SESSION_ID: str = os.environ.get("SESSION_ID", "brain_session")
STM_TTL: int = int(os.environ.get("STM_TTL", "3600"))

# ── ChromaDB ──────────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR: str = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_HOST: str = os.environ.get("CHROMA_HOST", "localhost")
CHROMA_PORT: int = int(os.environ.get("CHROMA_PORT", "8000"))

# ── Ollama / LLM provider selection ─────────────────────────────────────────────
LLM_PROVIDER: str = os.environ.get("LLM_PROVIDER", "openai").lower()
OLLAMA_URL: str = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "llama2")
OLLAMA_EMBEDDING_MODEL: str = os.environ.get("OLLAMA_EMBEDDING_MODEL", "text2vec")

# ── Database ──────────────────────────────────────────────────────────────────
DATABASE_URL: str = os.environ.get("DATABASE_URL", "sqlite:///./habits.db")


# ── Memory Consolidation ──────────────────────────────────────────────────────
CONSOLIDATION_INTERVAL_SECONDS: int = int(os.environ.get("CONSOLIDATION_INTERVAL_SECONDS", "3600"))
CONSOLIDATION_WINDOW: int = int(os.environ.get("CONSOLIDATION_WINDOW", "50"))
# ── Rate limiting ──────────────────────────────────────────────────────────────
RATE_LIMIT_MAX_CALLS: int = int(os.environ.get("RATE_LIMIT_MAX_CALLS", "10"))
RATE_LIMIT_WINDOW_SECONDS: int = int(os.environ.get("RATE_LIMIT_WINDOW_SECONDS", "60"))
