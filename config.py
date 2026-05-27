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

# ── Database ──────────────────────────────────────────────────────────────────
DATABASE_URL: str = os.environ.get("DATABASE_URL", "sqlite:///./habits.db")
