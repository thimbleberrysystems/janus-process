"""
config.py — centralised environment configuration.
All modules should import settings from here; never read os.environ directly.
"""
import os

from dotenv import load_dotenv

load_dotenv(override=False)  # .env values do NOT override vars already in environment

# ── LLM ───────────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.environ.get("OPENAI_MODEL", "gpt-4o")
OPENAI_EMBEDDING_MODEL: str = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

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

# ── PFC ↔ Basal Ganglia Recurrent Loop ────────────────────────────────────────
MAX_PFC_LOOPS: int = int(os.environ.get("MAX_PFC_LOOPS", "2"))
# ── Rate limiting ──────────────────────────────────────────────────────────────
RATE_LIMIT_MAX_CALLS: int = int(os.environ.get("RATE_LIMIT_MAX_CALLS", "10"))
RATE_LIMIT_WINDOW_SECONDS: int = int(os.environ.get("RATE_LIMIT_WINDOW_SECONDS", "60"))

# ── LLM Retry ─────────────────────────────────────────────────────────────────
LLM_RETRY_ATTEMPTS: int = int(os.environ.get("LLM_RETRY_ATTEMPTS", "3"))
LLM_RETRY_BACKOFF_MULTIPLIER: float = float(os.environ.get("LLM_RETRY_BACKOFF_MULTIPLIER", "0.5"))
LLM_RETRY_MAX_WAIT: float = float(os.environ.get("LLM_RETRY_MAX_WAIT", "10"))

# ── Amygdala ──────────────────────────────────────────────────────────────────
AMYGDALA_MODEL: str = os.environ.get("AMYGDALA_MODEL", "gpt-4o-mini")
AMYGDALA_THRESHOLD_LOW: float = float(os.environ.get("AMYGDALA_THRESHOLD_LOW", "0.2"))
AMYGDALA_THRESHOLD_MID: float = float(os.environ.get("AMYGDALA_THRESHOLD_MID", "0.5"))
AMYGDALA_THRESHOLD_HIGH: float = float(os.environ.get("AMYGDALA_THRESHOLD_HIGH", "0.7"))
AMYGDALA_TEMP_NEUTRAL: float = float(os.environ.get("AMYGDALA_TEMP_NEUTRAL", "0.7"))
AMYGDALA_TEMP_MILD: float = float(os.environ.get("AMYGDALA_TEMP_MILD", "0.5"))
AMYGDALA_TEMP_STRESSED: float = float(os.environ.get("AMYGDALA_TEMP_STRESSED", "0.3"))
AMYGDALA_TEMP_CRISIS: float = float(os.environ.get("AMYGDALA_TEMP_CRISIS", "0.1"))

# ── PFC ────────────────────────────────────────────────────────────────────────
PFC_MODEL: str = os.environ.get("PFC_MODEL", "gpt-4o")
PFC_CONSOLIDATION_THRESHOLD: float = float(os.environ.get("PFC_CONSOLIDATION_THRESHOLD", "0.5"))
PFC_STM_WINDOW: int = int(os.environ.get("PFC_STM_WINDOW", "10"))
PFC_DEFAULT_TEMPERATURE: float = float(os.environ.get("PFC_DEFAULT_TEMPERATURE", "0.3"))

# ── Basal Ganglia ─────────────────────────────────────────────────────────────
BG_MODEL: str = os.environ.get("BG_MODEL", "gpt-4o-mini")

# ── Hippocampus ────────────────────────────────────────────────────────────────
HIPPOCAMPUS_THRESHOLD: float = float(os.environ.get("HIPPOCAMPUS_THRESHOLD", "0.5"))
HIPPOCAMPUS_K: int = int(os.environ.get("HIPPOCAMPUS_K", "5"))
HIPPOCAMPUS_SEARCH_TYPE: str = os.environ.get("HIPPOCAMPUS_SEARCH_TYPE", "mmr")
HIPPOCAMPUS_COLLECTION: str = os.environ.get("HIPPOCAMPUS_COLLECTION", "episodic")

# ── Consolidator (extended) ───────────────────────────────────────────────────
CONSOLIDATION_COLLECTION: str = os.environ.get("CONSOLIDATION_COLLECTION", "episodic")
CONSOLIDATION_LLM_TEMPERATURE: float = float(os.environ.get("CONSOLIDATION_LLM_TEMPERATURE", "0.3"))
CONSOLIDATION_MODEL: str = os.environ.get("CONSOLIDATION_MODEL", "gpt-4o")
