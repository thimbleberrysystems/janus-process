"""
config.py — centralised configuration.
All modules should import settings from here; never read os.environ directly.

Priority: environment variable > config.yaml > built-in default.
"""
import os
from typing import Any

import yaml as _yaml

# ── Load config.yaml (if present) ────────────────────────────────────────────
_CONFIG: dict[str, Any] = {}
_config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
if os.path.exists(_config_path):
    with open(_config_path) as _f:
        _CONFIG = _yaml.safe_load(_f) or {}


def _get(key: str, default: str = "") -> str:
    """Return env var (uppercase) → config.yaml (lowercase) → built-in default."""
    return os.environ.get(key.upper(), str(_CONFIG.get(key.lower(), default)))


def _get_int(key: str, default: int) -> int:
    return int(_get(key, str(default)))


def _get_float(key: str, default: float) -> float:
    return float(_get(key, str(default)))


# ── LLM ───────────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = _get("openai_api_key", "")
OPENAI_MODEL: str = _get("openai_model", "gpt-4o")
OPENAI_EMBEDDING_MODEL: str = _get("openai_embedding_model", "text-embedding-3-small")

# ── LangSmith ─────────────────────────────────────────────────────────────────
LANGSMITH_API_KEY: str = _get("langsmith_api_key", "")
LANGSMITH_PROJECT: str = _get("langsmith_project", "janus-process")

# ── Redis ─────────────────────────────────────────────────────────────────────
REDIS_URL: str = _get("redis_url", "redis://localhost:6379")
SESSION_ID: str = _get("session_id", "brain_session")
STM_TTL: int = _get_int("stm_ttl", 7200)  # 2 hours; must be > CONSOLIDATION_INTERVAL_SECONDS

# ── ChromaDB ──────────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR: str = _get("chroma_persist_dir", "./chroma_db")
CHROMA_HOST: str = _get("chroma_host", "localhost")
CHROMA_PORT: int = _get_int("chroma_port", 8000)

# ── Ollama / LLM provider selection ──────────────────────────────────────────
LLM_PROVIDER: str = _get("llm_provider", "openai").lower()
OLLAMA_URL: str = _get("ollama_url", "http://127.0.0.1:11434")
OLLAMA_MODEL: str = _get("ollama_model", "llama2")
OLLAMA_EMBEDDING_MODEL: str = _get("ollama_embedding_model", "text2vec")

# ── Database ──────────────────────────────────────────────────────────────────
DATABASE_URL: str = _get("database_url", "sqlite:///./habits.db")

# ── Memory Consolidation ──────────────────────────────────────────────────────
CONSOLIDATION_INTERVAL_SECONDS: int = _get_int("consolidation_interval_seconds", 1800)  # 30 min; < STM_TTL
CONSOLIDATION_WINDOW: int = _get_int("consolidation_window", 50)

# ── PFC ↔ Basal Ganglia Recurrent Loop ────────────────────────────────────────
MAX_PFC_LOOPS: int = _get_int("max_pfc_loops", 2)

# ── Rate limiting ─────────────────────────────────────────────────────────────
RATE_LIMIT_MAX_CALLS: int = _get_int("rate_limit_max_calls", 10)
RATE_LIMIT_WINDOW_SECONDS: int = _get_int("rate_limit_window_seconds", 60)

# ── LLM Retry ─────────────────────────────────────────────────────────────────
LLM_RETRY_ATTEMPTS: int = _get_int("llm_retry_attempts", 3)
LLM_RETRY_BACKOFF_MULTIPLIER: float = _get_float("llm_retry_backoff_multiplier", 0.5)
LLM_RETRY_MAX_WAIT: float = _get_float("llm_retry_max_wait", 10)

# ── Amygdala ──────────────────────────────────────────────────────────────────
AMYGDALA_MODEL: str = _get("amygdala_model", "gpt-4o-mini")
AMYGDALA_THRESHOLD_LOW: float = _get_float("amygdala_threshold_low", 0.2)
AMYGDALA_THRESHOLD_MID: float = _get_float("amygdala_threshold_mid", 0.5)
AMYGDALA_THRESHOLD_HIGH: float = _get_float("amygdala_threshold_high", 0.7)
AMYGDALA_TEMP_NEUTRAL: float = _get_float("amygdala_temp_neutral", 0.7)
AMYGDALA_TEMP_MILD: float = _get_float("amygdala_temp_mild", 0.5)
AMYGDALA_TEMP_STRESSED: float = _get_float("amygdala_temp_stressed", 0.3)
AMYGDALA_TEMP_CRISIS: float = _get_float("amygdala_temp_crisis", 0.1)

# ── PFC ───────────────────────────────────────────────────────────────────────
PFC_MODEL: str = _get("pfc_model", "gpt-4o")
PFC_CONSOLIDATION_THRESHOLD: float = _get_float("pfc_consolidation_threshold", 0.5)
PFC_STM_WINDOW: int = _get_int("pfc_stm_window", 10)
PFC_DEFAULT_TEMPERATURE: float = _get_float("pfc_default_temperature", 0.3)

# ── Basal Ganglia ─────────────────────────────────────────────────────────────
BG_MODEL: str = _get("bg_model", "gpt-4o-mini")

# ── Hippocampus ───────────────────────────────────────────────────────────────
HIPPOCAMPUS_THRESHOLD: float = _get_float("hippocampus_threshold", 0.5)
HIPPOCAMPUS_K: int = _get_int("hippocampus_k", 5)
HIPPOCAMPUS_SEARCH_TYPE: str = _get("hippocampus_search_type", "mmr")
HIPPOCAMPUS_COLLECTION: str = _get("hippocampus_collection", "episodic")

# ── Consolidator ──────────────────────────────────────────────────────────────
CONSOLIDATION_COLLECTION: str = _get("consolidation_collection", "episodic")
CONSOLIDATION_LLM_TEMPERATURE: float = _get_float("consolidation_llm_temperature", 0.3)
CONSOLIDATION_MODEL: str = _get("consolidation_model", "gpt-4o")
LTM_SUMMARIZATION_INTERVAL_SECONDS: int = _get_int("ltm_summarization_interval_seconds", 86400)  # daily
LTM_SUMMARIZATION_MIN_DOCS: int = _get_int("ltm_summarization_min_docs", 20)

# ── System Prompts (source of truth: config.yaml) ─────────────────────────────
AMYGDALA_SYSTEM_PROMPT: str = _get("amygdala_system_prompt", "")
BG_SYSTEM_PROMPT: str = _get("bg_system_prompt", "")
BG_GATE_SYSTEM_PROMPT: str = _get("bg_gate_system_prompt", "")
PFC_SYSTEM_PROMPT: str = _get("pfc_system_prompt", "")
