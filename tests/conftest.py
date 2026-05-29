"""
tests/conftest.py — shared pytest fixtures and configuration.
"""
import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers so pytest --markers lists them cleanly."""
    config.addinivalue_line(
        "markers",
        "integration: marks tests that require running Docker services "
        "(Redis on :6379, ChromaDB on :8000). Skip with: pytest -m 'not integration'",
    )


@pytest.fixture()
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Remove real secret keys from the environment for a test, so config
    values are fully controlled by the test's own patch.dict / monkeypatch calls.
    """
    for key in ("OPENAI_API_KEY", "LANGSMITH_API_KEY", "REDIS_URL",
                "CHROMA_PERSIST_DIR", "LANGSMITH_PROJECT",
                "SESSION_ID", "STM_TTL", "CHROMA_PORT", "DATABASE_URL"):
        monkeypatch.delenv(key, raising=False)
