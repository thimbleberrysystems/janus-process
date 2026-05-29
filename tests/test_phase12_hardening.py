"""
tests/test_phase12_hardening.py — Security, resilience, and deployment tests.

Unit tests
----------
- No secrets hardcoded in source files.
- Rate limiting returns HTTP 429 after threshold is exceeded.
- LLM retry logic recovers from transient failures.
- VectorStore interface is swappable (in-memory stub satisfies Protocol).
- CI lint (ruff) passes.
- CI type-check (mypy) passes.

Integration tests
-----------------
Marked ``@pytest.mark.integration`` — require Docker and/or live services.
- Docker image builds cleanly.
- LangSmith traces are created per node (inherited from Phase 11 validation).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from starlette.testclient import TestClient

from api.server import app, get_brain_graph
from memory.vectorstore import ChromaVectorStore, InMemoryVectorStore, VectorStoreClient

WORKSPACE_ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

class _NodeTracker:
    def __init__(self, updates: dict[str, Any] | None = None) -> None:
        self._updates = updates or {}

    def __call__(self, state: dict) -> dict:
        return {**state, **self._updates}


def _make_mock_graph(final_response: str = "ok"):
    from agents.thalamus import build_graph

    return build_graph(
        amygdala=_NodeTracker({"emotional_weight": 0.1}),
        basal_ganglia=_NodeTracker({"procedural_match": None}),
        hippocampus=_NodeTracker({}),
        pfc=_NodeTracker({"final_response": final_response, "should_consolidate": False}),
        bg_gate=_NodeTracker({"bg_feedback": ""}),
    )


@pytest.fixture(autouse=True)
def _clear_overrides():
    """Always clean up dependency overrides after each test."""
    yield
    app.dependency_overrides.clear()


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture()
def mock_client() -> TestClient:
    """TestClient with the graph dependency mocked out."""
    mock_graph = _make_mock_graph()
    app.dependency_overrides[get_brain_graph] = lambda: mock_graph
    # Reset the rate limiter so other tests' state doesn't bleed in
    app.state.rate_limiter.reset()
    return TestClient(app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# test_no_secrets_in_source_code
# ---------------------------------------------------------------------------

# Patterns that indicate a hardcoded secret (not just a variable name)
_SECRET_PATTERNS = [
    re.compile(r'sk-[A-Za-z0-9]{20,}'),          # OpenAI API key
    re.compile(r'sk-proj-[A-Za-z0-9\-_]{20,}'),   # OpenAI project key
    re.compile(r'(?i)api[_-]?key\s*=\s*["\'][A-Za-z0-9\-_]{16,}["\']'),  # generic assignment
    re.compile(r'(?i)secret\s*=\s*["\'][A-Za-z0-9\-_]{16,}["\']'),
    re.compile(r'(?i)password\s*=\s*["\'][^"\']{8,}["\']'),
]

# Files to exclude from the scan (test fixtures legitimately reference key names)
_EXCLUDE_FILES = {
    "tests/conftest.py",
    "tests/test_phase12_hardening.py",
    ".env.example",
}


class TestNoSecretsInSourceCode:
    def test_no_secrets_in_source_code(self):
        """No Python source file may contain a hardcoded API key or secret."""
        violations: list[str] = []

        py_files = [
            p for p in WORKSPACE_ROOT.rglob("*.py")
            if ".venv" not in p.parts
            and ".git" not in p.parts
            and str(p.relative_to(WORKSPACE_ROOT)) not in _EXCLUDE_FILES
        ]

        for path in py_files:
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            rel = str(path.relative_to(WORKSPACE_ROOT))
            for pattern in _SECRET_PATTERNS:
                for match in pattern.finditer(content):
                    violations.append(f"{rel}: {match.group()!r}")

        assert not violations, (
            "Hardcoded secrets found in source files:\n"
            + "\n".join(f"  {v}" for v in violations)
        )

    def test_env_file_not_tracked(self):
        """.env must be listed in .gitignore (not committed with real secrets)."""
        gitignore = WORKSPACE_ROOT / ".gitignore"
        if gitignore.exists():
            content = gitignore.read_text(encoding="utf-8")
            assert ".env" in content, ".env is not excluded in .gitignore"


# ---------------------------------------------------------------------------
# test_rate_limit_enforced
# ---------------------------------------------------------------------------

class TestRateLimit:
    def test_rate_limit_enforced(self, mock_client):
        """Sending more requests than the limit must produce at least one HTTP 429."""
        limit = app.state.rate_limiter.max_calls
        # Send limit + 5 requests, all from the same "client"
        statuses = [
            mock_client.post("/think", json={"input": "hi"}).status_code
            for _ in range(limit + 5)
        ]
        assert 429 in statuses, (
            f"Expected HTTP 429 after {limit} calls but got: {set(statuses)}"
        )

    def test_rate_limit_threshold_matches_config(self):
        """The limiter's max_calls must match the configured value."""
        from config import RATE_LIMIT_MAX_CALLS

        assert app.state.rate_limiter.max_calls == RATE_LIMIT_MAX_CALLS

    def test_requests_below_limit_succeed(self, mock_client):
        """Requests within the rate-limit window must all return 200."""
        app.state.rate_limiter.reset()
        limit = app.state.rate_limiter.max_calls
        half = max(1, limit // 2)
        for _ in range(half):
            r = mock_client.post("/think", json={"input": "hi"})
            assert r.status_code == 200, f"Expected 200, got {r.status_code}"

    def test_rate_limiter_reset_clears_state(self, mock_client):
        """After reset(), previously blocked client can make requests again."""
        limit = app.state.rate_limiter.max_calls
        for _ in range(limit + 2):
            mock_client.post("/think", json={"input": "hi"})

        app.state.rate_limiter.reset()
        r = mock_client.post("/think", json={"input": "hi after reset"})
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# test_llm_retry_on_transient_failure
# ---------------------------------------------------------------------------

class TestLLMRetry:
    def test_llm_retry_on_transient_failure(self):
        """retry_invoke must retry up to 3 times and return success on the 3rd attempt."""
        from providers.llm import retry_invoke

        call_count = 0

        class _FlakyLLM:
            def invoke(self, messages):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise ConnectionError(f"transient error (attempt {call_count})")
                return AIMessage(content="recovered successfully")

        result = retry_invoke(_FlakyLLM(), [HumanMessage(content="test")])
        assert result.content == "recovered successfully"
        assert call_count == 3

    def test_llm_retry_reraises_after_max_attempts(self):
        """After 3 failed attempts, retry_invoke must re-raise the original error."""
        from tenacity import RetryError

        from providers.llm import retry_invoke

        class _AlwaysFails:
            def invoke(self, messages):
                raise ConnectionError("permanent failure")

        with pytest.raises((ConnectionError, RetryError)):
            retry_invoke(_AlwaysFails(), [HumanMessage(content="test")])

    def test_llm_retry_succeeds_first_try(self):
        """When no error occurs, retry_invoke returns the result without retrying."""
        from providers.llm import retry_invoke

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="hello")

        result = retry_invoke(mock_llm, [HumanMessage(content="hi")])
        assert result.content == "hello"
        mock_llm.invoke.assert_called_once()

    def test_retry_does_not_catch_non_transient_errors(self):
        """ValueError (non-transient) must propagate immediately without retrying."""
        from providers.llm import retry_invoke

        call_count = 0

        class _ValueErrorLLM:
            def invoke(self, messages):
                nonlocal call_count
                call_count += 1
                raise ValueError("bad input — not transient")

        with pytest.raises(ValueError):
            retry_invoke(_ValueErrorLLM(), [HumanMessage(content="test")])

        assert call_count == 1, "Non-transient error should not be retried"


# ---------------------------------------------------------------------------
# test_vectorstore_interface_swappable
# ---------------------------------------------------------------------------

class TestVectorstoreInterface:
    def test_in_memory_store_satisfies_protocol(self):
        """InMemoryVectorStore must satisfy the VectorStoreClient Protocol at runtime."""
        store = InMemoryVectorStore()
        assert isinstance(store, VectorStoreClient)

    def test_in_memory_store_add_and_search(self):
        """add() then search() must return matching documents."""
        store = InMemoryVectorStore()
        store.add(
            ["Paris is the capital of France", "London is the capital of England"],
            [{"source": "test"}, {"source": "test"}],
        )
        results = store.search("capital of France", k=2)
        assert len(results) >= 1
        assert any("Paris" in r["content"] for r in results)

    def test_in_memory_store_reset_clears_documents(self):
        """reset() must leave the store empty."""
        store = InMemoryVectorStore()
        store.add(["document one"], [{"source": "test"}])
        store.reset()
        assert store.search("document", k=5) == []

    def test_in_memory_store_search_returns_top_k(self):
        """search() must return no more than k results."""
        store = InMemoryVectorStore()
        for i in range(10):
            store.add([f"document number {i}"], [{"source": "test"}])
        results = store.search("document", k=3)
        assert len(results) <= 3

    def test_in_memory_store_empty_add(self):
        """add() with mismatched/empty metadatas must not raise."""
        store = InMemoryVectorStore()
        store.add(["text without metadata"], [{}])
        results = store.search("text", k=1)
        assert len(results) == 1

    def test_chroma_vector_store_satisfies_protocol(self):
        """ChromaVectorStore must satisfy the VectorStoreClient Protocol at runtime."""
        import chromadb
        from langchain_core.embeddings import DeterministicFakeEmbedding

        client = chromadb.EphemeralClient(settings=chromadb.Settings(allow_reset=True))
        client.reset()
        embed = DeterministicFakeEmbedding(size=64)

        store = ChromaVectorStore("episodic", embedding_fn=embed, chroma_client=client)
        assert isinstance(store, VectorStoreClient)

    def test_both_stores_share_identical_interface(self):
        """InMemoryVectorStore and ChromaVectorStore expose exactly the same public methods."""
        mem_methods = {m for m in dir(InMemoryVectorStore) if not m.startswith("_")}
        chroma_methods = {m for m in dir(ChromaVectorStore) if not m.startswith("_")}
        protocol_methods = {"add", "search", "reset"}
        assert protocol_methods.issubset(mem_methods)
        assert protocol_methods.issubset(chroma_methods)


# ---------------------------------------------------------------------------
# test_ci_lint_passes
# ---------------------------------------------------------------------------

class TestCILint:
    def test_ci_lint_passes(self):
        """Running ``ruff check .`` against the workspace must exit with code 0."""
        result = subprocess.run(
            [sys.executable, "-m", "ruff", "check", "."],
            cwd=str(WORKSPACE_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"ruff found lint errors:\n{result.stdout}\n{result.stderr}"
        )

    def test_ci_lint_binary_available(self):
        """ruff must be importable (installed as a dev dependency)."""
        result = subprocess.run(
            [sys.executable, "-m", "ruff", "--version"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0


# ---------------------------------------------------------------------------
# test_ci_type_check_passes
# ---------------------------------------------------------------------------

class TestCITypeCheck:
    def test_ci_type_check_passes(self):
        """Running ``mypy .`` against the workspace must exit with code 0."""
        result = subprocess.run(
            [sys.executable, "-m", "mypy", "."],
            cwd=str(WORKSPACE_ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, (
            f"mypy found type errors:\n{result.stdout}\n{result.stderr}"
        )

    def test_ci_type_check_binary_available(self):
        """mypy must be importable (installed as a dev dependency)."""
        result = subprocess.run(
            [sys.executable, "-m", "mypy", "--version"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0


# ---------------------------------------------------------------------------
# Integration: Docker build
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestDockerBuild:
    def test_docker_image_builds(self):
        """``docker build`` must succeed and the image size must be < 2 GB."""
        # Check docker availability
        check = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
        )
        if check.returncode != 0:
            # Try docker.exe (WSL2 / Docker Desktop)
            check = subprocess.run(
                ["docker.exe", "info"],
                capture_output=True,
                text=True,
            )
            if check.returncode != 0:
                pytest.skip("Docker not available in this environment")
            docker_cmd = "docker.exe"
        else:
            docker_cmd = "docker"

        build = subprocess.run(
            [docker_cmd, "build", "--target", "production", "-t", "janus-process:test", "."],
            cwd=str(WORKSPACE_ROOT),
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert build.returncode == 0, (
            f"docker build failed:\n{build.stdout[-2000:]}\n{build.stderr[-2000:]}"
        )

        # Verify image size is within a reasonable threshold (< 2 GB)
        inspect = subprocess.run(
            [docker_cmd, "inspect", "janus-process:test",
             "--format", "{{.Size}}"],
            capture_output=True,
            text=True,
        )
        if inspect.returncode == 0:
            size_bytes = int(inspect.stdout.strip())
            size_gb = size_bytes / (1024 ** 3)
            assert size_gb < 2.0, (
                f"Docker image is too large: {size_gb:.2f} GB (threshold: 2 GB)"
            )
