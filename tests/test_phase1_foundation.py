"""
tests/test_phase1_foundation.py

Phase 1 — Foundation & Scaffolding
Tests verify:
  - BrainState schema structure and defaults
  - Runtime validation (validate_brain_state)
  - config.py reads environment variables correctly
  - Docker services are reachable (integration, marked)
  - Stub LangGraph passes state through unchanged
"""
import importlib
import socket

import pytest

from models.brain_state import (
    default_brain_state,
    validate_brain_state,
)

# ── 1. BrainState schema ──────────────────────────────────────────────────────

class TestBrainStateSchema:
    def test_all_fields_present(self) -> None:
        """default_brain_state returns a dict with every BrainState field."""
        state = default_brain_state("hello")
        expected_keys = {
            "input", "emotional_weight", "llm_temperature", "emotional_directive",
            "working_memory", "retrieved_memories", "procedural_match",
            "pfc_proposals", "bg_feedback", "loop_count",
            "final_response", "should_consolidate",
        }
        assert expected_keys == set(state.keys())

    def test_field_types_match_schema(self) -> None:
        """Each field has the correct Python type after default construction."""
        state = default_brain_state("hello")
        assert isinstance(state["input"], str)
        assert isinstance(state["emotional_weight"], float)
        assert isinstance(state["working_memory"], list)
        assert isinstance(state["retrieved_memories"], list)
        assert state["procedural_match"] is None
        assert state["final_response"] is None
        assert isinstance(state["should_consolidate"], bool)

    def test_default_values(self) -> None:
        """Defaults are zeroed / empty as specified in design."""
        state = default_brain_state("test input")
        assert state["input"] == "test input"
        assert state["emotional_weight"] == 0.0
        assert state["working_memory"] == []
        assert state["retrieved_memories"] == []
        assert state["should_consolidate"] is False

    def test_input_preserved(self) -> None:
        """The user_input argument is stored verbatim in the 'input' field."""
        text = "What is the capital of France?"
        state = default_brain_state(text)
        assert state["input"] == text


# ── 2. validate_brain_state ───────────────────────────────────────────────────

class TestValidateBrainState:
    def test_valid_state_passes_through(self) -> None:
        """A fully-populated state dict is returned unchanged."""
        state = default_brain_state("hello")
        result = validate_brain_state(dict(state))
        assert result["input"] == "hello"

    def test_missing_input_raises_type_error(self) -> None:
        """Omitting 'input' raises TypeError naming the missing field."""
        bad = {
            "emotional_weight": 0.5,
            "working_memory": [],
            "retrieved_memories": [],
            "should_consolidate": False,
        }
        with pytest.raises(TypeError, match="input"):
            validate_brain_state(bad)

    def test_multiple_missing_fields_raises_type_error(self) -> None:
        """Omitting several required fields is reported in one error."""
        with pytest.raises(TypeError, match="missing required fields"):
            validate_brain_state({})

    def test_emotional_weight_out_of_range_raises_value_error(self) -> None:
        """emotional_weight outside [0.0, 1.0] raises ValueError."""
        state = dict(default_brain_state("hi"))
        state["emotional_weight"] = 1.5
        with pytest.raises(ValueError, match="emotional_weight"):
            validate_brain_state(state)

    def test_negative_emotional_weight_raises_value_error(self) -> None:
        state = dict(default_brain_state("hi"))
        state["emotional_weight"] = -0.1
        with pytest.raises(ValueError, match="emotional_weight"):
            validate_brain_state(state)

    def test_wrong_type_working_memory_raises_value_error(self) -> None:
        state = dict(default_brain_state("hi"))
        state["working_memory"] = "not a list"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="working_memory"):
            validate_brain_state(state)

    def test_wrong_type_should_consolidate_raises_value_error(self) -> None:
        state = dict(default_brain_state("hi"))
        state["should_consolidate"] = "yes"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="should_consolidate"):
            validate_brain_state(state)


# ── 3. config.py reads environment variables ──────────────────────────────────

class TestConfigLoadsEnvVars:
    def test_reads_openai_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-123")
        import config
        importlib.reload(config)
        assert config.OPENAI_API_KEY == "sk-test-key-123"

    def test_reads_redis_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REDIS_URL", "redis://testhost:6380")
        import config
        importlib.reload(config)
        assert config.REDIS_URL == "redis://testhost:6380"

    def test_reads_chroma_persist_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CHROMA_PERSIST_DIR", "/tmp/test_chroma")
        import config
        importlib.reload(config)
        assert config.CHROMA_PERSIST_DIR == "/tmp/test_chroma"

    def test_reads_langsmith_project(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LANGSMITH_PROJECT", "my-test-project")
        import config
        importlib.reload(config)
        assert config.LANGSMITH_PROJECT == "my-test-project"

    def test_defaults_when_env_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Fallback defaults are sensible when env vars are not set."""
        for key in ("REDIS_URL", "CHROMA_PERSIST_DIR", "LANGSMITH_PROJECT",
                    "STM_TTL", "CHROMA_PORT"):
            monkeypatch.delenv(key, raising=False)
        import config
        importlib.reload(config)
        assert config.REDIS_URL == "redis://localhost:6379"
        assert config.CHROMA_PERSIST_DIR == "./chroma_db"
        assert config.LANGSMITH_PROJECT == "janus-process"
        assert config.STM_TTL == 3600
        assert config.CHROMA_PORT == 8000

    def test_stm_ttl_cast_to_int(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STM_TTL", "7200")
        import config
        importlib.reload(config)
        assert config.STM_TTL == 7200
        assert isinstance(config.STM_TTL, int)


# ── 4. Docker services reachable (integration) ────────────────────────────────

@pytest.mark.integration
class TestDockerServicesReachable:
    def _can_connect(self, host: str, port: int, timeout: float = 2.0) -> bool:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            return False

    def test_redis_reachable(self) -> None:
        """Redis must be listening on localhost:6379."""
        assert self._can_connect("localhost", 6379), (
            "Redis not reachable on localhost:6379 — is Docker running? "
            "Run: docker compose up -d"
        )

    def test_chromadb_reachable(self) -> None:
        """ChromaDB must be listening on localhost:8000."""
        assert self._can_connect("localhost", 8000), (
            "ChromaDB not reachable on localhost:8000 — is Docker running? "
            "Run: docker compose up -d"
        )


# ── 5. Stub LangGraph passes state through unchanged ─────────────────────────

class _PassthroughGraph:
    """Minimal fake compiled graph: returns state dict unchanged."""
    def invoke(self, state: dict) -> dict:
        return dict(state)


class TestStubGraph:
    def test_returns_brain_state(self) -> None:
        """think() returns a dict with all BrainState keys."""
        from unittest.mock import patch

        import main
        from main import think
        with patch.object(main, "brain", _PassthroughGraph()):
            result = think("hello world")
        assert isinstance(result, dict)
        assert "input" in result
        assert "final_response" in result

    def test_input_preserved(self) -> None:
        """The input field in the returned state matches what was passed in."""
        from unittest.mock import patch

        import main
        from main import think
        with patch.object(main, "brain", _PassthroughGraph()):
            result = think("stub test input")
        assert result["input"] == "stub test input"

    def test_defaults_unchanged_by_stub(self) -> None:
        """think() with a passthrough graph preserves all default field values."""
        from unittest.mock import patch

        import main
        from main import think
        with patch.object(main, "brain", _PassthroughGraph()):
            result = think("anything")
        assert result["emotional_weight"] == 0.0
        assert result["working_memory"] == []
        assert result["retrieved_memories"] == []
        assert result["procedural_match"] is None
        assert result["final_response"] is None
        assert result["should_consolidate"] is False

    def test_graph_compiles(self) -> None:
        """build_graph() must return a compiled runnable without raising."""
        from main import build_graph
        graph = build_graph()
        assert graph is not None

    def test_multiple_turns_are_independent(self) -> None:
        """Each call to default_brain_state() produces an independent dict — no shared state."""
        s1 = default_brain_state("first")
        s2 = default_brain_state("second")
        assert s1["input"] == "first"
        assert s2["input"] == "second"
        # Mutating one must not affect the other (lists must be distinct objects)
        s1["working_memory"].append("item")
        assert s2["working_memory"] == []
