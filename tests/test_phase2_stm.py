"""
tests/test_phase2_stm.py — Phase 2: Short-Term Memory test suite.

Unit tests mock Redis; integration tests (marked `integration`) require a
real Redis on localhost:6379 (provided by docker compose).

Run only unit tests:
    pytest tests/test_phase2_stm.py -m "not integration"

Run all including integration:
    pytest tests/test_phase2_stm.py
"""
import time
import uuid

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unique_session() -> str:
    """Return a unique session ID to isolate each integration test."""
    return f"test-stm-{uuid.uuid4().hex}"


# ---------------------------------------------------------------------------
# Unit tests  (no real Redis required)
# ---------------------------------------------------------------------------

class TestSTMUnit:
    """Tests that fully mock the Redis layer."""

    def test_add_and_read_message(self, mocker) -> None:
        """write one message; assert get_short_term_memory() returns it."""
        mock_history = mocker.MagicMock()
        from langchain_core.messages import HumanMessage
        mock_history.messages = [HumanMessage(content="hello")]
        mock_history.redis_client.ping.return_value = True

        mocker.patch(
            "memory.short_term.RedisChatMessageHistory",
            return_value=mock_history,
        )

        from memory.short_term import add_to_short_term, get_short_term_memory

        add_to_short_term("hello", session_id="test-session")
        result = get_short_term_memory(session_id="test-session")

        assert result == ["hello"]

    def test_window_limit(self, mocker) -> None:
        """write 20 messages; assert get_short_term_memory(n=5) returns only the last 5."""
        from langchain_core.messages import HumanMessage

        all_messages = [HumanMessage(content=f"msg-{i}") for i in range(20)]
        mock_history = mocker.MagicMock()
        mock_history.messages = all_messages
        mock_history.redis_client.ping.return_value = True

        mocker.patch(
            "memory.short_term.RedisChatMessageHistory",
            return_value=mock_history,
        )

        from memory.short_term import get_short_term_memory

        result = get_short_term_memory(5, session_id="test-session")

        assert len(result) == 5
        assert result == [f"msg-{i}" for i in range(15, 20)]

    def test_empty_buffer_returns_empty_list(self, mocker) -> None:
        """fresh session; assert get_short_term_memory() returns []."""
        mock_history = mocker.MagicMock()
        mock_history.messages = []
        mock_history.redis_client.ping.return_value = True

        mocker.patch(
            "memory.short_term.RedisChatMessageHistory",
            return_value=mock_history,
        )

        from memory.short_term import get_short_term_memory

        result = get_short_term_memory(session_id="empty-session")

        assert result == []

    def test_redis_unavailable_raises_gracefully(self, mocker) -> None:
        """point to bad Redis URL; assert a clear ConnectionError is raised."""
        import redis as redis_lib

        mocker.patch(
            "memory.short_term.RedisChatMessageHistory",
            side_effect=redis_lib.exceptions.ConnectionError("Connection refused"),
        )

        from memory.short_term import get_short_term_memory

        with pytest.raises(ConnectionError, match="Cannot connect to Redis"):
            get_short_term_memory(session_id="bad-session")


# ---------------------------------------------------------------------------
# Integration tests  (require running Redis on localhost:6379)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestSTMIntegration:
    """Tests that hit a real Redis instance."""

    def test_add_and_read_message_real(self) -> None:
        """write one message; assert it is returned by get_short_term_memory()."""
        import importlib

        import memory.short_term as stm_mod
        importlib.reload(stm_mod)

        session = _unique_session()
        stm_mod.add_to_short_term("integration hello", session_id=session)
        result = stm_mod.get_short_term_memory(session_id=session)
        stm_mod.clear_short_term(session_id=session)

        assert "integration hello" in result

    def test_window_limit_real(self) -> None:
        """write 20 messages; assert get_short_term_memory(n=5) returns only 5."""
        import importlib

        import memory.short_term as stm_mod
        importlib.reload(stm_mod)

        session = _unique_session()
        for i in range(20):
            stm_mod.add_to_short_term(f"msg-{i}", session_id=session)

        result = stm_mod.get_short_term_memory(5, session_id=session)
        stm_mod.clear_short_term(session_id=session)

        assert len(result) == 5
        assert result[-1] == "msg-19"

    def test_ttl_expiry(self) -> None:
        """write message with TTL=1s; sleep 2s; assert buffer is empty."""
        import importlib

        import memory.short_term as stm_mod
        importlib.reload(stm_mod)

        session = _unique_session()
        stm_mod.add_to_short_term("expires soon", session_id=session, ttl=1)

        time.sleep(2)

        result = stm_mod.get_short_term_memory(session_id=session, ttl=1)
        assert result == []

    def test_empty_buffer_returns_empty_list_real(self) -> None:
        """fresh session; assert get_short_term_memory() returns []."""
        import importlib

        import memory.short_term as stm_mod
        importlib.reload(stm_mod)

        session = _unique_session()
        result = stm_mod.get_short_term_memory(session_id=session)

        assert result == []

    def test_redis_unavailable_raises_gracefully_real(self, monkeypatch) -> None:
        """point to bad Redis URL; assert ConnectionError is raised."""
        import memory.short_term as stm_mod

        # Patch the module-level variable directly; reload would not re-execute
        # config.py's module-level code, so the URL would stay at the real Redis.
        monkeypatch.setattr(stm_mod, "REDIS_URL", "redis://127.0.0.1:19999")

        with pytest.raises(ConnectionError, match="Cannot connect to Redis"):
            stm_mod.get_short_term_memory(session_id="bad")
