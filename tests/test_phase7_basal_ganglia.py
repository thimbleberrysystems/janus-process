"""
tests/test_phase7_basal_ganglia.py — Phase 7: Basal Ganglia Agent test suite.

All unit tests use an in-memory SQLite DB and a mock LLM — no containers or
API calls required.

Integration tests (``@pytest.mark.integration``) call the live LLM.

Run unit tests only:
    pytest tests/test_phase7_basal_ganglia.py -m "not integration"

Run all:
    pytest tests/test_phase7_basal_ganglia.py
"""
from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock

import pytest

from agents.basal_ganglia import (
    _ensure_schema,
    basal_ganglia_node,
    register_habit,
)
from models.brain_state import default_brain_state

# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture()
def db() -> sqlite3.Connection:  # type: ignore[misc]
    """Fresh in-memory SQLite DB with habits schema."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)
    yield conn
    conn.close()


def _mock_llm_match(habit_id: int) -> MagicMock:
    """LLM that returns a specific habit ID (simulates a match)."""
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=str(habit_id))
    return llm


def _mock_llm_miss() -> MagicMock:
    """LLM that returns NONE (simulates no match)."""
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="NONE")
    return llm


# ── DB helper tests ────────────────────────────────────────────────────────────


class TestRegisterHabit:
    def test_register_habit(self, db):
        """Inserted habit must be retrievable with hit_count=0."""
        habit_id = register_habit("greet user", "Hello! How can I help?", conn=db)
        row = db.execute(
            "SELECT * FROM habits WHERE id = ?", (habit_id,)
        ).fetchone()
        assert row is not None
        assert row["pattern"] == "greet user"
        assert row["response"] == "Hello! How can I help?"
        assert row["hit_count"] == 0

    def test_register_multiple_habits(self, db):
        """Each inserted habit gets a distinct ID."""
        id1 = register_habit("greet", "hi", conn=db)
        id2 = register_habit("farewell", "bye", conn=db)
        assert id1 != id2

    def test_empty_table_on_fresh_db(self, db):
        """Fresh in-memory DB must contain no habits."""
        rows = db.execute("SELECT * FROM habits").fetchall()
        assert rows == []


# ── basal_ganglia_node unit tests ──────────────────────────────────────────────


class TestBasalGangliaNode:
    def test_unknown_pattern_miss(self, db):
        """Empty habits table → procedural_match must be None."""
        state = default_brain_state("hello there")
        result = basal_ganglia_node(state, llm=_mock_llm_miss(), conn=db)
        assert result["procedural_match"] is None

    def test_known_pattern_match(self, db):
        """Matching input → procedural_match is the habit's response string."""
        habit_id = register_habit(
            "greet user", "Hello! How can I help you today?", conn=db
        )
        state = default_brain_state("hello there")
        result = basal_ganglia_node(state, llm=_mock_llm_match(habit_id), conn=db)
        assert result["procedural_match"] == "Hello! How can I help you today?"

    def test_hit_count_increments(self, db):
        """Same habit matched twice → hit_count=2 in the DB."""
        habit_id = register_habit("greet user", "Hi there!", conn=db)
        state = default_brain_state("hey")

        basal_ganglia_node(state, llm=_mock_llm_match(habit_id), conn=db)
        basal_ganglia_node(state, llm=_mock_llm_match(habit_id), conn=db)

        row = db.execute(
            "SELECT hit_count FROM habits WHERE id = ?", (habit_id,)
        ).fetchone()
        assert row["hit_count"] == 2

    def test_hit_count_not_incremented_on_miss(self, db):
        """Miss must not touch hit_count."""
        habit_id = register_habit("greet user", "Hi!", conn=db)
        state = default_brain_state("unrelated input")

        basal_ganglia_node(state, llm=_mock_llm_miss(), conn=db)

        row = db.execute(
            "SELECT hit_count FROM habits WHERE id = ?", (habit_id,)
        ).fetchone()
        assert row["hit_count"] == 0

    def test_match_bypasses_ltm(self, db):
        """
        When a habit matches, procedural_match is set (non-None).
        The Thalamus uses this signal to skip the Hippocampus (LTM retrieval).
        Assert that the match signal is correctly surfaced.
        """
        habit_id = register_habit("farewell", "Goodbye! Have a great day!", conn=db)
        state = default_brain_state("goodbye")
        result = basal_ganglia_node(state, llm=_mock_llm_match(habit_id), conn=db)

        # non-None procedural_match = signal to skip hippocampus in the router
        assert result["procedural_match"] is not None

    def test_miss_passes_through_unchanged(self, db):
        """On a miss all state fields (including procedural_match) are unchanged."""
        state = default_brain_state("quantum mechanics")
        state["emotional_weight"] = 0.3
        state["working_memory"] = ["earlier message"]

        result = basal_ganglia_node(state, llm=_mock_llm_miss(), conn=db)

        assert result["procedural_match"] is None
        assert result["input"] == "quantum mechanics"
        assert result["emotional_weight"] == pytest.approx(0.3)
        assert result["working_memory"] == ["earlier message"]

    def test_hallucinated_id_treated_as_miss(self, db):
        """If LLM returns an ID that doesn't exist, treat as miss (no crash)."""
        register_habit("greet user", "Hi!", conn=db)
        state = default_brain_state("hello")

        # LLM hallucinated ID 9999 which doesn't exist
        result = basal_ganglia_node(state, llm=_mock_llm_match(9999), conn=db)
        assert result["procedural_match"] is None

    def test_procedural_match_is_only_changed_field_on_match(self, db):
        """On a match, only procedural_match differs from the input state."""
        habit_id = register_habit("greet", "Hello!", conn=db)
        state = default_brain_state("hi")
        state["emotional_weight"] = 0.1
        state["retrieved_memories"] = ["old memory"]

        result = basal_ganglia_node(state, llm=_mock_llm_match(habit_id), conn=db)

        diff = {k for k in state if result.get(k) != state.get(k)}
        assert diff == {"procedural_match"}

    def test_no_habits_skips_llm(self, db):
        """Empty habits table must not invoke the LLM at all."""
        llm = _mock_llm_miss()
        state = default_brain_state("anything")
        basal_ganglia_node(state, llm=llm, conn=db)
        llm.invoke.assert_not_called()


# ── Integration tests (require live LLM) ──────────────────────────────────────


@pytest.mark.integration
class TestBasalGangliaIntegration:
    """Requires a running LLM (Ollama or configured OpenAI key)."""

    def test_greeting_matches_greet_habit(self):
        """Live LLM should recognise 'hi there' as matching a 'greet user' habit."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        _ensure_schema(conn)
        register_habit("greet user", "Hello! How can I help you today?", conn=conn)

        state = default_brain_state("hi there")
        result = basal_ganglia_node(state, conn=conn)
        assert result["procedural_match"] == "Hello! How can I help you today?"

    def test_unrelated_input_returns_none(self):
        """Input unrelated to any habit must yield procedural_match=None."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        _ensure_schema(conn)
        register_habit("greet user", "Hello!", conn=conn)

        state = default_brain_state("explain quantum entanglement")
        result = basal_ganglia_node(state, conn=conn)
        assert result["procedural_match"] is None
