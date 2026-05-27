"""
tests/test_phase6_pfc.py — Phase 6: PFC Agent test suite.

Unit tests inject a mock LLM — no API calls needed.
Integration tests (``@pytest.mark.integration``) call the live LLM.

Run unit tests only:
    pytest tests/test_phase6_pfc.py -m "not integration"

Run all:
    pytest tests/test_phase6_pfc.py
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agents.pfc import CONSOLIDATION_THRESHOLD, _build_context_block, pfc_node
from models.brain_state import default_brain_state

# ── Helper ─────────────────────────────────────────────────────────────────────


def _mock_llm(response_text: str = "Here is my response.") -> MagicMock:
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=response_text)
    return llm


def _state_with(**overrides):
    state = default_brain_state("test input")
    state.update(overrides)
    return state


# ── _build_context_block unit tests ───────────────────────────────────────────


class TestBuildContextBlock:
    def test_contains_input(self):
        state = default_brain_state("What is consciousness?")
        block = _build_context_block(state)
        assert "What is consciousness?" in block

    def test_contains_working_memory(self):
        state = _state_with(
            input="follow-up question",
            working_memory=["earlier turn A", "earlier turn B"],
        )
        block = _build_context_block(state)
        assert "earlier turn A" in block
        assert "earlier turn B" in block

    def test_contains_retrieved_memories(self):
        state = _state_with(
            input="something",
            retrieved_memories=["fact about Paris", "note about Rome"],
        )
        block = _build_context_block(state)
        assert "fact about Paris" in block
        assert "note about Rome" in block

    def test_contains_procedural_match(self):
        state = _state_with(input="hello", procedural_match="Greet the user warmly.")
        block = _build_context_block(state)
        assert "Greet the user warmly." in block

    def test_empty_context_does_not_crash(self):
        state = default_brain_state("anything")
        # All optional context fields are empty/None by default
        block = _build_context_block(state)
        assert "anything" in block  # input still present


# ── pfc_node unit tests ────────────────────────────────────────────────────────


class TestPfcNode:
    def test_final_response_is_populated(self):
        """Mock LLM reply must appear in state['final_response']."""
        state = default_brain_state("What time is it?")
        result = pfc_node(state, llm=_mock_llm("It is 3pm."))
        assert result["final_response"] == "It is 3pm."

    def test_final_response_is_non_empty_string(self):
        """final_response must always be a non-empty string."""
        state = default_brain_state("hello")
        result = pfc_node(state, llm=_mock_llm("Hello back!"))
        assert isinstance(result["final_response"], str)
        assert len(result["final_response"]) > 0

    def test_should_consolidate_true_when_high_emotion(self):
        """emotional_weight=0.8 (above threshold) → should_consolidate=True."""
        state = _state_with(input="I'm devastated", emotional_weight=0.8)
        result = pfc_node(state, llm=_mock_llm("I'm sorry to hear that."))
        assert result["should_consolidate"] is True

    def test_should_consolidate_false_when_low_emotion(self):
        """emotional_weight=0.2 (below threshold) → should_consolidate=False."""
        state = _state_with(input="What is 2+2?", emotional_weight=0.2)
        result = pfc_node(state, llm=_mock_llm("It is 4."))
        assert result["should_consolidate"] is False

    def test_should_consolidate_true_at_exact_threshold(self):
        """emotional_weight exactly at threshold must consolidate."""
        state = _state_with(input="test", emotional_weight=CONSOLIDATION_THRESHOLD)
        result = pfc_node(state, llm=_mock_llm("ok"))
        assert result["should_consolidate"] is True

    def test_all_context_injected_into_prompt(self):
        """working_memory, retrieved_memories, and procedural_match must all
        appear in the HumanMessage sent to the LLM."""
        llm = _mock_llm("response")
        state = _state_with(
            input="current question",
            working_memory=["stm_token_xyz"],
            retrieved_memories=["ltm_token_abc"],
            procedural_match="procedural_token_def",
        )
        pfc_node(state, llm=llm)

        call_args = llm.invoke.call_args
        messages = call_args[0][0]
        human_content = next(
            m.content for m in messages if m.__class__.__name__ == "HumanMessage"
        )
        assert "stm_token_xyz" in human_content
        assert "ltm_token_abc" in human_content
        assert "procedural_token_def" in human_content
        assert "current question" in human_content

    def test_empty_context_does_not_crash(self):
        """Node must not raise when working_memory, retrieved_memories, and
        procedural_match are all empty/None."""
        state = default_brain_state("simple question")
        result = pfc_node(state, llm=_mock_llm("Simple answer."))
        assert result["final_response"] == "Simple answer."

    def test_other_state_fields_unchanged(self):
        """Fields other than final_response and should_consolidate must not change."""
        state = _state_with(
            input="hello",
            emotional_weight=0.3,
            working_memory=["msg"],
            retrieved_memories=["mem"],
            procedural_match=None,
        )
        result = pfc_node(state, llm=_mock_llm("hi"))

        assert result["input"] == "hello"
        assert result["emotional_weight"] == pytest.approx(0.3)
        assert result["working_memory"] == ["msg"]
        assert result["retrieved_memories"] == ["mem"]
        assert result["procedural_match"] is None

    def test_whitespace_trimmed_from_response(self):
        """Leading/trailing whitespace in LLM output must be stripped."""
        state = default_brain_state("test")
        result = pfc_node(state, llm=_mock_llm("  trimmed response  \n"))
        assert result["final_response"] == "trimmed response"


# ── Integration tests (require live LLM) ──────────────────────────────────────


@pytest.mark.integration
class TestPfcIntegration:
    """Requires a running LLM (Ollama or configured OpenAI key)."""

    def test_final_response_is_populated_live(self):
        """Live LLM must return a non-empty final_response."""
        state = default_brain_state("What is the capital of France?")
        result = pfc_node(state)
        assert isinstance(result["final_response"], str)
        assert len(result["final_response"]) > 0

    def test_response_references_retrieved_memory(self):
        """Key fact in retrieved_memories must influence the live response."""
        state = _state_with(
            input="What do you know about the janus_canary_fact?",
            retrieved_memories=[
                "The janus_canary_fact is that blue whales are the largest animals."
            ],
        )
        result = pfc_node(state)
        # The response should reference something from the injected memory
        assert isinstance(result["final_response"], str)
        assert len(result["final_response"]) > 0
