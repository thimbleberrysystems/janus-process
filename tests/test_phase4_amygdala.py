"""
tests/test_phase4_amygdala.py — Phase 4: Amygdala Agent test suite.

Unit tests (default, no external services) inject a mock LLM so no API
calls are made.  Integration tests (``@pytest.mark.integration``) call
the live LLM configured via .env.

Run unit tests only:
    pytest tests/test_phase4_amygdala.py -m "not integration"

Run all:
    pytest tests/test_phase4_amygdala.py
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agents.amygdala import _parse_score, amygdala_node
from models.brain_state import default_brain_state

# ── Helper ─────────────────────────────────────────────────────────────────────


def _mock_llm(score_str: str) -> MagicMock:
    """Return a mock LLM whose .invoke() yields *score_str* as content."""
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=score_str)
    return llm


# ── _parse_score unit tests ────────────────────────────────────────────────────


class TestParseScore:
    def test_plain_decimal(self):
        assert _parse_score("0.85") == pytest.approx(0.85)

    def test_with_trailing_text(self):
        assert _parse_score("0.3 (mildly emotional)") == pytest.approx(0.3)

    def test_integer_one(self):
        assert _parse_score("1") == pytest.approx(1.0)

    def test_integer_zero(self):
        assert _parse_score("0") == pytest.approx(0.0)

    def test_clamped_above_one(self):
        assert _parse_score("1.5") == pytest.approx(1.0)

    def test_clamped_below_zero(self):
        assert _parse_score("-0.5") == pytest.approx(0.0)

    def test_no_number_defaults_zero(self):
        assert _parse_score("neutral") == pytest.approx(0.0)

    def test_whitespace_only(self):
        assert _parse_score("   ") == pytest.approx(0.0)


# ── amygdala_node unit tests ───────────────────────────────────────────────────


class TestAmygdalaNode:
    def test_llm_mocked_returns_valid_score(self):
        """Mock LLM returns '0.85' — node must parse and store it."""
        state = default_brain_state("I just lost my job")
        result = amygdala_node(state, llm=_mock_llm("0.85"))
        assert result["emotional_weight"] == pytest.approx(0.85)

    def test_state_updated_correctly(self):
        """Only emotional_weight changes; all other fields are preserved."""
        state = default_brain_state("hello world")
        state["working_memory"] = ["earlier message"]
        state["retrieved_memories"] = ["old memory"]
        result = amygdala_node(state, llm=_mock_llm("0.2"))

        assert result["emotional_weight"] == pytest.approx(0.2)
        assert result["input"] == "hello world"
        assert result["working_memory"] == ["earlier message"]
        assert result["retrieved_memories"] == ["old memory"]
        assert result["should_consolidate"] is False
        assert result["final_response"] is None

    def test_score_clamped_when_llm_returns_out_of_range(self):
        """Values > 1.0 hallucinated by the LLM must be clamped to 1.0."""
        state = default_brain_state("test")
        result = amygdala_node(state, llm=_mock_llm("2.5"))
        assert result["emotional_weight"] == pytest.approx(1.0)

    def test_malformed_response_defaults_to_zero(self):
        """Non-numeric LLM response must not raise; defaults to 0.0."""
        state = default_brain_state("test")
        result = amygdala_node(state, llm=_mock_llm("I cannot determine the score"))
        assert result["emotional_weight"] == pytest.approx(0.0)

    def test_score_is_float(self):
        """emotional_weight must be a Python float."""
        state = default_brain_state("test input")
        result = amygdala_node(state, llm=_mock_llm("0.55"))
        assert isinstance(result["emotional_weight"], float)

    def test_llm_called_with_input_text(self):
        """Node must pass state['input'] to the LLM (not some other field)."""
        llm = _mock_llm("0.5")
        state = default_brain_state("unique canary text")
        amygdala_node(state, llm=llm)

        call_args = llm.invoke.call_args
        messages = call_args[0][0]  # first positional arg = messages list
        human_msg = next(m for m in messages if m.__class__.__name__ == "HumanMessage")
        assert "unique canary text" in human_msg.content


# ── Integration tests (require live LLM) ──────────────────────────────────────


@pytest.mark.integration
class TestAmygdalaIntegration:
    """Requires a running LLM (Ollama or configured OpenAI key)."""

    def test_neutral_input_scores_low(self):
        """Plain factual question should score below 0.3."""
        state = default_brain_state("What time is it?")
        result = amygdala_node(state)
        assert isinstance(result["emotional_weight"], float)
        assert 0.0 <= result["emotional_weight"] <= 1.0
        assert result["emotional_weight"] < 0.3

    def test_emotional_input_scores_high(self):
        """Clearly distressing input should score above 0.7."""
        state = default_brain_state("I just lost my job and I don't know what to do")
        result = amygdala_node(state)
        assert isinstance(result["emotional_weight"], float)
        assert result["emotional_weight"] > 0.7

    def test_score_is_float_in_range(self):
        """Score must be a float in [0.0, 1.0] for 10 varied inputs."""
        inputs = [
            "What is 2 + 2?",
            "I'm really excited about my promotion!",
            "The weather is cloudy today.",
            "My mother passed away last night.",
            "Can you help me write a function?",
            "I feel completely hopeless.",
            "What's the capital of France?",
            "I'm terrified about my surgery tomorrow.",
            "Set a reminder for 3pm.",
            "I haven't eaten in two days and I'm scared.",
        ]
        for text in inputs:
            result = amygdala_node(default_brain_state(text))
            score = result["emotional_weight"]
            assert isinstance(score, float), f"Score for {text!r} is not a float"
            assert 0.0 <= score <= 1.0, f"Score {score!r} out of range for {text!r}"
