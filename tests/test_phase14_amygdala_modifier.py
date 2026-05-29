"""
tests/test_phase14_amygdala_modifier.py — Phase 14: Amygdala Global State Modifier.

Verifies that the Amygdala now derives ``llm_temperature`` and
``emotional_directive`` from ``emotional_weight``, and that the PFC uses
those values when building its LLM call.

All tests are pure unit tests — no API calls, no external services.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agents.amygdala import _derive_modifiers, amygdala_node
from agents.pfc import pfc_node
from models.brain_state import default_brain_state

# ── Helpers ────────────────────────────────────────────────────────────────────

def _mock_llm(content: str) -> MagicMock:
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=content)
    return llm


# ── _derive_modifiers unit tests ───────────────────────────────────────────────

class TestDeriveModifiers:
    def test_neutral_gives_high_temperature(self):
        temp, directive = _derive_modifiers(0.1)
        assert temp >= 0.6
        assert directive == ""

    def test_neutral_boundary_below_0_2(self):
        temp, directive = _derive_modifiers(0.19)
        assert temp >= 0.6
        assert directive == ""

    def test_mild_gives_medium_temperature(self):
        temp, _ = _derive_modifiers(0.3)
        assert 0.3 < temp < 0.7

    def test_mild_gives_non_empty_directive(self):
        _, directive = _derive_modifiers(0.3)
        assert directive != ""

    def test_stressed_gives_low_temperature(self):
        temp, _ = _derive_modifiers(0.6)
        assert temp <= 0.4

    def test_stressed_directive_mentions_feelings(self):
        _, directive = _derive_modifiers(0.6)
        assert "feelings" in directive.lower() or "stress" in directive.lower()

    def test_crisis_gives_lowest_temperature(self):
        temp, _ = _derive_modifiers(0.9)
        assert temp <= 0.2

    def test_crisis_directive_mentions_distress(self):
        _, directive = _derive_modifiers(0.9)
        assert "distress" in directive.lower() or "compassion" in directive.lower()

    def test_temperature_monotonically_decreasing(self):
        weights = [0.1, 0.3, 0.6, 0.9]
        temps = [_derive_modifiers(w)[0] for w in weights]
        assert temps == sorted(temps, reverse=True)

    def test_temperature_in_valid_range(self):
        for w in [0.0, 0.2, 0.5, 0.7, 1.0]:
            temp, _ = _derive_modifiers(w)
            assert 0.0 <= temp <= 1.0


# ── amygdala_node integration with new fields ──────────────────────────────────

class TestAmygdalaNodeModifiers:
    def test_node_sets_llm_temperature(self):
        state = default_brain_state("hello")
        result = amygdala_node(state, llm=_mock_llm("0.1"))
        assert "llm_temperature" in result
        assert isinstance(result["llm_temperature"], float)

    def test_node_sets_emotional_directive(self):
        state = default_brain_state("hello")
        result = amygdala_node(state, llm=_mock_llm("0.1"))
        assert "emotional_directive" in result
        assert isinstance(result["emotional_directive"], str)

    def test_neutral_score_sets_high_temperature(self):
        state = default_brain_state("what time is it?")
        result = amygdala_node(state, llm=_mock_llm("0.05"))
        assert result["llm_temperature"] >= 0.6

    def test_crisis_score_sets_low_temperature(self):
        state = default_brain_state("I want to end it all")
        result = amygdala_node(state, llm=_mock_llm("0.95"))
        assert result["llm_temperature"] <= 0.2

    def test_neutral_score_sets_empty_directive(self):
        state = default_brain_state("what time is it?")
        result = amygdala_node(state, llm=_mock_llm("0.1"))
        assert result["emotional_directive"] == ""

    def test_crisis_score_sets_non_empty_directive(self):
        state = default_brain_state("I'm in crisis")
        result = amygdala_node(state, llm=_mock_llm("0.85"))
        assert result["emotional_directive"] != ""

    def test_other_state_fields_unchanged(self):
        state = default_brain_state("hi")
        state["working_memory"] = ["prev message"]
        result = amygdala_node(state, llm=_mock_llm("0.4"))
        assert result["working_memory"] == ["prev message"]
        assert result["input"] == "hi"


# ── PFC uses state temperature and directive ───────────────────────────────────

class TestPfcConsumesModifiers:
    def _run_pfc(self, temperature: float, directive: str) -> tuple[MagicMock, object]:
        """Run pfc_node with given modifiers; return (mock_llm, result)."""
        state = default_brain_state("test input")
        state["llm_temperature"] = temperature
        state["emotional_directive"] = directive

        mock_llm = _mock_llm("some response")

        with patch("agents.pfc.get_llm", return_value=mock_llm) as mock_get_llm:
            result = pfc_node(state, llm=None)
            return mock_get_llm, result

    def test_pfc_uses_state_temperature(self):
        state = default_brain_state("test input")
        state["llm_temperature"] = 0.1

        captured_temps: list[float] = []

        def fake_get_llm(model_name: str | None = None, temperature: float = 0.5) -> MagicMock:
            captured_temps.append(temperature)
            return _mock_llm("response")

        with patch("agents.pfc.get_llm", side_effect=fake_get_llm):
            pfc_node(state, llm=None)

        assert captured_temps, "get_llm was never called"
        assert captured_temps[0] == pytest.approx(0.1)

    def test_pfc_injects_directive_into_system_message(self):
        state = default_brain_state("I feel awful")
        state["llm_temperature"] = 0.1
        state["emotional_directive"] = "The user is in distress. Be compassionate."

        captured_messages: list = []
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = lambda msgs: (
            captured_messages.extend(msgs) or MagicMock(content="ok")
        )

        pfc_node(state, llm=mock_llm)

        system_msgs = [m for m in captured_messages if hasattr(m, "content") and "distress" in str(m.content)]
        assert system_msgs, "emotional_directive not found in system message"

    def test_pfc_no_directive_does_not_prepend_blank_line(self):
        state = default_brain_state("hello")
        state["llm_temperature"] = 0.7
        state["emotional_directive"] = ""

        captured_messages: list = []
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = lambda msgs: (
            captured_messages.extend(msgs) or MagicMock(content="ok")
        )

        pfc_node(state, llm=mock_llm)

        system_content = next(
            (str(m.content) for m in captured_messages if hasattr(m, "type") and m.type == "system"),
            None,
        )
        # With no directive, system prompt should start with "You are"
        if system_content:
            assert not system_content.startswith("\n")
