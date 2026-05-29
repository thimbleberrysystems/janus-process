"""
tests/test_phase15_bg_gate.py — Phase 15: Basal Ganglia Response Gate.

Verifies:
- Gate node accepts good responses (bg_feedback = None)
- Gate node requests refinement with feedback string
- proposal is appended to pfc_proposals on every gate call
- PFC incorporates bg_feedback into the context block
- Gate does not overwrite final_response
- Thalamus routes habit path to END (gate never called)
- Thalamus routes deliberate path through gate

All tests are pure unit tests — no external services required.
"""
from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from agents.basal_ganglia import basal_ganglia_gate_node
from agents.pfc import _build_context_block
from agents.thalamus import build_graph
from models.brain_state import default_brain_state


# ── Helpers ────────────────────────────────────────────────────────────────────

def _mock_llm(content: str) -> MagicMock:
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=content)
    return llm


def _state_with_proposal(proposal: str = "Some response.", **overrides) -> dict:
    s = dict(default_brain_state("test input"))
    s["final_response"] = proposal
    s.update(overrides)
    return s  # type: ignore[return-value]


# ── Gate node: decision parsing ────────────────────────────────────────────────

class TestGateNodeDecision:
    def test_accept_sets_bg_feedback_to_empty(self):
        state = _state_with_proposal()
        result = basal_ganglia_gate_node(state, llm=_mock_llm("ACCEPT"))
        assert not result["bg_feedback"]

    def test_accept_case_insensitive(self):
        state = _state_with_proposal()
        result = basal_ganglia_gate_node(state, llm=_mock_llm("accept"))
        assert not result["bg_feedback"]

    def test_refine_sets_feedback_string(self):
        state = _state_with_proposal()
        result = basal_ganglia_gate_node(
            state, llm=_mock_llm("REFINE: Be more concise.")
        )
        assert result["bg_feedback"] == "Be more concise."

    def test_refine_without_colon_still_captured(self):
        state = _state_with_proposal()
        result = basal_ganglia_gate_node(
            state, llm=_mock_llm("REFINE Be more empathetic please")
        )
        assert result["bg_feedback"] == "Be more empathetic please"

    def test_refine_empty_body_gets_default_message(self):
        state = _state_with_proposal()
        result = basal_ganglia_gate_node(state, llm=_mock_llm("REFINE:"))
        assert result["bg_feedback"] == "Please improve the response."

    def test_unknown_decision_treated_as_accept(self):
        state = _state_with_proposal()
        result = basal_ganglia_gate_node(state, llm=_mock_llm("LOOKS GOOD"))
        assert not result["bg_feedback"]


# ── Gate node: proposal tracking ──────────────────────────────────────────────

class TestGateNodeProposals:
    def test_proposal_appended_on_accept(self):
        state = _state_with_proposal("Great answer.")
        result = basal_ganglia_gate_node(state, llm=_mock_llm("ACCEPT"))
        assert "Great answer." in result["pfc_proposals"]

    def test_proposal_appended_on_refine(self):
        state = _state_with_proposal("Bad answer.")
        result = basal_ganglia_gate_node(
            state, llm=_mock_llm("REFINE: Too vague.")
        )
        assert "Bad answer." in result["pfc_proposals"]

    def test_proposals_accumulate_across_calls(self):
        state = _state_with_proposal("First answer.")
        state["pfc_proposals"] = ["zeroth answer."]
        result = basal_ganglia_gate_node(state, llm=_mock_llm("ACCEPT"))
        assert result["pfc_proposals"] == ["zeroth answer.", "First answer."]

    def test_final_response_not_overwritten_by_gate(self):
        state = _state_with_proposal("My response.")
        result = basal_ganglia_gate_node(state, llm=_mock_llm("ACCEPT"))
        assert result["final_response"] == "My response."

    def test_gate_does_not_mutate_input_state(self):
        state = _state_with_proposal()
        original_proposals = list(state["pfc_proposals"])
        basal_ganglia_gate_node(state, llm=_mock_llm("ACCEPT"))
        assert state["pfc_proposals"] == original_proposals


# ── PFC incorporates bg_feedback ───────────────────────────────────────────────

class TestPfcFeedbackInjection:
    def test_bg_feedback_appears_in_context_block(self):
        state = dict(default_brain_state("I need help."))
        state["bg_feedback"] = "Be more empathetic in your tone."
        block = _build_context_block(state)  # type: ignore[arg-type]
        assert "Refinement Instruction" in block
        assert "Be more empathetic in your tone." in block

    def test_no_feedback_section_when_bg_feedback_is_none(self):
        state = dict(default_brain_state("Hello."))
        state["bg_feedback"] = ""
        block = _build_context_block(state)  # type: ignore[arg-type]
        assert "Refinement Instruction" not in block

    def test_no_feedback_section_when_bg_feedback_is_missing(self):
        state = dict(default_brain_state("Hello."))
        state.pop("bg_feedback", None)
        block = _build_context_block(state)  # type: ignore[arg-type]
        assert "Refinement Instruction" not in block


# ── Thalamus routing ───────────────────────────────────────────────────────────

class TestThalamousGateRouting:
    """Integration-style tests using build_graph with mock callables."""

    def _build(self, amygdala_fn, bg_fn, basal_ganglia_fn=None, pfc_fn=None, hippocampus_fn=None):
        def _noop_amygdala(s):
            return {**s, "emotional_weight": 0.1, "llm_temperature": 0.7, "emotional_directive": ""}

        def _noop_bg(s):
            return {**s, "procedural_match": None}

        def _noop_hippocampus(s):
            return {**s, "retrieved_memories": []}

        def _noop_pfc(s):
            return {**s, "final_response": "response", "should_consolidate": False, "working_memory": []}

        return build_graph(
            amygdala=amygdala_fn or _noop_amygdala,
            basal_ganglia=basal_ganglia_fn or _noop_bg,
            hippocampus=hippocampus_fn or _noop_hippocampus,
            pfc=pfc_fn or _noop_pfc,
            bg_gate=bg_fn,
            trace=False,
        )

    def test_gate_called_on_deliberate_path(self):
        gate_calls: list = []

        def gate(s):
            gate_calls.append(s)
            return {**s, "bg_feedback": "", "pfc_proposals": [s.get("final_response", "")]}

        graph = self._build(amygdala_fn=None, bg_fn=gate)
        graph.invoke(default_brain_state("hello"))
        assert len(gate_calls) == 1

    def test_gate_skipped_on_habit_path(self):
        gate_calls: list = []

        def gate(s):
            gate_calls.append(s)
            return {**s, "bg_feedback": "", "pfc_proposals": []}

        def habit_bg(s):
            return {**s, "procedural_match": "Hello back!"}

        graph = self._build(amygdala_fn=None, bg_fn=gate, basal_ganglia_fn=habit_bg)
        graph.invoke(default_brain_state("hi"))
        assert len(gate_calls) == 0

    def test_gate_output_contains_bg_feedback_key(self):
        gate_outputs: list = []

        def gate(s):
            out = {**s, "bg_feedback": "", "pfc_proposals": [s.get("final_response", "")]}
            gate_outputs.append(out)
            return out

        graph = self._build(amygdala_fn=None, bg_fn=gate)
        graph.invoke(default_brain_state("hello"))
        assert "bg_feedback" in gate_outputs[0]
        assert not gate_outputs[0]["bg_feedback"]

    def test_gate_output_contains_pfc_proposals_key(self):
        gate_outputs: list = []

        def gate(s):
            out = {**s, "bg_feedback": "", "pfc_proposals": [s.get("final_response", "r")]}
            gate_outputs.append(out)
            return out

        graph = self._build(amygdala_fn=None, bg_fn=gate)
        graph.invoke(default_brain_state("hello"))
        assert isinstance(gate_outputs[0]["pfc_proposals"], list)
        assert len(gate_outputs[0]["pfc_proposals"]) >= 1
