"""
tests/test_phase16_recurrent_loop.py — Phase 16: PFC ↔ Basal Ganglia Recurrent Loop

Tests cover:
- Single ACCEPT exits after one PFC iteration.
- REFINE then ACCEPT loops once and exits.
- Exhausting MAX_PFC_LOOPS exits without an ACCEPT.
- pfc_proposals audit trail captures all attempts.
- bg_feedback is cleared by PFC on each iteration.
- Habit path bypasses the loop entirely.
- _route_after_bg_gate routing logic.
"""
from __future__ import annotations

from agents.thalamus import _route_after_bg_gate, build_graph
from config import MAX_PFC_LOOPS
from models.brain_state import BrainState, default_brain_state

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _noop_amygdala(s: BrainState) -> BrainState:
    return {**s, "emotional_weight": 0.0, "llm_temperature": 0.5, "emotional_directive": ""}


def _noop_bg(s: BrainState) -> BrainState:
    return {**s, "procedural_match": None}


def _noop_hippocampus(s: BrainState) -> BrainState:
    return {**s, "retrieved_memories": []}


def _noop_pfc(s: BrainState) -> BrainState:
    loop_count = int(s.get("loop_count") or 0) + 1
    return {**s, "final_response": "PFC answer", "loop_count": loop_count,
            "bg_feedback": "", "working_memory": [], "should_consolidate": False}


def _build(
    *,
    bg_fn=None,
    pfc_fn=None,
    amygdala_fn=None,
    basal_ganglia_fn=None,
):
    return build_graph(
        amygdala=amygdala_fn or _noop_amygdala,
        basal_ganglia=basal_ganglia_fn or _noop_bg,
        hippocampus=_noop_hippocampus,
        pfc=pfc_fn or _noop_pfc,
        bg_gate=bg_fn,
        trace=False,
    )


# ---------------------------------------------------------------------------
# _route_after_bg_gate unit tests
# ---------------------------------------------------------------------------

class TestRouteAfterBgGate:
    def test_accept_exits(self):
        state = default_brain_state("hi")
        state["bg_feedback"] = ""
        state["loop_count"] = 1
        assert _route_after_bg_gate(state) == "end"

    def test_accept_none_exits(self):
        """None bg_feedback is also treated as accepted."""
        state = default_brain_state("hi")
        # bg_feedback not set at all
        state.pop("bg_feedback", None)
        state["loop_count"] = 1
        assert _route_after_bg_gate(state) == "end"

    def test_refine_within_budget_loops(self):
        state = default_brain_state("hi")
        state["bg_feedback"] = "Be more concise."
        state["loop_count"] = 1  # 1 < MAX_PFC_LOOPS (2)
        assert _route_after_bg_gate(state) == "pfc"

    def test_refine_at_max_budget_exits(self):
        state = default_brain_state("hi")
        state["bg_feedback"] = "Still needs work."
        state["loop_count"] = MAX_PFC_LOOPS  # budget exhausted
        assert _route_after_bg_gate(state) == "end"

    def test_refine_over_budget_exits(self):
        state = default_brain_state("hi")
        state["bg_feedback"] = "Still needs work."
        state["loop_count"] = MAX_PFC_LOOPS + 1
        assert _route_after_bg_gate(state) == "end"


# ---------------------------------------------------------------------------
# Integration: loop behaviour via build_graph
# ---------------------------------------------------------------------------

class TestRecurrentLoopIntegration:
    def test_single_accept_pfc_invoked_once(self):
        """Gate immediately accepts → PFC runs exactly once."""
        pfc_calls: list = []

        def pfc(s):
            pfc_calls.append(s)
            loop_count = int(s.get("loop_count") or 0) + 1
            return {**s, "final_response": "answer", "loop_count": loop_count,
                    "bg_feedback": "", "working_memory": [], "should_consolidate": False}

        def gate(s):
            return {**s, "bg_feedback": "", "pfc_proposals": [s.get("final_response", "")]}

        graph = _build(pfc_fn=pfc, bg_fn=gate)
        graph.invoke(default_brain_state("hello"))
        assert len(pfc_calls) == 1

    def test_refine_then_accept_pfc_invoked_twice(self):
        """Gate returns REFINE on first call, ACCEPT on second → PFC runs twice."""
        pfc_calls: list = []
        gate_calls: list = []

        def pfc(s):
            pfc_calls.append(s)
            loop_count = int(s.get("loop_count") or 0) + 1
            return {**s, "final_response": f"answer{loop_count}", "loop_count": loop_count,
                    "bg_feedback": "", "working_memory": [], "should_consolidate": False}

        def gate(s):
            gate_calls.append(s)
            if len(gate_calls) == 1:
                return {**s, "bg_feedback": "Be more concise.", "pfc_proposals": [s.get("final_response", "")]}
            return {**s, "bg_feedback": "", "pfc_proposals": list(s.get("pfc_proposals") or []) + [s.get("final_response", "")]}

        graph = _build(pfc_fn=pfc, bg_fn=gate)
        graph.invoke(default_brain_state("hello"))
        assert len(pfc_calls) == 2
        assert len(gate_calls) == 2

    def test_max_loops_exits_without_accept(self):
        """Gate always returns REFINE → exits after MAX_PFC_LOOPS iterations."""
        pfc_calls: list = []

        def pfc(s):
            pfc_calls.append(s)
            loop_count = int(s.get("loop_count") or 0) + 1
            return {**s, "final_response": "answer", "loop_count": loop_count,
                    "bg_feedback": "", "working_memory": [], "should_consolidate": False}

        def gate(s):
            return {**s, "bg_feedback": "Needs improvement.", "pfc_proposals": [s.get("final_response", "")]}

        graph = _build(pfc_fn=pfc, bg_fn=gate)
        graph.invoke(default_brain_state("hello"))
        assert len(pfc_calls) == MAX_PFC_LOOPS

    def test_pfc_proposals_captures_all_attempts(self):
        """All PFC proposals are accumulated in pfc_proposals."""
        proposals_seen: list = []

        def pfc(s):
            loop_count = int(s.get("loop_count") or 0) + 1
            return {**s, "final_response": f"attempt{loop_count}", "loop_count": loop_count,
                    "bg_feedback": "", "working_memory": [], "should_consolidate": False}

        def gate(s):
            proposals = list(s.get("pfc_proposals") or []) + [s.get("final_response", "")]
            proposals_seen.append(list(proposals))
            if len(proposals_seen) == 1:
                return {**s, "bg_feedback": "Try again.", "pfc_proposals": proposals}
            return {**s, "bg_feedback": "", "pfc_proposals": proposals}

        graph = _build(pfc_fn=pfc, bg_fn=gate)
        graph.invoke(default_brain_state("hello"))
        # Gate was called twice; final proposals list should have 2 entries
        assert len(proposals_seen[-1]) == 2
        assert proposals_seen[-1][0] == "attempt1"
        assert proposals_seen[-1][1] == "attempt2"

    def test_loop_count_increments_per_pfc_call(self):
        """loop_count grows by 1 on each PFC invocation."""
        counts_at_entry: list[int] = []

        def pfc(s):
            counts_at_entry.append(int(s.get("loop_count") or 0))
            loop_count = int(s.get("loop_count") or 0) + 1
            return {**s, "final_response": "ans", "loop_count": loop_count,
                    "bg_feedback": "", "working_memory": [], "should_consolidate": False}

        call_n = [0]

        def gate(s):
            call_n[0] += 1
            if call_n[0] == 1:
                return {**s, "bg_feedback": "Refine it.", "pfc_proposals": []}
            return {**s, "bg_feedback": "", "pfc_proposals": []}

        graph = _build(pfc_fn=pfc, bg_fn=gate)
        graph.invoke(default_brain_state("hello"))
        # PFC was invoked twice; loop_count at entry was 0 then 1
        assert counts_at_entry == [0, 1]

    def test_bg_feedback_cleared_by_pfc(self):
        """PFC returns bg_feedback='' so gate always sees a fresh slate."""
        pfc_outputs: list = []

        def pfc(s):
            loop_count = int(s.get("loop_count") or 0) + 1
            out = {**s, "final_response": "ans", "loop_count": loop_count,
                   "bg_feedback": "", "working_memory": [], "should_consolidate": False}
            pfc_outputs.append(out)
            return out

        call_n = [0]

        def gate(s):
            call_n[0] += 1
            if call_n[0] == 1:
                return {**s, "bg_feedback": "Fix it.", "pfc_proposals": []}
            return {**s, "bg_feedback": "", "pfc_proposals": []}

        graph = _build(pfc_fn=pfc, bg_fn=gate)
        graph.invoke(default_brain_state("hello"))
        # Each PFC output should have empty bg_feedback
        for out in pfc_outputs:
            assert not out["bg_feedback"]

    def test_habit_path_bypasses_loop_entirely(self):
        """Habit match → PFC runs once, gate never called, loop_count stays 1."""
        gate_calls: list = []
        pfc_calls: list = []

        def pfc(s):
            pfc_calls.append(s)
            loop_count = int(s.get("loop_count") or 0) + 1
            return {**s, "final_response": "habit ans", "loop_count": loop_count,
                    "bg_feedback": "", "working_memory": [], "should_consolidate": False}

        def gate(s):
            gate_calls.append(s)
            return {**s, "bg_feedback": "", "pfc_proposals": []}

        def habit_bg(s):
            return {**s, "procedural_match": "Habit response!"}

        graph = _build(pfc_fn=pfc, bg_fn=gate, basal_ganglia_fn=habit_bg)
        graph.invoke(default_brain_state("hi"))
        assert len(gate_calls) == 0
        assert len(pfc_calls) == 1
