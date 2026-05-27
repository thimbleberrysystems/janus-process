"""tests/test_phase8_thalamus.py — Unit tests for the Thalamus Router."""

from __future__ import annotations

from typing import Any, Dict

import pytest

from agents.thalamus import HIPPOCAMPUS_THRESHOLD, build_graph, _route_after_basal_ganglia
from models.brain_state import default_brain_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class NodeTracker:
    """Lightweight callable that tracks invocations and injects state updates.

    Use this in place of a real agent node to verify routing without any LLM
    or service calls.
    """

    def __init__(self, updates: Dict[str, Any] | None = None) -> None:
        self.called = False
        self.call_count = 0
        self.last_received_state: dict | None = None
        self._updates: Dict[str, Any] = updates or {}

    def __call__(self, state: dict) -> dict:
        self.called = True
        self.call_count += 1
        self.last_received_state = dict(state)
        return {**state, **self._updates}


def _make_nodes(
    *,
    emotional_weight: float = 0.0,
    procedural_match: str | None = None,
    final_response: str = "mock response",
):
    """Return a (amygdala, basal_ganglia, hippocampus, pfc) tuple of NodeTrackers.

    The amygdala tracker sets emotional_weight; basal_ganglia sets
    procedural_match; pfc sets final_response.
    """
    amygdala = NodeTracker(updates={"emotional_weight": emotional_weight})
    basal_ganglia = NodeTracker(updates={"procedural_match": procedural_match})
    hippocampus = NodeTracker()
    pfc = NodeTracker(updates={"final_response": final_response, "should_consolidate": False})
    return amygdala, basal_ganglia, hippocampus, pfc


# ---------------------------------------------------------------------------
# Routing-function unit tests (no graph compilation)
# ---------------------------------------------------------------------------

class TestRouteAfterBasalGanglia:
    def test_habit_match_routes_to_pfc(self):
        state = {**default_brain_state("hi"), "procedural_match": "Hello!", "emotional_weight": 0.9}
        assert _route_after_basal_ganglia(state) == "pfc"

    def test_high_emotion_no_match_routes_to_hippocampus(self):
        state = {**default_brain_state("hi"), "procedural_match": None, "emotional_weight": 0.9}
        assert _route_after_basal_ganglia(state) == "hippocampus"

    def test_low_emotion_no_match_routes_to_pfc(self):
        state = {**default_brain_state("hi"), "procedural_match": None, "emotional_weight": 0.2}
        assert _route_after_basal_ganglia(state) == "pfc"

    def test_exact_threshold_routes_to_hippocampus(self):
        state = {
            **default_brain_state("hi"),
            "procedural_match": None,
            "emotional_weight": HIPPOCAMPUS_THRESHOLD,
        }
        assert _route_after_basal_ganglia(state) == "hippocampus"

    def test_just_below_threshold_routes_to_pfc(self):
        state = {
            **default_brain_state("hi"),
            "procedural_match": None,
            "emotional_weight": HIPPOCAMPUS_THRESHOLD - 0.01,
        }
        assert _route_after_basal_ganglia(state) == "pfc"


# ---------------------------------------------------------------------------
# Graph tests
# ---------------------------------------------------------------------------

class TestGraphCompiles:
    def test_graph_compiles_with_real_nodes(self):
        """build_graph() must return a runnable without raising."""
        compiled = build_graph()
        assert compiled is not None
        assert hasattr(compiled, "invoke"), "Compiled graph must have an invoke() method"

    def test_graph_compiles_with_mock_nodes(self):
        amygdala, basal_ganglia, hippocampus, pfc = _make_nodes()
        compiled = build_graph(
            amygdala=amygdala,
            basal_ganglia=basal_ganglia,
            hippocampus=hippocampus,
            pfc=pfc,
        )
        assert hasattr(compiled, "invoke")


class TestRoutingIntegration:
    def test_low_emotion_routes_to_pfc_direct(self):
        """score=0.2, no habit match → Hippocampus must NOT be invoked."""
        amygdala, basal_ganglia, hippocampus, pfc = _make_nodes(
            emotional_weight=0.2, procedural_match=None
        )
        compiled = build_graph(
            amygdala=amygdala,
            basal_ganglia=basal_ganglia,
            hippocampus=hippocampus,
            pfc=pfc,
        )
        initial = default_brain_state("what time is it?")
        compiled.invoke(initial)

        assert amygdala.called, "Amygdala must always run"
        assert basal_ganglia.called, "Basal Ganglia must always run"
        assert not hippocampus.called, "Hippocampus must NOT run on low-emotion, no-habit path"
        assert pfc.called, "PFC must always run"

    def test_high_emotion_routes_via_hippocampus(self):
        """score=0.9, no habit match → Hippocampus IS invoked before PFC."""
        amygdala, basal_ganglia, hippocampus, pfc = _make_nodes(
            emotional_weight=0.9, procedural_match=None
        )
        compiled = build_graph(
            amygdala=amygdala,
            basal_ganglia=basal_ganglia,
            hippocampus=hippocampus,
            pfc=pfc,
        )
        initial = default_brain_state("I just lost my job")
        compiled.invoke(initial)

        assert hippocampus.called, "Hippocampus must run on high-emotion path"
        assert pfc.called, "PFC must run after Hippocampus"

    def test_habit_match_routes_via_basal_ganglia(self):
        """Procedural match found → Basal Ganglia IS invoked; Hippocampus skipped."""
        amygdala, basal_ganglia, hippocampus, pfc = _make_nodes(
            emotional_weight=0.3, procedural_match="Hello! How can I help you?"
        )
        compiled = build_graph(
            amygdala=amygdala,
            basal_ganglia=basal_ganglia,
            hippocampus=hippocampus,
            pfc=pfc,
        )
        initial = default_brain_state("hello there")
        compiled.invoke(initial)

        assert basal_ganglia.called, "Basal Ganglia must run"
        assert not hippocampus.called, "Hippocampus must be skipped when a habit is matched"
        assert pfc.called, "PFC must run after Basal Ganglia"

    def test_habit_match_with_high_emotion_still_skips_hippocampus(self):
        """Habit match takes priority over high emotion — no RAG retrieval needed."""
        amygdala, basal_ganglia, hippocampus, pfc = _make_nodes(
            emotional_weight=0.95, procedural_match="Bye!"
        )
        compiled = build_graph(
            amygdala=amygdala,
            basal_ganglia=basal_ganglia,
            hippocampus=hippocampus,
            pfc=pfc,
        )
        compiled.invoke(default_brain_state("goodbye"))
        assert not hippocampus.called


class TestFullGraphEndToEnd:
    def test_full_graph_sets_final_response(self):
        """All nodes mocked; graph must return a state with final_response set."""
        amygdala, basal_ganglia, hippocampus, pfc = _make_nodes(
            emotional_weight=0.8,
            procedural_match=None,
            final_response="Here is your answer.",
        )
        compiled = build_graph(
            amygdala=amygdala,
            basal_ganglia=basal_ganglia,
            hippocampus=hippocampus,
            pfc=pfc,
        )
        result = compiled.invoke(default_brain_state("tell me something"))

        assert result["final_response"] == "Here is your answer."

    def test_full_graph_all_nodes_run_on_high_emotion(self):
        """High emotion: all four nodes must be invoked."""
        amygdala, basal_ganglia, hippocampus, pfc = _make_nodes(
            emotional_weight=0.85,
            procedural_match=None,
        )
        compiled = build_graph(
            amygdala=amygdala,
            basal_ganglia=basal_ganglia,
            hippocampus=hippocampus,
            pfc=pfc,
        )
        compiled.invoke(default_brain_state("I am in crisis"))

        assert amygdala.call_count == 1
        assert basal_ganglia.call_count == 1
        assert hippocampus.call_count == 1
        assert pfc.call_count == 1

    def test_full_graph_low_emotion_no_habit_three_nodes(self):
        """Low emotion + no habit: amygdala, basal_ganglia, pfc — NOT hippocampus."""
        amygdala, basal_ganglia, hippocampus, pfc = _make_nodes(
            emotional_weight=0.1,
            procedural_match=None,
        )
        compiled = build_graph(
            amygdala=amygdala,
            basal_ganglia=basal_ganglia,
            hippocampus=hippocampus,
            pfc=pfc,
        )
        compiled.invoke(default_brain_state("what is 2 + 2?"))

        assert amygdala.call_count == 1
        assert basal_ganglia.call_count == 1
        assert hippocampus.call_count == 0
        assert pfc.call_count == 1


class TestStateFlowBetweenNodes:
    def test_graph_state_flows_through_nodes(self):
        """Each node must receive the state returned by the previous node."""
        received: list[dict] = []

        class RecordingNode:
            def __init__(self, updates: dict) -> None:
                self._updates = updates

            def __call__(self, state: dict) -> dict:
                received.append(dict(state))
                return {**state, **self._updates}

        amygdala_node = RecordingNode({"emotional_weight": 0.9})
        basal_ganglia_node = RecordingNode({"procedural_match": None})
        hippocampus_node = RecordingNode({"retrieved_memories": ["memory A"]})
        pfc_node = RecordingNode({"final_response": "done", "should_consolidate": True})

        compiled = build_graph(
            amygdala=amygdala_node,
            basal_ganglia=basal_ganglia_node,
            hippocampus=hippocampus_node,
            pfc=pfc_node,
        )
        compiled.invoke(default_brain_state("test input"))

        # Four nodes ran (amygdala, basal_ganglia, hippocampus, pfc)
        assert len(received) == 4, f"Expected 4 node invocations, got {len(received)}"

        # Basal Ganglia must see the emotional_weight set by Amygdala.
        assert received[1]["emotional_weight"] == 0.9

        # Hippocampus must see the procedural_match (None) set by Basal Ganglia.
        assert received[2]["procedural_match"] is None
        assert received[2]["emotional_weight"] == 0.9

        # PFC must see retrieved_memories set by Hippocampus.
        assert received[3]["retrieved_memories"] == ["memory A"]

    def test_state_input_preserved_throughout(self):
        """The original 'input' field must survive unmodified through all nodes."""
        original_input = "preserve this text"
        captured_final: list[dict] = []

        class PassthroughNode:
            def __init__(self, updates: dict) -> None:
                self._updates = updates

            def __call__(self, state: dict) -> dict:
                return {**state, **self._updates}

        pfc_node = PassthroughNode({"final_response": "ok", "should_consolidate": False})

        compiled = build_graph(
            amygdala=PassthroughNode({"emotional_weight": 0.1}),
            basal_ganglia=PassthroughNode({"procedural_match": None}),
            hippocampus=PassthroughNode({}),
            pfc=pfc_node,
        )
        result = compiled.invoke(default_brain_state(original_input))
        assert result["input"] == original_input


# ---------------------------------------------------------------------------
# Integration tests (require live LLM)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestThalamusIntegration:
    def test_graph_invoke_with_real_agents(self):
        """Run the full graph with real agents against live services."""
        compiled = build_graph()
        result = compiled.invoke(default_brain_state("what is the capital of France?"))

        assert isinstance(result.get("emotional_weight"), float)
        assert isinstance(result.get("final_response"), str)
        assert len(result["final_response"]) > 0

    def test_high_emotion_input_populates_retrieved_memories(self):
        """High-emotion input should trigger Hippocampus and populate retrieved_memories."""
        compiled = build_graph()
        result = compiled.invoke(default_brain_state("I am really anxious and scared right now"))

        # Hippocampus ran → field is a list (may be empty on cold start)
        assert isinstance(result.get("retrieved_memories"), list)
