"""
main.py — application entry point.

Phase 1: stub LangGraph that passes BrainState through unchanged.
Each subsequent phase replaces the stub node with a real agent node.
"""
import logging

from langgraph.graph import END, StateGraph

from logging_config import logger  # noqa: F401 — configures logging on import
from models.brain_state import BrainState, default_brain_state

_log = logging.getLogger("janus.main")


# ── Stub node ─────────────────────────────────────────────────────────────────

def _stub_node(state: BrainState) -> BrainState:
    """Pass-through node.  Replaced agent-by-agent in later phases."""
    _log.info("stub_node received input: %r", state["input"])
    return state


# ── Graph factory ─────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Compile and return the LangGraph StateGraph.

    Phase 1: single stub node → END.
    Later phases add real agent nodes and conditional edges.
    """
    graph = StateGraph(BrainState)
    graph.add_node("stub", _stub_node)
    graph.set_entry_point("stub")
    graph.add_edge("stub", END)
    return graph.compile()


# ── Public API ────────────────────────────────────────────────────────────────

brain = build_graph()


def think(user_input: str) -> BrainState:
    """Run the brain graph for a single turn and return the final state."""
    state = default_brain_state(user_input)
    result: BrainState = brain.invoke(state)
    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Janus Process — Phase 1 stub  (Ctrl-C to exit)\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        result = think(user_input)
        response = result.get("final_response") or "(no response — agents not yet wired)"
        print(f"Brain: {response}\n")
